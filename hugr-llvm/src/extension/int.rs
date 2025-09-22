use hugr_core::{
    HugrView, Node,
    extension::{
        prelude::{ConstError, sum_with_error},
        simple_op::MakeExtensionOp,
    },
    ops::{ExtensionOp, Value, constant::CustomConst},
    std_extensions::arithmetic::{
        int_ops::IntOpDef,
        int_types::{self, ConstInt},
    },
    types::{CustomType, Type, TypeArg},
};
use inkwell::{
    IntPredicate,
    types::{BasicType, BasicTypeEnum, IntType},
    values::{BasicValue, BasicValueEnum, IntValue},
};
use std::sync::LazyLock;

use crate::{
    CodegenExtension,
    custom::CodegenExtsBuilder,
    emit::{
        EmitOpArgs, emit_value,
        func::EmitFuncContext,
        get_intrinsic,
        ops::{emit_custom_binary_op, emit_custom_unary_op},
    },
    sum::{LLVMSumType, LLVMSumValue},
    types::{HugrSumType, TypingSession},
};

use anyhow::{Result, anyhow, bail};

use super::{DefaultPreludeCodegen, PreludeCodegen, conversions::int_type_bounds};

#[derive(Clone, Debug, Default)]
pub struct IntCodegenExtension<PCG>(PCG);

impl<PCG: PreludeCodegen> IntCodegenExtension<PCG> {
    pub fn new(ccg: PCG) -> Self {
        Self(ccg)
    }
}

impl<CCG: PreludeCodegen> From<CCG> for IntCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}

impl<CCG: PreludeCodegen> CodegenExtension for IntCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_const(emit_const_int)
            .custom_type((int_types::EXTENSION_ID, "int".into()), llvm_type)
            .simple_extension_op::<IntOpDef>(move |context, args, op| {
                emit_int_op(context, &self.0, args, op)
            })
    }
}

static ERR_NARROW: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Can't narrow into bounds".to_string(),
});
static ERR_IU_TO_S: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "iu_to_s argument out of bounds".to_string(),
});
static ERR_IS_TO_U: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "is_to_u called on negative value".to_string(),
});
static ERR_DIV_0: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Attempted division by 0".to_string(),
});

#[derive(Debug, Eq, PartialEq)]
enum DivOrMod {
    Div,
    Mod,
    DivMod,
}

struct DivModOp {
    op: DivOrMod,
    signed: bool,
    panic: bool,
}

impl DivModOp {
    fn emit<'c, H: HugrView<Node = Node>>(
        self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        pcg: &impl PreludeCodegen,
        log_width: u64,
        numerator: IntValue<'c>,
        denominator: IntValue<'c>,
    ) -> Result<Vec<BasicValueEnum<'c>>> {
        // Hugr semantics say that div and mod are equivalent to doing a divmod,
        // then projecting out an element from the pair, so that's what we do.
        let quotrem = make_divmod(
            ctx,
            pcg,
            log_width,
            numerator,
            denominator,
            self.panic,
            self.signed,
        )?;

        if self.op == DivOrMod::DivMod {
            if self.panic {
                // Unpack the tuple into two values.
                Ok(quotrem.build_untag(ctx.builder(), 0).unwrap())
            } else {
                Ok(vec![quotrem.as_basic_value_enum()])
            }
        } else {
            // Which field we should project out from the result of divmod.
            let index = match self.op {
                DivOrMod::Div => 0,
                DivOrMod::Mod => 1,
                _ => unreachable!(),
            };
            // If we emitted a panicking divmod, the result is just a tuple type.
            if self.panic {
                Ok(vec![
                    quotrem
                        .build_untag(ctx.builder(), 0)?
                        .into_iter()
                        .nth(index)
                        .unwrap(),
                ])
            }
            // Otherwise, we have a sum type `err + [int,int]`, which we need to
            // turn into a `err + int`.
            else {
                // Get the data out the the divmod result.
                let int_ty = numerator.get_type().as_basic_type_enum();
                let tuple_ty =
                    LLVMSumType::try_new(ctx.iw_context(), vec![vec![int_ty, int_ty]]).unwrap();
                let tuple = quotrem
                    .build_untag(ctx.builder(), 1)?
                    .into_iter()
                    .next()
                    .unwrap();
                let tuple_val = LLVMSumValue::try_new(tuple, tuple_ty)?;
                let data_val = tuple_val
                    .build_untag(ctx.builder(), 0)?
                    .into_iter()
                    .nth(index)
                    .unwrap();
                let err_val = quotrem
                    .build_untag(ctx.builder(), 0)?
                    .into_iter()
                    .next()
                    .unwrap();

                let tag_val = quotrem.build_get_tag(ctx.builder())?;
                tag_val.set_name("tag");

                // Build a new struct with the old tag and error data.
                let int_ty = int_types::INT_TYPES[log_width as usize].clone();
                let out_ty = LLVMSumType::try_from_hugr_type(
                    &ctx.typing_session(),
                    sum_with_error(vec![int_ty.clone()]),
                )
                .unwrap();

                let data_variant = out_ty.build_tag(ctx.builder(), 1, vec![data_val])?;
                data_variant.set_name("data_variant");
                let err_variant = out_ty.build_tag(ctx.builder(), 0, vec![err_val])?;
                err_variant.set_name("err_variant");

                let result = ctx
                    .builder()
                    .build_select(tag_val, data_variant, err_variant, "")?;
                Ok(vec![result])
            }
        }
    }
}

/// `ConstError` an integer comparison operation.
fn emit_icmp<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    pred: inkwell::IntPredicate,
) -> Result<()> {
    let true_val = emit_value(context, &Value::true_val())?;
    let false_val = emit_value(context, &Value::false_val())?;

    emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
        // get result as an i1
        let r = ctx.builder().build_int_compare(
            pred,
            lhs.into_int_value(),
            rhs.into_int_value(),
            "",
        )?;
        // convert to whatever bool_t is
        Ok(vec![
            ctx.builder().build_select(r, true_val, false_val, "")?,
        ])
    })
}

/// Emit an ipow operation. This isn't directly supported in llvm, so we do a
/// loop over the exponent, performing `imul`s instead.
/// The insertion pointer is expected to be pointing to the end of `launch_bb`.
fn emit_ipow<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
) -> Result<()> {
    emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
        let done_bb = ctx.new_basic_block("done", None);
        let pow_body_bb = ctx.new_basic_block("pow_body", Some(done_bb));
        let return_one_bb = ctx.new_basic_block("power_of_zero", Some(pow_body_bb));
        let pow_bb = ctx.new_basic_block("pow", Some(return_one_bb));

        let acc_p = ctx.builder().build_alloca(lhs.get_type(), "acc_ptr")?;
        let exp_p = ctx.builder().build_alloca(rhs.get_type(), "exp_ptr")?;
        ctx.builder().build_store(acc_p, lhs)?;
        ctx.builder().build_store(exp_p, rhs)?;
        ctx.builder().build_unconditional_branch(pow_bb)?;

        let zero = rhs.get_type().into_int_type().const_int(0, false);
        // Assumes RHS type is the same as output type (which it should be)
        let one = rhs.get_type().into_int_type().const_int(1, false);

        // Block for just returning one
        ctx.builder().position_at_end(return_one_bb);
        ctx.builder().build_store(acc_p, one)?;
        ctx.builder().build_unconditional_branch(done_bb)?;

        ctx.builder().position_at_end(pow_bb);
        let acc = ctx.builder().build_load(acc_p, "acc")?;
        let exp = ctx.builder().build_load(exp_p, "exp")?;

        // Special case if the exponent is 0 or 1
        ctx.builder().build_switch(
            exp.into_int_value(),
            pow_body_bb,
            &[(one, done_bb), (zero, return_one_bb)],
        )?;

        // Block that performs one `imul` and modifies the values in the store
        ctx.builder().position_at_end(pow_body_bb);
        let new_acc =
            ctx.builder()
                .build_int_mul(acc.into_int_value(), lhs.into_int_value(), "new_acc")?;
        let new_exp = ctx
            .builder()
            .build_int_sub(exp.into_int_value(), one, "new_exp")?;
        ctx.builder().build_store(acc_p, new_acc)?;
        ctx.builder().build_store(exp_p, new_exp)?;
        ctx.builder().build_unconditional_branch(pow_bb)?;

        ctx.builder().position_at_end(done_bb);
        let result = ctx.builder().build_load(acc_p, "result")?;
        Ok(vec![result.as_basic_value_enum()])
    })
}

fn emit_int_op<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    pcg: &impl PreludeCodegen,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: IntOpDef,
) -> Result<()> {
    match op {
        IntOpDef::iadd => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::imul => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::isub => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::idiv_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Div,
                    signed: true,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idiv_u => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Div,
                    signed: false,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::imod_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Mod,
                    signed: true,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::imod_u => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Mod,
                    signed: false,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idivmod_u => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::DivMod,
                    signed: false,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idivmod_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::DivMod,
                    signed: true,
                    panic: true,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idiv_checked_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Div,
                    signed: true,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idiv_checked_u => {
            let log_width = get_width_arg(&args, &op)?;

            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Div,
                    signed: false,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::imod_checked_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Mod,
                    signed: true,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::imod_checked_u => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::Mod,
                    signed: false,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idivmod_checked_u => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::DivMod,
                    signed: false,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::idivmod_checked_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let op = DivModOp {
                    op: DivOrMod::DivMod,
                    signed: true,
                    panic: false,
                };
                op.emit(
                    ctx,
                    pcg,
                    log_width,
                    lhs.into_int_value(),
                    rhs.into_int_value(),
                )
            })
        }
        IntOpDef::ineg => emit_custom_unary_op(context, args, |ctx, arg, _| {
            Ok(vec![
                ctx.builder()
                    .build_int_neg(arg.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::iabs => emit_custom_unary_op(context, args, |ctx, arg, _| {
            let intr = get_intrinsic(
                ctx.get_current_module(),
                "llvm.abs.i64",
                [ctx.iw_context().i64_type().as_basic_type_enum()],
            )?;
            let true_ = ctx.iw_context().bool_type().const_all_ones();
            let r = ctx
                .builder()
                .build_call(intr, &[arg.into_int_value().into(), true_.into()], "")?
                .try_as_basic_value()
                .unwrap_left();
            Ok(vec![r])
        }),
        IntOpDef::imax_s => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            let intr = get_intrinsic(
                ctx.get_current_module(),
                "llvm.smax.i64",
                [ctx.iw_context().i64_type().as_basic_type_enum()],
            )?;
            let r = ctx
                .builder()
                .build_call(
                    intr,
                    &[lhs.into_int_value().into(), rhs.into_int_value().into()],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_left();
            Ok(vec![r])
        }),
        IntOpDef::imax_u => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            let intr = get_intrinsic(
                ctx.get_current_module(),
                "llvm.umax.i64",
                [ctx.iw_context().i64_type().as_basic_type_enum()],
            )?;
            let r = ctx
                .builder()
                .build_call(
                    intr,
                    &[lhs.into_int_value().into(), rhs.into_int_value().into()],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_left();
            Ok(vec![r])
        }),
        IntOpDef::imin_s => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            let intr = get_intrinsic(
                ctx.get_current_module(),
                "llvm.smin.i64",
                [ctx.iw_context().i64_type().as_basic_type_enum()],
            )?;
            let r = ctx
                .builder()
                .build_call(
                    intr,
                    &[lhs.into_int_value().into(), rhs.into_int_value().into()],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_left();
            Ok(vec![r])
        }),
        IntOpDef::imin_u => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            let intr = get_intrinsic(
                ctx.get_current_module(),
                "llvm.umin.i64",
                [ctx.iw_context().i64_type().as_basic_type_enum()],
            )?;
            let r = ctx
                .builder()
                .build_call(
                    intr,
                    &[lhs.into_int_value().into(), rhs.into_int_value().into()],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_left();
            Ok(vec![r])
        }),
        IntOpDef::ishl => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::ishr => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), false, "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::ieq => emit_icmp(context, args, inkwell::IntPredicate::EQ),
        IntOpDef::ine => emit_icmp(context, args, inkwell::IntPredicate::NE),
        IntOpDef::ilt_s => emit_icmp(context, args, inkwell::IntPredicate::SLT),
        IntOpDef::igt_s => emit_icmp(context, args, inkwell::IntPredicate::SGT),
        IntOpDef::ile_s => emit_icmp(context, args, inkwell::IntPredicate::SLE),
        IntOpDef::ige_s => emit_icmp(context, args, inkwell::IntPredicate::SGE),
        IntOpDef::ilt_u => emit_icmp(context, args, inkwell::IntPredicate::ULT),
        IntOpDef::igt_u => emit_icmp(context, args, inkwell::IntPredicate::UGT),
        IntOpDef::ile_u => emit_icmp(context, args, inkwell::IntPredicate::ULE),
        IntOpDef::ige_u => emit_icmp(context, args, inkwell::IntPredicate::UGE),
        IntOpDef::ixor => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_xor(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::ior => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_or(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::inot => emit_custom_unary_op(context, args, |ctx, arg, _| {
            Ok(vec![
                ctx.builder()
                    .build_not(arg.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::iand => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_and(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::ipow => emit_ipow(context, args),
        // Type args are width of input, width of output
        IntOpDef::iwiden_u => emit_custom_unary_op(context, args, |ctx, arg, outs| {
            let [out] = outs.try_into()?;
            Ok(vec![
                ctx.builder()
                    .build_int_cast_sign_flag(arg.into_int_value(), out.into_int_type(), false, "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::iwiden_s => emit_custom_unary_op(context, args, |ctx, arg, outs| {
            let [out] = outs.try_into()?;

            Ok(vec![
                ctx.builder()
                    .build_int_cast_sign_flag(arg.into_int_value(), out.into_int_type(), true, "")?
                    .as_basic_value_enum(),
            ])
        }),
        IntOpDef::inarrow_s => {
            let Some(TypeArg::BoundedNat(out_log_width)) = args.node().args().last().cloned()
            else {
                bail!("Type arg to inarrow_s wasn't a Nat");
            };
            let (_, out_ty) = args.node.out_value_types().next().unwrap();
            emit_custom_unary_op(context, args, |ctx, arg, outs| {
                let result = make_narrow(
                    ctx,
                    arg,
                    outs,
                    out_log_width,
                    true,
                    out_ty.as_sum().unwrap().clone(),
                )?;
                Ok(vec![result])
            })
        }
        IntOpDef::inarrow_u => {
            let Some(TypeArg::BoundedNat(out_log_width)) = args.node().args().last().cloned()
            else {
                bail!("Type arg to inarrow_u wasn't a Nat");
            };
            let (_, out_ty) = args.node.out_value_types().next().unwrap();
            emit_custom_unary_op(context, args, |ctx, arg, outs| {
                let result = make_narrow(
                    ctx,
                    arg,
                    outs,
                    out_log_width,
                    false,
                    out_ty.as_sum().unwrap().clone(),
                )?;
                Ok(vec![result])
            })
        }
        IntOpDef::iu_to_s => {
            let log_width = get_width_arg(&args, &op)?;
            emit_custom_unary_op(context, args, |ctx, arg, _| {
                let (_, max_val, _) = int_type_bounds(u32::pow(2, log_width as u32));
                let max = arg
                    .get_type()
                    .into_int_type()
                    .const_int(max_val as u64, false);

                let within_bounds = ctx.builder().build_int_compare(
                    IntPredicate::ULE,
                    arg.into_int_value(),
                    max,
                    "bounds_check",
                )?;

                Ok(vec![val_or_panic(
                    ctx,
                    pcg,
                    within_bounds,
                    &ERR_IU_TO_S,
                    |_| Ok(arg),
                )?])
            })
        }
        IntOpDef::is_to_u => emit_custom_unary_op(context, args, |ctx, arg, _| {
            let zero = arg.get_type().into_int_type().const_zero();

            let within_bounds = ctx.builder().build_int_compare(
                IntPredicate::SGE,
                arg.into_int_value(),
                zero,
                "bounds_check",
            )?;

            Ok(vec![val_or_panic(
                ctx,
                pcg,
                within_bounds,
                &ERR_IS_TO_U,
                |_| Ok(arg),
            )?])
        }),
        _ => Err(anyhow!("IntOpEmitter: unimplemented op: {}", op.op_id())),
    }
}

// Helper to get the log width arg to an int op when it's the only argument
// panic if there's not exactly one nat arg
pub(crate) fn get_width_arg<H: HugrView<Node = Node>>(
    args: &EmitOpArgs<'_, '_, ExtensionOp, H>,
    op: &impl MakeExtensionOp,
) -> Result<u64> {
    let [TypeArg::BoundedNat(log_width)] = args.node.args() else {
        bail!(
            "Expected exactly one BoundedNat parameter to {}",
            op.op_id()
        )
    };
    Ok(*log_width)
}

// The semantics of the hugr operation specify that the divisor argument is
// always unsigned, and the signed/unsigned variants affect the types of the
// dividend and quotient only.
//
// LLVM's semantics for `srem`, however, have both operands being the same type.
// Moreover, llvm's `srem` does not implement the modulo operation: the
// remainder will have the same sign as the dividend instead of the divisor.
//
// See discussion at: https://github.com/CQCL/hugr/pull/2025#discussion_r2012537992
fn make_divmod<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    pcg: &impl PreludeCodegen,
    log_width: u64,
    numerator: IntValue<'c>,
    denominator: IntValue<'c>,
    panic: bool,
    signed: bool,
) -> Result<LLVMSumValue<'c>> {
    let int_arg_ty = int_types::INT_TYPES[log_width as usize].clone();
    let tuple_sum_ty = HugrSumType::new_tuple(vec![int_arg_ty.clone(), int_arg_ty.clone()]);

    let pair_ty = LLVMSumType::try_from_hugr_type(&ctx.typing_session(), tuple_sum_ty.clone())?;

    let build_divmod = |ctx: &mut EmitFuncContext<'c, '_, H>| -> Result<BasicValueEnum<'c>> {
        if signed {
            let max_signed_value = u64::pow(2, u32::pow(2, log_width as u32) - 1) - 1;
            let max_signed = numerator.get_type().const_int(max_signed_value, false);
            // Determine whether the divisor is "big" or "smol" for special casing.
            // Here, "big" means the divisor is larger than the biggest value
            // that could be represented by the type of the dividend.
            let large_divisor_bool = ctx.builder().build_int_compare(
                IntPredicate::UGT,
                denominator,
                max_signed,
                "is_divisor_large",
            )?;
            let large_divisor =
                ctx.builder()
                    .build_int_z_extend(large_divisor_bool, denominator.get_type(), "")?;
            let negative_numerator_bool = ctx.builder().build_int_compare(
                IntPredicate::SLT,
                numerator,
                numerator.get_type().const_zero(),
                "is_dividend_negative",
            )?;
            let negative_numerator = ctx.builder().build_int_z_extend(
                negative_numerator_bool,
                denominator.get_type(),
                "",
            )?;
            let tag = ctx.builder().build_left_shift(
                large_divisor,
                denominator.get_type().const_int(1, false),
                "",
            )?;

            let tag = ctx.builder().build_or(tag, negative_numerator, "tag")?;

            let quot = ctx
                .builder()
                .build_int_signed_div(numerator, denominator, "quotient")?;
            let rem = ctx
                .builder()
                .build_int_signed_rem(numerator, denominator, "remainder")?;

            let result_ptr = ctx.builder().build_alloca(pair_ty.clone(), "result")?;

            let finish = ctx.new_basic_block("finish", None);
            let negative_bigdiv = ctx.new_basic_block("negative_bigdiv", Some(finish));
            let negative_smoldiv = ctx.new_basic_block("negative_smoldiv", Some(finish));
            let non_negative_bigdiv = ctx.new_basic_block("non_negative_bigdiv", Some(finish));
            let non_negative_smoldiv = ctx.new_basic_block("non_negative_smoldiv", Some(finish));

            ctx.builder().build_switch(
                tag,
                non_negative_smoldiv,
                &[
                    (denominator.get_type().const_int(1, false), negative_smoldiv),
                    (
                        denominator.get_type().const_int(2, false),
                        non_negative_bigdiv,
                    ),
                    (denominator.get_type().const_int(3, false), negative_bigdiv),
                ],
            )?;

            let build_and_store_result =
                |ctx: &mut EmitFuncContext<'c, '_, H>, vs: Vec<BasicValueEnum<'c>>| -> Result<()> {
                    let result = pair_ty
                        .build_tag(ctx.builder(), 0, vs)?
                        //.build_tag(ctx.builder(), 0, vec![tag.as_basic_value_enum(), tag.as_basic_value_enum()])?
                        .as_basic_value_enum();
                    ctx.builder().build_store(result_ptr, result)?;
                    ctx.builder().build_unconditional_branch(finish)?;
                    Ok(())
                };

            // Default case (although it should only be reached by one branch).
            // When the divisor is smol and the dividend is positive, we can
            // rely on LLVM intrinsics.
            ctx.builder().position_at_end(non_negative_smoldiv);
            build_and_store_result(
                ctx,
                vec![quot.as_basic_value_enum(), rem.as_basic_value_enum()],
            )?;

            // When the divisor is smol and the dividend is negative,
            // we have two cases:
            ctx.builder().position_at_end(negative_smoldiv);
            {
                // If the remainder is 0, we can use the results of LLVM's `srem`
                let if_rem_zero = pair_ty
                    .build_tag(
                        ctx.builder(),
                        0,
                        vec![
                            quot.as_basic_value_enum(),
                            rem.get_type().const_zero().as_basic_value_enum(),
                        ],
                    )?
                    .as_basic_value_enum();

                // Otherwise, we return `(quotient - 1, divisor + remainder)`
                let if_rem_nonzero = pair_ty
                    .build_tag(
                        ctx.builder(),
                        0,
                        vec![
                            ctx.builder()
                                .build_int_sub(quot, quot.get_type().const_int(1, true), "")?
                                .as_basic_value_enum(),
                            ctx.builder()
                                .build_int_add(denominator, rem, "")?
                                .as_basic_value_enum(),
                        ],
                    )?
                    .as_basic_value_enum();

                let is_rem_zero = ctx.builder().build_int_compare(
                    IntPredicate::EQ,
                    rem,
                    rem.get_type().const_zero(),
                    "is_rem_0",
                )?;
                let result =
                    ctx.builder()
                        .build_select(is_rem_zero, if_rem_zero, if_rem_nonzero, "")?;
                ctx.builder().build_store(result_ptr, result)?;
                ctx.builder().build_unconditional_branch(finish)?;
            }

            // The (unsigned) divisor is bigger than the (signed) dividend could
            // possibly be, so it's safe to return quotient 0 and remainder = dividend
            ctx.builder().position_at_end(non_negative_bigdiv);
            build_and_store_result(
                ctx,
                vec![
                    numerator.get_type().const_zero().as_basic_value_enum(),
                    numerator.as_basic_value_enum(),
                ],
            )?;

            // The divisor is larger than the dividend can possibly be, and the
            // dividend is negative. This means we have to return `quotient - 1`
            // and the remainder is `dividend + divisor`.
            ctx.builder().position_at_end(negative_bigdiv);
            build_and_store_result(
                ctx,
                vec![
                    numerator.get_type().const_all_ones().as_basic_value_enum(),
                    ctx.builder()
                        .build_int_add(numerator, denominator, "")?
                        .as_basic_value_enum(),
                ],
            )?;

            ctx.builder().position_at_end(finish);
            let result = ctx.builder().build_load(result_ptr, "result")?;
            Ok(result)
        } else {
            let quot = ctx
                .builder()
                .build_int_unsigned_div(numerator, denominator, "quotient")?;
            let rem = ctx
                .builder()
                .build_int_unsigned_rem(numerator, denominator, "remainder")?;
            Ok(pair_ty
                .build_tag(
                    ctx.builder(),
                    0,
                    vec![quot.as_basic_value_enum(), rem.as_basic_value_enum()],
                )?
                .as_basic_value_enum())
        }
    };

    let int_ty = numerator.get_type();
    let zero = int_ty.const_zero();
    let lower_bounds_check =
        ctx.builder()
            .build_int_compare(IntPredicate::NE, denominator, zero, "valid_div")?;

    let sum_ty = LLVMSumType::try_from_hugr_type(
        &ctx.typing_session(),
        sum_with_error(vec![Type::from(tuple_sum_ty)]),
    )?;

    if panic {
        LLVMSumValue::try_new(
            val_or_panic(ctx, pcg, lower_bounds_check, &ERR_DIV_0, |ctx| {
                build_divmod(ctx)
            })?,
            pair_ty,
        )
    } else {
        let result = build_divmod(ctx)?;
        LLVMSumValue::try_new(
            val_or_error(ctx, lower_bounds_check, result, &ERR_DIV_0, sum_ty.clone())?,
            sum_ty,
        )
    }
}

fn make_narrow<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    arg: BasicValueEnum<'c>,
    outs: &[BasicTypeEnum<'c>],
    out_log_width: u64,
    signed: bool,
    sum_type: HugrSumType,
) -> Result<BasicValueEnum<'c>> {
    let [out] = TryInto::<[BasicTypeEnum; 1]>::try_into(outs)?;
    let width = 1 << out_log_width;
    let arg_int_ty: IntType = arg.get_type().into_int_type();
    let (int_min_value_s, int_max_value_s, int_max_value_u) = int_type_bounds(width);
    let out_int_ty = out
        .into_struct_type()
        .get_field_type_at_index(2)
        .unwrap()
        .into_int_type();
    let outside_range = if signed {
        let too_big = ctx.builder().build_int_compare(
            IntPredicate::SGT,
            arg.into_int_value(),
            arg_int_ty.const_int(int_max_value_s as u64, true),
            "upper_bounds_check",
        )?;
        let too_small = ctx.builder().build_int_compare(
            IntPredicate::SLT,
            arg.into_int_value(),
            arg_int_ty.const_int(int_min_value_s as u64, true),
            "lower_bounds_check",
        )?;
        ctx.builder()
            .build_or(too_big, too_small, "outside_range")?
    } else {
        ctx.builder().build_int_compare(
            IntPredicate::UGT,
            arg.into_int_value(),
            arg_int_ty.const_int(int_max_value_u, false),
            "upper_bounds_check",
        )?
    };

    let inbounds = ctx.builder().build_not(outside_range, "inbounds")?;
    let narrowed_val = ctx
        .builder()
        .build_int_cast_sign_flag(arg.into_int_value(), out_int_ty, signed, "")?
        .as_basic_value_enum();
    val_or_error(
        ctx,
        inbounds,
        narrowed_val,
        &ERR_NARROW,
        LLVMSumType::try_from_hugr_type(&ctx.typing_session(), sum_type).unwrap(),
    )
}

fn val_or_panic<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    pcg: &impl PreludeCodegen,
    dont_panic: IntValue<'c>,
    err: &ConstError,
    // Returned value must be same int type as `dont_panic`.
    go: impl Fn(&mut EmitFuncContext<'c, '_, H>) -> Result<BasicValueEnum<'c>>,
) -> Result<BasicValueEnum<'c>> {
    let exit_bb = ctx.new_basic_block("exit", None);
    let go_bb = ctx.new_basic_block("panic_if_0", Some(exit_bb));
    let panic_bb = ctx.new_basic_block("panic", Some(exit_bb));
    ctx.builder().build_unconditional_branch(go_bb)?;

    ctx.builder().position_at_end(panic_bb);
    let err = ctx.emit_custom_const(err)?;
    pcg.emit_panic(ctx, err)?;
    ctx.builder().build_unconditional_branch(exit_bb)?;

    ctx.builder().position_at_end(go_bb);
    ctx.builder().build_switch(
        dont_panic,
        panic_bb,
        &[(dont_panic.get_type().const_int(1, false), exit_bb)],
    )?;

    ctx.builder().position_at_end(exit_bb);

    go(ctx)
}

fn val_or_error<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    should_succeed: IntValue<'c>,
    val: BasicValueEnum<'c>,
    err: &ConstError,
    ty: LLVMSumType<'c>,
) -> Result<BasicValueEnum<'c>> {
    let err_val = ctx.emit_custom_const(err)?;

    let err_variant = ty.build_tag(ctx.builder(), 0, vec![err_val])?;
    let ok_variant = ty.build_tag(ctx.builder(), 1, vec![val])?;

    Ok(ctx
        .builder()
        .build_select(should_succeed, ok_variant, err_variant, "")?)
}

fn llvm_type<'c>(
    context: TypingSession<'c, '_>,
    hugr_type: &CustomType,
) -> Result<BasicTypeEnum<'c>> {
    if let [TypeArg::BoundedNat(n)] = hugr_type.args() {
        let m = *n as usize;
        if m < int_types::INT_TYPES.len() && int_types::INT_TYPES[m] == hugr_type.clone().into() {
            return Ok(match m {
                0..=3 => context.iw_context().i8_type(),
                4 => context.iw_context().i16_type(),
                5 => context.iw_context().i32_type(),
                6 => context.iw_context().i64_type(),
                _ => Err(anyhow!(
                    "IntTypesCodegenExtension: unsupported log_width: {}",
                    m
                ))?,
            }
            .into());
        }
    }
    Err(anyhow!(
        "IntTypesCodegenExtension: unsupported type: {}",
        hugr_type
    ))
}

fn emit_const_int<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    k: &ConstInt,
) -> Result<BasicValueEnum<'c>> {
    let ty: IntType = context.llvm_type(&k.get_type())?.try_into().unwrap();
    // k.value_u() is in two's complement representation of the exactly
    // correct bit width, so we are safe to unconditionally retrieve the
    // unsigned value and do no sign extension.
    Ok(ty.const_int(k.value_u(), false).as_basic_value_enum())
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Populates a [`CodegenExtsBuilder`] with all extensions needed to lower int
    /// ops, types, and constants.
    ///
    /// Any ops that panic will do so using [`DefaultPreludeCodegen`].
    #[must_use]
    pub fn add_default_int_extensions(self) -> Self {
        self.add_extension(IntCodegenExtension::new(DefaultPreludeCodegen))
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use hugr_core::builder::DataflowHugr;
    use hugr_core::extension::prelude::{ConstError, UnwrapBuilder, error_type};
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::{
        Hugr,
        builder::{Dataflow, DataflowSubContainer, SubContainer, handle::Outputs},
        extension::prelude::bool_t,
        ops::{DataflowOpTrait, ExtensionOp},
        std_extensions::arithmetic::{
            int_ops::{self, IntOpDef},
            int_types::{ConstInt, INT_TYPES},
        },
        types::{SumType, Type, TypeRow},
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        emit::test::{DFGW, SimpleHugrConfig},
        test::{TestContext, exec_ctx, llvm_ctx, single_op_hugr},
    };

    #[rstest::fixture]
    fn int_exec_ctx(mut exec_ctx: TestContext) -> TestContext {
        exec_ctx.add_extensions(|cem| {
            cem.add_default_int_extensions()
                .add_default_prelude_extensions()
        });
        exec_ctx
    }

    #[rstest::fixture]
    fn int_llvm_ctx(mut llvm_ctx: TestContext) -> TestContext {
        llvm_ctx.add_extensions(|cem| {
            cem.add_default_int_extensions()
                .add_default_prelude_extensions()
        });
        llvm_ctx
    }

    // Instantiate an extension op which takes one width argument
    fn make_int_op(name: impl AsRef<str>, log_width: u8) -> ExtensionOp {
        int_ops::EXTENSION
            .instantiate_extension_op(name.as_ref(), [u64::from(log_width).into()])
            .unwrap()
    }

    fn test_binary_int_op(ext_op: ExtensionOp, log_width: u8) -> Hugr {
        let ty = &INT_TYPES[log_width as usize];
        test_int_op_with_results::<2>(ext_op, log_width, None, ty.clone())
    }

    fn test_binary_icmp_op(ext_op: ExtensionOp, log_width: u8) -> Hugr {
        test_int_op_with_results::<2>(ext_op, log_width, None, bool_t())
    }

    fn test_int_op_with_results<const N: usize>(
        ext_op: ExtensionOp,
        log_width: u8,
        inputs: Option<[ConstInt; N]>,
        output_type: Type,
    ) -> Hugr {
        test_int_op_with_results_processing(ext_op, log_width, inputs, output_type, |_, a| Ok(a))
    }

    fn test_int_op_with_results_processing<const N: usize>(
        // N is the number of inputs to the hugr
        ext_op: ExtensionOp,
        log_width: u8,
        inputs: Option<[ConstInt; N]>, // If inputs are provided, they'll be wired into the op, otherwise the inputs to the hugr will be wired into the op
        output_type: Type,
        process: impl Fn(&mut DFGW, Outputs) -> Result<Outputs>,
    ) -> Hugr {
        let ty = &INT_TYPES[log_width as usize];
        let input_tys = if inputs.is_some() {
            vec![]
        } else {
            let input_tys = itertools::repeat_n(ty.clone(), N).collect();
            assert_eq!(input_tys, ext_op.signature().input.to_vec());
            input_tys
        };
        SimpleHugrConfig::new()
            .with_ins(input_tys)
            .with_outs(vec![output_type])
            .with_extensions(STD_REG.clone())
            .finish(|mut hugr_builder| {
                let input_wires = match inputs {
                    None => hugr_builder.input_wires_arr::<N>().to_vec(),
                    Some(inputs) => {
                        let mut input_wires = Vec::new();
                        for i in inputs.into_iter() {
                            let w = hugr_builder.add_load_value(i);
                            input_wires.push(w);
                        }
                        input_wires
                    }
                };
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, input_wires)
                    .unwrap()
                    .outputs();
                let processed_outputs = process(&mut hugr_builder, outputs).unwrap();
                hugr_builder
                    .finish_hugr_with_outputs(processed_outputs)
                    .unwrap()
            })
    }

    #[rstest]
    #[case(IntOpDef::iu_to_s, &[3])]
    #[case(IntOpDef::is_to_u, &[3])]
    #[case(IntOpDef::ineg, &[2])]
    #[case::idiv_checked_u("idiv_checked_u", &[3])]
    #[case::idiv_checked_s("idiv_checked_s", &[3])]
    #[case::imod_checked_u("imod_checked_u", &[6])]
    #[case::imod_checked_s("imod_checked_s", &[6])]
    #[case::idivmod_u("idivmod_u", &[3])]
    #[case::idivmod_s("idivmod_s", &[3])]
    #[case::idivmod_checked_u("idivmod_checked_u", &[6])]
    #[case::idivmod_checked_s("idivmod_checked_s", &[6])]
    fn test_emission(int_llvm_ctx: TestContext, #[case] op: IntOpDef, #[case] args: &[u8]) {
        use hugr_core::extension::simple_op::MakeExtensionOp as _;

        let mut insta = insta::Settings::clone_current();
        insta.set_snapshot_suffix(format!(
            "{}_{}_{:?}",
            insta.snapshot_suffix().unwrap_or(""),
            op.op_id(),
            args,
        ));
        let concrete = match *args {
            [] => op.without_log_width(),
            [log_width] => op.with_log_width(log_width),
            [lw1, lw2] => op.with_two_log_widths(lw1, lw2),
            _ => panic!("unexpected number of args to the op!"),
        };
        insta.bind(|| {
            let hugr = single_op_hugr(concrete.into());
            check_emission!(hugr, int_llvm_ctx);
        });
    }

    #[rstest]
    #[case::iadd("iadd", 3)]
    #[case::isub("isub", 6)]
    #[case::ipow("ipow", 3)]
    #[case::idiv_u("idiv_u", 3)]
    #[case::idiv_s("idiv_s", 3)]
    #[case::imod_u("imod_u", 3)]
    #[case::imod_s("imod_s", 3)]
    fn test_binop_emission(int_llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        let ext_op = make_int_op(op.clone(), width);
        let hugr = test_binary_int_op(ext_op, width);
        check_emission!(op.clone(), hugr, int_llvm_ctx);
    }

    #[rstest]
    #[case::signed_2_3("iwiden_s", 2, 3)]
    #[case::signed_1_6("iwiden_s", 1, 6)]
    #[case::unsigned_2_3("iwiden_u", 2, 3)]
    #[case::unsigned_1_6("iwiden_u", 1, 6)]
    fn test_widen_emission(
        int_llvm_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
    ) {
        let out_ty = INT_TYPES[to as usize].clone();
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(&op, [u64::from(from).into(), u64::from(to).into()])
            .unwrap();
        let hugr = test_int_op_with_results::<1>(ext_op, from, None, out_ty);

        check_emission!(
            format!("{}_{}_{}", op.clone(), from, to),
            hugr,
            int_llvm_ctx
        );
    }

    #[rstest]
    #[case::signed("inarrow_s", 3, 2)]
    #[case::unsigned("inarrow_u", 6, 4)]
    fn test_narrow_emission(
        int_llvm_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
    ) {
        let out_ty = SumType::new([vec![error_type()], vec![INT_TYPES[to as usize].clone()]]);
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(&op, [u64::from(from).into(), u64::from(to).into()])
            .unwrap();
        let hugr = test_int_op_with_results::<1>(ext_op, from, None, out_ty.into());

        check_emission!(
            format!("{}_{}_{}", op.clone(), from, to),
            hugr,
            int_llvm_ctx
        );
    }

    #[rstest]
    #[case::ieq("ieq", 1)]
    #[case::ilt_s("ilt_s", 0)]
    fn test_cmp_emission(int_llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        let ext_op = make_int_op(op.clone(), width);
        let hugr = test_binary_icmp_op(ext_op, width);
        check_emission!(op.clone(), hugr, int_llvm_ctx);
    }

    #[rstest]
    #[case::imax("imax_u", 1, 2, 2)]
    #[case::imax("imax_u", 2, 1, 2)]
    #[case::imax("imax_u", 2, 2, 2)]
    #[case::imin("imin_u", 1, 2, 1)]
    #[case::imin("imin_u", 2, 1, 1)]
    #[case::imin("imin_u", 2, 2, 2)]
    #[case::ishl("ishl", 73, 1, 146)]
    // (2^64 - 1) << 1 = (2^64 - 2)
    #[case::ishl("ishl", 18446744073709551615, 1, 18446744073709551614)]
    #[case::ishr("ishr", 73, 1, 36)]
    #[case::ior("ior", 6, 9, 15)]
    #[case::ior("ior", 6, 15, 15)]
    #[case::ixor("ixor", 6, 9, 15)]
    #[case::ixor("ixor", 6, 15, 9)]
    #[case::ixor("ixor", 15, 6, 9)]
    #[case::iand("iand", 6, 15, 6)]
    #[case::iand("iand", 15, 6, 6)]
    #[case::iand("iand", 15, 15, 15)]
    #[case::ipow("ipow", 2, 3, 8)]
    #[case::ipow("ipow", 42, 1, 42)]
    #[case::ipow("ipow", 42, 0, 1)]
    #[case::idiv("idiv_u", 42, 2, 21)]
    #[case::idiv("idiv_u", 42, 5, 8)]
    #[case::imod("imod_u", 42, 2, 0)]
    #[case::imod("imod_u", 42, 5, 2)]
    fn test_exec_unsigned_bin_op(
        int_exec_ctx: TestContext,
        #[case] op: String,
        #[case] lhs: u64,
        #[case] rhs: u64,
        #[case] result: u64,
    ) {
        let ty = &INT_TYPES[6].clone();
        let inputs = [
            ConstInt::new_u(6, lhs).unwrap(),
            ConstInt::new_u(6, rhs).unwrap(),
        ];
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<2>(ext_op, 6, Some(inputs), ty.clone());
        assert_eq!(int_exec_ctx.exec_hugr_u64(hugr, "main"), result);
    }

    #[rstest]
    #[case::imax("imax_s", 1, 2, 2)]
    #[case::imax("imax_s", 2, 1, 2)]
    #[case::imax("imax_s", 2, 2, 2)]
    #[case::imax("imax_s", -1, -2, -1)]
    #[case::imax("imax_s", -2, -1, -1)]
    #[case::imax("imax_s", -2, -2, -2)]
    #[case::imin("imin_s", 1, 2, 1)]
    #[case::imin("imin_s", 2, 1, 1)]
    #[case::imin("imin_s", 2, 2, 2)]
    #[case::imin("imin_s", -1, -2, -2)]
    #[case::imin("imin_s", -2, -1, -2)]
    #[case::imin("imin_s", -2, -2, -2)]
    #[case::ipow("ipow", -2, 1, -2)]
    #[case::ipow("ipow", -2, 2, 4)]
    #[case::ipow("ipow", -2, 3, -8)]
    fn test_exec_signed_bin_op(
        int_exec_ctx: TestContext,
        #[case] op: String,
        #[case] lhs: i64,
        #[case] rhs: i64,
        #[case] result: i64,
    ) {
        let ty = &INT_TYPES[6].clone();
        let inputs = [
            ConstInt::new_s(6, lhs).unwrap(),
            ConstInt::new_s(6, rhs).unwrap(),
        ];
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<2>(ext_op, 6, Some(inputs), ty.clone());
        assert_eq!(int_exec_ctx.exec_hugr_i64(hugr, "main"), result);
    }

    #[rstest]
    #[case::iabs("iabs", 42, 42)]
    #[case::iabs("iabs", -42, 42)]
    fn test_exec_signed_unary_op(
        int_exec_ctx: TestContext,
        #[case] op: String,
        #[case] arg: i64,
        #[case] result: i64,
    ) {
        let input = ConstInt::new_s(6, arg).unwrap();
        let ty = INT_TYPES[6].clone();
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<1>(ext_op, 6, Some([input]), ty.clone());
        assert_eq!(int_exec_ctx.exec_hugr_i64(hugr, "main"), result);
    }

    #[rstest]
    #[case::inot("inot", 9223372036854775808, !9223372036854775808u64)]
    #[case::inot("inot", 42, !42u64)]
    #[case::inot("inot", !0u64, 0)]
    fn test_exec_unsigned_unary_op(
        int_exec_ctx: TestContext,
        #[case] op: String,
        #[case] arg: u64,
        #[case] result: u64,
    ) {
        let input = ConstInt::new_u(6, arg).unwrap();
        let ty = INT_TYPES[6].clone();
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<1>(ext_op, 6, Some([input]), ty.clone());
        assert_eq!(int_exec_ctx.exec_hugr_u64(hugr, "main"), result);
    }

    #[rstest]
    #[case(-127)]
    #[case(-1)]
    #[case(0)]
    #[case(1)]
    #[case(127)]
    fn test_exec_widen(int_exec_ctx: TestContext, #[case] num: i16) {
        let from: u8 = 3;
        let to: u8 = 6;
        let ty = INT_TYPES[to as usize].clone();

        if num >= 0 {
            let input = ConstInt::new_u(from, num as u64).unwrap();

            let ext_op = int_ops::EXTENSION
                .instantiate_extension_op(
                    "iwiden_u".as_ref(),
                    [(from as u64).into(), (to as u64).into()],
                )
                .unwrap();

            let hugr = test_int_op_with_results::<1>(ext_op, to, Some([input]), ty.clone());

            assert_eq!(int_exec_ctx.exec_hugr_u64(hugr, "main"), num as u64);
        }

        let input = ConstInt::new_s(from, num as i64).unwrap();

        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(
                "iwiden_s".as_ref(),
                [(from as u64).into(), (to as u64).into()],
            )
            .unwrap();

        let hugr = test_int_op_with_results::<1>(ext_op, to, Some([input]), ty.clone());

        assert_eq!(int_exec_ctx.exec_hugr_i64(hugr, "main"), num as i64);
    }

    #[rstest]
    #[case("inarrow_s", 6, 2, 4)]
    #[case("inarrow_s", 6, 5, (1 << 5) - 1)]
    #[case("inarrow_s", 6, 4, -1)]
    #[case("inarrow_s", 6, 4, -(1 << 4) - 1)]
    #[case("inarrow_s", 6, 4, -(1 <<15))]
    #[case("inarrow_s", 6, 5, (1 << 31) - 1)]
    fn test_narrow_s(
        int_exec_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
        #[case] arg: i64,
    ) {
        let input = ConstInt::new_s(from, arg).unwrap();
        let to_ty = INT_TYPES[to as usize].clone();
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(op.as_ref(), [u64::from(from).into(), u64::from(to).into()])
            .unwrap();

        let hugr = test_int_op_with_results_processing::<1>(
            ext_op,
            to,
            Some([input]),
            to_ty.clone(),
            |builder, outs| {
                let [out] = outs.to_array();

                let err_row = TypeRow::from(vec![error_type()]);
                let ty_row = TypeRow::from(vec![to_ty.clone()]);
                // Handle the sum type returned by narrow by building a conditional.
                // We're only testing the happy path here, so insert a panic in the
                // "error" branch, knowing that it wont come up.
                //
                // Negative results can be tested manually, but we lack the testing
                // infrastructure to detect execution crashes without crashing the
                // test process.
                let mut cond_b = builder.conditional_builder(
                    ([err_row, ty_row], out),
                    [],
                    vec![to_ty.clone()].into(),
                )?;
                let mut sad_b = cond_b.case_builder(0)?;
                let err = ConstError::new(2, "This shouldn't happen");
                let w = sad_b.add_load_value(ConstInt::new_s(to, 0)?);
                sad_b.add_panic(err, vec![to_ty.clone()], [(w, to_ty.clone())])?;
                sad_b.finish_with_outputs([w])?;

                let happy_b = cond_b.case_builder(1)?;
                let [w] = happy_b.input_wires_arr();
                happy_b.finish_with_outputs([w])?;

                let handle = cond_b.finish_sub_container()?;
                Ok(handle.outputs())
            },
        );
        assert_eq!(int_exec_ctx.exec_hugr_i64(hugr, "main"), arg);
    }

    #[rstest]
    #[case(6, 42)]
    #[case(4, 7)]
    //#[case(4, 256)] -- crashes because a panic is emitted (good)
    fn test_u_to_s(int_exec_ctx: TestContext, #[case] log_width: u8, #[case] val: u64) {
        let ty = &INT_TYPES[log_width as usize].clone();
        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![ty.clone()])
            .with_extensions(STD_REG.clone())
            .finish(|mut hugr_builder| {
                let unsigned =
                    hugr_builder.add_load_value(ConstInt::new_u(log_width, val).unwrap());
                let iu_to_s = make_int_op("iu_to_s", log_width);
                let [signed] = hugr_builder
                    .add_dataflow_op(iu_to_s, [unsigned])
                    .unwrap()
                    .outputs_arr();
                hugr_builder.finish_hugr_with_outputs([signed]).unwrap()
            });
        let act = int_exec_ctx.exec_hugr_i64(hugr, "main");
        assert_eq!(act, val as i64);
    }

    #[rstest]
    #[case(3, 0)]
    #[case(4, 255)]
    // #[case(3, -1)] -- crashes because a panic is emitted (good)
    fn test_s_to_u(int_exec_ctx: TestContext, #[case] log_width: u8, #[case] val: i64) {
        let ty = &INT_TYPES[log_width as usize].clone();
        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![ty.clone()])
            .with_extensions(STD_REG.clone())
            .finish(|mut hugr_builder| {
                let signed = hugr_builder.add_load_value(ConstInt::new_s(log_width, val).unwrap());
                let is_to_u = make_int_op("is_to_u", log_width);
                let [unsigned] = hugr_builder
                    .add_dataflow_op(is_to_u, [signed])
                    .unwrap()
                    .outputs_arr();
                let num = hugr_builder.add_load_value(ConstInt::new_u(log_width, 42).unwrap());
                let [res] = hugr_builder
                    .add_dataflow_op(make_int_op("iadd", log_width), [unsigned, num])
                    .unwrap()
                    .outputs_arr();
                hugr_builder.finish_hugr_with_outputs([res]).unwrap()
            });
        let act = int_exec_ctx.exec_hugr_u64(hugr, "main");
        assert_eq!(act, (val as u64) + 42);
    }

    // Log width fixed at 3 (i.e. divmod : Fn(i8, u8) -> (i8, u8)
    #[rstest]
    #[case::bigdiv_non_negative(127, 255, (0, 127))] // Big divisor, positive dividend
    #[case::bigdiv_negative(-42, 255, (-1, 213))] // Big divisor, negative dividend
    #[case::smoldiv_non_negative(42, 10, (4, 2))] // Normal divisor, positive dividend
    #[case::smoldiv_negative_rem0(-42, 21, (-2, 0))] // Normal divisor, negative dividend, remainder 0
    #[case::smoldiv_negative_rem_nonzero(-42, 10, (-5, 8))] // Normal divisor, negative dividend, remainder >0
    fn test_divmod_s(
        int_exec_ctx: TestContext,
        #[case] dividend: i64,
        #[case] divisor: u64,
        #[case] expected_result: (i64, u64),
    ) {
        let int_ty = INT_TYPES[3].clone();
        let k_dividend = ConstInt::new_s(3, dividend).unwrap();
        let k_divisor = ConstInt::new_u(3, divisor).unwrap();
        let quot_hugr = test_int_op_with_results(
            make_int_op("idiv_s", 3),
            3,
            Some([k_dividend.clone(), k_divisor.clone()]),
            int_ty.clone(),
        );
        let rem_hugr = test_int_op_with_results(
            make_int_op("imod_s", 3),
            3,
            Some([k_dividend, k_divisor]),
            int_ty,
        );
        let quot = int_exec_ctx.exec_hugr_i64(quot_hugr, "main");
        let rem = int_exec_ctx.exec_hugr_u64(rem_hugr, "main");
        assert_eq!((quot, rem), expected_result);
    }
}

use hugr_core::{
    extension::prelude::ConstError,
    ops::{constant::CustomConst, ExtensionOp, NamedOp, Value},
    std_extensions::arithmetic::{
        int_ops::IntOpDef,
        int_types::{self, ConstInt},
    },
    types::{CustomType, TypeArg},
    HugrView, Node,
};
use inkwell::{
    types::{BasicType, BasicTypeEnum, IntType},
    values::{BasicValue, BasicValueEnum, IntValue},
    IntPredicate,
};

use crate::{
    custom::CodegenExtsBuilder,
    emit::{
        emit_value,
        func::EmitFuncContext,
        get_intrinsic,
        ops::{emit_custom_binary_op, emit_custom_unary_op},
        EmitOpArgs,
    },
    sum::LLVMSumType,
    types::{HugrSumType, TypingSession},
};

use anyhow::{anyhow, bail, Result};

enum RuntimeError {
    Narrow,
}

impl RuntimeError {
    fn show(&self) -> &str {
        match self {
            RuntimeError::Narrow => "Can't narrow into bounds",
        }
    }
}

/// Emit an integer comparison operation.
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
        Ok(vec![ctx
            .builder()
            .build_select(r, true_val, false_val, "")?])
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
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: IntOpDef,
) -> Result<()> {
    match op {
        IntOpDef::iadd => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::imul => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::isub => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::idiv_s => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::idiv_u => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::imod_s => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::ineg => emit_custom_unary_op(context, args, |ctx, arg, _| {
            Ok(vec![ctx
                .builder()
                .build_int_neg(arg.into_int_value(), "")?
                .as_basic_value_enum()])
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
            Ok(vec![ctx
                .builder()
                .build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::ishr => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_right_shift(lhs.into_int_value(), rhs.into_int_value(), false, "")?
                .as_basic_value_enum()])
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
            Ok(vec![ctx
                .builder()
                .build_xor(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::ior => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_or(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::inot => emit_custom_unary_op(context, args, |ctx, arg, _| {
            Ok(vec![ctx
                .builder()
                .build_not(arg.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::iand => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![ctx
                .builder()
                .build_and(lhs.into_int_value(), rhs.into_int_value(), "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::ipow => emit_ipow(context, args),
        // Type args are width of input, width of output
        IntOpDef::iwiden_u => emit_custom_unary_op(context, args, |ctx, arg, outs| {
            let [out] = outs.try_into()?;
            Ok(vec![ctx
                .builder()
                .build_int_cast_sign_flag(arg.into_int_value(), out.into_int_type(), false, "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::iwiden_s => emit_custom_unary_op(context, args, |ctx, arg, outs| {
            let [out] = outs.try_into()?;

            Ok(vec![ctx
                .builder()
                .build_int_cast_sign_flag(arg.into_int_value(), out.into_int_type(), true, "")?
                .as_basic_value_enum()])
        }),
        IntOpDef::inarrow_s => {
            let Some(TypeArg::BoundedNat { n: out_log_width }) = args.node().args().last().cloned()
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
            let Some(TypeArg::BoundedNat { n: out_log_width }) = args.node().args().last().cloned()
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
        _ => Err(anyhow!("IntOpEmitter: unimplemented op: {}", op.name())),
    }
}

fn make_narrow<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    arg: BasicValueEnum<'c>,
    outs: &[BasicTypeEnum<'c>],
    log_width: u64,
    signed: bool,
    sum_type: HugrSumType,
) -> Result<BasicValueEnum<'c>> {
    let [out] = TryInto::<[BasicTypeEnum; 1]>::try_into(outs)?;
    let width = 1 << log_width;
    //arg.get_type().into_int_type().
    let max_val = if signed {
        (i64::MAX >> (64 - width)) as u64
    } else {
        u64::MAX >> (64 - width)
    };
    let out_int_ty = out
        .into_struct_type()
        .get_field_type_at_index(2)
        .unwrap()
        .into_int_type();
    let max = arg
        .get_type()
        .into_int_type()
        .const_int(max_val, true);
    let pred = if signed {
        IntPredicate::SGT
    } else {
        IntPredicate::UGT
    };
    let bigger_than_max =
        ctx.builder()
            .build_int_compare(pred, arg.into_int_value(), max, "upper_bounds_check")?;

    // Contains true if the input is out of bounds
    let should_fail = if signed {
        let min = ctx.builder().build_int_neg(max, "min")?;
        let less_than_min = ctx.builder().build_int_compare(
            IntPredicate::SLT,
            arg.into_int_value(),
            min,
            "lower_bounds_check",
        )?;
        ctx.builder()
            .build_or(less_than_min, bigger_than_max, "oob")?
    } else {
        bigger_than_max
    };
    let narrowed_val = ctx
        .builder()
        .build_int_cast_sign_flag(arg.into_int_value(), out_int_ty, signed, "")?
        .as_basic_value_enum();
    val_or_error(
        ctx,
        should_fail,
        narrowed_val,
        RuntimeError::Narrow,
        LLVMSumType::try_from_hugr_type(&ctx.typing_session(), sum_type).unwrap(),
    )
}

fn val_or_error<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    should_fail: IntValue<'c>,
    val: BasicValueEnum<'c>,
    msg: RuntimeError,
    ty: LLVMSumType<'c>,
) -> Result<BasicValueEnum<'c>> {
    let err_msg = Value::extension(ConstError::new(2, msg.show()));
    let err_val = emit_value(ctx, &err_msg)?;

    let err_variant = ty.build_tag(ctx.builder(), 0, vec![err_val])?;
    let ok_variant = ty.build_tag(ctx.builder(), 1, vec![val])?;

    Ok(ctx
        .builder()
        .build_select(should_fail, err_variant, ok_variant, "")?)
}

fn llvm_type<'c>(
    context: TypingSession<'c, '_>,
    hugr_type: &CustomType,
) -> Result<BasicTypeEnum<'c>> {
    if let [TypeArg::BoundedNat { n }] = hugr_type.args() {
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

/// Populates a [CodegenExtsBuilder] with all extensions needed to lower int
/// ops, types, and constants.
pub fn add_int_extensions<'a, H: HugrView<Node = Node> + 'a>(
    cem: CodegenExtsBuilder<'a, H>,
) -> CodegenExtsBuilder<'a, H> {
    cem.custom_const(emit_const_int)
        .custom_type((int_types::EXTENSION_ID, "int".into()), llvm_type)
        .simple_extension_op::<IntOpDef>(emit_int_op)
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Populates a [CodegenExtsBuilder] with all extensions needed to lower int
    /// ops, types, and constants.
    pub fn add_int_extensions(self) -> Self {
        add_int_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use anyhow::Result;
    use hugr_core::extension::prelude::{error_type, ConstError, UnwrapBuilder};
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::{
        builder::{handle::Outputs, Dataflow, DataflowSubContainer, SubContainer},
        extension::prelude::bool_t,
        ops::{DataflowOpTrait, ExtensionOp},
        std_extensions::arithmetic::{
            int_ops,
            int_types::{ConstInt, INT_TYPES},
        },
        types::{SumType, Type, TypeRow},
        Hugr,
    };
    use rstest::rstest;

    use crate::extension::DefaultPreludeCodegen;
    use crate::{
        check_emission,
        emit::test::{SimpleHugrConfig, DFGW},
        extension::{int::add_int_extensions, prelude::add_prelude_extensions},
        test::{exec_ctx, llvm_ctx, TestContext},
    };

    // Instantiate an extension op which takes one width argument
    fn make_int_op(name: impl AsRef<str>, log_width: u8) -> ExtensionOp {
        int_ops::EXTENSION
            .instantiate_extension_op(name.as_ref(), [(log_width as u64).into()])
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
                        inputs.into_iter().for_each(|i| {
                            let w = hugr_builder.add_load_value(i);
                            input_wires.push(w);
                        });
                        input_wires
                    }
                };
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, input_wires)
                    .unwrap()
                    .outputs();
                let processed_outputs = process(&mut hugr_builder, outputs).unwrap();
                hugr_builder.finish_with_outputs(processed_outputs).unwrap()
            })
    }

    #[rstest]
    fn test_neg_emission(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_int_extensions);
        let ty = INT_TYPES[2].clone();
        let ext_op = make_int_op("ineg", 2);
        let hugr = test_int_op_with_results::<1>(ext_op, 2, None, ty.clone());
        check_emission!("ineg", hugr, llvm_ctx);
    }

    #[rstest]
    #[case::iadd("iadd", 3)]
    #[case::isub("isub", 6)]
    #[case::ipow("ipow", 3)]
    fn test_binop_emission(mut llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        llvm_ctx.add_extensions(add_int_extensions);
        let ext_op = make_int_op(op.clone(), width);
        let hugr = test_binary_int_op(ext_op, width);
        check_emission!(op.clone(), hugr, llvm_ctx);
    }

    #[rstest]
    #[case::signed_2_3("iwiden_s", 2, 3)]
    #[case::signed_1_6("iwiden_s", 1, 6)]
    #[case::unsigned_2_3("iwiden_u", 2, 3)]
    #[case::unsigned_1_6("iwiden_u", 1, 6)]
    fn test_widen_emission(
        mut llvm_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
    ) {
        llvm_ctx.add_extensions(add_int_extensions);
        let out_ty = INT_TYPES[to as usize].clone();
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(&op, [(from as u64).into(), (to as u64).into()])
            .unwrap();
        let hugr = test_int_op_with_results::<1>(ext_op, from, None, out_ty);

        check_emission!(format!("{}_{}_{}", op.clone(), from, to), hugr, llvm_ctx);
    }

    #[rstest]
    #[case::signed("inarrow_s", 3, 2)]
    #[case::unsigned("inarrow_u", 6, 4)]
    fn test_narrow_emission(
        mut llvm_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
    ) {
        llvm_ctx.add_extensions(add_int_extensions);
        llvm_ctx.add_extensions(|cem| add_prelude_extensions(cem, DefaultPreludeCodegen));
        let out_ty = SumType::new([vec![error_type()], vec![INT_TYPES[to as usize].clone()]]);
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(&op, [(from as u64).into(), (to as u64).into()])
            .unwrap();
        let hugr = test_int_op_with_results::<1>(ext_op, from, None, out_ty.into());

        check_emission!(format!("{}_{}_{}", op.clone(), from, to), hugr, llvm_ctx);
    }

    #[rstest]
    #[case::ieq("ieq", 1)]
    #[case::ilt_s("ilt_s", 0)]
    fn test_cmp_emission(mut llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        llvm_ctx.add_extensions(add_int_extensions);
        let ext_op = make_int_op(op.clone(), width);
        let hugr = test_binary_icmp_op(ext_op, width);
        check_emission!(op.clone(), hugr, llvm_ctx);
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
    fn test_exec_unsigned_bin_op(
        mut exec_ctx: TestContext,
        #[case] op: String,
        #[case] lhs: u64,
        #[case] rhs: u64,
        #[case] result: u64,
    ) {
        exec_ctx.add_extensions(add_int_extensions);
        let ty = &INT_TYPES[6].clone();
        let inputs = [
            ConstInt::new_u(6, lhs).unwrap(),
            ConstInt::new_u(6, rhs).unwrap(),
        ];
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<2>(ext_op, 6, Some(inputs), ty.clone());
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), result);
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
        mut exec_ctx: TestContext,
        #[case] op: String,
        #[case] lhs: i64,
        #[case] rhs: i64,
        #[case] result: i64,
    ) {
        exec_ctx.add_extensions(add_int_extensions);
        let ty = &INT_TYPES[6].clone();
        let inputs = [
            ConstInt::new_s(6, lhs).unwrap(),
            ConstInt::new_s(6, rhs).unwrap(),
        ];
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<2>(ext_op, 6, Some(inputs), ty.clone());
        assert_eq!(exec_ctx.exec_hugr_i64(hugr, "main"), result);
    }

    #[rstest]
    #[case::iabs("iabs", 42, 42)]
    #[case::iabs("iabs", -42, 42)]
    fn test_exec_signed_unary_op(
        mut exec_ctx: TestContext,
        #[case] op: String,
        #[case] arg: i64,
        #[case] result: i64,
    ) {
        exec_ctx.add_extensions(add_int_extensions);
        let input = ConstInt::new_s(6, arg).unwrap();
        let ty = INT_TYPES[6].clone();
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<1>(ext_op, 6, Some([input]), ty.clone());
        assert_eq!(exec_ctx.exec_hugr_i64(hugr, "main"), result);
    }

    #[rstest]
    #[case::inot("inot", 9223372036854775808, !9223372036854775808u64)]
    #[case::inot("inot", 42, !42u64)]
    #[case::inot("inot", !0u64, 0)]
    fn test_exec_unsigned_unary_op(
        mut exec_ctx: TestContext,
        #[case] op: String,
        #[case] arg: u64,
        #[case] result: u64,
    ) {
        exec_ctx.add_extensions(add_int_extensions);
        let input = ConstInt::new_u(6, arg).unwrap();
        let ty = INT_TYPES[6].clone();
        let ext_op = make_int_op(&op, 6);

        let hugr = test_int_op_with_results::<1>(ext_op, 6, Some([input]), ty.clone());
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), result);
    }

    #[rstest]
    #[case("inarrow_s", 6, 2, 4)]
    #[case("inarrow_s", 6, 5, 2^5 - 1)]
    #[case("inarrow_s", 6, 4, -1)]
    #[case("inarrow_s", 6, 4, -(2^4 - 1))]
    fn test_narrow_s(
        mut exec_ctx: TestContext,
        #[case] op: String,
        #[case] from: u8,
        #[case] to: u8,
        #[case] arg: i64,
    ) {
        exec_ctx.add_extensions(add_int_extensions);
        exec_ctx.add_extensions(|cem| add_prelude_extensions(cem, DefaultPreludeCodegen));
        let input = ConstInt::new_s(from, arg).unwrap();
        let to_ty = INT_TYPES[to as usize].clone();
        let ext_op = int_ops::EXTENSION
            .instantiate_extension_op(op.as_ref(), [(from as u64).into(), (to as u64).into()])
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
        assert_eq!(exec_ctx.exec_hugr_i64(hugr, "main"), arg);
    }
}

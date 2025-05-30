use anyhow::{Result, anyhow};
use hugr_core::Node;
use hugr_core::ops::ExtensionOp;
use hugr_core::ops::{Value, constant::CustomConst};
use hugr_core::std_extensions::arithmetic::float_ops::FloatOps;
use hugr_core::{
    HugrView,
    std_extensions::arithmetic::float_types::{self, ConstF64},
};
use inkwell::{
    types::{BasicType, FloatType},
    values::{BasicValue, BasicValueEnum},
};

use crate::emit::ops::{emit_custom_binary_op, emit_custom_unary_op};
use crate::emit::{EmitOpArgs, func::EmitFuncContext};
use crate::emit::{emit_value, get_intrinsic};

use crate::custom::CodegenExtsBuilder;

/// Emit a float comparison operation.
fn emit_fcmp<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    pred: inkwell::FloatPredicate,
) -> Result<()> {
    let true_val = emit_value(context, &Value::true_val())?;
    let false_val = emit_value(context, &Value::false_val())?;

    emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
        // get result as an i1
        let r = ctx.builder().build_float_compare(
            pred,
            lhs.into_float_value(),
            rhs.into_float_value(),
            "",
        )?;
        // convert to whatever bool_t is
        Ok(vec![
            ctx.builder().build_select(r, true_val, false_val, "")?,
        ])
    })
}

fn emit_float_op<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: FloatOps,
) -> Result<()> {
    // We emit the float comparison variants where NaN is an absorbing value.
    // Any comparison with NaN is always false.
    #[allow(clippy::wildcard_in_or_patterns)]
    match op {
        FloatOps::feq => emit_fcmp(context, args, inkwell::FloatPredicate::OEQ),
        FloatOps::fne => emit_fcmp(context, args, inkwell::FloatPredicate::ONE),
        FloatOps::flt => emit_fcmp(context, args, inkwell::FloatPredicate::OLT),
        FloatOps::fgt => emit_fcmp(context, args, inkwell::FloatPredicate::OGT),
        FloatOps::fle => emit_fcmp(context, args, inkwell::FloatPredicate::OLE),
        FloatOps::fge => emit_fcmp(context, args, inkwell::FloatPredicate::OGE),
        FloatOps::fadd => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        FloatOps::fsub => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        FloatOps::fneg => emit_custom_unary_op(context, args, |ctx, v, _| {
            Ok(vec![
                ctx.builder()
                    .build_float_neg(v.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        FloatOps::fmul => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        FloatOps::fdiv => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        FloatOps::fpow => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            let float_ty = ctx.iw_context().f64_type().as_basic_type_enum();
            let func = get_intrinsic(ctx.get_current_module(), "llvm.pow.f64", [float_ty])?;
            Ok(vec![
                ctx.builder()
                    .build_call(func, &[lhs.into(), rhs.into()], "")?
                    .try_as_basic_value()
                    .unwrap_left()
                    .as_basic_value_enum(),
            ])
        }),
        // Missing ops, not supported by inkwell
        FloatOps::fmax
        | FloatOps::fmin
        | FloatOps::fabs
        | FloatOps::ffloor
        | FloatOps::fceil
        | FloatOps::ftostring
        | _ => {
            let name: &str = op.into();
            Err(anyhow!("FloatOpEmitter: unimplemented op: {name}"))
        }
    }
}

fn emit_constf64<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    k: &ConstF64,
) -> Result<BasicValueEnum<'c>> {
    let ty: FloatType = context.llvm_type(&k.get_type())?.try_into().unwrap();
    Ok(ty.const_float(k.value()).as_basic_value_enum())
}

pub fn add_float_extensions<'a, H: HugrView<Node = Node> + 'a>(
    cem: CodegenExtsBuilder<'a, H>,
) -> CodegenExtsBuilder<'a, H> {
    cem.custom_type(
        (
            float_types::EXTENSION_ID,
            float_types::FLOAT_TYPE_ID.clone(),
        ),
        |ts, _custom_type| Ok(ts.iw_context().f64_type().as_basic_type_enum()),
    )
    .custom_const(emit_constf64)
    .simple_extension_op::<FloatOps>(emit_float_op)
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    #[must_use]
    pub fn add_float_extensions(self) -> Self {
        add_float_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use hugr_core::Hugr;
    use hugr_core::builder::DataflowHugr;
    use hugr_core::extension::SignatureFunc;
    use hugr_core::extension::simple_op::MakeOpDef;
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::std_extensions::arithmetic::float_ops::FloatOps;
    use hugr_core::types::TypeRow;
    use hugr_core::{
        builder::Dataflow,
        std_extensions::arithmetic::float_types::{ConstF64, float64_type},
    };
    use rstest::rstest;

    use super::add_float_extensions;
    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        test::{TestContext, llvm_ctx},
    };

    fn test_float_op(op: FloatOps) -> Hugr {
        let SignatureFunc::PolyFuncType(poly_sig) = op.signature() else {
            panic!("Expected PolyFuncType");
        };
        let sig = poly_sig.body();
        let inp: TypeRow = sig.input.clone().try_into().unwrap();
        let out: TypeRow = sig.output.clone().try_into().unwrap();

        SimpleHugrConfig::new()
            .with_ins(inp)
            .with_outs(out)
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let outputs = builder
                    .add_dataflow_op(op, builder.input_wires())
                    .unwrap()
                    .outputs();
                builder.finish_hugr_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    fn const_float(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_float_extensions);
        let hugr = SimpleHugrConfig::new()
            .with_outs(float64_type())
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let c = builder.add_load_value(ConstF64::new(3.12));
                builder.finish_hugr_with_outputs([c]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    #[case::feq(FloatOps::feq)]
    #[case::fne(FloatOps::fne)]
    #[case::flt(FloatOps::flt)]
    #[case::fgt(FloatOps::fgt)]
    #[case::fle(FloatOps::fle)]
    #[case::fge(FloatOps::fge)]
    #[case::fadd(FloatOps::fadd)]
    #[case::fsub(FloatOps::fsub)]
    #[case::fneg(FloatOps::fneg)]
    #[case::fmul(FloatOps::fmul)]
    #[case::fdiv(FloatOps::fdiv)]
    #[case::fpow(FloatOps::fpow)]
    fn float_operations(mut llvm_ctx: TestContext, #[case] op: FloatOps) {
        let name: &str = op.into();
        let hugr = test_float_op(op);
        llvm_ctx.add_extensions(add_float_extensions);
        check_emission!(name, hugr, llvm_ctx);
    }
}

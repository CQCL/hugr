//! Double-precision math operations and lowering to LLVM intrinsics.
//!
//! Angles are in radians. `fmod` lowers to LLVM's floating-point remainder instruction.

use anyhow::{Result, anyhow};
use hugr_core::{HugrView, Node, ops::ExtensionOp, std_extensions::arithmetic::math::MathOps};
use inkwell::{types::BasicType, values::BasicValue};

use crate::{
    custom::CodegenExtsBuilder,
    emit::{
        EmitOpArgs,
        func::EmitFuncContext,
        get_intrinsic,
        ops::{emit_custom_binary_op, emit_custom_unary_op},
    },
};

fn emit_math_op<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: MathOps,
) -> Result<()> {
    match op {
        MathOps::fmod => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
            Ok(vec![
                ctx.builder()
                    .build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        // `pow` and the float extension's `fpow` have the same LLVM lowering to
        // `llvm.pow.f64`. Their lowerings could be shared in a future refactor.
        MathOps::pow | MathOps::atan2 => {
            emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                let float_ty = ctx.iw_context().f64_type().as_basic_type_enum();
                let name: &str = op.into();
                let func = get_intrinsic(
                    ctx.get_current_module(),
                    format!("llvm.{name}.f64"),
                    [float_ty],
                )?;
                Ok(vec![
                    ctx.builder()
                        .build_call(func, &[lhs.into(), rhs.into()], "")?
                        .try_as_basic_value()
                        .unwrap_basic(),
                ])
            })
        }
        MathOps::sin
        | MathOps::cos
        | MathOps::tan
        | MathOps::atan
        | MathOps::asin
        | MathOps::acos
        | MathOps::exp
        | MathOps::exp2
        | MathOps::log
        | MathOps::log2
        | MathOps::log10 => emit_custom_unary_op(context, args, |ctx, value, _| {
            let float_ty = ctx.iw_context().f64_type().as_basic_type_enum();
            let name: &str = op.into();
            let func = get_intrinsic(
                ctx.get_current_module(),
                format!("llvm.{name}.f64"),
                [float_ty],
            )?;
            Ok(vec![
                ctx.builder()
                    .build_call(func, &[value.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic(),
            ])
        }),
        _ => Err(anyhow!("MathOpEmitter: unimplemented op: {op:?}")),
    }
}

/// Register math lowering and its floating-point types and constants.
pub fn add_math_extensions<'a, H: HugrView<Node = Node> + 'a>(
    builder: CodegenExtsBuilder<'a, H>,
) -> CodegenExtsBuilder<'a, H> {
    builder
        .add_float_extensions()
        .simple_extension_op::<MathOps>(emit_math_op)
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Register math lowering and its floating-point types and constants.
    #[must_use]
    pub fn add_math_extensions(self) -> Self {
        add_math_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use hugr_core::{
        builder::{Dataflow, DataflowHugr},
        extension::{SignatureFunc, simple_op::MakeOpDef},
        std_extensions::{
            STD_REG,
            arithmetic::{
                float_types::{ConstF64, float64_type},
                math::MathOpBuilder,
            },
        },
        types::TypeRow,
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        test::{TestContext, exec_ctx, llvm_ctx},
    };

    #[rstest]
    #[case::sin(MathOps::sin)]
    #[case::cos(MathOps::cos)]
    #[case::tan(MathOps::tan)]
    #[case::atan(MathOps::atan)]
    #[case::atan2(MathOps::atan2)]
    #[case::asin(MathOps::asin)]
    #[case::acos(MathOps::acos)]
    #[case::exp(MathOps::exp)]
    #[case::exp2(MathOps::exp2)]
    #[case::log(MathOps::log)]
    #[case::log2(MathOps::log2)]
    #[case::log10(MathOps::log10)]
    #[case::pow(MathOps::pow)]
    #[case::fmod(MathOps::fmod)]
    fn math_emission(mut llvm_ctx: TestContext, #[case] op: MathOps) {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_math_extensions);
        let SignatureFunc::PolyFuncType(signature) = op.signature() else {
            panic!("Expected PolyFuncType");
        };
        let inputs: TypeRow = signature.body().input.clone().try_into().unwrap();
        let outputs: TypeRow = signature.body().output.clone().try_into().unwrap();
        let mut hugr = SimpleHugrConfig::new()
            .with_ins(inputs)
            .with_outs(outputs)
            .with_extensions(STD_REG.clone())
            .finish(|mut builder| {
                let inputs: Vec<_> = builder.input_wires().collect();
                let output = builder.add_math_op(op, inputs).unwrap();
                builder.finish_hugr_with_outputs([output]).unwrap()
            });
        let name: &str = op.into();
        check_emission!(name, hugr, llvm_ctx);
    }

    #[rstest]
    #[case(MathOps::sin, &[0.5], 0.5_f64.sin())]
    #[case(MathOps::cos, &[0.5], 0.5_f64.cos())]
    #[case(MathOps::tan, &[0.5], 0.5_f64.tan())]
    #[case(MathOps::atan, &[-2.0], (-2.0_f64).atan())]
    #[case(MathOps::atan2, &[2.0, 1.0], 2.0_f64.atan2(1.0))]
    #[case(MathOps::atan2, &[2.0, -1.0], 2.0_f64.atan2(-1.0))]
    #[case(MathOps::atan2, &[-2.0, -1.0], (-2.0_f64).atan2(-1.0))]
    #[case(MathOps::atan2, &[-2.0, 1.0], (-2.0_f64).atan2(1.0))]
    #[case(MathOps::asin, &[0.5], 0.5_f64.asin())]
    #[case(MathOps::acos, &[0.5], 0.5_f64.acos())]
    #[case(MathOps::exp, &[1.0], 1.0_f64.exp())]
    #[case(MathOps::exp2, &[3.0], 8.0)]
    #[case(MathOps::log, &[2.0], 2.0_f64.ln())]
    #[case(MathOps::log2, &[8.0], 3.0)]
    #[case(MathOps::log10, &[100.0], 2.0)]
    #[case(MathOps::pow, &[2.0, 3.0], 8.0)]
    #[case(MathOps::fmod, &[-7.0, 2.0], -1.0)]
    fn math_exec(
        mut exec_ctx: TestContext,
        #[case] op: MathOps,
        #[case] inputs: &[f64],
        #[case] expected: f64,
    ) {
        exec_ctx.add_extensions(add_math_extensions);
        let hugr = SimpleHugrConfig::new()
            .with_outs([float64_type()])
            .with_extensions(STD_REG.clone())
            .finish(|mut builder| {
                let inputs: Vec<_> = inputs
                    .iter()
                    .map(|&v| builder.add_load_value(ConstF64::new(v)))
                    .collect();
                let output = builder.add_math_op(op, inputs).unwrap();
                builder.finish_hugr_with_outputs([output]).unwrap()
            });
        let actual = exec_ctx.exec_hugr_f64(hugr, "main");
        assert!(
            (actual - expected).abs() < 1e-12,
            "{op:?}: {actual} != {expected}"
        );
    }
}

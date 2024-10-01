use std::{any::TypeId, collections::HashSet};

use anyhow::{anyhow, bail, Result};
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::ops::{constant::CustomConst, Value};
use hugr::std_extensions::arithmetic::float_ops::FloatOps;
use hugr::types::CustomType;
use hugr::{
    std_extensions::arithmetic::{
        float_ops,
        float_types::{self, ConstF64, FLOAT64_CUSTOM_TYPE},
    },
    HugrView,
};
use inkwell::types::BasicTypeEnum;
use inkwell::{
    types::{BasicType, FloatType},
    values::{BasicValue, BasicValueEnum},
};

use crate::emit::emit_value;
use crate::emit::ops::{emit_custom_binary_op, emit_custom_unary_op};
use crate::emit::{func::EmitFuncContext, EmitOpArgs};
use crate::types::TypingSession;

use super::{CodegenExtension, CodegenExtsMap};

/// A [CodegenExtension] for the [hugr::std_extensions::arithmetic::float_types] extension.
pub struct FloatTypesCodegenExtension;

impl<H: HugrView> CodegenExtension<H> for FloatTypesCodegenExtension {
    fn extension(&self) -> ExtensionId {
        float_types::EXTENSION_ID
    }

    fn llvm_type<'c>(
        &self,
        context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> anyhow::Result<BasicTypeEnum<'c>> {
        if hugr_type == &FLOAT64_CUSTOM_TYPE {
            Ok(context.iw_context().f64_type().as_basic_type_enum())
        } else {
            Err(anyhow!(
                "FloatCodegenExtension: Unsupported type: {}",
                hugr_type
            ))
        }
    }

    fn emit_extension_op<'c>(
        &self,
        _: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        bail!("FloatTypesCodegenExtension does not implement any extension ops, try FloatOpsCodegenExtension for: {:?}", args.node().as_ref())
    }
    fn supported_consts(&self) -> HashSet<TypeId> {
        [TypeId::of::<ConstF64>()].into_iter().collect()
    }

    fn load_constant<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn hugr::ops::constant::CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        let Some(k) = konst.downcast_ref::<ConstF64>() else {
            return Ok(None);
        };
        let ty: FloatType<'c> = context.llvm_type(&k.get_type())?.try_into().unwrap();
        Ok(Some(ty.const_float(k.value()).as_basic_value_enum()))
    }
}

struct FloatOpsCodegenExtension;

impl<H: HugrView> CodegenExtension<H> for FloatOpsCodegenExtension {
    fn extension(&self) -> hugr::extension::ExtensionId {
        float_ops::EXTENSION_ID
    }

    fn llvm_type<'c>(
        &self,
        _context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> anyhow::Result<BasicTypeEnum<'c>> {
        Err(anyhow!(
            "FloatOpsCodegenExtension: unsupported type: {hugr_type}"
        ))
    }

    fn emit_extension_op<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let op = FloatOps::from_optype(&args.node().generalise()).ok_or(anyhow!(
            "FloatOpEmitter: from_optype_failed: {:?}",
            args.node().as_ref()
        ))?;
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
                Ok(vec![ctx
                    .builder()
                    .build_float_add(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum()])
            }),
            FloatOps::fsub => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                Ok(vec![ctx
                    .builder()
                    .build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum()])
            }),
            FloatOps::fneg => emit_custom_unary_op(context, args, |ctx, v, _| {
                Ok(vec![ctx
                    .builder()
                    .build_float_neg(v.into_float_value(), "")?
                    .as_basic_value_enum()])
            }),
            FloatOps::fmul => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                Ok(vec![ctx
                    .builder()
                    .build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum()])
            }),
            FloatOps::fdiv => emit_custom_binary_op(context, args, |ctx, (lhs, rhs), _| {
                Ok(vec![ctx
                    .builder()
                    .build_float_div(lhs.into_float_value(), rhs.into_float_value(), "")?
                    .as_basic_value_enum()])
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
}

/// Emit a float comparison operation.
fn emit_fcmp<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
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
        // convert to whatever BOOL_T is
        Ok(vec![ctx
            .builder()
            .build_select(r, true_val, false_val, "")?])
    })
}

pub fn add_float_extensions<H: HugrView>(cem: CodegenExtsMap<'_, H>) -> CodegenExtsMap<'_, H> {
    cem.add_cge(FloatTypesCodegenExtension)
        .add_cge(FloatOpsCodegenExtension)
}

impl<H: HugrView> CodegenExtsMap<'_, H> {
    pub fn add_float_extensions(self) -> Self {
        add_float_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use hugr::extension::simple_op::MakeOpDef;
    use hugr::extension::SignatureFunc;
    use hugr::std_extensions::arithmetic::float_ops::{self, FloatOps};
    use hugr::types::TypeRow;
    use hugr::Hugr;
    use hugr::{
        builder::{Dataflow, DataflowSubContainer},
        std_extensions::arithmetic::{
            float_ops::FLOAT_OPS_REGISTRY,
            float_types::{ConstF64, FLOAT64_TYPE},
        },
    };
    use rstest::rstest;

    use super::add_float_extensions;
    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        test::{llvm_ctx, TestContext},
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
            .with_extensions(float_ops::FLOAT_OPS_REGISTRY.to_owned())
            .finish(|mut builder| {
                let outputs = builder
                    .add_dataflow_op(op, builder.input_wires())
                    .unwrap()
                    .outputs();
                builder.finish_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    fn const_float(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_float_extensions);
        let hugr = SimpleHugrConfig::new()
            .with_outs(FLOAT64_TYPE)
            .with_extensions(FLOAT_OPS_REGISTRY.to_owned())
            .finish(|mut builder| {
                let c = builder.add_load_value(ConstF64::new(3.12));
                builder.finish_with_outputs([c]).unwrap()
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
    fn float_operations(mut llvm_ctx: TestContext, #[case] op: FloatOps) {
        let name: &str = op.into();
        let hugr = test_float_op(op);
        llvm_ctx.add_extensions(add_float_extensions);
        check_emission!(name, hugr, llvm_ctx);
    }
}

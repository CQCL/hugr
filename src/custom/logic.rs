use hugr::{
    extension::{simple_op::MakeExtensionOp, ExtensionId},
    ops::{ExtensionOp, Value},
    std_extensions::logic::{self, LogicOp},
    types::{CustomType, SumType},
    HugrView,
};
use inkwell::{types::BasicTypeEnum, IntPredicate};

use crate::{
    emit::{emit_value, func::EmitFuncContext, EmitOpArgs},
    sum::LLVMSumValue,
    types::TypingSession,
};

use super::{CodegenExtension, CodegenExtsMap};
use anyhow::{anyhow, Result};

/// A [CodegenExtension] for the [hugr::std_extensions::logic]
/// extension.
pub struct LogicCodegenExtension;

impl<H: HugrView> CodegenExtension<H> for LogicCodegenExtension {
    fn extension(&self) -> ExtensionId {
        logic::EXTENSION_ID
    }

    fn llvm_type<'c>(
        &self,
        _context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        Err(anyhow!(
            "LogicCodegenExtension: unsupported type: {}",
            hugr_type
        ))
    }

    fn emit_extension_op<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let lot = LogicOp::from_optype(&args.node().generalise()).ok_or(anyhow!(
            "LogicOpEmitter: from_optype_failed: {:?}",
            args.node().as_ref()
        ))?;
        let builder = context.builder();
        // Turn bool sum inputs into i1's
        let mut inputs = vec![];
        for inp in args.inputs {
            let bool_ty = context.llvm_sum_type(SumType::Unit { size: 2 })?;
            let bool_val = LLVMSumValue::try_new(inp, bool_ty)?;
            inputs.push(bool_val.build_get_tag(builder)?);
        }
        let res = match lot {
            LogicOp::And => {
                let mut acc = inputs[0];
                for inp in inputs.into_iter().skip(1) {
                    acc = builder.build_and(acc, inp, "")?;
                }
                acc
            }
            LogicOp::Or => {
                let mut acc = inputs[0];
                for inp in inputs.into_iter().skip(1) {
                    acc = builder.build_or(acc, inp, "")?;
                }
                acc
            }
            LogicOp::Eq => {
                let x = inputs.pop().unwrap();
                let y = inputs.pop().unwrap();
                let mut acc = builder.build_int_compare(IntPredicate::EQ, x, y, "")?;
                for inp in inputs {
                    let eq = builder.build_int_compare(IntPredicate::EQ, inp, x, "")?;
                    acc = builder.build_and(acc, eq, "")?;
                }
                acc
            }
            op => {
                return Err(anyhow!("LogicOpEmitter: Unknown op: {op:?}"));
            }
        };
        // Turn result back into sum
        let res = builder.build_int_cast(res, context.iw_context().bool_type(), "")?;
        let true_val = emit_value(context, &Value::true_val())?;
        let false_val = emit_value(context, &Value::false_val())?;
        let res = context
            .builder()
            .build_select(res, true_val, false_val, "")?;
        args.outputs.finish(context.builder(), vec![res])
    }
}

/// Populates a [CodegenExtsMap] with all extensions needed to lower logic ops,
/// types, and constants.
pub fn add_logic_extensions<H: HugrView>(cem: CodegenExtsMap<'_, H>) -> CodegenExtsMap<'_, H> {
    cem.add_cge(LogicCodegenExtension)
}

impl<H: HugrView> CodegenExtsMap<'_, H> {
    /// Populates a [CodegenExtsMap] with all extensions needed to lower logic ops,
    /// types, and constants.
    pub fn add_logic_extensions(self) -> Self {
        add_logic_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Dataflow, DataflowSubContainer},
        extension::{prelude::BOOL_T, ExtensionRegistry},
        std_extensions::logic::{self, LogicOp},
        Hugr,
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        custom::logic::add_logic_extensions,
        emit::test::SimpleHugrConfig,
        test::{llvm_ctx, TestContext},
    };

    fn test_logic_op(op: LogicOp, arity: usize) -> Hugr {
        SimpleHugrConfig::new()
            .with_ins(vec![BOOL_T; arity])
            .with_outs(vec![BOOL_T])
            .with_extensions(ExtensionRegistry::try_new(vec![logic::EXTENSION.to_owned()]).unwrap())
            .finish(|mut builder| {
                let outputs = builder
                    .add_dataflow_op(op, builder.input_wires())
                    .unwrap()
                    .outputs();
                builder.finish_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    fn and(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(LogicOp::And, 2);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn or(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(LogicOp::Or, 2);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn eq(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(LogicOp::Eq, 2);
        check_emission!(hugr, llvm_ctx);
    }
}

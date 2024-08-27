use hugr::{
    extension::{simple_op::MakeExtensionOp, ExtensionId},
    ops::{CustomOp, Value},
    std_extensions::logic::{self, ConcreteLogicOp, NaryLogic},
    types::{CustomType, SumType},
    HugrView,
};
use inkwell::{types::BasicTypeEnum, IntPredicate};

use crate::{
    emit::{emit_value, func::EmitFuncContext, EmitOp, EmitOpArgs},
    sum::LLVMSumValue,
    types::TypingSession,
};

use super::{CodegenExtension, CodegenExtsMap};
use anyhow::{anyhow, Result};

struct LogicOpEmitter<'c, 'd, H>(&'d mut EmitFuncContext<'c, H>);

impl<'c, H: HugrView> EmitOp<'c, CustomOp, H> for LogicOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        let lot = ConcreteLogicOp::from_optype(&args.node().generalise()).ok_or(anyhow!(
            "LogicOpEmitter: from_optype_failed: {:?}",
            args.node().as_ref()
        ))?;
        let builder = self.0.builder();
        // Turn bool sum inputs into i1's
        let mut inputs = vec![];
        for inp in args.inputs {
            let bool_ty = self.0.llvm_sum_type(SumType::Unit { size: 2 })?;
            let bool_val = LLVMSumValue::try_new(inp, bool_ty)?;
            inputs.push(bool_val.build_get_tag(builder)?);
        }
        let res = match lot.0 {
            NaryLogic::And => {
                let mut acc = inputs[0];
                for inp in inputs.into_iter().skip(1) {
                    acc = builder.build_and(acc, inp, "")?;
                }
                acc
            }
            NaryLogic::Or => {
                let mut acc = inputs[0];
                for inp in inputs.into_iter().skip(1) {
                    acc = builder.build_or(acc, inp, "")?;
                }
                acc
            }
            NaryLogic::Eq => {
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
        let res = builder.build_int_cast(res, self.0.iw_context().bool_type(), "")?;
        let true_val = emit_value(self.0, &Value::true_val())?;
        let false_val = emit_value(self.0, &Value::false_val())?;
        let res = self
            .0
            .builder()
            .build_select(res, true_val, false_val, "")?;
        args.outputs.finish(self.0.builder(), vec![res])
    }
}

/// A [CodegenExtension] for the [hugr::std_extensions::logic]
/// extension.
pub struct LogicCodegenExtension;

impl<'c, H: HugrView> CodegenExtension<'c, H> for LogicCodegenExtension {
    fn extension(&self) -> ExtensionId {
        logic::EXTENSION_ID
    }

    fn llvm_type<'d>(
        &self,
        _context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        Err(anyhow!(
            "LogicCodegenExtension: unsupported type: {}",
            hugr_type
        ))
    }

    fn emitter<'a>(
        &self,
        context: &'a mut EmitFuncContext<'c, H>,
    ) -> Box<dyn EmitOp<'c, CustomOp, H> + 'a> {
        Box::new(LogicOpEmitter(context))
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
        std_extensions::logic::{self, NaryLogic},
        Hugr,
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        custom::logic::add_logic_extensions,
        emit::test::SimpleHugrConfig,
        test::{llvm_ctx, TestContext},
    };

    fn test_logic_op(op: NaryLogic, arity: usize) -> Hugr {
        SimpleHugrConfig::new()
            .with_ins(vec![BOOL_T; arity])
            .with_outs(vec![BOOL_T])
            .with_extensions(ExtensionRegistry::try_new(vec![logic::EXTENSION.to_owned()]).unwrap())
            .finish(|mut builder| {
                let outputs = builder
                    .add_dataflow_op(op.with_n_inputs(arity as u64), builder.input_wires())
                    .unwrap()
                    .outputs();
                builder.finish_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    fn and(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(NaryLogic::And, 3);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn or(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(NaryLogic::Or, 3);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn eq(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(NaryLogic::Eq, 3);
        check_emission!(hugr, llvm_ctx);
    }
}

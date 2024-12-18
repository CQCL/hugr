use hugr_core::{
    extension::simple_op::MakeExtensionOp,
    ops::{ExtensionOp, NamedOp, Value},
    std_extensions::logic::{self, LogicOp},
    types::SumType,
    HugrView,
};
use inkwell::IntPredicate;

use crate::{
    custom::CodegenExtsBuilder,
    emit::{emit_value, func::EmitFuncContext, EmitOpArgs},
    sum::LLVMSumValue,
};

use anyhow::{anyhow, Result};

fn emit_logic_op<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
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
        let bool_ty = context.llvm_sum_type(SumType::new_unary(2))?;
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
        LogicOp::Not => {
            let x = inputs[0];
            builder.build_not(x, "")?
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

/// Populates a [CodegenExtsBuilder] with all extensions needed to lower logic
/// ops.
pub fn add_logic_extensions<'a, H: HugrView + 'a>(
    cem: CodegenExtsBuilder<'a, H>,
) -> CodegenExtsBuilder<'a, H> {
    cem.extension_op(logic::EXTENSION_ID, LogicOp::Eq.name(), emit_logic_op)
        .extension_op(logic::EXTENSION_ID, LogicOp::And.name(), emit_logic_op)
        .extension_op(logic::EXTENSION_ID, LogicOp::Or.name(), emit_logic_op)
        .extension_op(logic::EXTENSION_ID, LogicOp::Not.name(), emit_logic_op)
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    /// Populates a [CodegenExtsBuilder] with all extensions needed to lower
    /// logic ops.
    pub fn add_logic_extensions(self) -> Self {
        add_logic_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{Dataflow, DataflowSubContainer},
        extension::{prelude::bool_t, ExtensionRegistry},
        std_extensions::logic::{self, LogicOp},
        Hugr,
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        extension::logic::add_logic_extensions,
        test::{llvm_ctx, TestContext},
    };

    fn test_logic_op(op: LogicOp, arity: usize) -> Hugr {
        SimpleHugrConfig::new()
            .with_ins(vec![bool_t(); arity])
            .with_outs(vec![bool_t()])
            .with_extensions(ExtensionRegistry::new(vec![logic::EXTENSION.to_owned()]))
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

    #[rstest]
    fn not(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_logic_extensions);
        let hugr = test_logic_op(LogicOp::Not, 1);
        check_emission!(hugr, llvm_ctx);
    }
}

use std::{any::TypeId, collections::HashSet};

use hugr::{
    extension::{simple_op::MakeExtensionOp, ExtensionId},
    ops::{constant::CustomConst, CustomOp, NamedOp, Value},
    std_extensions::arithmetic::{
        int_ops::{self, ConcreteIntOp, IntOpDef},
        int_types::{self, ConstInt},
    },
    types::{CustomType, TypeArg},
    HugrView,
};
use inkwell::{
    types::{BasicTypeEnum, IntType},
    values::{BasicValue, BasicValueEnum},
};

use crate::emit::{
    emit_value, func::EmitFuncContext, ops::emit_custom_binary_op, ops::emit_custom_unary_op,
    EmitOp, EmitOpArgs, NullEmitLlvm,
};
use crate::types::TypingSession;

use super::{CodegenExtension, CodegenExtsMap};
use anyhow::{anyhow, Result};

struct IntOpEmitter<'c, 'd, H>(&'d mut EmitFuncContext<'c, H>);

/// Emit an integer comparison operation.
fn emit_icmp<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, CustomOp, H>,
    pred: inkwell::IntPredicate,
) -> Result<()> {
    let true_val = emit_value(context, &Value::true_val())?;
    let false_val = emit_value(context, &Value::false_val())?;

    emit_custom_binary_op(context, args, |builder, (lhs, rhs), _| {
        // get result as an i1
        let r = builder.build_int_compare(pred, lhs.into_int_value(), rhs.into_int_value(), "")?;
        // convert to whatever BOOL_T is
        Ok(vec![builder.build_select(r, true_val, false_val, "")?])
    })
}

impl<'c, H: HugrView> EmitOp<'c, CustomOp, H> for IntOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        let iot = ConcreteIntOp::from_optype(&args.node().generalise()).ok_or(anyhow!(
            "IntOpEmitter from_optype_failed: {:?}",
            args.node().as_ref()
        ))?;
        match iot.def {
            IntOpDef::iadd => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::imul => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::isub => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::idiv_s => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::idiv_u => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::imod_s => emit_custom_binary_op(self.0, args, |builder, (lhs, rhs), _| {
                Ok(vec![builder
                    .build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::ineg => emit_custom_unary_op(self.0, args, |builder, arg, _| {
                Ok(vec![builder
                    .build_int_neg(arg.into_int_value(), "")?
                    .as_basic_value_enum()])
            }),
            IntOpDef::ieq => emit_icmp(self.0, args, inkwell::IntPredicate::EQ),
            IntOpDef::ilt_s => emit_icmp(self.0, args, inkwell::IntPredicate::SLT),
            IntOpDef::igt_s => emit_icmp(self.0, args, inkwell::IntPredicate::SGT),
            IntOpDef::ile_s => emit_icmp(self.0, args, inkwell::IntPredicate::SLE),
            IntOpDef::ige_s => emit_icmp(self.0, args, inkwell::IntPredicate::SGE),
            IntOpDef::ilt_u => emit_icmp(self.0, args, inkwell::IntPredicate::ULT),
            IntOpDef::igt_u => emit_icmp(self.0, args, inkwell::IntPredicate::UGT),
            IntOpDef::ile_u => emit_icmp(self.0, args, inkwell::IntPredicate::ULE),
            IntOpDef::ige_u => emit_icmp(self.0, args, inkwell::IntPredicate::UGE),
            _ => Err(anyhow!("IntOpEmitter: unimplemented op: {}", iot.name())),
        }
    }
}

/// A [CodegenExtension] for the [hugr::std_extensions::arithmetic::int_ops]
/// extension.
///
/// TODO: very incomplete
pub struct IntOpsCodegenExtension;

impl<'c, H: HugrView> CodegenExtension<'c, H> for IntOpsCodegenExtension {
    fn extension(&self) -> ExtensionId {
        int_ops::EXTENSION_ID
    }

    fn llvm_type<'d>(
        &self,
        _context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        Err(anyhow!(
            "IntOpsCodegenExtension: unsupported type: {}",
            hugr_type
        ))
    }

    fn emitter<'a>(
        &self,
        context: &'a mut EmitFuncContext<'c, H>,
    ) -> Box<dyn EmitOp<'c, CustomOp, H> + 'a> {
        Box::new(IntOpEmitter(context))
    }
}

/// A [CodegenExtension] for the [hugr::std_extensions::arithmetic::int_types]
/// extension.
///
/// TODO: very incomplete
pub struct IntTypesCodegenExtension;

impl<'c, H: HugrView> EmitOp<'c, CustomOp, H> for IntTypesCodegenExtension {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        Err(anyhow!(
            "IntTypesCodegenExtension: unsupported op: {}",
            args.node().name()
        ))
    }
}

impl<'c, H: HugrView> CodegenExtension<'c, H> for IntTypesCodegenExtension {
    fn extension(&self) -> ExtensionId {
        int_types::EXTENSION_ID
    }

    fn llvm_type<'d>(
        &self,
        context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        if let [TypeArg::BoundedNat { n }] = hugr_type.args() {
            let m = *n as usize;
            if m < int_types::INT_TYPES.len() && int_types::INT_TYPES[m] == hugr_type.clone().into()
            {
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

    fn emitter<'a>(
        &self,
        _: &'a mut EmitFuncContext<'c, H>,
    ) -> Box<dyn EmitOp<'c, CustomOp, H> + 'a> {
        Box::new(NullEmitLlvm)
    }

    fn supported_consts(&self) -> HashSet<TypeId> {
        [TypeId::of::<ConstInt>()].into_iter().collect()
    }

    fn load_constant(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn hugr::ops::constant::CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        let Some(k) = konst.downcast_ref::<ConstInt>() else {
            return Ok(None);
        };
        let ty: IntType<'c> = context
            .llvm_type(&k.get_type())?
            .try_into()
            .map_err(|_| anyhow!("Failed to get ConstInt as IntType"))?;
        // k.value_u() is in two's complement representation of the exactly
        // correct bit width, so we are safe to unconditionally retrieve the
        // unsigned value and do no sign extension.
        Ok(Some(ty.const_int(k.value_u(), false).into()))
    }
}

/// Populates a [CodegenExtsMap] with all extensions needed to lower int ops,
/// types, and constants.
pub fn add_int_extensions<H: HugrView>(cem: CodegenExtsMap<'_, H>) -> CodegenExtsMap<'_, H> {
    cem.add_cge(IntOpsCodegenExtension)
        .add_cge(IntTypesCodegenExtension)
}

impl<H: HugrView> CodegenExtsMap<'_, H> {
    /// Populates a [CodegenExtsMap] with all extensions needed to lower int ops,
    /// types, and constants.
    pub fn add_int_extensions(self) -> Self {
        add_int_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Dataflow, DataflowSubContainer},
        extension::prelude::BOOL_T,
        std_extensions::arithmetic::{int_ops, int_types::INT_TYPES},
        types::TypeRow,
        Hugr,
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        custom::int::add_int_extensions,
        emit::test::SimpleHugrConfig,
        test::{llvm_ctx, TestContext},
    };

    fn test_binary_int_op(name: impl AsRef<str>, log_width: u8) -> Hugr {
        let ty = &INT_TYPES[log_width as usize];
        test_binary_int_op_with_results(name, log_width, vec![ty.clone()])
    }

    fn test_binary_icmp_op(name: impl AsRef<str>, log_width: u8) -> Hugr {
        test_binary_int_op_with_results(name, log_width, vec![BOOL_T])
    }
    fn test_binary_int_op_with_results(
        name: impl AsRef<str>,
        log_width: u8,
        output_types: impl Into<TypeRow>,
    ) -> Hugr {
        let ty = &INT_TYPES[log_width as usize];
        SimpleHugrConfig::new()
            .with_ins(vec![ty.clone(), ty.clone()])
            .with_outs(output_types.into())
            .with_extensions(int_ops::INT_OPS_REGISTRY.clone())
            .finish(|mut hugr_builder| {
                let [in1, in2] = hugr_builder.input_wires_arr();
                let ext_op = int_ops::EXTENSION
                    .instantiate_extension_op(
                        name.as_ref(),
                        [(log_width as u64).into()],
                        &int_ops::INT_OPS_REGISTRY,
                    )
                    .unwrap();
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, [in1, in2])
                    .unwrap()
                    .outputs();
                hugr_builder.finish_with_outputs(outputs).unwrap()
            })
    }

    fn test_unary_int_op(name: impl AsRef<str>, log_width: u8) -> Hugr {
        let ty = &INT_TYPES[log_width as usize];
        SimpleHugrConfig::new()
            .with_ins(vec![ty.clone()])
            .with_outs(vec![ty.clone()])
            .with_extensions(int_ops::INT_OPS_REGISTRY.clone())
            .finish(|mut hugr_builder| {
                let [in1] = hugr_builder.input_wires_arr();
                let ext_op = int_ops::EXTENSION
                    .instantiate_extension_op(
                        name.as_ref(),
                        [(log_width as u64).into()],
                        &int_ops::INT_OPS_REGISTRY,
                    )
                    .unwrap();
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, [in1])
                    .unwrap()
                    .outputs();
                hugr_builder.finish_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    fn test_neg_emission(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_int_extensions);
        let hugr = test_unary_int_op("ineg", 2);
        check_emission!("ineg", hugr, llvm_ctx);
    }

    #[rstest]
    #[case::iadd("iadd", 3)]
    #[case::isub("isub", 6)]
    fn test_binop_emission(mut llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        llvm_ctx.add_extensions(add_int_extensions);
        let hugr = test_binary_int_op(op.clone(), width);
        check_emission!(op.clone(), hugr, llvm_ctx);
    }

    #[rstest]
    #[case::ieq("ieq", 1)]
    #[case::ilt_s("ilt_s", 0)]
    fn test_cmp_emission(mut llvm_ctx: TestContext, #[case] op: String, #[case] width: u8) {
        llvm_ctx.add_extensions(add_int_extensions);
        let hugr = test_binary_icmp_op(op.clone(), width);
        check_emission!(op.clone(), hugr, llvm_ctx);
    }
}

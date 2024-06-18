use std::{any::TypeId, collections::HashSet};

use hugr::{
    extension::{simple_op::MakeExtensionOp, ExtensionId},
    ops::{constant::CustomConst, CustomOp, NamedOp},
    std_extensions::arithmetic::{
        int_ops::{self, ConcreteIntOp},
        int_types::{self, ConstInt},
    },
    types::{CustomType, TypeArg},
    HugrView,
};
use inkwell::{
    types::{BasicTypeEnum, IntType},
    values::BasicValueEnum,
};

use crate::{
    emit::{func::EmitFuncContext, EmitOp, EmitOpArgs, NullEmitLlvm},
    types::TypingSession,
};

use super::{CodegenExtension, CodegenExtsMap};
use anyhow::{anyhow, Result};

struct IntOpEmitter<'c, 'd, H: HugrView>(&'d mut EmitFuncContext<'c, H>);

impl<'c, H: HugrView> EmitOp<'c, CustomOp, H> for IntOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        let iot = ConcreteIntOp::from_optype(&args.node().generalise())
            .ok_or(anyhow!("IntOpEmitter from_optype_failed"))?;
        match iot.name().as_str() {
            "iadd" => {
                let builder = self.0.builder();
                let [lhs, rhs] = TryInto::<[_; 2]>::try_into(args.inputs).unwrap();
                let a = builder.build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?;
                args.outputs.finish(builder, [a.into()])
            }
            _ => Err(anyhow!("IntOpEmitter: unknown name")),
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
}

/// Populates a [CodegenExtsMap] with all extensions needed to lower int ops,
/// types, and constants.
pub fn add_int_extensions<H: HugrView>(cem: CodegenExtsMap<'_, H>) -> CodegenExtsMap<'_, H> {
    cem.add_cge(IntOpsCodegenExtension)
        .add_cge(IntTypesCodegenExtension)
}

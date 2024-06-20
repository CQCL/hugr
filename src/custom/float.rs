use std::{any::TypeId, collections::HashSet};

use anyhow::{anyhow, Result};
use hugr::{
    ops::{constant::CustomConst, CustomOp},
    std_extensions::arithmetic::{
        float_ops,
        float_types::{self, ConstF64, FLOAT64_CUSTOM_TYPE},
    },
    HugrView,
};
use inkwell::{
    types::{BasicType, FloatType},
    values::{BasicValue, BasicValueEnum},
};

use crate::emit::{func::EmitFuncContext, EmitOp, EmitOpArgs, NullEmitLlvm};

use super::{CodegenExtension, CodegenExtsMap};

struct FloatTypesCodegenExtension;

impl<'c, H: HugrView> CodegenExtension<'c, H> for FloatTypesCodegenExtension {
    fn extension(&self) -> hugr::extension::ExtensionId {
        float_types::EXTENSION_ID
    }

    fn llvm_type(
        &self,
        context: &crate::types::TypingSession<'c, H>,
        hugr_type: &hugr::types::CustomType,
    ) -> anyhow::Result<inkwell::types::BasicTypeEnum<'c>> {
        if hugr_type == &FLOAT64_CUSTOM_TYPE {
            Ok(context.iw_context().f64_type().as_basic_type_enum())
        } else {
            Err(anyhow!(
                "FloatCodegenExtension: Unsupported type: {}",
                hugr_type
            ))
        }
    }

    fn emitter<'a>(
        &self,
        _context: &'a mut crate::emit::func::EmitFuncContext<'c, H>,
    ) -> Box<dyn crate::emit::EmitOp<'c, hugr::ops::CustomOp, H> + 'a> {
        Box::new(NullEmitLlvm)
    }

    fn supported_consts(&self) -> HashSet<TypeId> {
        [TypeId::of::<ConstF64>()].into_iter().collect()
    }

    fn load_constant(
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

impl<'c, H: HugrView> CodegenExtension<'c, H> for FloatOpsCodegenExtension {
    fn extension(&self) -> hugr::extension::ExtensionId {
        float_ops::EXTENSION_ID
    }

    fn llvm_type(
        &self,
        _context: &crate::types::TypingSession<'c, H>,
        hugr_type: &hugr::types::CustomType,
    ) -> anyhow::Result<inkwell::types::BasicTypeEnum<'c>> {
        Err(anyhow!(
            "FloatOpsCodegenExtension: unsupported type: {hugr_type}"
        ))
    }

    fn emitter<'a>(
        &self,
        context: &'a mut crate::emit::func::EmitFuncContext<'c, H>,
    ) -> Box<dyn crate::emit::EmitOp<'c, hugr::ops::CustomOp, H> + 'a> {
        Box::new(FloatOpEmitter(context))
    }
}

// we allow dead code for now, but once we implement the emitter, we should
// remove this
#[allow(dead_code)]
struct FloatOpEmitter<'c, 'd, H: HugrView>(&'d mut EmitFuncContext<'c, H>);

impl<'c, H: HugrView> EmitOp<'c, CustomOp, H> for FloatOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        use hugr::ops::NamedOp;
        let name = args.node().name();
        // This looks strange now, but we will add cases for ops piecemeal, as
        // in the analgous match expression in `IntOpEmitter`.
        #[allow(clippy::match_single_binding)]
        match name.as_str() {
            n => Err(anyhow!("FloatOpEmitter: unknown op: {n}")),
        }
    }
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
}

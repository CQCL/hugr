use std::{any::TypeId, collections::HashSet};

use anyhow::{anyhow, Result};
use hugr::{
    extension::prelude::{self, ConstExternalSymbol, ConstUsize, QB_T, USIZE_T},
    ops::constant::CustomConst,
    types::TypeEnum,
    HugrView,
};
use inkwell::{
    types::{BasicType, BasicTypeEnum, IntType},
    values::BasicValueEnum,
};

use crate::{emit::func::EmitFuncContext, types::TypingSession};

use super::{CodegenExtension, CodegenExtsMap};

/// A helper trait for implementing [CodegenExtension]s for
/// [hugr::extension::prelude].
///
/// All methods have sensible defaults provided, and [DefaultPreludeCodegen] is
/// trivial implementation o this trait, which delegates everything to those
/// default implementations.
///
/// One should use either [PreludeCodegenExtension::new], or
/// [CodegenExtsMap::add_prelude_extensions] to work with the
/// [CodegenExtension].
///
/// TODO several types and ops are unimplemented. We expect to add methods to
/// this trait as necessary, allowing downstream users to customise the lowering
/// of `prelude`.
pub trait PreludeCodegen {
    /// Return the llvm type of [hugr::extension::prelude::USIZE_T]. That type
    /// must be an [IntType].
    fn usize_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [hugr::extension::prelude::QB_T].
    fn qubit_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> impl BasicType<'c> {
        session.iw_context().i16_type()
    }
}

/// A trivial implementation of [PreludeCodegen] which passes all methods
/// through to their default implementations.
#[derive(Default)]
pub struct DefaultPreludeCodegen;

impl PreludeCodegen for DefaultPreludeCodegen {}

#[derive(Clone, Debug, Default)]
pub struct PreludeCodegenExtension<PCG>(PCG);

impl<PCG: PreludeCodegen> PreludeCodegenExtension<PCG> {
    pub fn new(pcg: PCG) -> Self {
        Self(pcg)
    }
}

impl<PCG: PreludeCodegen> From<PCG> for PreludeCodegenExtension<PCG> {
    fn from(pcg: PCG) -> Self {
        Self::new(pcg)
    }
}

impl<'c, H: HugrView, PCG: PreludeCodegen> CodegenExtension<'c, H>
    for PreludeCodegenExtension<PCG>
{
    fn extension(&self) -> hugr::extension::ExtensionId {
        prelude::PRELUDE_ID
    }

    fn llvm_type(
        &self,
        ts: &crate::types::TypingSession<'c, H>,
        hugr_type: &hugr::types::CustomType,
    ) -> anyhow::Result<BasicTypeEnum<'c>> {
        let TypeEnum::Extension(qubit_custom_type) = QB_T.as_type_enum().clone() else {
            panic!("Qubit is not a custom type: {QB_T:?}");
        };
        if &qubit_custom_type == hugr_type {
            return Ok(self.0.qubit_type(ts).as_basic_type_enum());
        }
        let TypeEnum::Extension(usize_custom_type) = USIZE_T.as_type_enum().clone() else {
            panic!("usize is not a custom type: {USIZE_T:?}");
        };
        if &usize_custom_type == hugr_type {
            return Ok(self.0.usize_type(ts).as_basic_type_enum());
        }
        Err(anyhow::anyhow!(
            "Type not supported by prelude extension: {hugr_type:?}"
        ))
    }

    fn emitter<'a>(
        &self,
        _: &'a mut crate::emit::func::EmitFuncContext<'c, H>,
    ) -> Box<dyn crate::emit::EmitOp<'c, hugr::ops::CustomOp, H> + 'a> {
        todo!()
    }

    fn supported_consts(&self) -> HashSet<TypeId> {
        [
            TypeId::of::<ConstUsize>(),
            TypeId::of::<ConstExternalSymbol>(),
        ]
        .into_iter()
        .collect()
    }

    fn load_constant(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        if let Some(k) = konst.downcast_ref::<ConstUsize>() {
            let ty: IntType<'c> = context
                .llvm_type(&k.get_type())?
                .try_into()
                .map_err(|_| anyhow!("Failed to get ConstUsize as IntType"))?;
            Ok(Some(ty.const_int(k.value(), false).into()))
        } else if let Some(k) = konst.downcast_ref::<ConstExternalSymbol>() {
            let llvm_type = context.llvm_type(&k.get_type())?;
            let global = context.get_global(&k.symbol, llvm_type, k.constant)?;
            Ok(Some(
                context
                    .builder()
                    .build_load(global.as_pointer_value(), &k.symbol)?,
            ))
        } else {
            Ok(None)
        }
    }
}

/// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using `pcg`
/// as the implementation.
pub fn add_prelude_extensions<'c, H: HugrView>(
    cem: CodegenExtsMap<'c, H>,
    pcg: impl PreludeCodegen + 'c,
) -> CodegenExtsMap<'c, H> {
    cem.add_cge(PreludeCodegenExtension(pcg))
}

/// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using
/// [DefaultPreludeCodegen] as the implementation.
pub fn add_default_prelude_extensions<H: HugrView>(cem: CodegenExtsMap<H>) -> CodegenExtsMap<H> {
    cem.add_cge(PreludeCodegenExtension::from(DefaultPreludeCodegen))
}

impl<'c, H: HugrView> CodegenExtsMap<'c, H> {
    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using `pcg`
    /// as the implementation.
    pub fn add_default_prelude_extensions(self) -> Self {
        add_default_prelude_extensions(self)
    }

    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using
    /// [DefaultPreludeCodegen] as the implementation.
    pub fn add_prelude_extensions(self, builder: impl PreludeCodegen + 'c) -> Self {
        add_prelude_extensions(self, builder)
    }
}

#[cfg(test)]
mod test {
    use hugr::builder::{Dataflow, DataflowSubContainer};
    use hugr::type_row;
    use hugr::types::Type;
    use rstest::rstest;

    use crate::check_emission;
    use crate::emit::test::SimpleHugrConfig;
    use crate::test::{llvm_ctx, TestContext};

    use super::*;

    struct TestPreludeCodegen;
    impl PreludeCodegen for TestPreludeCodegen {
        fn usize_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> IntType<'c> {
            session.iw_context().i32_type()
        }

        fn qubit_type<'c, H: HugrView>(
            &self,
            session: &TypingSession<'c, H>,
        ) -> impl BasicType<'c> {
            session.iw_context().f64_type()
        }
    }

    #[rstest]
    fn prelude_extension_types(llvm_ctx: TestContext) {
        let ctx = llvm_ctx.iw_context();
        let ext: PreludeCodegenExtension<TestPreludeCodegen> = TestPreludeCodegen.into();
        let tc = llvm_ctx.get_typing_session();

        let TypeEnum::Extension(qb_ct) = QB_T.as_type_enum().clone() else {
            unreachable!()
        };
        let TypeEnum::Extension(usize_ct) = USIZE_T.as_type_enum().clone() else {
            unreachable!()
        };

        assert_eq!(
            ctx.i32_type().as_basic_type_enum(),
            ext.llvm_type(&tc, &usize_ct).unwrap()
        );
        assert_eq!(
            ctx.f64_type().as_basic_type_enum(),
            ext.llvm_type(&tc, &qb_ct).unwrap()
        );
    }

    #[rstest]
    fn prelude_extension_types_in_test_context(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(|x| x.add_prelude_extensions(TestPreludeCodegen));
        let tc = llvm_ctx.get_typing_session();
        assert_eq!(
            llvm_ctx.iw_context().i32_type().as_basic_type_enum(),
            tc.llvm_type(&USIZE_T).unwrap()
        );
        assert_eq!(
            llvm_ctx.iw_context().f64_type().as_basic_type_enum(),
            tc.llvm_type(&QB_T).unwrap()
        );
    }

    #[rstest]
    fn prelude_const_usize(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_default_prelude_extensions);

        let hugr = SimpleHugrConfig::new()
            .with_outs(USIZE_T)
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(17));
                builder.finish_with_outputs([k]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_const_external_symbol(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_default_prelude_extensions);
        let konst1 = ConstExternalSymbol::new("sym1", USIZE_T, true);
        let konst2 = ConstExternalSymbol::new(
            "sym2",
            Type::new_sum([type_row![USIZE_T, Type::new_unit_sum(3)], type_row![]]),
            false,
        );

        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![konst1.get_type(), konst2.get_type()])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k1 = builder.add_load_value(konst1);
                let k2 = builder.add_load_value(konst2);
                builder.finish_with_outputs([k1, k2]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }
}

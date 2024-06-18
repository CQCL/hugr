use std::rc::Rc;

use hugr::Hugr;
use inkwell::{
    context::Context,
    module::Module,
    types::{BasicType, BasicTypeEnum},
};
use itertools::Itertools as _;
use rstest::fixture;

use crate::{
    custom::CodegenExtsMap,
    emit::{EmitHugr, EmitModuleContext, Namer},
    types::{TypeConverter, TypingSession},
};

pub type THugrView = Hugr;

pub struct InstaSettingsBuilder {
    settings: insta::Settings,
    llvm: Option<String>,
    id: Option<usize>,
    omit_expression: bool,
}

impl InstaSettingsBuilder {
    pub fn new_llvm(id: Option<usize>) -> Self {
        let mut r = Self::new(id);
        r.llvm = Some(super::llvm_version().into());
        r
    }

    pub fn new(id: Option<usize>) -> Self {
        Self {
            settings: insta::Settings::clone_current(),
            llvm: None,
            id,
            omit_expression: false,
        }
    }

    pub fn settings_mut(&mut self) -> &mut insta::Settings {
        &mut self.settings
    }

    pub fn finish(mut self) -> Option<insta::internals::SettingsBindDropGuard> {
        let suffix = self
            .llvm
            .into_iter()
            .chain(self.id.into_iter().map(|x| x.to_string()))
            .join("_");
        self.settings.set_snapshot_suffix(suffix);
        self.settings.set_omit_expression(self.omit_expression);
        Some(self.settings.bind_to_scope())
    }
}

pub fn no_extensions(id: CodegenExtsMap<'_, THugrView>) -> CodegenExtsMap<'_, THugrView> {
    id
}

type MakeCodegenExtsMapFn =
    Box<dyn for<'a> Fn(CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView>>;
pub struct TestContext {
    context: Context,
    mk_exts: MakeCodegenExtsMapFn,
    _insta: Option<insta::internals::SettingsBindDropGuard>,
}

impl TestContext {
    fn new(
        ext_builder: impl for<'a> Fn(CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView>
            + 'static,
        insta_settings: Option<InstaSettingsBuilder>,
    ) -> Self {
        let context = Context::create();
        Self {
            context,
            mk_exts: Box::new(ext_builder),
            _insta: insta_settings.and_then(InstaSettingsBuilder::finish),
        }
    }

    pub fn i32(&'_ self) -> BasicTypeEnum<'_> {
        self.context.i32_type().as_basic_type_enum()
    }

    pub fn type_converter(&'_ self) -> Rc<TypeConverter<'_>> {
        TypeConverter::new(&self.context)
    }

    pub fn extensions(&'_ self) -> Rc<CodegenExtsMap<'_, THugrView>> {
        Rc::new((self.mk_exts)(CodegenExtsMap::default()))
    }

    pub fn with_context<'c, T>(&'c self, f: impl FnOnce(&'c Context) -> T) -> T {
        f(&self.context)
    }

    pub fn with_tsesh<'c, T, F>(&'c self, f: F) -> T
    where
        F: for<'d> FnOnce(TypingSession<'c, THugrView>) -> T,
    {
        self.with_context(|ctx| {
            let tc = TypeConverter::new(ctx);

            f(tc.session(self.extensions()))
        })
    }

    pub fn with_emit_context<'c, T>(
        &'c self,
        f: impl FnOnce(EmitHugr<'c, THugrView>) -> (T, EmitHugr<'c, THugrView>),
    ) -> (T, Module<'c>) {
        self.with_context(|ctx| {
            let m = ctx.create_module("test_context");
            let exts = self.extensions();
            let (r, ectx) = f(EmitHugr::new(ctx, m, exts));
            (r, ectx.finish())
        })
    }

    pub fn with_emit_module_context<'c, T>(
        &'c self,
        f: impl FnOnce(EmitModuleContext<'c, THugrView>) -> T,
    ) -> T {
        self.with_context(|ctx| {
            let m = ctx.create_module("test_module");
            f(EmitModuleContext::new(
                m,
                Namer::default().into(),
                self.extensions(),
                TypeConverter::new(ctx),
            ))
        })
    }
}

#[fixture]
pub fn test_ctx(
    #[default(no_extensions)] exts_builder: impl for<'a> Fn(CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView>
        + 'static,
) -> TestContext {
    TestContext::new(exts_builder, None)
}

#[fixture]
pub fn llvm_ctx(
    #[default(-1)] id: i32,
    #[default(no_extensions)] exts_builder: impl for<'a> Fn(CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView>
        + 'static,
) -> TestContext {
    TestContext::new(
        exts_builder,
        Some(InstaSettingsBuilder::new_llvm(
            (id >= 0).then_some(id as usize),
        )),
    )
}

impl Default for InstaSettingsBuilder {
    fn default() -> Self {
        Self::new_llvm(None)
    }
}

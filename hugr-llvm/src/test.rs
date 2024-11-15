use std::rc::Rc;

use hugr_core::Hugr;
use inkwell::{
    context::Context,
    types::{BasicType, BasicTypeEnum},
};
use itertools::Itertools as _;
use rstest::fixture;

use crate::{
    custom::{CodegenExtsBuilder, CodegenExtsMap},
    emit::{test::Emission, EmitHugr, EmitModuleContext, Namer},
    types::{TypeConverter, TypingSession},
    utils::fat::FatExt as _,
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

pub trait MakeCodegenExtsMapFn:
    Fn(CodegenExtsBuilder<'static, THugrView>) -> CodegenExtsBuilder<'static, THugrView> + 'static
{
}

impl<
        F: Fn(CodegenExtsBuilder<'static, THugrView>) -> CodegenExtsBuilder<'static, THugrView>
            + ?Sized
            + 'static,
    > MakeCodegenExtsMapFn for F
{
}

type MakeCodegenExtsMapBox = Box<dyn MakeCodegenExtsMapFn>;
// We would like to just stor a CodegenExtsMap, but we can't because it's
// lifetime parameter would need to be the lifetime of TestContext, which is
// prohibitted. Instead, we store a factory function as below.
pub struct TestContext {
    context: Context,
    mk_exts: MakeCodegenExtsMapBox,
    _insta: Option<insta::internals::SettingsBindDropGuard>,
    namer: Namer,
}

impl TestContext {
    fn new(
        ext_builder: impl MakeCodegenExtsMapFn + 'static,
        insta_settings: Option<InstaSettingsBuilder>,
    ) -> Self {
        let context = Context::create();
        Self {
            context,
            mk_exts: Box::new(ext_builder),
            _insta: insta_settings.and_then(InstaSettingsBuilder::finish),
            namer: Default::default(),
        }
    }

    pub fn i32(&'_ self) -> BasicTypeEnum<'_> {
        self.context.i32_type().as_basic_type_enum()
    }

    pub fn type_converter(&self) -> Rc<TypeConverter<'static>> {
        self.extensions().type_converter
    }

    pub fn extensions(&self) -> CodegenExtsMap<'static, THugrView> {
        (self.mk_exts)(Default::default()).finish()
    }

    pub fn add_extensions(&'_ mut self, f: impl MakeCodegenExtsMapFn) {
        let old_mk_exts = {
            let mut tmp: MakeCodegenExtsMapBox = Box::new(|x| x);
            std::mem::swap(&mut self.mk_exts, &mut tmp);
            tmp
        };
        self.mk_exts = Box::new(move |cem| f(old_mk_exts(cem)))
    }

    pub fn iw_context(&self) -> &Context {
        &self.context
    }

    pub fn get_typing_session(&self) -> TypingSession<'_, 'static> {
        self.type_converter().session(&self.context)
    }

    pub fn get_emit_hugr(&'_ self) -> EmitHugr<'_, 'static, THugrView> {
        let ctx = self.iw_context();
        let m = ctx.create_module("test_context");
        let exts = self.extensions();
        EmitHugr::new(ctx, m, Rc::new(self.namer.clone()), Rc::new(exts))
    }

    pub fn set_namer(&mut self, namer: Namer) {
        self.namer = namer;
    }

    pub fn get_emit_module_context(&'_ self) -> EmitModuleContext<'_, 'static, THugrView> {
        let ctx = self.iw_context();
        let m = ctx.create_module("test_context");
        EmitModuleContext::new(
            &self.context,
            m,
            self.namer.clone().into(),
            self.extensions().into(),
        )
    }

    /// Lower `hugr` to LLVM, then JIT and execute the function named
    /// by `entry_point` in the inner module.
    ///
    /// That function must take no arguments and return an LLVM `i64`.
    pub fn exec_hugr_u64(&self, hugr: THugrView, entry_point: impl AsRef<str>) -> u64 {
        let emission = Emission::emit_hugr(hugr.fat_root().unwrap(), self.get_emit_hugr()).unwrap();
        emission.verify().unwrap();

        emission.exec_u64(entry_point).unwrap()
    }

    pub fn exec_hugr_f64(&self, hugr: THugrView, entry_point: impl AsRef<str>) -> f64 {
        let emission = Emission::emit_hugr(hugr.fat_root().unwrap(), self.get_emit_hugr()).unwrap();

        emission.exec_f64(entry_point).unwrap()
    }
}

#[fixture]
pub fn test_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    TestContext::new(
        |_| CodegenExtsBuilder::default(),
        Some(InstaSettingsBuilder::new(id)),
    )
}

#[fixture]
pub fn llvm_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    TestContext::new(
        |_| CodegenExtsBuilder::default(),
        Some(InstaSettingsBuilder::new_llvm(id)),
    )
}

#[fixture]
pub fn exec_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    let mut r = TestContext::new(
        |_| CodegenExtsBuilder::default(),
        Some(InstaSettingsBuilder::new_llvm(id)),
    );
    // we need to refer to functions by name, so no mangling
    r.set_namer(Namer::new("", false));
    r
}

impl Default for InstaSettingsBuilder {
    fn default() -> Self {
        Self::new_llvm(None)
    }
}

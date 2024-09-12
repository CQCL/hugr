use std::rc::Rc;

use hugr::Hugr;
use inkwell::{
    context::Context,
    types::{BasicType, BasicTypeEnum},
};
use itertools::Itertools as _;
use rstest::fixture;

use crate::{
    custom::CodegenExtsMap,
    emit::{test::Emission, EmitHugr, EmitModuleContext, Namer},
    fat::FatExt,
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

// We would like to just stor a CodegenExtsMap, but we can't because it's
// lifetime parameter would need to be the lifetime of TestContext, which is
// prohibitted. Instead, we store a factory function as below.
type MakeCodegenExtsMapFn = Box<dyn for<'a> Fn(&'a Context) -> CodegenExtsMap<'a, THugrView>>;
pub struct TestContext {
    context: Context,
    mk_exts: MakeCodegenExtsMapFn,
    _insta: Option<insta::internals::SettingsBindDropGuard>,
    namer: Namer,
}

impl TestContext {
    fn new(
        ext_builder: impl for<'a> Fn(&'a Context) -> CodegenExtsMap<'a, THugrView> + 'static,
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

    pub fn type_converter(&'_ self) -> Rc<TypeConverter<'_>> {
        TypeConverter::new(&self.context)
    }

    pub fn extensions(&'_ self) -> Rc<CodegenExtsMap<'_, THugrView>> {
        Rc::new((self.mk_exts)(&self.context))
    }

    pub fn add_extensions<
        F: for<'a> Fn(CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView> + 'static,
    >(
        &'_ mut self,
        f: F,
    ) {
        self.add_extensions_with_context(move |_, exts| f(exts))
    }

    pub fn add_extensions_with_context<
        F: for<'a> Fn(&'a Context, CodegenExtsMap<'a, THugrView>) -> CodegenExtsMap<'a, THugrView>
            + 'static,
    >(
        &'_ mut self,
        f: F,
    ) {
        fn dummy(_: &'_ Context) -> CodegenExtsMap<'_, THugrView> {
            unreachable!()
        }
        let mut old_mk_exts: MakeCodegenExtsMapFn = Box::new(dummy);
        std::mem::swap(&mut self.mk_exts, &mut old_mk_exts);
        let new_mk_exts: MakeCodegenExtsMapFn = Box::new(move |ctx| f(ctx, (old_mk_exts)(ctx)));
        self.mk_exts = new_mk_exts;
    }

    pub fn iw_context(&self) -> &Context {
        &self.context
    }

    pub fn get_typing_session(&'_ self) -> TypingSession<'_, THugrView> {
        self.type_converter().session(self.extensions())
    }

    pub fn get_emit_hugr(&'_ self) -> EmitHugr<'_, THugrView> {
        let ctx = self.iw_context();
        let m = ctx.create_module("test_context");
        let namer = self.namer.clone().into();
        let exts = self.extensions();
        EmitHugr::new(ctx, m, namer, exts)
    }

    pub fn set_namer(&mut self, namer: Namer) {
        self.namer = namer;
    }

    pub fn get_emit_module_context(&'_ self) -> EmitModuleContext<'_, THugrView> {
        let ctx = self.iw_context();
        let m = ctx.create_module("test_context");
        EmitModuleContext::new(
            m,
            self.namer.clone().into(),
            self.extensions(),
            self.type_converter(),
        )
    }

    /// Lower `hugr` to LLVM, then JIT and execute the function named
    /// by `entry_point` in the inner module.
    ///
    /// That function must take no arguments and return an LLVM `i64`.
    pub fn exec_hugr_u64(&self, hugr: THugrView, entry_point: impl AsRef<str>) -> u64 {
        let emission = Emission::emit_hugr(hugr.fat_root().unwrap(), self.get_emit_hugr()).unwrap();

        emission.exec_u64(entry_point).unwrap()
    }
}

#[fixture]
pub fn test_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    TestContext::new(
        |_| CodegenExtsMap::default(),
        Some(InstaSettingsBuilder::new(id)),
    )
}

#[fixture]
pub fn llvm_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    TestContext::new(
        |_| CodegenExtsMap::default(),
        Some(InstaSettingsBuilder::new_llvm(id)),
    )
}

#[fixture]
pub fn exec_ctx(#[default(-1)] id: i32) -> TestContext {
    let id = (id >= 0).then_some(id as usize);
    let mut r = TestContext::new(
        |_| CodegenExtsMap::default(),
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

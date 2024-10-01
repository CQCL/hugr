use std::{
    env,
    fmt::Display,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::Result;
use hugr::{
    extension::{prelude, ExtensionId, ExtensionRegistry},
    ops::{DataflowOpTrait as _, ExtensionOp},
    std_extensions::arithmetic::{float_types, int_ops, int_types},
    types::CustomType,
    Hugr, HugrView,
};
use hugr_llvm::{
    custom::{CodegenExtension, CodegenExtsMap},
    emit::{deaggregate_call_result, EmitFuncContext, EmitHugr, EmitOpArgs, Namer},
    fat::FatExt as _,
    types::TypingSession,
};
use inkwell::{context::Context, module::Module, types::BasicTypeEnum};
use itertools::Itertools as _;
use lazy_static::lazy_static;
use rstest::{fixture, rstest};

lazy_static! {
    static ref EXTENSION_REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        int_ops::EXTENSION.to_owned(),
        int_types::EXTENSION.to_owned(),
        prelude::PRELUDE.to_owned(),
        float_types::EXTENSION.to_owned(),
    ])
    .unwrap();
}

// A toy codegen extension for "quantum.tket2" ops.
struct Tket2CodegenExtension;

impl<H: HugrView> CodegenExtension<H> for Tket2CodegenExtension {
    fn extension(&self) -> ExtensionId {
        ExtensionId::new("quantum.tket2").unwrap()
    }

    fn llvm_type<'c>(
        &self,
        _context: &TypingSession<'c, H>,
        _hugr_type: &CustomType,
    ) -> anyhow::Result<BasicTypeEnum<'c>> {
        unimplemented!()
    }

    fn emit_extension_op<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        // we lower all ops by declaring an extern function of the same name
        // and signature, and calling that function.
        // let opaque = args.node().as_ref().clone().into_opaque();
        let node = args.node();
        let sig = node.signature();
        let func_type = context.llvm_func_type(&sig)?;
        let func = context.get_extern_func(node.def().name(), func_type)?;
        let call_args = args.inputs.into_iter().map_into().collect_vec();
        let builder = context.builder();
        let call_result = builder.build_call(func, &call_args, "")?;
        let call_result = deaggregate_call_result(builder, call_result, args.outputs.len())?;
        args.outputs.finish(builder, call_result)
    }
}

// drives `hugr-llvm` to produce an LLVM module from a Hugr.
fn hugr_to_module<'c>(context: &'c Context, hugr: &'c Hugr) -> Module<'c> {
    let module = context.create_module("test_context");
    let namer = Namer::default().into();
    let exts = CodegenExtsMap::default()
        .add_int_extensions()
        .add_logic_extensions()
        .add_default_prelude_extensions()
        .add_float_extensions()
        .add_cge(Tket2CodegenExtension);
    let root = hugr.fat_root().unwrap();
    EmitHugr::new(context, module, namer, exts.into())
        .emit_module(root)
        .unwrap()
        .finish()
}

struct TestConfig {
    test_dir: PathBuf,
}

impl TestConfig {
    pub fn new() -> TestConfig {
        let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/guppy_test_cases");
        assert!(test_dir.is_dir());
        TestConfig { test_dir }
    }
}

impl TestConfig {
    // invokes the path with python, expecting to recieve Hugr serialised as
    // JSON on stdout.
    fn get_guppy_output(&self, path: impl AsRef<Path>) -> Hugr {
        let mut hugr: Hugr =
            serde_json::from_reader(File::open(self.test_dir.join(path)).unwrap()).unwrap();
        hugr.update_validate(&EXTENSION_REGISTRY).unwrap();
        hugr
    }

    fn run<T>(&self, path: impl AsRef<Path>, opt: bool, go: impl FnOnce(String) -> T) -> T {
        let hugr = self.get_guppy_output(path);
        let context = Context::create();
        let module = hugr_to_module(&context, &hugr);
        module
            .verify()
            .unwrap_or_else(|pp| panic!("Failed to verify module: {pp}"));
        if opt {
            let pb = inkwell::passes::PassManager::create(());
            pb.add_promote_memory_to_register_pass();
            pb.run_on(&module);
        }
        go(module.to_string())
    }
}

#[fixture]
fn test_config() -> TestConfig {
    TestConfig::new()
}

fn with_suffix<R>(s: impl Display, go: impl FnOnce() -> R) -> R {
    let mut settings = insta::Settings::clone_current();
    let old_suffix = settings
        .snapshot_suffix()
        .map_or("".to_string(), |s| format!("{s}."));
    let llvm_str = hugr_llvm::llvm_version();
    settings.set_snapshot_suffix(format!("{old_suffix}{llvm_str}.{s}"));
    settings.bind(go)
}

macro_rules! guppy_test {
    ($filename:expr, $testname:ident) => {
        #[rstest]
        #[ignore = "Guppy has not yet been upgraded to hugr-0.12.0"]
        fn $testname(test_config: TestConfig) {
            with_suffix("noopt", || {
                test_config.run($filename, false, |module_string| {
                    insta::assert_snapshot!(module_string)
                });
            });
            with_suffix("opt", || {
                test_config.run($filename, true, |module_string| {
                    insta::assert_snapshot!(module_string)
                });
            });
        }
    };
}

guppy_test!("even_odd.py.json", even_odd);
guppy_test!("even_odd2.py.json", even_odd2);
guppy_test!("planqc-1.py.json", planqc1);
guppy_test!("planqc-2.py.json", planqc2);
guppy_test!("planqc-3.py.json", planqc3);

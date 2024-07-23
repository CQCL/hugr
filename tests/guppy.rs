use std::{
    env,
    fmt::Display,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Result;
use hugr::{
    extension::{prelude, ExtensionId, ExtensionRegistry},
    ops::{CustomOp, OpTrait},
    std_extensions::arithmetic::{float_types, int_ops, int_types},
    Hugr, HugrView,
};
use hugr_llvm::{
    custom::{CodegenExtension, CodegenExtsMap},
    emit::{deaggregate_call_result, EmitHugr, EmitOp, EmitOpArgs, Namer},
    fat::FatExt as _,
};
use inkwell::{context::Context, module::Module};
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

struct Tket2Emitter<'a, 'c, H>(&'a mut hugr_llvm::emit::EmitFuncContext<'c, H>);

impl<'a, 'c, H: HugrView> EmitOp<'c, CustomOp, H> for Tket2Emitter<'a, 'c, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, CustomOp, H>) -> Result<()> {
        // we lower all ops by declaring an extern function of the same name
        // and signature, and calling that function.
        let opaque = args.node().as_ref().clone().into_opaque();
        let sig = opaque.dataflow_signature().unwrap();
        let func_type = self.0.llvm_func_type(&sig)?;
        let func = self.0.get_extern_func(opaque.name(), func_type)?;
        let call_args = args.inputs.into_iter().map_into().collect_vec();
        let builder = self.0.builder();
        let call_result = builder.build_call(func, &call_args, "")?;
        let call_result = deaggregate_call_result(builder, call_result, args.outputs.len())?;
        args.outputs.finish(builder, call_result)
    }
}

// A toy codegen extension for "quantum.tket2" ops.
struct Tket2CodegenExtension;

impl<'c, H: HugrView> CodegenExtension<'c, H> for Tket2CodegenExtension {
    fn extension(&self) -> ExtensionId {
        ExtensionId::new("quantum.tket2").unwrap()
    }

    fn llvm_type(
        &self,
        _context: &hugr_llvm::types::TypingSession<'c, H>,
        _hugr_type: &hugr::types::CustomType,
    ) -> anyhow::Result<inkwell::types::BasicTypeEnum<'c>> {
        unimplemented!()
    }

    fn emitter<'a>(
        &self,
        context: &'a mut hugr_llvm::emit::EmitFuncContext<'c, H>,
    ) -> Box<dyn hugr_llvm::emit::EmitOp<'c, hugr::ops::CustomOp, H> + 'a> {
        Box::new(Tket2Emitter(context))
    }
}

// drives `hugr-llvm` to produce an LLVM module from a Hugr.
fn hugr_to_module<'c>(context: &'c Context, hugr: &'c Hugr) -> Module<'c> {
    let module = context.create_module("test_context");
    let namer = Namer::default().into();
    let exts = CodegenExtsMap::default()
        .add_int_extensions()
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
    python_bin: PathBuf,
    test_dir: PathBuf,
}

impl TestConfig {
    pub fn new() -> TestConfig {
        let python_bin = env::var("HUGR_LLVM_PYTHON_BIN")
            .map(Into::into)
            .ok()
            .or_else(|| pathsearch::find_executable_in_path("python"))
            .unwrap_or_else(|| panic!("Could not find python in PATH or HUGR_LLVM_PYTHON_BIN"));
        let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/guppy_test_cases");
        assert!(test_dir.is_dir());
        TestConfig {
            python_bin,
            test_dir,
        }
    }
}

impl TestConfig {
    // invokes the path with python, expecting to recieve Hugr serialised as
    // JSON on stdout.
    fn get_guppy_output(&self, path: impl AsRef<Path>) -> Hugr {
        let mut guppy_cmd = Command::new(&self.python_bin);
        guppy_cmd
            .arg(self.test_dir.join(path.as_ref()))
            .stdout(Stdio::piped());
        let mut guppy_proc = guppy_cmd.spawn().unwrap();
        let mut hugr: Hugr = serde_json::from_reader(guppy_proc.stdout.take().unwrap()).unwrap();
        assert!(guppy_proc.wait().unwrap().success());
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

guppy_test!("even_odd.py", even_odd);
guppy_test!("even_odd2.py", even_odd2);
guppy_test!("planqc-1.py", planqc1);
guppy_test!("planqc-2.py", planqc2);
guppy_test!("planqc-3.py", planqc3);

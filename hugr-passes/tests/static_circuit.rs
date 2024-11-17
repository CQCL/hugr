use std::{path::{Path, PathBuf}, process::{Command, Stdio}};

use hugr_core::{extension::ExtensionSet, hugr::IdentList, ops::NamedOp, package::Package, std_extensions::STD_REG, Hugr, HugrView};
use hugr_passes::{inline::InlinePass, static_circuit::{check_results, Gate, StaticCircuitContext, StaticCircuitPass}};
use insta::{self,glob};
use itertools::Itertools;

struct T {
    file: Box<Path>,
}

impl T {
    fn new(path: impl AsRef<Path>) -> Self {
        Self { file: path.as_ref().into() }
    }

    fn run(&self) -> Result<Package, Box<dyn std::error::Error>> {
        let guppy_output = {
            let mut cmd = Command::new("python3");
            cmd.arg(self.file.as_ref());
            eprintln!("{cmd:?}");
            cmd.output()?
        };

        let status = guppy_output.status;
        if !status.success() {
            Err(format!("Guppy failed for {}: {status}\n{}", self.file.as_ref().to_string_lossy(), String::from_utf8(guppy_output.stderr).unwrap()))?;
        }

        Package::from_json_reader(guppy_output.stdout.as_slice()).map_err(Into::into)
    }
}


fn scc<'a, H: HugrView>(hugr: &'a H) -> StaticCircuitContext<'a, H> {
    let extensions = ExtensionSet::from_iter([IdentList::new_unchecked("tket2.quantum")]);
    let alloc_ops = hugr.nodes().filter(|&x| hugr.get_optype(x).as_extension_op().map_or(false, |ext_op| ext_op.def().name() == "QAlloc")).collect();
    let free_ops = hugr.nodes().filter(|&x| hugr.get_optype(x).as_extension_op().map_or(false, |ext_op| ext_op.def().name() == "QFree")).collect();
    StaticCircuitContext { hugr, extensions, alloc_ops, free_ops }
}

fn gates_to_str(gates: Option<Vec<Gate>>) -> String {
    gates.map_or("No static circuit".into(), |gates| gates.into_iter().map(|gate|
                                                                    format!("{} {:?}", gate.1, gate.2)).join("\n"))

}

#[test]
fn guppy_examples() {
    let pass = StaticCircuitPass::default();
    glob!("static_circuit/guppy_examples/**/*.py", |path| {
        let mut package = T::new(path).run().unwrap();
        let mut reg = STD_REG.to_owned();
        package.update_validate(&mut reg).unwrap();
        let Some([mut hugr]): Option<[Hugr;1]> = package.modules.try_into().ok() else {
            panic!("Expected exactly one hugr")
        };

        InlinePass::default().run(&mut hugr, &reg).unwrap();

        let gates = check_results(pass.run(scc(&hugr), &reg).unwrap()).unwrap();

        insta::assert_snapshot!(gates_to_str(gates))
    });
}

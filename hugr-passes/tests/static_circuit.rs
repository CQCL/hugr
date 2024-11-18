use std::{
    fs, path::Path, process::Command,
};

use hugr_core::{
    extension::ExtensionSet, hugr::IdentList, ops::NamedOp, package::Package,
    std_extensions::STD_REG, Hugr, HugrView,
};
use hugr_passes::{
    inline::InlinePass,
    static_circuit::{Gate, StaticCircuitContext, StaticCircuitPass, StaticCircuitResult},
};
use insta::{self, glob};
use itertools::Itertools;

struct T {
    file: Box<Path>,
}

impl T {
    fn new(path: impl AsRef<Path>) -> Self {
        Self {
            file: path.as_ref().into(),
        }
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
            Err(format!(
                "Guppy failed for {}: {status}\n{}",
                self.file.as_ref().to_string_lossy(),
                String::from_utf8(guppy_output.stderr).unwrap()
            ))?;
        }

        Package::from_json_reader(guppy_output.stdout.as_slice()).map_err(Into::into)
    }
}

fn scc<H: HugrView>(hugr: &H) -> StaticCircuitContext<'_, H> {
    let extensions = ExtensionSet::from_iter([IdentList::new_unchecked("tket2.quantum")]);
    let alloc_ops = hugr
        .nodes()
        .filter(|&x| {
            hugr.get_optype(x)
                .as_extension_op()
                .map_or(false, |ext_op| ext_op.def().name() == "QAlloc")
        })
        .collect();
    let free_ops = hugr
        .nodes()
        .filter(|&x| {
            hugr.get_optype(x)
                .as_extension_op()
                .map_or(false, |ext_op| ext_op.def().name() == "QFree")
        })
        .collect();
    StaticCircuitContext {
        hugr,
        extensions,
        alloc_ops,
        free_ops,
    }
}

fn gates_to_str(gates: Option<Vec<Gate>>) -> String {
    gates.map_or("No static circuit".into(), |gates| {
        gates
            .iter()
            .map(Gate::show)
            .join("\n")
    })
}

#[test]
fn guppy_examples() {
    let pass = StaticCircuitPass::default();
    glob!("static_circuit/guppy_examples/**/*.py", |path| {
        let mut package = T::new(path).run().unwrap();
        let mut reg = STD_REG.to_owned();
        package.update_validate(&mut reg).unwrap();
        let Some([mut hugr]): Option<[Hugr; 1]> = package.modules.try_into().ok() else {
            panic!("Expected exactly one hugr")
        };

        let program_src = {
            let src = fs::read_to_string(path).unwrap();
            let start = src.lines().enumerate().find_map(|(i, line)| line.starts_with("## BEGIN").then_some(i));
            let end = src.lines().enumerate().find_map(|(i, line)| line.starts_with("## END").then_some(i));
            if let (Some(start), Some(end)) = (start,end) {
                assert!(start < end);
                src.lines().skip(start + 1).take(end - start - 1).join("\n")
            } else {
                src
            }
        };

        let mut settings = insta::Settings::clone_current();
        // settings.set_description(program_src);
        settings.set_omit_expression(true);
        settings.bind(|| {
            InlinePass::default().run(&mut hugr, &reg).unwrap();

            let scc = scc(&hugr);
            let scr = StaticCircuitPass::default().run(scc.clone(), &reg).unwrap();
            insta::assert_snapshot!(format!("{program_src}\n===========================\n{}",gates_to_str(scr.static_circuit(scc))));
        });
    });
}

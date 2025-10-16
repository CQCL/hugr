//! Convert between different HUGR envelope formats.
use std::io::BufReader;

use crate::hugr_io::HugrInputArgs;
use anyhow::Result;
use clap::Parser;
use hugr::NodeIndex;
use hugr::envelope::EnvelopeReader;
use hugr::envelope::description::{ExtensionDesc, ModuleDesc};
use hugr::extension::Version;
use hugr::ops::OpType;
use tabled::Tabled;
use tabled::derive::display;

/// Convert between different HUGR envelope formats.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Describe the contents of a HUGR envelope.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct DescribeArgs {
    /// Hugr input.
    #[command(flatten)]
    pub input_args: HugrInputArgs,
    /// enumerate packaged extensions
    #[arg(long, default_value = "false")]
    pub packaged_extensions: bool,

    #[command(flatten)]
    /// Configure module description
    pub module_args: ModuleArgs,
}

/// Arguments for reading a HUGR input.
#[derive(Debug, clap::Args)]
pub struct ModuleArgs {
    /// Describe specified module (0-based index) in detail.
    #[arg(long)]
    pub module: Option<usize>,

    #[arg(long, default_value = "true")]
    /// Display resolved extensions used by the module.
    /// Requires module to be specified.
    pub resolved_extensions: bool,

    #[arg(long, default_value = "true")]
    /// Display public symbols in the module.
    /// Requires module to be specified.
    pub public_symbols: bool,

    #[arg(long, default_value = "false")]
    /// Display claimed extensions set by generator in module metadata.
    ///  Requires module to be specified.
    pub generator_used_extensions: bool,
}

impl DescribeArgs {
    /// Convert a HUGR between different envelope formats
    pub fn run_describe(&mut self) -> Result<()> {
        // TODO reuse code from hugr_io
        let extensions = self.input_args.load_extensions()?;
        let buffer = BufReader::new(&mut self.input_args.input);

        let (desc, res) = EnvelopeReader::new(buffer, &extensions)?.read();

        if let Err(err) = res {
            eprintln!("{err}");

            println!("\nPartial description:");
        }
        let header = desc.header();
        println!(
            "{header}\nPackage contains {} module(s) and {} extension(s)",
            desc.n_modules(),
            desc.n_packaged_extensions()
        );
        let mut modules: Vec<_> = desc.modules().collect();
        if let Some(idx) = self.module_args.module {
            modules = vec![modules.remove(idx)];
        }

        let summaries: Vec<ModuleSummary> = modules
            .iter()
            .map(|m| (*m).clone().unwrap_or_default().into())
            .collect();

        let summary_table = tabled::Table::builder(summaries).index().build();
        println!("{summary_table}");

        if self.module_args.module.is_some() {
            // only show detailed info if a specific module is requested
            self.display_module(modules.remove(0).clone().unwrap())?;
        }

        Ok(())
    }

    fn display_module(&self, desc: ModuleDesc) -> Result<()> {
        let args = &self.module_args;
        match (args.resolved_extensions, desc.used_extensions_resolved) {
            (true, Some(exts)) => {
                let ext_rows: Vec<ExtensionRow> = exts.iter().cloned().map(Into::into).collect();
                let ext_table = tabled::Table::new(ext_rows);
                println!("Resolved extensions:\n{ext_table}");
            }
            (true, None) => {
                println!("No resolved extensions information available.");
            }
            _ => {}
        }

        match (args.public_symbols, desc.public_symbols) {
            (true, Some(syms)) => {
                let sym_table =
                    tabled::Table::new(syms.into_iter().map(|s| SymbolRow { symbol: s }));
                println!("Public symbols:\n{sym_table}");
            }
            (true, None) => {
                println!("No public symbols information available.");
            }
            _ => {}
        }

        match (
            args.generator_used_extensions,
            desc.used_extensions_generator,
        ) {
            (true, Some(exts)) => {
                let ext_rows: Vec<ExtensionRow> = exts.iter().cloned().map(Into::into).collect();
                let ext_table = tabled::Table::new(ext_rows);
                println!("Generator-claimed extensions:\n{ext_table}");
            }
            (true, None) => {
                println!("No generator-claimed extensions information available.");
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Tabled)]
struct ExtensionRow {
    name: String,
    version: Version,
}

#[derive(Tabled)]
struct SymbolRow {
    #[tabled(rename = "Symbol")]
    symbol: String,
}

impl From<ExtensionDesc> for ExtensionRow {
    fn from(desc: ExtensionDesc) -> Self {
        Self {
            name: desc.name,
            version: desc.version,
        }
    }
}

#[derive(Tabled, Default)]
struct ModuleSummary {
    #[tabled(display("display::option", "n/a"))]
    num_nodes: Option<usize>,
    #[tabled(display("display::option", "n/a"))]
    entrypoint_node: Option<usize>,
    #[tabled(display("display::option", "n/a"))]
    entrypoint_op: Option<String>,
    #[tabled(display("display::option", "n/a"))]
    generator: Option<String>,
}

fn op_string(op: OpType) -> String {
    match op {
        OpType::FuncDefn(defn) => format!("FuncDefn({})", defn.func_name()),
        OpType::FuncDecl(decl) => format!("FuncDecl({})", decl.func_name()),
        _ => format!("{op}"),
    }
}

impl From<ModuleDesc> for ModuleSummary {
    fn from(desc: ModuleDesc) -> Self {
        let (entrypoint_node, entrypoint_op) = if let Some((n, op)) = desc.entrypoint {
            (Some(n.index()), Some(op_string(op)))
        } else {
            (None, None)
        };
        Self {
            num_nodes: desc.num_nodes,
            entrypoint_node,
            entrypoint_op,
            generator: desc.generator,
        }
    }
}

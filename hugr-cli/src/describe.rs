//! Describe the contents of HUGR packages.
use std::io::Write;

use crate::hugr_io::HugrInputArgs;
use anyhow::Result;
use clap::Parser;
use clio::Output;
use hugr::NodeIndex;
use hugr::envelope::ReadError;
use hugr::envelope::description::{ExtensionDesc, ModuleDesc, PackageDesc};
use hugr::extension::Version;
use hugr::package::Package;
use tabled::Tabled;
use tabled::derive::display;

/// Describe the contents of a serialized HUGR package.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Describe the contents of a HUGR envelope.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct DescribeArgs {
    /// Hugr input.
    #[command(flatten)]
    pub input_args: HugrInputArgs,
    /// Enumerate packaged extensions
    #[arg(long, default_value = "false")]
    pub packaged_extensions: bool,

    #[command(flatten)]
    /// Configure module description
    pub module_args: ModuleArgs,

    #[arg(long, default_value = "false")]
    /// Output in json format
    pub json: bool,

    /// Output file. Use '-' for stdout.
    #[clap(short, long, value_parser, default_value = "-")]
    pub output: Output,
}

/// Arguments for reading a HUGR input.
#[derive(Debug, clap::Args)]
pub struct ModuleArgs {
    #[arg(long, default_value = "false")]
    /// Don't display resolved extensions used by the module.
    pub no_resolved_extensions: bool,

    #[arg(long, default_value = "false")]
    /// Display public symbols in the module.
    pub public_symbols: bool,

    #[arg(long, default_value = "false")]
    /// Display claimed extensions set by generator in module metadata.
    pub generator_claimed_extensions: bool,
}
impl ModuleArgs {
    fn filter_module(&self, module: &mut ModuleDesc) {
        if self.no_resolved_extensions {
            module.used_extensions_resolved = None;
        }
        if !self.public_symbols {
            module.public_symbols = None;
        }
        if !self.generator_claimed_extensions {
            module.used_extensions_generator = None;
        }
    }
}
impl DescribeArgs {
    /// Load and describe the HUGR package.
    pub fn run_describe(&mut self) -> Result<()> {
        let (mut desc, res) = match self.input_args.get_described_package() {
            Ok((desc, pkg)) => (desc, Ok(pkg)),
            Err(crate::CliError::ReadEnvelope(ReadError::Payload {
                source,
                partial_description,
            })) => (partial_description, Err(source)), // keep error for later
            Err(e) => return Err(e.into()),
        };

        // clear fields that have not been requested
        for module in desc.modules.iter_mut().flatten() {
            self.module_args.filter_module(module);
        }

        let res = res.map_err(anyhow::Error::from);
        if self.json {
            if !self.packaged_extensions {
                desc.packaged_extensions.clear();
            }
            self.output_json(desc, &res)?;
        } else {
            self.print_description(desc)?;
        }

        // bubble up any errors
        res.map(|_| ())
    }

    fn print_description(&mut self, desc: PackageDesc) -> Result<()> {
        let header = desc.header();
        writeln!(
            self.output,
            "{header}\nPackage contains {} module(s) and {} extension(s)",
            desc.n_modules(),
            desc.n_packaged_extensions()
        )?;
        let summaries: Vec<ModuleSummary> = desc
            .modules
            .iter()
            .map(|m| m.as_ref().map(Into::into).unwrap_or_default())
            .collect();
        let summary_table = tabled::Table::builder(summaries).index().build();
        writeln!(self.output, "{summary_table}")?;

        for (i, module) in desc.modules.into_iter().enumerate() {
            writeln!(self.output, "\nModule {i}:")?;
            if let Some(module) = module {
                self.display_module(module)?;
            }
        }
        if self.packaged_extensions {
            writeln!(self.output, "Packaged extensions:")?;
            let ext_rows: Vec<ExtensionRow> = desc
                .packaged_extensions
                .into_iter()
                .flatten()
                .map(Into::into)
                .collect();
            let ext_table = tabled::Table::new(ext_rows);
            writeln!(self.output, "{ext_table}")?;
        }
        Ok(())
    }

    fn output_json(&mut self, package_desc: PackageDesc, res: &Result<Package>) -> Result<()> {
        let err_str = res.as_ref().err().map(|e| format!("{e:?}"));
        let json_desc = JsonDescription {
            package_desc,
            error: err_str,
        };
        serde_json::to_writer_pretty(&mut self.output, &json_desc)?;
        Ok(())
    }

    fn display_module(&mut self, desc: ModuleDesc) -> Result<()> {
        if let Some(exts) = desc.used_extensions_resolved {
            let ext_rows: Vec<ExtensionRow> = exts.into_iter().map(Into::into).collect();
            let ext_table = tabled::Table::new(ext_rows);
            writeln!(self.output, "Resolved extensions:\n{ext_table}")?;
        }

        if let Some(syms) = desc.public_symbols {
            let sym_table = tabled::Table::new(syms.into_iter().map(|s| SymbolRow { symbol: s }));
            writeln!(self.output, "Public symbols:\n{sym_table}")?;
        }

        if let Some(exts) = desc.used_extensions_generator {
            let ext_rows: Vec<ExtensionRow> = exts.into_iter().map(Into::into).collect();
            let ext_table = tabled::Table::new(ext_rows);
            writeln!(self.output, "Generator claimed extensions:\n{ext_table}")?;
        }

        Ok(())
    }
}

#[derive(serde::Serialize)]
struct JsonDescription {
    #[serde(flatten)]
    package_desc: PackageDesc,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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

impl From<&ModuleDesc> for ModuleSummary {
    fn from(desc: &ModuleDesc) -> Self {
        let (entrypoint_node, entrypoint_op) = if let Some(ep) = &desc.entrypoint {
            (
                Some(ep.node.index()),
                Some(hugr::envelope::description::op_string(&ep.optype)),
            )
        } else {
            (None, None)
        };
        Self {
            num_nodes: desc.num_nodes,
            entrypoint_node,
            entrypoint_op,
            generator: desc.generator.clone(),
        }
    }
}

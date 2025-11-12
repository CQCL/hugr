//! Describe the contents of HUGR packages.
use crate::hugr_io::HugrInputArgs;
use anyhow::Result;
use clap::Parser;
use clio::Output;
use hugr::NodeIndex;
use hugr::envelope::ReadError;
use hugr::envelope::description::{ExtensionDesc, ModuleDesc, PackageDesc};
use hugr::extension::Version;
use hugr::package::Package;
use std::io::{Read, Write};
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
    #[arg(long, default_value = "false", help_heading = "Filter")]
    pub packaged_extensions: bool,

    #[command(flatten)]
    /// Configure module description
    pub module_args: ModuleArgs,

    #[arg(long, default_value = "false", help_heading = "JSON")]
    /// Output in json format
    pub json: bool,

    #[arg(long, default_value = "false", help_heading = "JSON")]
    /// Output JSON schema for the description format.
    /// Can't be combined with --json.
    pub json_schema: bool,

    /// Output file. Use '-' for stdout.
    #[clap(short, long, value_parser, default_value = "-")]
    pub output: Output,
}

/// Arguments for reading a HUGR input.
#[derive(Debug, clap::Args)]
pub struct ModuleArgs {
    #[arg(long, default_value = "false", help_heading = "Filter")]
    /// Don't display resolved extensions used by the module.
    pub no_resolved_extensions: bool,

    #[arg(long, default_value = "false", help_heading = "Filter")]
    /// Display public symbols in the module.
    pub public_symbols: bool,

    #[arg(long, default_value = "false", help_heading = "Filter")]
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
    /// Load and describe the HUGR package with optional input/output overrides.
    ///
    /// # Arguments
    ///
    /// * `input_override` - Optional reader to use instead of the CLI input argument.
    /// * `output_override` - Optional writer to use instead of the CLI output argument.
    pub fn run_describe_with_io<R: Read, W: Write>(
        &mut self,
        input_override: Option<R>,
        mut output_override: Option<W>,
    ) -> Result<()> {
        if self.json_schema {
            let schema = schemars::schema_for!(PackageDescriptionJson);
            let schema_json = serde_json::to_string_pretty(&schema)?;
            if let Some(ref mut writer) = output_override {
                writeln!(writer, "{schema_json}")?;
            } else {
                writeln!(self.output, "{schema_json}")?;
            }
            return Ok(());
        }

        let (mut desc, res) = match self
            .input_args
            .get_described_package_with_reader(input_override)
        {
            Ok((desc, pkg)) => (desc, Ok(pkg)),
            Err(crate::CliError::ReadEnvelope(ReadError::Payload {
                source,
                partial_description,
            })) => (partial_description, Err(source)),
            Err(e) => return Err(e.into()),
        };

        // clear fields that have not been requested
        for module in desc.modules.iter_mut().flatten() {
            self.module_args.filter_module(module);
        }

        let res = res.map_err(anyhow::Error::from);

        let writer: &mut dyn Write = if let Some(ref mut w) = output_override {
            w
        } else {
            &mut self.output
        };

        if self.json {
            if !self.packaged_extensions {
                desc.packaged_extensions.clear();
            }
            output_json(desc, &res, writer)?;
        } else {
            print_description(desc, self.packaged_extensions, writer)?;
        }

        // bubble up any errors
        res.map(|_| ())
    }

    /// Load and describe the HUGR package.
    pub fn run_describe(&mut self) -> Result<()> {
        self.run_describe_with_io(None::<&[u8]>, None::<Vec<u8>>)
    }
}

/// Print a human-readable description of a package.
fn print_description<W: Write + ?Sized>(
    desc: PackageDesc,
    show_packaged_extensions: bool,
    writer: &mut W,
) -> Result<()> {
    let header = desc.header();
    let n_modules = desc.n_modules();
    let n_extensions = desc.n_packaged_extensions();
    let module_str = if n_modules == 1 { "module" } else { "modules" };
    let extension_str = if n_extensions == 1 {
        "extension"
    } else {
        "extensions"
    };

    writeln!(
        writer,
        "{header}\nPackage contains {n_modules} {module_str} and {n_extensions} {extension_str}",
    )?;

    let summaries: Vec<ModuleSummary> = desc
        .modules
        .iter()
        .map(|m| m.as_ref().map(Into::into).unwrap_or_default())
        .collect();
    let summary_table = tabled::Table::builder(summaries).index().build();
    writeln!(writer, "{summary_table}")?;

    for (i, module) in desc.modules.into_iter().enumerate() {
        writeln!(writer, "\nModule {i}:")?;
        if let Some(module) = module {
            display_module(module, writer)?;
        }
    }
    if show_packaged_extensions {
        writeln!(writer, "Packaged extensions:")?;
        let ext_rows: Vec<ExtensionRow> = desc
            .packaged_extensions
            .into_iter()
            .flatten()
            .map(Into::into)
            .collect();
        let ext_table = tabled::Table::new(ext_rows);
        writeln!(writer, "{ext_table}")?;
    }
    Ok(())
}

/// Output a package description as JSON.
fn output_json<W: Write + ?Sized>(
    package_desc: PackageDesc,
    res: &Result<Package>,
    writer: &mut W,
) -> Result<()> {
    let err_str = res.as_ref().err().map(|e| format!("{e:?}"));
    let json_desc = PackageDescriptionJson {
        package_desc,
        error: err_str,
    };
    serde_json::to_writer_pretty(writer, &json_desc)?;
    Ok(())
}

/// Display information about a single module.
fn display_module<W: Write + ?Sized>(desc: ModuleDesc, writer: &mut W) -> Result<()> {
    if let Some(exts) = desc.used_extensions_resolved {
        let ext_rows: Vec<ExtensionRow> = exts.into_iter().map(Into::into).collect();
        let ext_table = tabled::Table::new(ext_rows);
        writeln!(writer, "Resolved extensions:\n{ext_table}")?;
    }

    if let Some(syms) = desc.public_symbols {
        let sym_table = tabled::Table::new(syms.into_iter().map(|s| SymbolRow { symbol: s }));
        writeln!(writer, "Public symbols:\n{sym_table}")?;
    }

    if let Some(exts) = desc.used_extensions_generator {
        let ext_rows: Vec<ExtensionRow> = exts.into_iter().map(Into::into).collect();
        let ext_table = tabled::Table::new(ext_rows);
        writeln!(writer, "Generator claimed extensions:\n{ext_table}")?;
    }

    Ok(())
}

#[derive(serde::Serialize, schemars::JsonSchema)]
struct PackageDescriptionJson {
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

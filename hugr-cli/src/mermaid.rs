//! Render mermaid diagrams.
use std::io::Write;

use crate::CliError;
use crate::hugr_io::HugrInputArgs;
use anyhow::Result;
use clap::Parser;
use clio::Output;
use hugr::HugrView;
use hugr::package::PackageValidationError;

/// Dump the standard extensions.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Render mermaid diagrams..")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct MermaidArgs {
    /// Hugr input.
    #[command(flatten)]
    pub input_args: HugrInputArgs,

    /// Validate package.
    #[arg(
        long,
        help = "Validate before rendering, includes extension inference."
    )]
    pub validate: bool,
    /// Output file '-' for stdout
    #[clap(long, short, value_parser, default_value = "-")]
    output: Output,
}

impl MermaidArgs {
    /// Write the mermaid diagram to the output.
    pub fn run_print(&mut self) -> Result<()> {
        if self.input_args.hugr_json {
            self.run_print_hugr()
        } else {
            self.run_print_envelope()
        }
    }

    /// Write the mermaid diagram for a HUGR envelope.
    pub fn run_print_envelope(&mut self) -> Result<()> {
        let package = self.input_args.get_package()?;

        if self.validate {
            package.validate().map_err(CliError::Validate)?;
        }

        for hugr in package.modules {
            writeln!(self.output, "{}", hugr.mermaid_string())?;
        }
        Ok(())
    }

    /// Write the mermaid diagram for a legacy HUGR json.
    pub fn run_print_hugr(&mut self) -> Result<()> {
        let hugr = self.input_args.get_hugr()?;

        if self.validate {
            hugr.validate()
                .map_err(PackageValidationError::Validation)?;
        }

        writeln!(self.output, "{}", hugr.mermaid_string())?;
        Ok(())
    }
}

//! Render mermaid diagrams.
use std::io::Write;

use clap::Parser;
use clap_verbosity_flag::log::Level;
use clio::Output;
use hugr::package::PackageValidationError;
use hugr::HugrView;

use crate::hugr_io::HugrInputArgs;
use crate::OtherArgs;

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

    /// Additional arguments
    #[command(flatten)]
    pub other_args: OtherArgs,
}

impl MermaidArgs {
    /// Write the mermaid diagram to the output.
    pub fn run_print(&mut self) -> Result<(), crate::CliError> {
        match self.input_args.hugr_json {
            true => self.run_print_hugr(),
            false => self.run_print_envelope(),
        }
    }

    /// Write the mermaid diagram for a HUGR envelope.
    pub fn run_print_envelope(&mut self) -> Result<(), crate::CliError> {
        let package = self.input_args.get_package()?;

        if self.validate {
            package.validate()?;
        }

        for hugr in package.modules {
            writeln!(self.output, "{}", hugr.mermaid_string())?;
        }
        Ok(())
    }

    /// Write the mermaid diagram for a legacy HUGR json.
    pub fn run_print_hugr(&mut self) -> Result<(), crate::CliError> {
        let hugr = self.input_args.get_hugr()?;

        if self.validate {
            hugr.validate()
                .map_err(PackageValidationError::Validation)?;
        }

        writeln!(self.output, "{}", hugr.mermaid_string())?;
        Ok(())
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.other_args.verbosity(level)
    }
}

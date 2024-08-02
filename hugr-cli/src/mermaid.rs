//! Render mermaid diagrams.
use std::io::Write;

use clap::Parser;
use clio::Output;
use hugr_core::HugrView;

/// Dump the standard extensions.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Render mermaid diagrams..")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct MermaidArgs {
    /// Common arguments
    #[command(flatten)]
    pub hugr_args: crate::HugrArgs,
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
    pub fn run_print(&mut self) -> Result<(), crate::CliError> {
        let hugrs = if self.validate {
            self.hugr_args.validate()?
        } else {
            self.hugr_args.get_package()?.modules
        };

        for hugr in hugrs {
            write!(self.output, "{}", hugr.mermaid_string())?;
        }
        Ok(())
    }
}

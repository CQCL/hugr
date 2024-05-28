//! Standard command line tools, used by the hugr binary.

pub use clap::Parser;
use clap_stdin::FileOrStdin;
use hugr_core::{extension::ExtensionRegistry, Hugr, HugrView};
use thiserror::Error;
/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
pub struct CmdLineArgs {
    input: FileOrStdin,
    /// Visualise with mermaid.
    #[arg(short, long, value_name = "MERMAID", help = "Visualise with mermaid.")]
    mermaid: bool,
    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    no_validate: bool,
    // TODO YAML extensions
}

/// Error type for the CLI.
#[derive(Error, Debug)]
pub enum CliError {
    /// Error reading input.
    #[error("Error reading input: {0}")]
    Input(#[from] clap_stdin::StdinError),
    /// Error parsing input.
    #[error("Error parsing input: {0}")]
    Parse(#[from] serde_json::Error),
    /// Error validating HUGR.
    #[error("Error validating HUGR: {0}")]
    Validate(#[from] hugr_core::hugr::ValidationError),
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl CmdLineArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&self, registry: &ExtensionRegistry) -> Result<(), CliError> {
        let mut hugr: Hugr = serde_json::from_reader(self.input.into_reader()?)?;
        if self.mermaid {
            println!("{}", hugr.mermaid_string());
        }

        if !self.no_validate {
            hugr.update_validate(registry)?;

            println!("{}", VALID_PRINT);
        }
        Ok(())
    }
}

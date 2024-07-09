//! Standard command line tools, used by the hugr binary.

use clap_stdin::FileOrStdin;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use thiserror::Error;
/// We reexport some clap types that are used in the public API.
pub use {clap::Parser, clap_verbosity_flag::Level};

use hugr_core::{extension::ExtensionRegistry, Hugr, HugrView};

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
pub struct CmdLineArgs {
    input: FileOrStdin,
    /// Visualise with mermaid.
    #[arg(short, long, value_name = "MERMAID", help = "Visualise with mermaid.")]
    mermaid: bool,
    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    no_validate: bool,
    /// Verbosity.
    #[command(flatten)]
    verbose: Verbosity<InfoLevel>,
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
    pub fn run(&self, registry: &ExtensionRegistry) -> Result<Hugr, CliError> {
        let mut hugr: Hugr = serde_json::from_reader(self.input.clone().into_reader()?)?;
        if self.mermaid {
            println!("{}", hugr.mermaid_string());
        }

        if !self.no_validate {
            hugr.update_validate(registry)?;
            if self.verbosity(Level::Info) {
                eprintln!("{}", VALID_PRINT);
            }
        }
        Ok(hugr)
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

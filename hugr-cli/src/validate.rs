//! The `validate` subcommand.
use std::path::PathBuf;

use clap::Parser;
use clap_stdin::FileOrStdin;
use clap_verbosity_flag::{InfoLevel, Level, Verbosity};
use hugr_core::{extension::ExtensionRegistry, Extension, Hugr, HugrView as _};
use thiserror::Error;

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct CliArgs {
    /// The input hugr to parse.
    pub input: FileOrStdin,
    /// Visualise with mermaid.
    #[arg(short, long, value_name = "MERMAID", help = "Visualise with mermaid.")]
    pub mermaid: bool,
    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    pub no_validate: bool,
    /// Verbosity.
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
    /// No standard extensions.
    #[arg(long, help = "Don't use standard extensions when validating.")]
    pub no_std: bool,
    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    pub extensions: Vec<PathBuf>,
}

/// Error type for the CLI.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CliError {
    /// Error reading input.
    #[error("Error reading input: {0}")]
    Input(#[from] clap_stdin::StdinError),
    /// Error reading input.
    #[error("Error reading from path: {0}")]
    InputFile(#[from] std::io::Error),
    /// Error parsing input.
    #[error("Error parsing input: {0}")]
    Parse(#[from] serde_json::Error),
    /// Error validating HUGR.
    #[error("Error validating HUGR: {0}")]
    Validate(#[from] hugr_core::hugr::ValidationError),
    /// Error registering extension.
    #[error("Error registering extension: {0}")]
    ExtReg(#[from] hugr_core::extension::ExtensionRegistryError),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Package of module HUGRs and extensions.
/// The HUGRs are validated against the extensions.
pub struct Package {
    modules: Vec<Hugr>,
    extensions: Vec<Extension>,
}

impl Package {
    /// Create a new package.
    pub fn new(modules: Vec<Hugr>, extensions: Vec<Extension>) -> Self {
        Self {
            modules,
            extensions,
        }
    }

    /// Modules in the package.
    pub fn modules(&self) -> &[Hugr] {
        &self.modules
    }

    /// Extensions in the package.
    pub fn extensions(&self) -> &[Extension] {
        &self.extensions
    }
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl CliArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&self) -> Result<Vec<Hugr>, CliError> {
        let rdr = self.input.clone().into_reader()?;
        let val: serde_json::Value = serde_json::from_reader(rdr)?;
        // read either a package or a single hugr
        let (mut modules, packed_exts) = if let Ok(Package {
            modules,
            extensions,
        }) = serde_json::from_value::<Package>(val.clone())
        {
            (modules, extensions)
        } else {
            let hugr: Hugr = serde_json::from_value(val)?;
            (vec![hugr], vec![])
        };

        let mut reg: ExtensionRegistry = if self.no_std {
            hugr_core::extension::PRELUDE_REGISTRY.to_owned()
        } else {
            hugr_core::std_extensions::std_reg()
        };

        // register packed extensions
        for ext in packed_exts {
            reg.register_updated(ext)?;
        }

        // register external extensions
        for ext in &self.extensions {
            let f = std::fs::File::open(ext)?;
            let ext: Extension = serde_json::from_reader(f)?;
            reg.register_updated(ext)?;
        }

        for hugr in modules.iter_mut() {
            if self.mermaid {
                println!("{}", hugr.mermaid_string());
            }

            if !self.no_validate {
                hugr.update_validate(&reg)?;
                if self.verbosity(Level::Info) {
                    eprintln!("{}", VALID_PRINT);
                }
            }
        }
        Ok(modules)
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

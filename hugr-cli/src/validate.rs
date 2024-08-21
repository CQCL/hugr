//! The `validate` subcommand.

use clap::Parser;
use clap_verbosity_flag::Level;
use hugr_core::{extension::ExtensionRegistry, Extension, Hugr};
use thiserror::Error;

use crate::{CliError, HugrArgs, Package};

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct ValArgs {
    #[command(flatten)]
    /// common arguments
    pub hugr_args: HugrArgs,
}

/// Error type for the CLI.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ValError {
    /// Error validating HUGR.
    #[error("Error validating HUGR: {0}")]
    Validate(#[from] hugr_core::hugr::ValidationError),
    /// Error registering extension.
    #[error("Error registering extension: {0}")]
    ExtReg(#[from] hugr_core::extension::ExtensionRegistryError),
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl ValArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&mut self) -> Result<Vec<Hugr>, CliError> {
        self.hugr_args.validate()
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.hugr_args.verbosity(level)
    }
}

impl HugrArgs {
    /// Load the package and validate against an extension registry.
    pub fn validate(&mut self) -> Result<Vec<Hugr>, CliError> {
        let Package {
            mut modules,
            extensions: packed_exts,
        } = self.get_package()?;

        let mut reg: ExtensionRegistry = if self.no_std {
            hugr_core::extension::PRELUDE_REGISTRY.to_owned()
        } else {
            hugr_core::std_extensions::STD_REG.to_owned()
        };

        // register packed extensions
        for ext in packed_exts {
            reg.register_updated(ext).map_err(ValError::ExtReg)?;
        }

        // register external extensions
        for ext in &self.extensions {
            let f = std::fs::File::open(ext)?;
            let ext: Extension = serde_json::from_reader(f)?;
            reg.register_updated(ext).map_err(ValError::ExtReg)?;
        }

        for hugr in modules.iter_mut() {
            hugr.update_validate(&reg).map_err(ValError::Validate)?;
            if self.verbosity(Level::Info) {
                eprintln!("{}", VALID_PRINT);
            }
        }
        Ok(modules)
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

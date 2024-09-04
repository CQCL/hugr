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
    pub fn run(&mut self) -> Result<(Vec<Hugr>, ExtensionRegistry), CliError> {
        let result = self.hugr_args.validate()?;
        if self.verbosity(Level::Info) {
            eprintln!("{}", VALID_PRINT);
        }
        Ok(result)
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.hugr_args.verbosity(level)
    }
}

impl Package {
    /// Validate the package against an extension registry.
    ///
    /// `reg` is updated with any new extensions.
    ///
    /// Returns the validated modules.
    pub fn validate(mut self, reg: &mut ExtensionRegistry) -> Result<Vec<Hugr>, ValError> {
        // register packed extension
        for ext in self.extensions {
            reg.register_updated(ext)?;
        }

        for hugr in self.modules.iter_mut() {
            hugr.update_validate(&reg)?;
        }

        Ok(self.modules)
    }
}

impl HugrArgs {
    /// Load the package and validate against an extension registry.
    ///
    /// Returns the validated modules and the extension registry the modules
    /// were validated against.
    pub fn validate(&mut self) -> Result<(Vec<Hugr>, ExtensionRegistry), CliError> {
        let package = self.get_package()?;

        let mut reg: ExtensionRegistry = if self.no_std {
            hugr_core::extension::PRELUDE_REGISTRY.to_owned()
        } else {
            hugr_core::std_extensions::STD_REG.to_owned()
        };

        // register external extensions
        for ext in &self.extensions {
            let f = std::fs::File::open(ext)?;
            let ext: Extension = serde_json::from_reader(f)?;
            reg.register_updated(ext).map_err(ValError::ExtReg)?;
        }

        let modules = package.validate(&mut reg)?;
        Ok((modules, reg))
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

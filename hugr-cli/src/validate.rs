//! The `validate` subcommand.

use clap::Parser;
use clap_verbosity_flag::Level;
use hugr::package::PackageValidationError;
use hugr::{extension::ExtensionRegistry, Extension, Hugr};

use crate::{CliError, HugrArgs};

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

impl HugrArgs {
    /// Load the package and validate against an extension registry.
    ///
    /// Returns the validated modules and the extension registry the modules
    /// were validated against.
    pub fn validate(&mut self) -> Result<(Vec<Hugr>, ExtensionRegistry), CliError> {
        let mut package = self.get_package_or_hugr()?;

        let mut reg: ExtensionRegistry = if self.no_std {
            hugr::extension::PRELUDE_REGISTRY.to_owned()
        } else {
            hugr::std_extensions::STD_REG.to_owned()
        };

        // register external extensions
        for ext in &self.extensions {
            let f = std::fs::File::open(ext)?;
            let ext: Extension = serde_json::from_reader(f)?;
            reg.register_updated(ext)
                .map_err(PackageValidationError::Extension)?;
        }

        package.update_validate(&mut reg)?;
        Ok((package.into_hugrs(), reg))
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

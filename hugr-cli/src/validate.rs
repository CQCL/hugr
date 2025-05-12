//! The `validate` subcommand.

use clap::Parser;
use clap_verbosity_flag::log::Level;
use hugr::package::PackageValidationError;
use hugr::{Hugr, HugrView};

use crate::hugr_io::HugrInputArgs;
use crate::{CliError, OtherArgs};

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct ValArgs {
    /// Hugr input.
    #[command(flatten)]
    pub input_args: HugrInputArgs,

    /// Additional arguments
    #[command(flatten)]
    pub other_args: OtherArgs,
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl ValArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&mut self) -> Result<Vec<Hugr>, CliError> {
        let result = match self.input_args.hugr_json {
            true => {
                let hugr = self.input_args.get_hugr()?;
                hugr.validate()
                    .map_err(PackageValidationError::Validation)?;
                vec![hugr]
            }
            false => {
                let package = self.input_args.get_package()?;
                package.validate()?;
                package.modules
            }
        };

        if self.verbosity(Level::Info) {
            eprintln!("{VALID_PRINT}");
        }

        Ok(result)
    }

    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.other_args.verbosity(level)
    }
}

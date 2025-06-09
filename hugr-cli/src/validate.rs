//! The `validate` subcommand.

use anyhow::Result;
use clap::Parser;
use hugr::HugrView;
use hugr::package::PackageValidationError;
use tracing::info;

use crate::hugr_io::HugrInputArgs;

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
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl ValArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&mut self) -> Result<()> {
        if self.input_args.hugr_json {
            let hugr = self.input_args.get_hugr()?;
            hugr.validate()
                .map_err(PackageValidationError::Validation)?;
        } else {
            let package = self.input_args.get_package()?;
            package.validate()?;
        };

        info!("{VALID_PRINT}");

        Ok(())
    }
}

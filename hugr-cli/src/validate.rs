//! The `validate` subcommand.

use anyhow::Result;
use clap::Parser;
use hugr::HugrView;
use hugr::package::PackageValidationError;
use tracing::info;

use crate::CliError;
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
            let generator = hugr::envelope::get_generator(&[&hugr]);

            hugr.validate()
                .map_err(PackageValidationError::Validation)
                .map_err(|val_err| wrap_generator(generator, val_err))?;
        } else {
            let package = self.input_args.get_package()?;
            let generator = hugr::envelope::get_generator(&package.modules);
            package
                .validate()
                .map_err(|val_err| wrap_generator(generator, val_err))?;
        };

        info!("{VALID_PRINT}");

        Ok(())
    }
}

fn wrap_generator(generator: Option<String>, val_err: PackageValidationError) -> CliError {
    if let Some(g) = generator {
        CliError::ValidateKnownGenerator {
            inner: val_err,
            generator: Box::new(g.to_string()),
        }
    } else {
        CliError::Validate(val_err)
    }
}

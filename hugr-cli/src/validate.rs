//! The `validate` subcommand.

use anyhow::Result;
use clap::Parser;
use hugr::package::PackageValidationError;
use hugr::{HugrView, Node};
use std::io::Read;
#[cfg(feature = "tracing")]
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
    ///
    /// # Arguments
    ///
    /// * `input_override` - Optional reader to use instead of the CLI input argument.
    ///   If provided, this reader will be used for input instead of
    ///   `self.input_args.input`.
    pub fn run_with_input<R: Read>(&mut self, input_override: Option<R>) -> Result<()> {
        if self.input_args.hugr_json {
            #[allow(deprecated)]
            let hugr = self.input_args.get_hugr_with_reader(input_override)?;
            let generator = get_generator(&hugr);

            hugr.validate()
                .map_err(PackageValidationError::Validation)
                .map_err(|val_err| CliError::validation(generator, val_err))?;
        } else {
            let (desc, package) = self
                .input_args
                .get_described_package_with_reader(input_override)?;
            let generator = desc.generator();
            package
                .validate()
                .map_err(|val_err| CliError::validation(generator, val_err))?;
        };

        #[cfg(feature = "tracing")]
        info!("{VALID_PRINT}");
        #[cfg(not(feature = "tracing"))]
        eprintln!("{VALID_PRINT}");

        Ok(())
    }

    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&mut self) -> Result<()> {
        self.run_with_input(None::<&[u8]>)
    }
}

fn get_generator(hugr: &impl HugrView<Node = Node>) -> Option<String> {
    let mod_desc: hugr::envelope::description::ModuleDesc = hugr.into();
    mod_desc.generator
}

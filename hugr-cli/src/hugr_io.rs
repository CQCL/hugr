//! Input/output arguments for the HUGR CLI.

use clio::Input;
use hugr::envelope::{read_envelope, EnvelopeError};
use hugr::extension::ExtensionRegistry;
use hugr::package::Package;
use hugr::{Extension, Hugr};
use std::io::BufReader;
use std::path::PathBuf;

use crate::CliError;

/// Arguments for reading a HUGR input.
#[derive(Debug, clap::Args)]
pub struct HugrInputArgs {
    /// Input file. Defaults to `-` for stdin.
    #[arg(value_parser, default_value = "-", help_heading = "Input")]
    pub input: Input,

    /// No standard extensions.
    #[arg(
        long,
        help_heading = "Input",
        help = "Don't use standard extensions when validating hugrs. Prelude is still used."
    )]
    pub no_std: bool,
    /// Extensions paths.
    #[arg(
        short,
        long,
        help_heading = "Input",
        help = "Paths to serialised extensions to validate against."
    )]
    pub extensions: Vec<PathBuf>,
    /// Read the input as a HUGR JSON file instead of an envelope.
    ///
    /// This is a legacy option for reading old HUGR files.
    #[clap(long, help_heading = "Input")]
    pub hugr_json: bool,
}

impl HugrInputArgs {
    /// Read a hugr envelope from the input and return the package encoded
    /// within.
    ///
    /// If [HugrInputArgs::hugr_json] is `true`, [HugrInputArgs::get_hugr] should be called instead as
    /// reading the input as a package will fail.
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        let extensions = self.load_extensions()?;
        let buffer = BufReader::new(&mut self.input);
        match read_envelope(buffer, &extensions) {
            Ok((_, pkg)) => Ok(pkg),
            Err(EnvelopeError::MagicNumber { .. }) => Err(CliError::NotAnEnvelope),
            Err(e) => Err(CliError::Envelope(e)),
        }
    }

    /// Read a hugr JSON file from the input.
    ///
    /// If [HugrInputArgs::hugr_json] is `false`, [HugrInputArgs::get_package] should be called instead as
    /// reading an input envelope as a HUGR json will fail.
    pub fn get_hugr(&mut self) -> Result<Hugr, CliError> {
        let extensions = self.load_extensions()?;
        let mut buffer = BufReader::new(&mut self.input);
        let hugr = Hugr::load_json(&mut buffer, &extensions)?;
        Ok(hugr)
    }

    /// Return a register with the selected extensions.
    ///
    /// This includes the standard extensions if [HugrInputArgs::no_std] is `false`,
    /// and the extensions loaded from the paths in [HugrInputArgs::extensions].
    pub fn load_extensions(&self) -> Result<ExtensionRegistry, CliError> {
        let mut reg = if self.no_std {
            hugr::extension::PRELUDE_REGISTRY.to_owned()
        } else {
            hugr::std_extensions::STD_REG.to_owned()
        };

        for ext in &self.extensions {
            let f = std::fs::File::open(ext)?;
            let ext: Extension = serde_json::from_reader(f)?;
            reg.register_updated(ext);
        }

        Ok(reg)
    }
}

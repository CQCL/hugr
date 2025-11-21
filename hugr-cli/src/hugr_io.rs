//! Input/output arguments for the HUGR CLI.

use clio::Input;
use hugr::envelope::description::PackageDesc;
use hugr::envelope::read_envelope;
use hugr::extension::ExtensionRegistry;
use hugr::package::Package;
use hugr::{Extension, Hugr};
use std::io::{BufReader, Read};
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
        help = "Paths to additional serialised extensions needed to load the Hugr."
    )]
    pub extensions: Vec<PathBuf>,
}

impl HugrInputArgs {
    /// Read a hugr envelope from the input and return the package encoded
    /// within.
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        self.get_described_package().map(|(_, package)| package)
    }

    /// Read a hugr envelope from the input and return the envelope
    /// description and the decoded package.
    pub fn get_described_package(&mut self) -> Result<(PackageDesc, Package), CliError> {
        self.get_described_package_with_reader::<&[u8]>(None)
    }

    /// Read a hugr envelope from an optional reader and return the envelope
    /// description and the decoded package.
    ///
    /// If `reader` is `None`, reads from the input specified in the args.
    pub fn get_described_package_with_reader<R: Read>(
        &mut self,
        reader: Option<R>,
    ) -> Result<(PackageDesc, Package), CliError> {
        let extensions = self.load_extensions()?;

        match reader {
            Some(r) => {
                let buffer = BufReader::new(r);
                Ok(read_envelope(buffer, &extensions)?)
            }
            None => {
                let buffer = BufReader::new(&mut self.input);
                Ok(read_envelope(buffer, &extensions)?)
            }
        }
    }

    /// Read a hugr JSON file from an optional reader.
    ///
    /// If `reader` is `None`, reads from the input specified in the args.
    /// This is a legacy option for reading old HUGR JSON files.
    pub(crate) fn get_hugr_with_reader<R: Read>(
        &mut self,
        reader: Option<R>,
    ) -> Result<Hugr, CliError> {
        let extensions = self.load_extensions()?;

        /// Wraps the hugr JSON so that it defines a valid envelope.
        const PREPEND: &str = r#"HUGRiHJv?@{"modules": ["#;
        const APPEND: &str = r#"],"extensions": []}"#;

        let mut envelope = PREPEND.to_string();

        match reader {
            Some(r) => {
                let mut buffer = BufReader::new(r);
                buffer.read_to_string(&mut envelope)?;
            }
            None => {
                let mut buffer = BufReader::new(&mut self.input);
                buffer.read_to_string(&mut envelope)?;
            }
        }

        envelope.push_str(APPEND);

        let hugr = Hugr::load_str(envelope, Some(&extensions))?;
        Ok(hugr)
    }

    /// Return a register with the selected extensions.
    ///
    /// This includes the standard extensions if [`HugrInputArgs::no_std`] is `false`,
    /// and the extensions loaded from the paths in [`HugrInputArgs::extensions`].
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

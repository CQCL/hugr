//! Input/output arguments for the HUGR CLI.

use clio::Input;
use hugr::envelope::description::PackageDesc;
use hugr::envelope::{EnvelopeConfig, read_described_envelope};
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
    /// # Errors
    ///
    /// If [`HugrInputArgs::hugr_json`] is `true`, [`HugrInputArgs::get_hugr`] should be called instead as
    /// reading the input as a package will fail.
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        self.get_described_package().map(|(_, package)| package)
    }

    /// Read a hugr envelope from a generic reader and return the package encoded
    /// within.
    ///
    /// This method allows reading from any source implementing `Read`, such as
    /// in-memory buffers or byte slices.
    ///
    /// # Errors
    ///
    /// If [`HugrInputArgs::hugr_json`] is `true`, this will fail as the hugr_json
    /// format requires special handling.
    pub fn get_package_from_reader<R: Read>(&self, reader: R) -> Result<Package, CliError> {
        self.get_described_package_from_reader(reader)
            .map(|(_, package)| package)
    }

    /// Read a hugr envelope from the input and return the envelope
    /// configuration and the package encoded within.
    ///
    /// # Errors
    ///
    /// If [`HugrInputArgs::hugr_json`] is `true`, [`HugrInputArgs::get_hugr`] should be called instead as
    /// reading the input as a package will fail.
    #[deprecated(since = "0.24.1", note = "Use get_described_envelope instead")]
    pub fn get_envelope(&mut self) -> Result<(EnvelopeConfig, Package), CliError> {
        let (desc, package) = self.get_described_package()?;
        Ok((desc.header.config(), package))
    }

    /// Read a hugr envelope from the input and return the envelope
    /// description and the decoded package.
    ///
    /// # Errors
    ///
    /// If [`HugrInputArgs::hugr_json`] is `true`, [`HugrInputArgs::get_hugr`] should be called instead as
    /// reading the input as a package will fail.
    pub fn get_described_package(&mut self) -> Result<(PackageDesc, Package), CliError> {
        let extensions = self.load_extensions()?;
        let buffer = BufReader::new(&mut self.input);

        Ok(read_described_envelope(buffer, &extensions)?)
    }

    /// Read a hugr envelope from a generic reader and return the envelope
    /// description and the decoded package.
    ///
    /// # Errors
    ///
    /// If [`HugrInputArgs::hugr_json`] is `true`, this will fail as the hugr_json
    /// format requires special handling.
    pub fn get_described_package_from_reader<R: Read>(
        &self,
        reader: R,
    ) -> Result<(PackageDesc, Package), CliError> {
        let extensions = self.load_extensions()?;
        let buffer = BufReader::new(reader);

        Ok(read_described_envelope(buffer, &extensions)?)
    }
    /// Read a hugr JSON file from the input.
    ///
    /// This is a legacy option for reading old HUGR JSON files when the
    /// [`HugrInputArgs::hugr_json`] flag is used.
    ///
    /// For most cases, [`HugrInputArgs::get_package`] should be called instead.
    #[deprecated(note = "Use `HugrInputArgs::get_package` instead.", since = "0.22.2")]
    pub fn get_hugr(&mut self) -> Result<Hugr, CliError> {
        let extensions = self.load_extensions()?;
        let mut buffer = BufReader::new(&mut self.input);

        /// Wraps the hugr JSON so that it defines a valid envelope.
        const PREPEND: &str = r#"HUGRiHJv?@{"modules": ["#;
        const APPEND: &str = r#"],"extensions": []}"#;

        let mut envelope = PREPEND.to_string();
        buffer.read_to_string(&mut envelope)?;
        envelope.push_str(APPEND);

        let hugr = Hugr::load_str(envelope, Some(&extensions))?;
        Ok(hugr)
    }

    /// Read a hugr JSON file from a generic reader.
    ///
    /// This is a legacy option for reading old HUGR JSON files.
    #[deprecated(
        note = "Use `HugrInputArgs::get_package_from_reader` instead.",
        since = "0.22.2"
    )]
    pub fn get_hugr_from_reader<R: Read>(&self, reader: R) -> Result<Hugr, CliError> {
        let extensions = self.load_extensions()?;
        let mut buffer = BufReader::new(reader);

        /// Wraps the hugr JSON so that it defines a valid envelope.
        const PREPEND: &str = r#"HUGRiHJv?@{"modules": ["#;
        const APPEND: &str = r#"],"extensions": []}"#;

        let mut envelope = PREPEND.to_string();
        buffer.read_to_string(&mut envelope)?;
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

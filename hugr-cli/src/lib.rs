//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::Input;
use derive_more::{Display, Error, From};
use hugr::extension::ExtensionRegistry;
use hugr::package::{PackageEncodingError, PackageValidationError};
use hugr::Hugr;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::{ffi::OsString, path::PathBuf};

pub mod extensions;
pub mod mermaid;
pub mod validate;

// TODO: Deprecated re-export. Remove on a breaking release.
#[doc(inline)]
#[deprecated(since = "0.13.2", note = "Use `hugr::package::Package` instead.")]
pub use hugr::package::Package;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "HUGR CLI tools.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub enum CliArgs {
    /// Validate and visualize a HUGR file.
    Validate(validate::ValArgs),
    /// Write standard extensions out in serialized form.
    GenExtensions(extensions::ExtArgs),
    /// Write HUGR as mermaid diagrams.
    Mermaid(mermaid::MermaidArgs),
    /// External commands
    #[command(external_subcommand)]
    External(Vec<OsString>),
}

/// Error type for the CLI.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum CliError {
    /// Error reading input.
    #[display("Error reading from path: {_0}")]
    InputFile(std::io::Error),
    /// Error parsing input.
    #[display("Error parsing package: {_0}")]
    Parse(serde_json::Error),
    /// Hugr load error.
    #[display("Error parsing package: {_0}")]
    HUGRLoad(PackageEncodingError),
    #[display("Error validating HUGR: {_0}")]
    /// Errors produced by the `validate` subcommand.
    Validate(PackageValidationError),
}

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
pub struct HugrArgs {
    /// Input HUGR file, use '-' for stdin
    #[clap(value_parser, default_value = "-")]
    pub input: Input,
    /// Verbosity.
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
    /// No standard extensions.
    #[arg(
        long,
        help = "Don't use standard extensions when validating. Prelude is still used."
    )]
    pub no_std: bool,
    /// Extensions paths.
    #[arg(
        short,
        long,
        help = "Paths to serialised extensions to validate against."
    )]
    pub extensions: Vec<PathBuf>,
}

/// A simple enum containing either a package or a single hugr.
///
/// This is required since `Package`s can only contain module-rooted hugrs.
#[derive(Debug, Clone, PartialEq)]
pub enum PackageOrHugr {
    /// A package with module-rooted HUGRs and some required extensions.
    Package(Package),
    /// An arbitrary HUGR.
    Hugr(Hugr),
}

impl PackageOrHugr {
    /// Returns the list of hugrs in the package.
    pub fn into_hugrs(self) -> Vec<Hugr> {
        match self {
            PackageOrHugr::Package(pkg) => pkg.modules,
            PackageOrHugr::Hugr(hugr) => vec![hugr],
        }
    }

    /// Validates the package or hugr.
    pub fn validate(&self) -> Result<(), PackageValidationError> {
        match self {
            PackageOrHugr::Package(pkg) => pkg.validate(),
            PackageOrHugr::Hugr(hugr) => Ok(hugr.validate()?),
        }
    }
}

impl AsRef<[Hugr]> for PackageOrHugr {
    fn as_ref(&self) -> &[Hugr] {
        match self {
            PackageOrHugr::Package(pkg) => &pkg.modules,
            PackageOrHugr::Hugr(hugr) => std::slice::from_ref(hugr),
        }
    }
}

impl HugrArgs {
    /// Read either a package or a single hugr from the input.
    pub fn get_package_or_hugr(
        &mut self,
        extensions: &ExtensionRegistry,
    ) -> Result<PackageOrHugr, CliError> {
        // We need to read the input twice; once to try to load it as a HUGR, and if that fails, as a package.
        // If `input` is a file, we can reuse the reader by seeking back to the start.
        // Else, we need to read the file into a buffer.
        match self.input.can_seek() {
            true => get_package_or_hugr_seek(&mut self.input, extensions),
            false => {
                let mut buffer = Vec::new();
                self.input.read_to_end(&mut buffer)?;
                get_package_or_hugr_seek(Cursor::new(buffer), extensions)
            }
        }
    }

    /// Read either a package from the input.
    ///
    /// deprecated: use [HugrArgs::get_package_or_hugr] instead.
    #[deprecated(
        since = "0.13.2",
        note = "Use `HugrArgs::get_package_or_hugr` instead."
    )]
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        let val: serde_json::Value = serde_json::from_reader(&mut self.input)?;
        let pkg = serde_json::from_value::<Package>(val.clone())?;
        Ok(pkg)
    }
}

/// Load a package or hugr from a seekable input.
fn get_package_or_hugr_seek<I: Seek + Read>(
    mut input: I,
    extensions: &ExtensionRegistry,
) -> Result<PackageOrHugr, CliError> {
    match Hugr::load_json(&mut input, extensions) {
        Ok(hugr) => Ok(PackageOrHugr::Hugr(hugr)),
        Err(_) => {
            input.seek(SeekFrom::Start(0))?;
            let pkg = Package::from_json_reader(input, extensions)?;
            Ok(PackageOrHugr::Package(pkg))
        }
    }
}

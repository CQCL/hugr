//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::Input;
use derive_more::{Display, Error, From};
use hugr::extension::ExtensionRegistry;
use hugr::package::PackageValidationError;
use hugr::Hugr;
use std::{ffi::OsString, path::PathBuf};

pub mod extensions;
pub mod mermaid;
pub mod validate;

// TODO: Deprecated re-export. Remove on a breaking release.
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
    ///
    /// Updates the extension registry with any new extensions defined in the package.
    pub fn update_validate(
        &mut self,
        reg: &mut ExtensionRegistry,
    ) -> Result<(), PackageValidationError> {
        match self {
            PackageOrHugr::Package(pkg) => pkg.update_validate(reg),
            PackageOrHugr::Hugr(hugr) => hugr.update_validate(reg).map_err(Into::into),
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
    pub fn get_package_or_hugr(&mut self) -> Result<PackageOrHugr, CliError> {
        let val: serde_json::Value = serde_json::from_reader(&mut self.input)?;
        if let Ok(hugr) = serde_json::from_value::<Hugr>(val.clone()) {
            return Ok(PackageOrHugr::Hugr(hugr));
        }
        let pkg = serde_json::from_value::<Package>(val.clone())?;
        Ok(PackageOrHugr::Package(pkg))
    }
}

//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::Input;
use derive_more::{Display, Error, From};
use hugr::package::{PackageLoadError, PackageValidationError};
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
    #[display("Error parsing input: {_0}")]
    Parse(serde_json::Error),
    /// Error loading a package.
    #[display("Error parsing package: {_0}")]
    Package(PackageLoadError),
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

impl HugrArgs {
    /// Read either a package or a single hugr from the input.
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        let pkg = Package::from_json_reader(&mut self.input)?;
        Ok(pkg)
    }
}

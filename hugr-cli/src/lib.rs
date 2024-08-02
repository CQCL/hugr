//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::Input;
use hugr_core::{Extension, Hugr};
use std::{ffi::OsString, path::PathBuf};
use thiserror::Error;

pub mod extensions;
pub mod mermaid;
pub mod validate;

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
#[derive(Debug, Error)]
#[error(transparent)]
#[non_exhaustive]
pub enum CliError {
    /// Error reading input.
    #[error("Error reading from path: {0}")]
    InputFile(#[from] std::io::Error),
    /// Error parsing input.
    #[error("Error parsing input: {0}")]
    Parse(#[from] serde_json::Error),
    /// Errors produced by the `validate` subcommand.
    Validate(#[from] validate::ValError),
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Package of module HUGRs and extensions.
/// The HUGRs are validated against the extensions.
pub struct Package {
    /// Module HUGRs included in the package.
    pub modules: Vec<Hugr>,
    /// Extensions to validate against.
    pub extensions: Vec<Extension>,
}

impl Package {
    /// Create a new package.
    pub fn new(modules: Vec<Hugr>, extensions: Vec<Extension>) -> Self {
        Self {
            modules,
            extensions,
        }
    }
}

impl HugrArgs {
    /// Read either a package or a single hugr from the input.
    pub fn get_package(&mut self) -> Result<Package, CliError> {
        let val: serde_json::Value = serde_json::from_reader(&mut self.input)?;
        // read either a package or a single hugr
        if let Ok(p) = serde_json::from_value::<Package>(val.clone()) {
            Ok(p)
        } else {
            let hugr: Hugr = serde_json::from_value(val)?;
            Ok(Package::new(vec![hugr], vec![]))
        }
    }
}

//! Standard command line tools, used by the hugr binary.

use clap::{crate_version, Parser};
use clap_verbosity_flag::log::Level;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use hugr::envelope::EnvelopeError;
use hugr::package::{PackageEncodingError, PackageValidationError};
use std::ffi::OsString;

pub mod extensions;
pub mod hugr_io;
pub mod mermaid;
pub mod validate;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = crate_version!(), long_about = None)]
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
#[derive(Debug, derive_more::Display, derive_more::Error, derive_more::From)]
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
    #[display("Error decoding HUGR envelope: {_0}")]
    /// Errors produced by the `validate` subcommand.
    Envelope(EnvelopeError),
}

/// Other arguments affecting the HUGR CLI runtime.
#[derive(Parser, Debug)]
pub struct OtherArgs {
    /// Verbosity.
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
}

impl OtherArgs {
    /// Test whether a `level` message should be output.
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

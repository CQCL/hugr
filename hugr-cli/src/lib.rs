//! Standard command line tools, used by the hugr binary.

use std::ffi::OsString;
use thiserror::Error;
/// We reexport some clap types that are used in the public API.
pub use {clap::Parser, clap_verbosity_flag::Level};

pub mod extensions;
pub mod validate;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "HUGR CLI tools.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub enum CliArgs {
    /// Validate and visualize a HUGR file.
    Validate(validate::CliArgs),
    /// Write standard extensions out in serialized form.
    GenExtension(extensions::ExtArgs),
    /// External commands
    #[command(external_subcommand)]
    External(Vec<OsString>),
}

/// Error type for the CLI.
#[derive(Debug, Error)]
#[error(transparent)]
#[non_exhaustive]
pub enum CliError {
    /// Errors produced by the `validate` subcommand.
    Validate(#[from] validate::CliError),
}

//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::Input;
use std::{ffi::OsString, path::PathBuf};
use thiserror::Error;

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
    Validate(validate::ValArgs),
    /// Write standard extensions out in serialized form.
    GenExtensions(extensions::ExtArgs),
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

//! Standard command line tools for the HUGR format.
//!
//! This library provides utilities for the HUGR CLI.
//!
//! ## CLI Usage
//!
//! Run `cargo install hugr-cli` to install the CLI tools. This will make the
//! `hugr` executable available in your shell as long as you have [cargo's bin
//! directory](https://doc.rust-lang.org/book/ch14-04-installing-binaries.html)
//! in your path.
//!
//! The CLI provides two subcommands:
//!
//! - `validate` for validating HUGR files.
//! - `mermaid` for visualizing HUGR files as mermaid diagrams.
//!
//! ### Validate
//!
//! Validate and visualize a HUGR file
//!
//! Usage: `hugr validate [OPTIONS] [INPUT]`
//!
//! ```text
//! Options:
//!   -v, --verbose...  Increase logging verbosity
//!   -q, --quiet...    Decrease logging verbosity
//!   -h, --help        Print help (see more with '--help')
//!   -V, --version     Print version
//!
//! Input:
//!       --no-std                   Don't use standard extensions when validating hugrs. Prelude is still used.
//!   -e, --extensions <EXTENSIONS>  Paths to serialised extensions to validate against.
//!       --hugr-json                Read the input as a HUGR JSON file instead of an envelope
//!   [INPUT]                    Input file. Defaults to `-` for stdin
//! ```
//!
//! ### Mermaid
//!
//! Write HUGR as mermaid diagrams
//!
//! Usage: `hugr mermaid [OPTIONS] [INPUT]`
//!
//! ```text
//! Options:
//!       --validate         Validate before rendering, includes extension inference.
//!   -o, --output <OUTPUT>  Output file '-' for stdout [default: -]
//!   -v, --verbose...       Increase logging verbosity
//!   -q, --quiet...         Decrease logging verbosity
//!   -h, --help             Print help (see more with '--help')
//!   -V, --version          Print version
//!
//! Input:
//!       --no-std                   Don't use standard extensions when validating hugrs. Prelude is still used.
//!   -e, --extensions <EXTENSIONS>  Paths to serialised extensions to validate against.
//!       --hugr-json                Read the input as a HUGR JSON file instead of an envelope
//!   [INPUT]                    Input file. Defaults to `-` for stdin.
//! ```

use clap::{Parser, crate_version};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use hugr::envelope::EnvelopeError;
use hugr::package::PackageValidationError;
use std::ffi::OsString;
use thiserror::Error;

pub mod convert;
pub mod extensions;
pub mod hugr_io;
pub mod mermaid;
pub mod validate;

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = crate_version!(), long_about = None)]
#[clap(about = "HUGR CLI tools.")]
#[group(id = "hugr")]
pub struct CliArgs {
    /// The command to be run.
    #[command(subcommand)]
    pub command: CliCommand,
    /// Verbosity.
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
}

/// The CLI subcommands.
#[derive(Debug, clap::Subcommand)]
#[non_exhaustive]
pub enum CliCommand {
    /// Validate and visualize a HUGR file.
    Validate(validate::ValArgs),
    /// Write standard extensions out in serialized form.
    GenExtensions(extensions::ExtArgs),
    /// Write HUGR as mermaid diagrams.
    Mermaid(mermaid::MermaidArgs),
    /// Convert between different HUGR envelope formats.
    Convert(convert::ConvertArgs),
    /// External commands
    #[command(external_subcommand)]
    External(Vec<OsString>),
}

/// Error type for the CLI.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CliError {
    /// Error reading input.
    #[error("Error reading from path.")]
    InputFile(#[from] std::io::Error),
    /// Error parsing input.
    #[error("Error parsing package.")]
    Parse(#[from] serde_json::Error),
    #[error("Error validating HUGR.")]
    /// Errors produced by the `validate` subcommand.
    Validate(#[from] PackageValidationError),
    #[error("Error decoding HUGR envelope.")]
    /// Errors produced by the `validate` subcommand.
    Envelope(#[from] EnvelopeError),
    /// Pretty error when the user passes a non-envelope file.
    #[error(
        "Input file is not a HUGR envelope. Invalid magic number.\n\nUse `--hugr-json` to read a raw HUGR JSON file instead."
    )]
    NotAnEnvelope,
    /// Invalid format string for conversion.
    #[error(
        "Invalid format: '{_0}'. Valid formats are: json, model, model-exts, model-text, model-text-exts"
    )]
    InvalidFormat(String),
}

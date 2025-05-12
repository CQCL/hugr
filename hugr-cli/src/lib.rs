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
use clap_verbosity_flag::log::Level;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use hugr::envelope::EnvelopeError;
use hugr::package::PackageValidationError;
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
    #[display("Error validating HUGR: {_0}")]
    /// Errors produced by the `validate` subcommand.
    Validate(PackageValidationError),
    #[display("Error decoding HUGR envelope: {_0}")]
    /// Errors produced by the `validate` subcommand.
    Envelope(EnvelopeError),
    /// Pretty error when the user passes a non-envelope file.
    #[display(
        "Input file is not a HUGR envelope. Invalid magic number.\n\nUse `--hugr-json` to read a raw HUGR JSON file instead."
    )]
    NotAnEnvelope,
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
    #[must_use]
    pub fn verbosity(&self, level: Level) -> bool {
        self.verbose.log_level_filter() >= level
    }
}

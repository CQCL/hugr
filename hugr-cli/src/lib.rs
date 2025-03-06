//! Standard command line tools, used by the hugr binary.

use clap::builder::{TypedValueParser as _, ValueParser};
use clap::{crate_version, FromArgMatches, Parser};
use clap_verbosity_flag::{InfoLevel, Verbosity};
use clio::{Input, Output};
use derive_more::{Display, Error, From};
use hugr::extension::ExtensionRegistry;
use hugr::package::{PackageEncodingError, PackageValidationError};
use hugr::envelope::{EnvelopeError, EnvelopeConfig, PayloadDescriptor};
use hugr::Hugr;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::{ffi::OsString, path::PathBuf};

pub mod extensions;
pub mod mermaid;
pub mod validate;

use hugr::package::Package;

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

    #[display("Error decoding HUGR envelope: {_0}")]
    /// Errors produced by the `validate` subcommand.
    Envelope(EnvelopeError),
}

/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
pub struct HugrArgs {

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

#[derive(Debug, clap::Args)]
pub struct HugrOutputArgs {
    #[arg(short, long, value_parser)]
    pub output: Option<Output>,

    // TODO clap stuff to make this work
    // #[arg(long)]
    // pub output_format: Option<EnvelopeConfig>,
}


#[derive(Debug, clap::Args)]
pub struct HugrInputArgs {
    #[arg(short, long, value_parser)]
    pub input: Option<Input>,

    #[clap(flatten)]
    pub input_format: Option<InputFormatArgs>,
}

#[derive(clap::Args)]
pub struct PayloadTypeArg {
    pub json: bool,
    pub zstd: bool
}


// TODO add clap annotations
#[derive(Debug, Clone)]
pub enum InputFormatArgs {
    /// Auto detect
    Auto,
    Envelope(PayloadDescriptor), // Fail if it's not an envelope with this descriptor
    // TODO JSON package, JSON HUGR, model package, model HUGR
}


impl clap::FromArgMatches for InputFormatArgs {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        todo!()
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        todo!()
    }
}
impl clap::Args for InputFormatArgs {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        todo!()
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        todo!()
    }
}

pub use hugr::envelope::PackageOrHugr;

impl HugrArgs {
    /// Read either a package or a single hugr from the input.
    pub fn get_package_or_hugr(
        &mut self,
        extensions: &ExtensionRegistry,
    ) -> Result<PackageOrHugr, CliError> {
        // TODO Implement this function
        todo!()
        // We need to read the input twice; once to try to load it as a HUGR, and if that fails, as a package.
        // If `input` is a file, we can reuse the reader by seeking back to the start.
        // Else, we need to read the file into a buffer.
        // match self.input.can_seek() {
        //     true => get_package_or_hugr_seek(&mut self.input, extensions),
        //     false => {
        //         let mut buffer = Vec::new();
        //         self.input.read_to_end(&mut buffer)?;
        //         get_package_or_hugr_seek(Cursor::new(buffer), extensions)
        //     }
        // }
    }
}

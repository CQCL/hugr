//! Convert between different HUGR envelope formats.
use anyhow::Result;
use clap::Parser;
use clio::Output;
use hugr::envelope::{EnvelopeConfig, EnvelopeFormat, ZstdConfig};
use std::io::{Read, Write};

use crate::CliError;
use crate::hugr_io::HugrInputArgs;

/// Convert between different HUGR envelope formats.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Convert a HUGR between different envelope formats.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct ConvertArgs {
    /// Hugr input.
    #[command(flatten)]
    pub input_args: HugrInputArgs,

    /// Output file. Use '-' for stdout.
    #[clap(short, long, value_parser, default_value = "-")]
    pub output: Output,

    /// Output format. One of: json, model, model-exts, model-text, model-text-exts
    #[clap(short, long, value_name = "FORMAT")]
    pub format: Option<String>,

    /// Use default text-based envelope configuration.
    /// Cannot be combined with --format or --binary.
    #[clap(long, conflicts_with_all = ["format", "binary"])]
    pub text: bool,

    /// Use default binary envelope configuration.
    /// Cannot be combined with --format or --text.
    #[clap(long, conflicts_with_all = ["format", "text"])]
    pub binary: bool,

    /// Enable zstd compression for the output
    #[clap(long)]
    pub compress: bool,

    /// Zstd compression level (1-22, where 1 is fastest and 22 is best compression)
    /// Uses the default level if not specified.
    #[clap(long, value_name = "LEVEL", requires = "compress")]
    pub compression_level: Option<u8>,
}

impl ConvertArgs {
    /// Convert a HUGR between different envelope formats with optional input/output overrides.
    ///
    /// # Arguments
    ///
    /// * `input_override` - Optional reader to use instead of the CLI input argument.
    /// * `output_override` - Optional writer to use instead of the CLI output argument.
    pub fn run_convert_with_io<R: Read, W: Write>(
        &mut self,
        input_override: Option<R>,
        mut output_override: Option<W>,
    ) -> Result<()> {
        let (env_config, package) = self
            .input_args
            .get_described_package_with_reader(input_override)?;

        // Handle text and binary format flags, which override the format option
        let mut config = if self.text {
            EnvelopeConfig::text()
        } else if self.binary {
            EnvelopeConfig::binary()
        } else {
            // Parse the requested format
            let format = match &self.format {
                Some(fmt) => match fmt.as_str() {
                    "json" => EnvelopeFormat::PackageJson,
                    "model" => EnvelopeFormat::Model,
                    "model-exts" => EnvelopeFormat::ModelWithExtensions,
                    "model-text" => EnvelopeFormat::ModelText,
                    "model-text-exts" => EnvelopeFormat::ModelTextWithExtensions,
                    _ => Err(CliError::InvalidFormat(fmt.clone()))?,
                },
                None => env_config.header.config().format, // Use input format if not specified
            };
            EnvelopeConfig::new(format)
        };

        // Configure compression
        if let Some(level) = self.compress.then_some(self.compression_level).flatten() {
            config = config.with_zstd(ZstdConfig::new(level));
        }

        // Write the package with the requested format
        if let Some(ref mut writer) = output_override {
            hugr::envelope::write_envelope(writer, &package, config)?;
        } else {
            hugr::envelope::write_envelope(&mut self.output, &package, config)?;
        }

        Ok(())
    }

    /// Convert a HUGR between different envelope formats
    pub fn run_convert(&mut self) -> Result<()> {
        self.run_convert_with_io(None::<&[u8]>, None::<Vec<u8>>)
    }
}

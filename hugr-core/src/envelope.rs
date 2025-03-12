//! Envelope format for HUGR packages.
//!
//! The binary format is designed to be extensible and backwards-compatible.
//! It consists of a header declaring the format used to encode the HUGR, followed
//! by the encoded HUGR itself.
//!
//! Use [`read_envelope`] and [`write_envelope`] for reading and writing envelopes
//! from/to readers and writers.
//!
//! ## Envelope header
//!
//! The binary header format is 10 bytes, with the following fields:
//!
//! | Field  | Size (bytes) | Description |
//! |--------|--------------|-------------|
//! | Magic  | 8            | [MAGIC_NUMBER] constant identifying the envelope format, little-endian. |
//! | Format | 1            | [PayloadFormat] describing the payload format. |
//! | Flags  | 1            | Additional configuration flags. |
//!
//! Flags:
//!
//! - Bit 0: Whether the payload is compressed with zstd.
//! - Bits 1-7: Reserved for future use.
//!

mod header;

pub use header::{EnvelopeConfig, PayloadFormat, ZstdConfig, MAGIC_NUMBER};

use crate::{
    extension::ExtensionRegistry,
    package::{Package, PackageEncodingError, PackageError},
};
use header::EnvelopeHeader;
use itertools::Itertools as _;
use std::io::BufRead;
use std::io::{BufReader, Write};

#[cfg(feature = "model_unstable")]
use crate::import::ImportError;

/// Read a HUGR envelope from a reader.
///
/// Returns the deserialized package and the configuration used to encode it.
///
/// Parameters:
/// - `reader`: The reader to read the envelope from.
/// - `registry`: An extension registry with additional extensions to use when
///     decoding the HUGR, if they are not already included in the package.
pub fn read_envelope(
    mut reader: impl BufRead,
    registry: &ExtensionRegistry,
) -> Result<(EnvelopeConfig, Package), EnvelopeError> {
    let header = EnvelopeHeader::read(&mut reader)?;

    let package = match header.zstd {
        #[cfg(feature = "zstd")]
        true => read_impl(
            BufReader::new(zstd::Decoder::new(reader)?),
            header,
            registry,
        ),
        #[cfg(not(feature = "zstd"))]
        true => Err(EnvelopeError::ZstdUnsupported),
        false => read_impl(reader, header, registry),
    }?;
    Ok((header.config(), package))
}

/// Write a HUGR package into an envelope, using the specified configuration.
///
/// It is recommended to use a buffered writer for better performance.
/// See [`std::io::BufWriter`] for more information.
pub fn write_envelope(
    mut writer: impl Write,
    package: &Package,
    config: EnvelopeConfig,
) -> Result<(), EnvelopeError> {
    let header = config.make_header();
    header.write(&mut writer)?;

    match config.zstd {
        #[cfg(feature = "zstd")]
        Some(zstd) => {
            let writer = zstd::Encoder::new(writer, zstd.level())?;
            write_impl(writer, package, config)?;
        }
        #[cfg(not(feature = "zstd"))]
        Some(_) => return Err(EnvelopeError::ZstdUnsupported),
        None => write_impl(writer, package, config)?,
    }

    Ok(())
}

/// Error type for envelope operations.
#[derive(derive_more::Display, derive_more::Error, Debug, derive_more::From)]
#[non_exhaustive]
pub enum EnvelopeError {
    /// Bad magic number.
    #[display("Bad magic number. expected '{expected:X}' found '{found:X}'")]
    #[from(ignore)]
    MagicNumber {
        /// The expected magic number.
        ///
        /// See [`MAGIC_NUMBER`].
        expected: u64,
        /// The magic number in the envelope.
        found: u64,
    },
    /// The specified payload format is invalid.
    #[display("Format descriptor {descriptor} is invalid.")]
    #[from(ignore)]
    InvalidFormatDescriptor {
        /// The unsupported format.
        descriptor: usize,
    },
    /// The specified payload format is not supported.
    #[display("Payload format {format} is not supported.{}",
        match feature {
            Some(f) => format!(" This requires the '{f}' feature for `hugr`."),
            None => "".to_string()
        },
    )]
    #[from(ignore)]
    FormatUnsupported {
        /// The unsupported format.
        format: PayloadFormat,
        /// Optionally, the feature required to support this format.
        feature: Option<&'static str>,
    },
    /// Envelope encoding required zstd compression, but the feature is not enabled.
    #[display("Zstd compression is not supported. This requires the 'zstd' feature for `hugr`.")]
    #[from(ignore)]
    ZstdUnsupported,
    /// Tried to encode a package with multiple HUGRs, when only 1 was expected.
    #[display(
        "Packages with multiple HUGRs are currently unsupported. Tried to encode {count} HUGRs, when 1 was expected."
    )]
    #[from(ignore)]
    MultipleHugrs {
        /// The number of HUGRs in the package.
        count: usize,
    },
    /// JSON serialization error.
    SerdeError {
        /// The source error.
        source: serde_json::Error,
    },
    /// IO read/write error.
    IO {
        /// The source error.
        source: std::io::Error,
    },
    /// Error decoding a package from the payload.
    Package {
        /// The source error.
        source: PackageError,
    },
    /// Error writing a json package to the payload.
    PackageEncoding {
        /// The source error.
        source: PackageEncodingError,
    },
    /// Error importing a HUGR from a hugr-model payload.
    #[cfg(feature = "model_unstable")]
    ModelImport {
        /// The source error.
        source: ImportError,
    },
    /// Error reading a HUGR model payload.
    #[cfg(feature = "model_unstable")]
    ModelRead {
        /// The source error.
        source: hugr_model::v0::binary::ReadError,
    },
    /// Error writing a HUGR model payload.
    #[cfg(feature = "model_unstable")]
    ModelWrite {
        /// The source error.
        source: hugr_model::v0::binary::WriteError,
    },
}

/// Internal implementation of [`read_envelope`] to call with/without the zstd decompression wrapper.
fn read_impl(
    payload: impl BufRead,
    header: EnvelopeHeader,
    registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    match header.format {
        PayloadFormat::PackageJson => Ok(Package::from_json_reader(payload, registry)?),
        #[cfg(feature = "model_unstable")]
        PayloadFormat::Model | PayloadFormat::ModelWithExtensions => {
            decode_model(payload, registry, header.format)
        }
        #[cfg(not(feature = "model_unstable"))]
        PayloadFormat::Model | PayloadFormat::ModelWithExtensions => {
            Err(EnvelopeError::FormatUnsupported {
                format: unsupported,
                feature: "model_unstable",
            })
        }
    }
}

/// Read a HUGR model payload from a reader.
///
/// Parameters:
/// - `stream`: The reader to read the envelope from.
/// - `extension_registry`: An extension registry with additional extensions to use when
///   decoding the HUGR, if they are not already included in the package.
/// - `format`: The format of the payload.
#[cfg(feature = "model_unstable")]
fn decode_model(
    mut stream: impl BufRead,
    extension_registry: &ExtensionRegistry,
    format: PayloadFormat,
) -> Result<Package, EnvelopeError> {
    use crate::{import::import_hugr, Extension};
    use hugr_model::v0::bumpalo::Bump;

    if format.model_version() != Some(0) {
        return Err(EnvelopeError::FormatUnsupported {
            format,
            feature: None,
        });
    }

    let bump = Bump::default();
    let module_list = hugr_model::v0::binary::read_from_reader(&mut stream, &bump)?;

    let mut extension_registry = extension_registry.clone();
    if format.append_extensions() {
        let extra_extensions: Vec<Extension> =
            serde_json::from_reader::<_, Vec<Extension>>(stream)?;
        for ext in extra_extensions {
            extension_registry.register_updated(ext);
        }
    }

    // TODO: Import multiple hugrs from the model?
    let hugr = import_hugr(&module_list, &extension_registry)?;
    let mut package = Package::new([hugr])?;

    // Ensure the package contains all extensions from the registry,
    // even if they were not used in the hugr.
    package.extensions = extension_registry;

    Ok(package)
}

/// Internal implementation of [`write_envelope`] to call with/without the zstd compression wrapper.
fn write_impl(
    writer: impl Write,
    package: &Package,
    config: EnvelopeConfig,
) -> Result<(), EnvelopeError> {
    match config.format {
        PayloadFormat::PackageJson => package.to_json_writer(writer)?,
        #[cfg(feature = "model_unstable")]
        PayloadFormat::Model | PayloadFormat::ModelWithExtensions => {
            encode_model(writer, package, config.format)?
        }
        #[cfg(not(feature = "model_unstable"))]
        PayloadFormat::Model | PayloadFormat::ModelWithExtensions => {
            Err(EnvelopeError::FormatUnsupported {
                format: unsupported,
                feature: "model_unstable",
            })
        }
    }
    Ok(())
}

#[cfg(feature = "model_unstable")]
fn encode_model(
    mut writer: impl Write,
    package: &Package,
    format: PayloadFormat,
) -> Result<(), EnvelopeError> {
    use crate::export::export_hugr;
    use hugr_model::v0::{binary::write_to_writer, bumpalo::Bump};

    if format.model_version() != Some(0) {
        return Err(EnvelopeError::FormatUnsupported {
            format,
            feature: None,
        });
    }

    // TODO: Export multiple hugrs to the model?
    if package.modules.len() != 1 {
        return Err(EnvelopeError::MultipleHugrs {
            count: package.modules.len(),
        });
    }
    let bump = Bump::default();
    let module = export_hugr(&package.modules[0], &bump);
    write_to_writer(&module, &mut writer)?;

    if format.append_extensions() {
        serde_json::to_writer(writer, &package.extensions.iter().collect_vec())?;
    }

    Ok(())
}

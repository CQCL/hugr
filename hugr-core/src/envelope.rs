//! Envelope format for HUGR packages.
//!
//! The format is designed to be extensible and backwards-compatible. It
//! consists of a header declaring the format used to encode the HUGR, followed
//! by the encoded HUGR itself.
//!
//! Use [`read_envelope`] and [`write_envelope`] for reading and writing
//! envelopes from/to readers and writers, or call [Package::load] and
//! [Package::store] directly.
//!
//! ## Payload formats
//!
//! The envelope may encode the HUGR in different formats, listed in
//! [`EnvelopeFormat`]. The payload may also be compressed with zstd.
//!
//! Some formats can be represented as ASCII, as indicated by the
//! [`EnvelopeFormat::ascii_printable`] method. When this is the case, the
//! whole envelope can be stored in a string.
//!
//! ## Envelope header
//!
//! The binary header format is 10 bytes, with the following fields:
//!
//! | Field  | Size (bytes) | Description |
//! |--------|--------------|-------------|
//! | Magic  | 8            | [MAGIC_NUMBERS] constant identifying the envelope format. |
//! | Format | 1            | [EnvelopeFormat] describing the payload format. |
//! | Flags  | 1            | Additional configuration flags. |
//!
//! Flags:
//!
//! - Bit 0: Whether the payload is compressed with zstd.
//! - Bits 1-5: Reserved for future use.
//! - Bit 7,6: Constant "01" to make some headers ascii-printable.
//!

#![allow(deprecated)]
// TODO: Due to a bug in `derive_more`
// (https://github.com/JelteF/derive_more/issues/419) we need to deactivate
// deprecation warnings here. We can reactivate them once the bug is fixed by
// https://github.com/JelteF/derive_more/pull/454.

mod header;
mod package_json;
pub mod serde_with;

pub use header::{EnvelopeConfig, EnvelopeFormat, ZstdConfig, MAGIC_NUMBERS};
pub use package_json::PackageEncodingError;

use crate::Hugr;
use crate::{extension::ExtensionRegistry, package::Package};
use header::EnvelopeHeader;
use std::io::BufRead;
use std::io::Write;
use std::str::FromStr;

#[allow(unused_imports)]
use itertools::Itertools as _;

use crate::import::ImportError;

/// Read a HUGR envelope from a reader.
///
/// Returns the deserialized package and the configuration used to encode it.
///
/// Parameters:
/// - `reader`: The reader to read the envelope from.
/// - `registry`: An extension registry with additional extensions to use when
///   decoding the HUGR, if they are not already included in the package.
pub fn read_envelope(
    mut reader: impl BufRead,
    registry: &ExtensionRegistry,
) -> Result<(EnvelopeConfig, Package), EnvelopeError> {
    let header = EnvelopeHeader::read(&mut reader)?;

    let package = match header.zstd {
        #[cfg(feature = "zstd")]
        true => read_impl(
            std::io::BufReader::new(zstd::Decoder::new(reader)?),
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
    writer: impl Write,
    package: &Package,
    config: EnvelopeConfig,
) -> Result<(), EnvelopeError> {
    write_envelope_impl(writer, &package.modules, &package.extensions, config)
}

/// Write a deconstructed HUGR package into an envelope, using the specified configuration.
///
/// It is recommended to use a buffered writer for better performance.
/// See [`std::io::BufWriter`] for more information.
pub(crate) fn write_envelope_impl<'h>(
    mut writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    config: EnvelopeConfig,
) -> Result<(), EnvelopeError> {
    let header = config.make_header();
    header.write(&mut writer)?;

    match config.zstd {
        #[cfg(feature = "zstd")]
        Some(zstd) => {
            let writer = zstd::Encoder::new(writer, zstd.level())?.auto_finish();
            write_impl(writer, hugrs, extensions, config)?;
        }
        #[cfg(not(feature = "zstd"))]
        Some(_) => return Err(EnvelopeError::ZstdUnsupported),
        None => write_impl(writer, hugrs, extensions, config)?,
    }

    Ok(())
}

/// Error type for envelope operations.
#[derive(derive_more::Display, derive_more::Error, Debug, derive_more::From)]
#[non_exhaustive]
pub enum EnvelopeError {
    /// Bad magic number.
    #[display(
        "Bad magic number. expected 0x{:X} found 0x{:X}",
        u64::from_be_bytes(*expected),
        u64::from_be_bytes(*found)
    )]
    #[from(ignore)]
    MagicNumber {
        /// The expected magic number.
        ///
        /// See [`MAGIC_NUMBERS`].
        expected: [u8; 8],
        /// The magic number in the envelope.
        found: [u8; 8],
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
        format: EnvelopeFormat,
        /// Optionally, the feature required to support this format.
        feature: Option<&'static str>,
    },
    /// Not all envelope formats can be represented as ASCII.
    ///
    /// This error is used when trying to store the envelope into a string.
    #[display("Envelope format {format} cannot be represented as ASCII.")]
    #[from(ignore)]
    NonASCIIFormat {
        /// The unsupported format.
        format: EnvelopeFormat,
    },
    /// Envelope encoding required zstd compression, but the feature is not enabled.
    #[display("Zstd compression is not supported. This requires the 'zstd' feature for `hugr`.")]
    #[from(ignore)]
    ZstdUnsupported,
    /// Expected the envelope to contain a single HUGR.
    #[display("Expected an envelope containing a single hugr, but it contained {}.", if *count == 0 {
        "none".to_string()
    } else {
        count.to_string()
    })]
    #[from(ignore)]
    ExpectedSingleHugr {
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
    /// Error writing a json package to the payload.
    PackageEncoding {
        /// The source error.
        source: PackageEncodingError,
    },
    /// Error importing a HUGR from a hugr-model payload.
    ModelImport {
        /// The source error.
        source: ImportError,
    },
    /// Error reading a HUGR model payload.
    ModelRead {
        /// The source error.
        source: hugr_model::v0::binary::ReadError,
    },
    /// Error writing a HUGR model payload.
    ModelWrite {
        /// The source error.
        source: hugr_model::v0::binary::WriteError,
    },
    /// Error reading a HUGR model payload.
    ModelTextRead {
        /// The source error.
        source: hugr_model::v0::ast::ParseError,
    },
    /// Error reading a HUGR model payload.
    ModelTextResolve {
        /// The source error.
        source: hugr_model::v0::ast::ResolveError,
    },
}

/// Internal implementation of [`read_envelope`] to call with/without the zstd decompression wrapper.
fn read_impl(
    payload: impl BufRead,
    header: EnvelopeHeader,
    registry: &ExtensionRegistry,
) -> Result<Package, EnvelopeError> {
    match header.format {
        #[allow(deprecated)]
        EnvelopeFormat::PackageJson => Ok(package_json::from_json_reader(payload, registry)?),
        EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions => {
            decode_model(payload, registry, header.format)
        }
        EnvelopeFormat::ModelText => decode_model_ast(payload, registry, header.format),
    }
}

/// Read a HUGR model payload from a reader.
///
/// Parameters:
/// - `stream`: The reader to read the envelope from.
/// - `extension_registry`: An extension registry with additional extensions to use when
///   decoding the HUGR, if they are not already included in the package.
/// - `format`: The format of the payload.
fn decode_model(
    mut stream: impl BufRead,
    extension_registry: &ExtensionRegistry,
    format: EnvelopeFormat,
) -> Result<Package, EnvelopeError> {
    use crate::{import::import_package, Extension};
    use hugr_model::v0::bumpalo::Bump;

    if format.model_version() != Some(0) {
        return Err(EnvelopeError::FormatUnsupported {
            format,
            feature: None,
        });
    }

    let bump = Bump::default();
    let model_package = hugr_model::v0::binary::read_from_reader(&mut stream, &bump)?;

    let mut extension_registry = extension_registry.clone();
    if format.append_extensions() {
        let extra_extensions: Vec<Extension> =
            serde_json::from_reader::<_, Vec<Extension>>(stream)?;
        for ext in extra_extensions {
            extension_registry.register_updated(ext);
        }
    }

    Ok(import_package(&model_package, &extension_registry)?)
}

/// Read a HUGR model text payload from a reader.
///
/// Parameters:
/// - `stream`: The reader to read the envelope from.
/// - `extension_registry`: An extension registry with additional extensions to use when
///   decoding the HUGR, if they are not already included in the package.
/// - `format`: The format of the payload.
fn decode_model_ast(
    mut stream: impl BufRead,
    extension_registry: &ExtensionRegistry,
    format: EnvelopeFormat,
) -> Result<Package, EnvelopeError> {
    use crate::import::import_package;
    use hugr_model::v0::bumpalo::Bump;

    if format.model_version() != Some(0) {
        return Err(EnvelopeError::FormatUnsupported {
            format,
            feature: None,
        });
    }

    // Read the package into a string, then parse it.
    //
    // Due to how `to_string` works, we cannot append extensions after the package.
    let mut buffer = String::new();
    stream.read_to_string(&mut buffer)?;
    let ast_package = hugr_model::v0::ast::Package::from_str(&buffer)?;

    let bump = Bump::default();
    let model_package = ast_package.resolve(&bump)?;

    Ok(import_package(&model_package, extension_registry)?)
}

/// Internal implementation of [`write_envelope`] to call with/without the zstd compression wrapper.
fn write_impl<'h>(
    writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    config: EnvelopeConfig,
) -> Result<(), EnvelopeError> {
    match config.format {
        #[allow(deprecated)]
        EnvelopeFormat::PackageJson => package_json::to_json_writer(hugrs, extensions, writer)?,
        EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions | EnvelopeFormat::ModelText => {
            encode_model(writer, hugrs, extensions, config.format)?
        }
    }
    Ok(())
}

fn encode_model<'h>(
    mut writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    format: EnvelopeFormat,
) -> Result<(), EnvelopeError> {
    use hugr_model::v0::{binary::write_to_writer, bumpalo::Bump};

    use crate::export::export_package;

    if format.model_version() != Some(0) {
        return Err(EnvelopeError::FormatUnsupported {
            format,
            feature: None,
        });
    }

    let bump = Bump::default();
    let model_package = export_package(hugrs, extensions, &bump);

    match format {
        EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions => {
            write_to_writer(&model_package, &mut writer)?;
        }
        EnvelopeFormat::ModelText => {
            let model_package = model_package.as_ast().unwrap();
            writeln!(writer, "{model_package}")?;
        }
        _ => unreachable!(),
    }

    if format.append_extensions() {
        serde_json::to_writer(writer, &extensions.iter().collect_vec())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cool_asserts::assert_matches;
    use rstest::rstest;
    use std::io::BufReader;

    use crate::builder::test::{multi_module_package, simple_package};
    use crate::extension::PRELUDE_REGISTRY;

    #[rstest]
    fn errors() {
        let package = simple_package();
        assert_matches!(
            package.store_str(EnvelopeConfig::binary()),
            Err(EnvelopeError::NonASCIIFormat { .. })
        );
    }

    #[rstest]
    #[case::empty(Package::default())]
    #[case::simple(simple_package())]
    #[case::multi(multi_module_package())]
    fn text_roundtrip(#[case] package: Package) {
        let envelope = package.store_str(EnvelopeConfig::text()).unwrap();
        let new_package = Package::load_str(&envelope, None).unwrap();
        assert_eq!(package, new_package);
    }

    #[rstest]
    #[case::empty(Package::default())]
    #[case::simple(simple_package())]
    #[case::multi(multi_module_package())]
    #[cfg_attr(all(miri, feature = "zstd"), ignore)] // FFI calls (required to compress with zstd) are not supported in miri
    fn compressed_roundtrip(#[case] package: Package) {
        let mut buffer = Vec::new();
        let config = EnvelopeConfig {
            format: EnvelopeFormat::PackageJson,
            zstd: Some(ZstdConfig::default()),
        };
        let res = package.store(&mut buffer, config);

        match cfg!(feature = "zstd") {
            true => res.unwrap(),
            false => {
                assert_matches!(res, Err(EnvelopeError::ZstdUnsupported));
                return;
            }
        }

        let (decoded_config, new_package) =
            read_envelope(BufReader::new(buffer.as_slice()), &PRELUDE_REGISTRY).unwrap();

        assert_eq!(config.format, decoded_config.format);
        assert_eq!(config.zstd.is_some(), decoded_config.zstd.is_some());
        assert_eq!(package, new_package);
    }

    #[rstest]
    //#[case::empty(Package::default())] // Not currently supported
    #[case::simple(simple_package())]
    //#[case::multi(multi_module_package())] // Not currently supported
    fn module_exts_roundtrip(#[case] package: Package) {
        let mut buffer = Vec::new();
        let config = EnvelopeConfig {
            format: EnvelopeFormat::ModelWithExtensions,
            zstd: None,
        };
        package.store(&mut buffer, config).unwrap();
        let (decoded_config, new_package) =
            read_envelope(BufReader::new(buffer.as_slice()), &PRELUDE_REGISTRY).unwrap();

        assert_eq!(config.format, decoded_config.format);
        assert_eq!(config.zstd.is_some(), decoded_config.zstd.is_some());
        assert_eq!(package, new_package);
    }

    #[rstest]
    //#[case::empty(Package::default())] // Not currently supported
    #[case::simple(simple_package())]
    //#[case::multi(multi_module_package())] // Not currently supported
    fn module_roundtrip(#[case] package: Package) {
        let mut buffer = Vec::new();
        let config = EnvelopeConfig {
            format: EnvelopeFormat::Model,
            zstd: None,
        };
        package.store(&mut buffer, config).unwrap();

        let (decoded_config, new_package) =
            read_envelope(BufReader::new(buffer.as_slice()), &PRELUDE_REGISTRY).unwrap();

        assert_eq!(config.format, decoded_config.format);
        assert_eq!(config.zstd.is_some(), decoded_config.zstd.is_some());

        assert_eq!(package, new_package);
    }
}

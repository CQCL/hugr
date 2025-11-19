use std::io::Write;

use itertools::Itertools as _;
use thiserror::Error;

use crate::Hugr;
use crate::extension::ExtensionRegistry;

use super::header::{EnvelopeConfig, EnvelopeFormat, HeaderError};
use super::package_json::PackageEncodingError;
use super::{FormatUnsupportedError, check_model_version};

/// Write a package to an envelope with the specified configuration.
///
/// # Errors
///
/// - If the header cannot be written.
/// - If the payload cannot be encoded.
/// - If zstd compression is requested but the `zstd` feature is not enabled.
pub(super) fn write_envelope<'h>(
    mut writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    config: EnvelopeConfig,
) -> Result<(), WriteError> {
    let header = config.make_header();
    header.write(&mut writer)?;

    match config.zstd {
        #[cfg(feature = "zstd")]
        Some(zstd) => {
            let writer = zstd::Encoder::new(writer, zstd.level())?.auto_finish();
            write_impl(writer, hugrs, extensions, config)?;
        }
        #[cfg(not(feature = "zstd"))]
        Some(_) => return Err(WriteErrorInner::ZstdUnsupported.into()),
        None => write_impl(writer, hugrs, extensions, config)?,
    }

    Ok(())
}

/// Internal implementation of write to call with/without the zstd compression wrapper.
fn write_impl<'h>(
    writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    config: EnvelopeConfig,
) -> Result<(), WriteError> {
    match config.format {
        EnvelopeFormat::PackageJson => {
            super::package_json::to_json_writer(hugrs, extensions, writer)?
        }
        EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions => {
            check_model_version(config.format)?;
            encode_model_binary(writer, hugrs, extensions, config.format)?;
        }
        EnvelopeFormat::ModelText | EnvelopeFormat::ModelTextWithExtensions => {
            check_model_version(config.format)?;
            encode_model_text(writer, hugrs, extensions, config.format)?;
        }
    }
    Ok(())
}

/// Encode the package as a binary HUGR model.
fn encode_model_binary<'h>(
    mut writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    format: EnvelopeFormat,
) -> Result<(), ModelBinaryWriteError> {
    use hugr_model::v0::{binary::write_to_writer, bumpalo::Bump};

    use crate::export::export_package;

    let bump = Bump::default();
    let model_package = export_package(hugrs, extensions, &bump);

    write_to_writer(&model_package, &mut writer)?;

    // Append extensions for binary model.
    if format == EnvelopeFormat::ModelWithExtensions {
        serde_json::to_writer(writer, &extensions.iter().collect_vec())?;
    }

    Ok(())
}

/// Encode the package as a text HUGR model.
fn encode_model_text<'h>(
    mut writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    format: EnvelopeFormat,
) -> Result<(), ModelTextWriteError> {
    use hugr_model::v0::bumpalo::Bump;

    use crate::export::export_package;

    // Prepend extensions for text model.
    if format == EnvelopeFormat::ModelTextWithExtensions {
        serde_json::to_writer(&mut writer, &extensions.iter().collect_vec())?;
    }

    let bump = Bump::default();
    let model_package = export_package(hugrs, extensions, &bump);

    let model_package = model_package.as_ast().unwrap();
    writeln!(writer, "{model_package}")?;

    Ok(())
}

/// Error encoding an envelope payload.
#[derive(Error, Debug)]
#[non_exhaustive]
#[error(transparent)]
pub struct WriteError(pub(crate) WriteErrorInner);

impl WriteError {
    /// Create a new error for a non-ASCII format.
    pub(crate) fn non_ascii_format(format: EnvelopeFormat) -> Self {
        WriteErrorInner::NonASCIIFormat { format }.into()
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
#[error(transparent)]
/// Error encoding an envelope payload with enumerated variants.
pub(crate) enum WriteErrorInner {
    /// Error encoding a JSON format package.
    JsonWrite(#[from] PackageEncodingError),
    /// Error encoding a binary model format package.
    ModelBinary(#[from] ModelBinaryWriteError),
    /// Error encoding a text model format package.
    ModelText(#[from] ModelTextWriteError),
    /// Error writing the envelope header.
    Header(#[from] HeaderError),
    /// The specified payload format is not supported.
    FormatUnsupported(#[from] FormatUnsupportedError),
    /// Not all envelope formats can be represented as ASCII.
    ///
    /// This error is used when trying to store the envelope into a string.
    #[error("Envelope format {format} cannot be represented as ASCII.")]
    NonASCIIFormat {
        /// The unsupported format.
        format: EnvelopeFormat,
    },
    /// IO read/write error.
    #[error(transparent)]
    IO(#[from] std::io::Error),
    /// Envelope encoding required zstd compression, but the feature is not enabled.
    #[error("Zstd compression is not supported. This requires the 'zstd' feature for `hugr`.")]
    #[cfg_attr(feature = "zstd", allow(dead_code))]
    ZstdUnsupported,
}

impl<T: Into<WriteErrorInner>> From<T> for WriteError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) enum ModelTextWriteError {
    JsonSerialize(#[from] serde_json::Error),
    StringWrite(#[from] std::io::Error),
}

#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) enum ModelBinaryWriteError {
    WriteBinary(#[from] hugr_model::v0::binary::WriteError),
    JsonSerialize(#[from] serde_json::Error),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::extension::ExtensionRegistry;
    use std::io::Cursor;

    #[test]
    fn test_write_empty_package() {
        let config = EnvelopeConfig {
            format: EnvelopeFormat::PackageJson,
            zstd: None,
        };
        let cursor = Cursor::new(Vec::new());
        let hugrs: Vec<&Hugr> = vec![];
        let extensions = ExtensionRegistry::new([]);

        let result = write_envelope(cursor, hugrs, &extensions, config);
        // Empty JSON package should succeed
        assert!(result.is_ok());
    }

    #[test]
    fn test_non_ascii_format_error() {
        let format = EnvelopeFormat::Model;
        let error = WriteError::non_ascii_format(format);
        let error_msg = error.to_string();
        assert!(error_msg.contains("cannot be represented as ASCII"));
    }
}

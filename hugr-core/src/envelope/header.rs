//! Definitions for the header of an envelope.

use std::io::{Read, Write};
use std::num::NonZeroU8;

use super::EnvelopeError;

/// Magic number identifying the start of an envelope.
///
/// In ascii, this is "HUGRiHJv". The second half is a randomly generated string
/// to avoid accidental collisions with other file formats.
pub const MAGIC_NUMBERS: &[u8] = "HUGRiHJv".as_bytes();

/// Header at the start of a binary envelope file.
///
/// See the [crate::envelope] module documentation for the binary format.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Default, derive_more::Display)]
#[display("EnvelopeHeader({format}{})",
    if *zstd { ", zstd compressed" } else { "" },
)]
pub(super) struct EnvelopeHeader {
    /// The format used for the payload.
    pub format: EnvelopeFormat,
    /// Whether the payload is compressed with zstd.
    pub zstd: bool,
}

/// Encoded format of an envelope payload.
#[derive(
    Clone, Copy, Eq, PartialEq, Debug, Default, Hash, derive_more::Display, strum::FromRepr,
)]
#[non_exhaustive]
pub enum EnvelopeFormat {
    /// `hugr-model` v0 binary capnproto message.
    Model = 1,
    /// `hugr-model` v0 binary capnproto message followed by a json-encoded [crate::extension::ExtensionRegistry].
    //
    // This is a temporary format required until the model adds support for extensions.
    ModelWithExtensions = 2,
    /// Json-encoded [crate::package::Package]
    ///
    /// Uses a printable ascii value as the discriminant so the envelope can be
    /// read as text.
    #[default]
    PackageJson = 63, // '?' in ascii
}

// We use a u8 to represent EnvelopeFormat in the binary format, so we should not
// add any non-unit variants or ones with discriminants > 255.
static_assertions::assert_eq_size!(EnvelopeFormat, u8);

impl EnvelopeFormat {
    /// Returns whether to encode the extensions as json after the hugr payload.
    pub fn append_extensions(self) -> bool {
        matches!(self, Self::ModelWithExtensions)
    }

    /// If the format is a model format, returns its version number.
    pub fn model_version(self) -> Option<u32> {
        match self {
            Self::Model | Self::ModelWithExtensions => Some(0),
            _ => None,
        }
    }

    /// Returns whether the encoding format is ASCII-printable.
    ///
    /// If true, the encoded envelope can be read as text.
    pub fn ascii_printable(self) -> bool {
        matches!(self, Self::PackageJson)
    }
}

/// Configuration for encoding an envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct EnvelopeConfig {
    /// The format to use for the payload.
    pub format: EnvelopeFormat,
    /// Whether to compress the payload with zstd, and the compression level to
    /// use.
    pub zstd: Option<ZstdConfig>,
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        let format = Default::default();
        let zstd = if cfg!(feature = "zstd") {
            Some(ZstdConfig::default())
        } else {
            None
        };
        Self { format, zstd }
    }
}

impl EnvelopeConfig {
    /// Create a new envelope header with the specified configuration.
    pub(super) fn make_header(&self) -> EnvelopeHeader {
        EnvelopeHeader {
            format: self.format,
            zstd: self.zstd.is_some(),
        }
    }

    /// Default configuration for a plain-text envelope.
    pub const fn text() -> Self {
        Self {
            format: EnvelopeFormat::PackageJson,
            zstd: None,
        }
    }

    /// Default configuration for a binary envelope.
    ///
    /// If the `zstd` feature is enabled, this will use zstd compression.
    pub const fn binary() -> Self {
        Self {
            format: EnvelopeFormat::Model,
            zstd: None,
        }
    }
}

/// Configuration for zstd compression.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ZstdConfig {
    /// The compression level to use.
    ///
    /// The current range is 1-22, where 1 is fastest and 22 is best
    /// compression. Values above 20 should be used with caution, as they
    /// require additional memory.
    ///
    /// If `None`, zstd's default level is used.
    pub level: Option<NonZeroU8>,
}

impl ZstdConfig {
    /// Returns the zstd compression level to pass to the zstd library.
    ///
    /// Uses [zstd::DEFAULT_COMPRESSION_LEVEL] if the level is not set.
    pub fn level(&self) -> i32 {
        #[allow(unused_assignments, unused_mut)]
        let mut default = 0;
        #[cfg(feature = "zstd")]
        {
            default = zstd::DEFAULT_COMPRESSION_LEVEL;
        }
        self.level.map_or(default, |l| l.get() as i32)
    }
}

impl EnvelopeHeader {
    /// Returns the envelope configuration corresponding to this header.
    ///
    /// Note that zstd compression level is not stored in the header.
    pub fn config(&self) -> EnvelopeConfig {
        EnvelopeConfig {
            format: self.format,
            zstd: match self.zstd {
                true => Some(ZstdConfig { level: None }),
                false => None,
            },
        }
    }

    /// Write an envelope header to a writer.
    ///
    /// See the [crate::envelope] module documentation for the binary format.
    pub fn write(&self, writer: &mut impl Write) -> Result<(), EnvelopeError> {
        // The first 8 bytes are the magic number in little-endian.
        writer.write_all(MAGIC_NUMBERS)?;
        // Next is the format descriptor.
        let format_bytes = [self.format as u8];
        writer.write_all(&format_bytes)?;
        // Next is the flags byte.
        let mut flags = 0b01000000u8;
        flags |= self.zstd as u8;
        writer.write_all(&[flags])?;

        Ok(())
    }

    /// Reads an envelope header from a reader.
    ///
    /// Consumes exactly 10 bytes from the reader.
    /// See the [crate::envelope] module documentation for the binary format.
    pub fn read(reader: &mut impl Read) -> Result<EnvelopeHeader, EnvelopeError> {
        // The first 8 bytes are the magic number in little-endian.
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC_NUMBERS {
            return Err(EnvelopeError::MagicNumber {
                expected: MAGIC_NUMBERS.try_into().unwrap(),
                found: magic,
            });
        }

        // Next is the format descriptor.
        let mut format_bytes = [0; 1];
        reader.read_exact(&mut format_bytes)?;
        let format_discriminant = format_bytes[0] as usize;
        let Some(format) = EnvelopeFormat::from_repr(format_discriminant) else {
            return Err(EnvelopeError::InvalidFormatDescriptor {
                descriptor: format_discriminant,
            });
        };

        // Next is the flags byte.
        let mut flags_bytes = [0; 1];
        reader.read_exact(&mut flags_bytes)?;
        let zstd = flags_bytes[0] & 0x1 != 0;

        Ok(Self { format, zstd })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(EnvelopeFormat::Model)]
    #[case(EnvelopeFormat::ModelWithExtensions)]
    #[case(EnvelopeFormat::PackageJson)]
    fn header_round_trip(#[case] format: EnvelopeFormat) {
        // With zstd compression
        let header = EnvelopeHeader { format, zstd: true };

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();
        let read_header = EnvelopeHeader::read(&mut buffer.as_slice()).unwrap();
        assert_eq!(header, read_header);

        // Without zstd compression
        let header = EnvelopeHeader {
            format,
            zstd: false,
        };

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();
        let read_header = EnvelopeHeader::read(&mut buffer.as_slice()).unwrap();
        assert_eq!(header, read_header);
    }
}

//! Definitions for the header of an envelope.

use std::io::{Read, Write};
use std::num::NonZeroU8;

use itertools::Itertools;

use super::EnvelopeError;

/// Magic number identifying the start of an envelope.
///
/// In ascii, this is "`HUGRiHJv`". The second half is a randomly generated string
/// to avoid accidental collisions with other file formats.
pub const MAGIC_NUMBERS: &[u8] = "HUGRiHJv".as_bytes();

/// The all-unset header flags configuration.
/// Bit 7 is always set to ensure we have a printable ASCII character.
const DEFAULT_FLAGS: u8 = 0b0100_0000u8;
/// The ZSTD flag bit in the header's flags.
const ZSTD_FLAG: u8 = 0b0000_0001;

/// Header at the start of a binary envelope file.
///
/// See the [`crate::envelope`] module documentation for the binary format.
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
    /// `hugr-model` v0 binary capnproto message followed by a json-encoded
    /// [`crate::extension::ExtensionRegistry`].
    ///
    /// This is a temporary format required until the model adds support for
    /// extensions.
    ModelWithExtensions = 2,
    /// Human-readable S-expression encoding using [`hugr_model::v0`].
    ///
    /// Uses a printable ascii value as the discriminant so the envelope can be
    /// read as text.
    ///
    /// :caution: This format does not yet support extension encoding, so it should
    /// be avoided.
    //
    // TODO: Update comment once extension encoding is supported.
    ModelText = 40, // '(' in ascii
    /// Human-readable S-expression encoding using [`hugr_model::v0`].
    ///
    /// Uses a printable ascii value as the discriminant so the envelope can be
    /// read as text.
    ///
    /// This is a temporary format required until the model adds support for
    /// extensions.
    ModelTextWithExtensions = 41, // ')' in ascii
    /// Json-encoded [`crate::package::Package`]
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
    /// If the format is a model format, returns its version number.
    #[must_use]
    pub fn model_version(self) -> Option<u32> {
        match self {
            Self::Model
            | Self::ModelWithExtensions
            | Self::ModelText
            | Self::ModelTextWithExtensions => Some(0),
            _ => None,
        }
    }

    /// Returns whether the encoding format is ASCII-printable.
    ///
    /// If true, the encoded envelope can be read as text.
    #[must_use]
    pub fn ascii_printable(self) -> bool {
        matches!(
            self,
            Self::PackageJson | Self::ModelText | Self::ModelTextWithExtensions
        )
    }
}

/// Configuration for encoding an envelope.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct EnvelopeConfig {
    /// The format to use for the payload.
    pub format: EnvelopeFormat,
    /// Whether to compress the payload with zstd, and the compression level to
    /// use.
    pub zstd: Option<ZstdConfig>,
}

impl EnvelopeConfig {
    /// Create a new envelope configuration with the specified format.
    /// `zstd` compression is disabled by default.
    pub fn new(format: EnvelopeFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set the zstd compression configuration for the envelope.
    pub fn with_zstd(self, zstd: ZstdConfig) -> Self {
        Self {
            zstd: Some(zstd),
            ..self
        }
    }

    /// Disable zstd compression in the envelope configuration.
    pub fn disable_compression(self) -> Self {
        Self { zstd: None, ..self }
    }

    /// Create a new envelope header with the specified configuration.
    pub(super) fn make_header(&self) -> EnvelopeHeader {
        EnvelopeHeader {
            format: self.format,
            zstd: self.zstd.is_some(),
        }
    }

    /// Default configuration for a plain-text envelope.
    #[must_use]
    pub const fn text() -> Self {
        Self {
            format: EnvelopeFormat::PackageJson,
            zstd: None,
        }
    }

    /// Default configuration for a binary envelope.
    ///
    /// If the `zstd` feature is enabled, this will use zstd compression.
    #[must_use]
    pub const fn binary() -> Self {
        Self {
            format: EnvelopeFormat::ModelWithExtensions,
            zstd: Some(ZstdConfig::default_level()),
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
    /// Create a new zstd configuration with the specified compression level.
    pub fn new(level: u8) -> Self {
        Self {
            level: NonZeroU8::new(level),
        }
    }
    /// Create a new zstd configuration with default compression level.
    #[must_use]
    pub const fn default_level() -> Self {
        Self { level: None }
    }

    /// Returns the zstd compression level to pass to the zstd library.
    ///
    /// Uses [`zstd::DEFAULT_COMPRESSION_LEVEL`] if the level is not set.
    #[must_use]
    pub fn level(&self) -> i32 {
        #[allow(unused_assignments, unused_mut)]
        let mut default = 0;
        #[cfg(feature = "zstd")]
        {
            default = zstd::DEFAULT_COMPRESSION_LEVEL;
        }
        self.level.map_or(default, |l| i32::from(l.get()))
    }
}

impl EnvelopeHeader {
    /// Returns the envelope configuration corresponding to this header.
    ///
    /// Note that zstd compression level is not stored in the header.
    pub fn config(&self) -> EnvelopeConfig {
        EnvelopeConfig {
            format: self.format,
            zstd: if self.zstd {
                Some(ZstdConfig { level: None })
            } else {
                None
            },
        }
    }

    /// Write an envelope header to a writer.
    ///
    /// See the [`crate::envelope`] module documentation for the binary format.
    pub fn write(&self, writer: &mut impl Write) -> Result<(), EnvelopeError> {
        // The first 8 bytes are the magic number in little-endian.
        writer.write_all(MAGIC_NUMBERS)?;
        // Next is the format descriptor.
        let format_bytes = [self.format as u8];
        writer.write_all(&format_bytes)?;
        // Next is the flags byte.
        let mut flags = DEFAULT_FLAGS;
        if self.zstd {
            flags |= ZSTD_FLAG;
        }
        writer.write_all(&[flags])?;

        Ok(())
    }

    /// Reads an envelope header from a reader.
    ///
    /// Consumes exactly 10 bytes from the reader.
    /// See the [`crate::envelope`] module documentation for the binary format.
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
        let flags: u8 = flags_bytes[0];

        let zstd = flags & ZSTD_FLAG != 0;

        // Check if there's any unrecognized flags.
        let other_flags = (flags ^ DEFAULT_FLAGS) & !ZSTD_FLAG;
        if other_flags != 0 {
            let flag_ids = (0..8).filter(|i| other_flags & (1 << i) != 0).collect_vec();
            return Err(EnvelopeError::FlagUnsupported { flag_ids });
        }

        Ok(Self { format, zstd })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cool_asserts::assert_matches;
    use rstest::rstest;

    #[rstest]
    #[case(EnvelopeFormat::Model)]
    #[case(EnvelopeFormat::ModelWithExtensions)]
    #[case(EnvelopeFormat::ModelText)]
    #[case(EnvelopeFormat::ModelTextWithExtensions)]
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

    #[rstest]
    fn header_errors() {
        let header = EnvelopeHeader {
            format: EnvelopeFormat::Model,
            zstd: false,
        };
        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 10);
        let flags = buffer[9];
        assert_eq!(flags, DEFAULT_FLAGS);

        // Invalid magic
        let mut invalid_magic = buffer.clone();
        invalid_magic[7] = 0xFF;
        assert_matches!(
            EnvelopeHeader::read(&mut invalid_magic.as_slice()),
            Err(EnvelopeError::MagicNumber { .. })
        );

        // Unrecognised flags
        let mut unrecognised_flags = buffer.clone();
        unrecognised_flags[9] |= 0b0001_0010;
        assert_matches!(
            EnvelopeHeader::read(&mut unrecognised_flags.as_slice()),
            Err(EnvelopeError::FlagUnsupported { flag_ids })
            => assert_eq!(flag_ids, vec![1, 4])
        );
    }
}

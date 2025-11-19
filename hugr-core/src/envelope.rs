//! Envelope format for HUGR packages.
//!
//! The format is designed to be extensible and backwards-compatible. It
//! consists of a header declaring the format used to encode the HUGR, followed
//! by the encoded HUGR itself.
//!
//! Use [`read_envelope`] and [`write_envelope`] for reading and writing
//! envelopes from/to readers and writers, or call [`Package::load`] and
//! [`Package::store`] directly.
//!
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
//! | Magic  | 8            | [`MAGIC_NUMBERS`] constant identifying the envelope format. |
//! | Format | 1            | [`EnvelopeFormat`] describing the payload format. |
//! | Flags  | 1            | Additional configuration flags. |
//!
//! Flags:
//!
//! - Bit 0: Whether the payload is compressed with zstd.
//! - Bits 1-5: Reserved for future use.
//! - Bit 7,6: Constant "01" to make some headers ascii-printable.
//!

pub mod description;
mod header;
mod package_json;
mod reader;
mod writer;
use reader::EnvelopeReader;
pub use reader::PayloadError;
pub use writer::WriteError;

pub mod serde_with;

pub use header::{EnvelopeConfig, EnvelopeFormat, EnvelopeHeader, MAGIC_NUMBERS, ZstdConfig};
pub use package_json::PackageEncodingError;

use crate::Hugr;
use crate::envelope::description::PackageDesc;
use crate::envelope::header::HeaderError;
use crate::{
    extension::{ExtensionRegistry, Version},
    package::Package,
};
use std::io::BufRead;
use std::io::Write;
use thiserror::Error;

#[allow(unused_imports)]
use itertools::Itertools as _;

// TODO centralise all core metadata keys in one place.
// https://github.com/CQCL/hugr/issues/2651

/// Key used to store the name of the generator that produced the envelope.
pub const GENERATOR_KEY: &str = "core.generator";
/// Key used to store the list of used extensions in the metadata of a HUGR.
pub const USED_EXTENSIONS_KEY: &str = "core.used_extensions";

/// Format a generator value from the metadata.
pub fn format_generator(json_val: &serde_json::Value) -> String {
    match json_val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(obj) => {
            if let (Some(name), version) = (
                obj.get("name").and_then(|v| v.as_str()),
                obj.get("version").and_then(|v| v.as_str()),
            ) {
                if let Some(version) = version {
                    // Expected format: {"name": "generator", "version": "1.0.0"}
                    format!("{name}-v{version}")
                } else {
                    name.to_string()
                }
            } else {
                // just print the whole object as a string
                json_val.to_string()
            }
        }
        // Raw JSON string fallback
        _ => json_val.to_string(),
    }
}

/// Read a HUGR envelope from a reader.
///
/// Returns the deserialized package and a high level description of the envelope.
///
/// Parameters:
/// - `reader`: The reader to read the envelope from.
/// - `registry`: An extension registry with additional extensions to use when
///   decoding the HUGR, if they are not already included in the package.
///
/// # Errors
/// - [`ReadError::EnvelopeHeader`] if there was an error reading the envelope header.
/// - [`ReadError::Payload`] if there was an error reading the package payload,
///   including a partial description of the envelope read before the error occurred.
pub fn read_envelope(
    reader: impl BufRead,
    registry: &ExtensionRegistry,
) -> Result<(PackageDesc, Package), ReadError> {
    let reader = EnvelopeReader::new(reader, registry).map_err(Box::new)?;
    let (desc, res) = reader.read();
    match res {
        Ok(pkg) => Ok((desc, pkg)),
        Err(e) => Err(ReadError::Payload {
            source: Box::new(e),
            partial_description: desc,
        }),
    }
}

/// Deprecated alias for [`read_envelope`].
#[deprecated(since = "0.25.0", note = "Use `read_envelope` instead.")]
pub fn read_described_envelope(
    reader: impl BufRead,
    registry: &ExtensionRegistry,
) -> Result<(PackageDesc, Package), ReadError> {
    read_envelope(reader, registry)
}

/// Errors during reading a HUGR envelope.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReadError {
    /// Error reading the envelope header.
    #[error(transparent)]
    EnvelopeHeader(#[from] Box<HeaderError>),
    /// Error reading the package payload.
    #[error("Error reading package payload in envelope.")]
    Payload {
        /// The source error.
        source: Box<PayloadError>,
        /// Partial description of the envelope read before the error occurred.
        partial_description: PackageDesc,
    },
    /// Expected the envelope to contain a single HUGR.
    #[error("Expected an envelope containing a single hugr, but it contained {}.", if *count == 0 {
        "none".to_string()
    } else {
        count.to_string()
    })]
    ExpectedSingleHugr {
        /// The number of HUGRs in the package.
        count: usize,
    },
}

/// Write a HUGR package into an envelope, using the specified configuration.
///
/// It is recommended to use a buffered writer for better performance.
/// See [`std::io::BufWriter`] for more information.
pub fn write_envelope(
    writer: impl Write,
    package: &Package,
    config: EnvelopeConfig,
) -> Result<(), WriteError> {
    write_envelope_impl(writer, &package.modules, &package.extensions, config)
}

/// Write a deconstructed HUGR package into an envelope, using the specified configuration.
///
/// It is recommended to use a buffered writer for better performance.
/// See [`std::io::BufWriter`] for more information.
pub(crate) fn write_envelope_impl<'h>(
    writer: impl Write,
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    config: EnvelopeConfig,
) -> Result<(), WriteError> {
    writer::write_envelope(writer, hugrs, extensions, config)
}

#[derive(Debug, Error)]
#[error(
    "The envelope format {format} is not supported.{}",
    match feature {
        Some(f) => format!(" This requires the '{f}' feature for `hugr`."),
        None => String::new()
    },
)]
pub(crate) struct FormatUnsupportedError {
    /// The unsupported format.
    format: EnvelopeFormat,
    /// Optionally, the feature required to support this format.
    feature: Option<&'static str>,
}

fn check_model_version(format: EnvelopeFormat) -> Result<(), FormatUnsupportedError> {
    if format.model_version() != Some(0) {
        return Err(FormatUnsupportedError {
            format,
            feature: None,
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
#[error(
    "Extension '{name}' version mismatch: registered version is {registered}, but used version is {used}"
)]
/// Error raised when the reported used version of an extension
/// does not match the registered version in the extension registry.
pub struct ExtensionVersionMismatch {
    /// The name of the extension.
    pub name: String,
    /// The registered version of the extension in the loaded registry.
    pub registered: Version,
    /// The version of the extension reported as used in the HUGR metadata.
    pub used: Version,
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Error raised when checking for breaking changes in used extensions.
pub enum ExtensionBreakingError {
    /// The extension version in the metadata does not match the registered version.
    #[error("{0}")]
    ExtensionVersionMismatch(ExtensionVersionMismatch),

    /// Error deserializing the used extensions metadata.
    #[error("Failed to deserialize used extensions metadata")]
    Deserialization(#[from] serde_json::Error),
}

/// If HUGR metadata contains a list of used extensions, under the key [`USED_EXTENSIONS_KEY`],
/// and extension is registered in the given registry, check that the
/// version of the extension in the metadata matches the registered version.
/// Version compatibility is defined by [`compatible_versions`].
fn check_breaking_extensions(
    registry: &ExtensionRegistry,
    used_exts: impl IntoIterator<Item = description::ExtensionDesc>,
) -> Result<(), ExtensionBreakingError> {
    for ext in used_exts {
        let Some(registered) = registry.get(ext.name.as_str()) else {
            continue; // Extension not registered, ignore
        };
        if !compatible_versions(registered.version(), &ext.version) {
            // This is a breaking change, raise an error.

            return Err(ExtensionBreakingError::ExtensionVersionMismatch(
                ExtensionVersionMismatch {
                    name: ext.name,
                    registered: registered.version().clone(),
                    used: ext.version,
                },
            ));
        }
    }

    Ok(())
}

/// Check if two versions are compatible according to:
/// - Major version must match.
/// - If major version is 0, minor version must match.
/// - The registered version must be greater than or equal to the used version.
fn compatible_versions(registered: &Version, used: &Version) -> bool {
    if used.major != registered.major {
        return false;
    }
    if used.major == 0 && used.minor != registered.minor {
        return false;
    }

    registered >= used
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use cool_asserts::assert_matches;
    use rstest::rstest;
    use std::borrow::Cow;
    use std::io::BufReader;

    use crate::HugrView;
    use crate::builder::test::{multi_module_package, simple_package};
    use crate::envelope::writer::WriteErrorInner;
    use crate::extension::{Extension, ExtensionRegistry, Version};
    use crate::extension::{ExtensionId, PRELUDE_REGISTRY};
    use crate::hugr::HugrMut;
    use crate::hugr::test::check_hugr_equality;
    use crate::std_extensions::STD_REG;
    use serde_json::json;
    use std::sync::Arc;
    /// Returns an `ExtensionRegistry` with the extensions from both
    /// sets. Avoids cloning if the first one already contains all
    /// extensions from the second one.
    fn join_extensions<'a>(
        extensions: &'a ExtensionRegistry,
        other: &ExtensionRegistry,
    ) -> Cow<'a, ExtensionRegistry> {
        if other.iter().all(|e| extensions.contains(e.name())) {
            Cow::Borrowed(extensions)
        } else {
            let mut extensions = extensions.clone();
            extensions.extend(other);
            Cow::Owned(extensions)
        }
    }

    /// Serialize and deserialize a HUGR into an envelope with the given config,
    /// and check that the result is the same as the original.
    ///
    /// We do not compare the before and after `Hugr`s for equality directly,
    /// because impls of `CustomConst` are not required to implement equality
    /// checking.
    ///
    /// Returns the deserialized HUGR.
    pub(crate) fn check_hugr_roundtrip(hugr: &Hugr, config: EnvelopeConfig) -> Hugr {
        let mut buffer = Vec::new();
        hugr.store(&mut buffer, config).unwrap();

        let extensions = join_extensions(&STD_REG, hugr.extensions());

        let reader = BufReader::new(buffer.as_slice());
        let extracted = Hugr::load(reader, Some(&extensions)).unwrap();

        check_hugr_equality(&extracted, hugr);
        extracted
    }

    #[rstest]
    fn errors() {
        let package = simple_package();
        // The binary format is not ASCII-printable, so store_str should fail
        assert_matches!(
            package.store_str(EnvelopeConfig::binary()),
            Err(WriteError(WriteErrorInner::NonASCIIFormat { .. }))
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
                // ZstdUnsupported error should be raised
                assert_matches!(res, Err(WriteError(WriteErrorInner::ZstdUnsupported)));
                return;
            }
        }

        let (desc, new_package) =
            read_envelope(BufReader::new(buffer.as_slice()), &PRELUDE_REGISTRY).unwrap();
        let decoded_config = desc.header.config();
        assert_eq!(config.format, decoded_config.format);
        assert_eq!(config.zstd.is_some(), decoded_config.zstd.is_some());
        assert_eq!(package, new_package);
    }

    #[rstest]
    // Empty packages
    #[case::empty_model(Package::default(), EnvelopeFormat::Model)]
    #[case::empty_model_exts(Package::default(), EnvelopeFormat::ModelWithExtensions)]
    #[case::empty_text(Package::default(), EnvelopeFormat::ModelText)]
    #[case::empty_text_exts(Package::default(), EnvelopeFormat::ModelTextWithExtensions)]
    // Single hugrs
    #[case::simple_bin(simple_package(), EnvelopeFormat::Model)]
    #[case::simple_bin_exts(simple_package(), EnvelopeFormat::ModelWithExtensions)]
    #[case::simple_text(simple_package(), EnvelopeFormat::ModelText)]
    #[case::simple_text_exts(simple_package(), EnvelopeFormat::ModelTextWithExtensions)]
    // Multiple hugrs
    #[case::multi_bin(multi_module_package(), EnvelopeFormat::Model)]
    #[case::multi_bin_exts(multi_module_package(), EnvelopeFormat::ModelWithExtensions)]
    #[case::multi_text(multi_module_package(), EnvelopeFormat::ModelText)]
    #[case::multi_text_exts(multi_module_package(), EnvelopeFormat::ModelTextWithExtensions)]
    fn model_roundtrip(#[case] package: Package, #[case] format: EnvelopeFormat) {
        let mut buffer = Vec::new();
        let config = EnvelopeConfig { format, zstd: None };
        package.store(&mut buffer, config).unwrap();

        let (desc, new_package) =
            read_envelope(BufReader::new(buffer.as_slice()), &PRELUDE_REGISTRY).unwrap();
        let decoded_config = desc.header.config();

        assert_eq!(config.format, decoded_config.format);
        assert_eq!(config.zstd.is_some(), decoded_config.zstd.is_some());

        assert_eq!(package, new_package);
    }

    /// Test helper to call `check_breaking_extensions_against_registry`
    fn check(hugr: &Hugr, registry: &ExtensionRegistry) -> Result<(), ExtensionBreakingError> {
        let mut desc = description::ModuleDesc::default();
        desc.load_used_extensions_generator(&hugr)?;
        let Some(used_exts) = desc.used_extensions_generator else {
            return Ok(());
        };
        check_breaking_extensions(registry, used_exts)
    }

    #[rstest]
    #[case::simple(simple_package())]
    fn test_check_breaking_extensions(#[case] mut package: Package) {
        // extension with major version 0
        let test_ext_v0 =
            Extension::new(ExtensionId::new_unchecked("test-v0"), Version::new(0, 2, 3));
        //  extension with major version > 0
        let test_ext_v1 =
            Extension::new(ExtensionId::new_unchecked("test-v1"), Version::new(1, 2, 3));

        // Create a registry with the test extensions
        let registry =
            ExtensionRegistry::new([Arc::new(test_ext_v0.clone()), Arc::new(test_ext_v1.clone())]);
        let mut hugr = package.modules.remove(0);

        // No metadata - should pass
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Matching version for v0 - should pass
        let used_exts = json!([{ "name": "test-v0", "version": "0.2.3" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Matching major/minor but lower patch for v0 - should pass
        let used_exts = json!([{ "name": "test-v0", "version": "0.2.2" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        //Different minor version for v0 - should fail
        let used_exts = json!([{ "name": "test-v0", "version": "0.3.3" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v0" && registered == Version::new(0, 2, 3) && used == Version::new(0, 3, 3)
        );

        assert!(
            check(&hugr, hugr.extensions()).is_ok(),
            "Extension is not actually used in the HUGR, should be ignored by full check"
        );

        // Different major version for v0 - should fail
        let used_exts = json!([{ "name": "test-v0", "version": "1.2.3" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v0" && registered == Version::new(0, 2, 3) && used == Version::new(1, 2, 3)
        );

        // Higher patch version for v0 - should fail
        let used_exts = json!([{ "name": "test-v0", "version": "0.2.4" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v0" && registered == Version::new(0, 2, 3) && used == Version::new(0, 2, 4)
        );

        // Matching version for v1 - should pass
        let used_exts = json!([{ "name": "test-v1", "version": "1.2.3" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Lower minor version for v1 - should pass
        let used_exts = json!([{ "name": "test-v1", "version": "1.1.0" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Lower patch for v1 - should pass
        let used_exts = json!([{ "name": "test-v1", "version": "1.2.2" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Different major version for v1 - should fail
        let used_exts = json!([{ "name": "test-v1", "version": "2.2.3" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v1" && registered == Version::new(1, 2, 3) && used == Version::new(2, 2, 3)
        );

        // Higher minor version for v1 - should fail
        let used_exts = json!([{ "name": "test-v1", "version": "1.3.0" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v1" && registered == Version::new(1, 2, 3) && used == Version::new(1, 3, 0)
        );

        // Higher patch version for v1 - should fail
        let used_exts = json!([{ "name": "test-v1", "version": "1.2.4" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v1" && registered == Version::new(1, 2, 3) && used == Version::new(1, 2, 4)
        );

        // Non-registered extension - should pass
        let used_exts = json!([{ "name": "unknown", "version": "1.0.0" }]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));

        // Multiple extensions - one mismatch should fail
        let used_exts = json!([
            { "name": "unknown", "version": "1.0.0" },
            { "name": "test-v1", "version": "2.0.0" }
        ]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::ExtensionVersionMismatch(ExtensionVersionMismatch {
                name,
                registered,
                used
            })) if name == "test-v1" && registered == Version::new(1, 2, 3) && used == Version::new(2, 0, 0)
        );

        // Invalid metadata format - should fail with deserialization error
        hugr.set_metadata(
            hugr.module_root(),
            USED_EXTENSIONS_KEY,
            json!("not an array"),
        );
        assert_matches!(
            check(&hugr, &registry),
            Err(ExtensionBreakingError::Deserialization(_))
        );

        //  Multiple extensions with all compatible versions - should pass
        let used_exts = json!([
            { "name": "test-v0", "version": "0.2.2" },
            { "name": "test-v1", "version": "1.1.9" }
        ]);
        hugr.set_metadata(hugr.module_root(), USED_EXTENSIONS_KEY, used_exts);
        assert_matches!(check(&hugr, &registry), Ok(()));
    }
}

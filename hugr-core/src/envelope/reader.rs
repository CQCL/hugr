use std::io::{BufRead, Read};
use std::str::FromStr as _;

use hugr_model::v0::table;
use itertools::{Either, Itertools as _};

use crate::HugrView as _;
use crate::envelope::description::{PackageDesc, PackageDescResult};
use crate::envelope::header::{EnvelopeFormat, HeaderError};
use crate::envelope::{
    EnvelopeError, EnvelopeHeader, ExtensionBreakingError, FormatUnsupportedError,
};
use crate::extension::resolution::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::{Extension, ExtensionRegistry};
use crate::import::{ImportError, import_described_hugr};
use crate::package::Package;

use super::{check_breaking_extensions, check_model_version, package_json::PackageEncodingError};
use thiserror::Error;

use hugr_model::v0::bumpalo::Bump;
#[cfg(feature = "zstd")]
type RightType<R> = std::io::BufReader<zstd::Decoder<'static, std::io::BufReader<R>>>;
#[cfg(not(feature = "zstd"))]
type RightType<R> = std::io::BufReader<R>;

pub(crate) struct MaybeZstdRead<R>(Either<R, RightType<R>>);

impl<R> std::io::Read for MaybeZstdRead<R>
where
    R: std::io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match &mut self.0 {
            Either::Left(r) => r.read(buf),
            Either::Right(r) => r.read(buf),
        }
    }
}

impl<R> std::io::BufRead for MaybeZstdRead<R>
where
    R: std::io::BufRead,
{
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        match &mut self.0 {
            Either::Left(r) => r.fill_buf(),
            Either::Right(r) => r.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match &mut self.0 {
            Either::Left(r) => r.consume(amt),
            Either::Right(r) => r.consume(amt),
        }
    }
}

/// Reader for HUGR envelopes.
///
/// To read a package from an envelope, first create an `EnvelopeReader` using
/// [`EnvelopeReader::new`], then call [`EnvelopeReader::read`].
pub(super) struct EnvelopeReader<R> {
    description: PackageDesc,
    reader: MaybeZstdRead<R>,
    registry: ExtensionRegistry,
}

impl<R: BufRead> EnvelopeReader<R> {
    /// Create a new `EnvelopeReader` from a reader and an extension registry.
    ///
    /// # Errors
    ///
    /// - If the header is invalid.
    /// - If zstd decompression is requested but the `zstd` feature is not
    ///   enabled.
    pub(super) fn new(mut reader: R, registry: &ExtensionRegistry) -> Result<Self, HeaderError> {
        let header = EnvelopeHeader::read(&mut reader)?;
        let reader = match header.zstd {
            #[cfg(feature = "zstd")]
            true => Either::Right(std::io::BufReader::new(zstd::Decoder::new(reader)?)),
            #[cfg(not(feature = "zstd"))]
            true => Err(super::header::HeaderErrorInner::ZstdUnsupported)?,
            false => Either::Left(reader),
        };
        Ok(Self {
            description: PackageDesc::new(header),
            reader: MaybeZstdRead(reader),
            registry: registry.clone(),
        })
    }

    pub(crate) fn description(&self) -> &PackageDesc {
        &self.description
    }

    fn header(&self) -> &EnvelopeHeader {
        &self.description.header
    }

    fn register_packaged(&mut self, extensions: &ExtensionRegistry) {
        self.registry.extend(extensions);
    }

    fn read_impl(&mut self) -> Result<Package, PayloadError> {
        let mut package = match self.header().format {
            EnvelopeFormat::PackageJson => self.decode_json()?,
            EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions => self.decode_model()?,
            EnvelopeFormat::ModelText | EnvelopeFormat::ModelTextWithExtensions => {
                self.decode_model_ast()?
            }
        };
        self.description.set_n_modules(package.modules.len());
        for (index, module) in package.modules.iter_mut().enumerate() {
            let desc = &mut self.description.modules[index];
            let desc = desc.get_or_insert_default();
            desc.load_used_extensions_generator(module)
                .map_err(ExtensionBreakingError::from)?;
            if let Some(used_exts) = &mut desc.used_extensions_generator {
                check_breaking_extensions(module.extensions(), used_exts.drain(..))?;
            }

            module.resolve_extension_defs(&self.registry)?;
            // overwrite the description with the actual module read,
            // cheap so ok to repeat.
            desc.load_from_hugr(&module);
        }

        for (index, ext) in package.extensions.iter().enumerate() {
            self.description.set_packaged_extension(index, ext);
        }
        Ok(package)
    }

    /// Read the package and return the description and the package or an error.
    ///
    /// The description is always returned, even if reading the package fails,
    /// it may be incomplete. Minimally it contains the header, but may also
    /// contain any information gathered prior to the error.
    ///
    /// # Errors
    ///
    /// - If reading the package payload fails.
    pub(super) fn read(mut self) -> PackageDescResult<Package, PayloadError> {
        let res = self.read_impl();

        self.description.wrap(res)
    }

    /// Read a Package in json format from an io reader.
    /// Returns package and the combined extension registry
    /// of the provided registry and the package extensions.
    fn decode_json(&mut self) -> Result<Package, PackageEncodingError> {
        let super::package_json::PackageDeser {
            modules,
            extensions: pkg_extensions,
        } = serde_json::from_reader(&mut self.reader)?;
        let modules = modules.into_iter().map(|h| h.0).collect_vec();
        let pkg_extensions = ExtensionRegistry::new_with_extension_resolution(
            pkg_extensions,
            &WeakExtensionRegistry::from(&self.registry),
        )?;

        // Resolve the operations in the modules using the defined registries.
        self.register_packaged(&pkg_extensions);
        Ok(Package {
            modules,
            extensions: pkg_extensions,
        })
    }
    /// Read a HUGR model payload from a reader.
    fn decode_model(&mut self) -> Result<Package, ModelBinaryReadError> {
        check_model_version(self.header().format)?;
        let bump = Bump::default();
        let model_package = hugr_model::v0::binary::read_from_reader(&mut self.reader, &bump)?;

        let packaged_extensions = if self.header().format == EnvelopeFormat::ModelWithExtensions {
            ExtensionRegistry::load_json(&mut self.reader, &self.registry)?
        } else {
            ExtensionRegistry::new([])
        };
        self.register_packaged(&packaged_extensions);

        self.import_package(&model_package, packaged_extensions)
            .map_err(Into::into)
    }

    /// Read a HUGR model text payload from a reader.
    fn decode_model_ast(&mut self) -> Result<Package, ModelTextReadError> {
        let format = self.header().format;
        check_model_version(format)?;

        let packaged_extensions = if format == EnvelopeFormat::ModelTextWithExtensions {
            let deserializer = serde_json::Deserializer::from_reader(&mut self.reader);
            // Deserialize the first json object, leaving the rest of the reader unconsumed.
            let extra_extensions = deserializer
                .into_iter::<Vec<Extension>>()
                .next()
                .unwrap_or(Ok(vec![]))?;
            ExtensionRegistry::new(extra_extensions.into_iter().map(std::sync::Arc::new))
        } else {
            ExtensionRegistry::new([])
        };

        // Read the package into a string, then parse it.
        //
        // Due to how `to_string` works, we cannot append extensions after the package.
        let mut buffer = String::new();
        self.reader.read_to_string(&mut buffer)?;
        let ast_package = hugr_model::v0::ast::Package::from_str(&buffer)?;

        let bump = Bump::default();
        let model_package = ast_package.resolve(&bump)?;

        self.import_package(&model_package, packaged_extensions)
            .map_err(Into::into)
    }

    fn import_package(
        &mut self,
        package: &table::Package,
        packaged_extensions: ExtensionRegistry,
    ) -> Result<Package, crate::import::ImportError> {
        self.description.set_n_modules(package.modules.len());

        let modules = package
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| {
                let (result, desc) = import_described_hugr(module, &self.registry).into_parts();
                self.description.set_module(index, desc);
                result
            })
            .collect::<Result<Vec<_>, _>>()?;

        // This does not panic since the import already requires a module root.
        let mut package = Package::new(modules);
        package.extensions = packaged_extensions;
        Ok(package)
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
/// Error decoding an envelope payload.
#[error(transparent)]
pub struct PayloadError(PayloadErrorInner);

#[derive(Error, Debug)]
#[non_exhaustive]
#[error(transparent)]
/// Error decoding an envelope payload with enumerated variants.
enum PayloadErrorInner {
    /// Error decoding a JSON format package.
    JsonRead(#[from] PackageEncodingError),
    /// Error decoding a binary model format package.
    ModelBinary(#[from] ModelBinaryReadError),
    /// Error decoding a text model format package.
    ModelText(#[from] ModelTextReadError),
    /// Error raised while checking for breaking extension version mismatch.
    ExtensionsBreaking(#[from] ExtensionBreakingError),
    /// Error resolving extensions while decoding the payload.
    ExtensionResolution(#[from] ExtensionResolutionError),
}
impl From<PayloadError> for EnvelopeError {
    fn from(value: PayloadError) -> Self {
        match value.0 {
            PayloadErrorInner::JsonRead(e) => e.into(),
            PayloadErrorInner::ModelBinary(e) => e.into(),
            PayloadErrorInner::ModelText(e) => e.into(),
            #[expect(deprecated)]
            PayloadErrorInner::ExtensionsBreaking(e) => super::WithGenerator {
                inner: Box::new(e),
                generator: None,
            }
            .into(),
            PayloadErrorInner::ExtensionResolution(e) => e.into(),
        }
    }
}

impl<T: Into<PayloadErrorInner>> From<T> for PayloadError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
#[non_exhaustive]
enum ModelTextReadError {
    ParseString(#[from] hugr_model::v0::ast::ParseError),
    Import(#[from] ImportError),
    ExtensionLoad(#[from] crate::extension::ExtensionRegistryLoadError),
    FormatUnsupported(#[from] FormatUnsupportedError),
    ExtensionDeserialize(#[from] serde_json::Error),
    StringRead(#[from] std::io::Error),
    ResolveError(#[from] hugr_model::v0::ast::ResolveError),
}
impl From<ModelTextReadError> for EnvelopeError {
    fn from(value: ModelTextReadError) -> Self {
        match value {
            ModelTextReadError::FormatUnsupported(e) => EnvelopeError::FormatUnsupported {
                format: e.format,
                feature: e.feature,
            },
            ModelTextReadError::ParseString(e) => e.into(),
            ModelTextReadError::Import(e) => e.into(),
            ModelTextReadError::ExtensionLoad(e) => e.into(),
            ModelTextReadError::ExtensionDeserialize(e) => e.into(),
            ModelTextReadError::StringRead(e) => e.into(),
            ModelTextReadError::ResolveError(e) => e.into(),
        }
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
#[non_exhaustive]
enum ModelBinaryReadError {
    ParseString(#[from] hugr_model::v0::ast::ParseError),
    ReadBinary(#[from] hugr_model::v0::binary::ReadError),
    Import(#[from] ImportError),
    Extensions(#[from] crate::extension::ExtensionRegistryLoadError),
    FormatUnsupported(#[from] FormatUnsupportedError),
}

impl From<ModelBinaryReadError> for EnvelopeError {
    fn from(value: ModelBinaryReadError) -> Self {
        match value {
            ModelBinaryReadError::FormatUnsupported(e) => EnvelopeError::FormatUnsupported {
                format: e.format,
                feature: e.feature,
            },
            ModelBinaryReadError::ParseString(e) => e.into(),
            ModelBinaryReadError::ReadBinary(e) => e.into(),
            ModelBinaryReadError::Import(e) => e.into(),
            ModelBinaryReadError::Extensions(e) => e.into(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::extension::ExtensionRegistry;

    use crate::envelope::header::EnvelopeHeader;
    use cool_asserts::assert_matches;

    use std::io::{Cursor, Write as _};

    #[test]
    fn test_read_invalid_header() {
        let cursor = Cursor::new(Vec::new()); // Empty cursor simulates invalid header
        let registry = ExtensionRegistry::new([]);
        let result = EnvelopeReader::new(cursor, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_invalid_json_payload() {
        let header = EnvelopeHeader {
            format: EnvelopeFormat::PackageJson,
            ..Default::default()
        };
        let mut cursor = Cursor::new(Vec::new());
        header.write(&mut cursor).unwrap();
        cursor.write_all(b"invalid json").unwrap(); // Write invalid JSON payload
        cursor.set_position(0);

        let registry = ExtensionRegistry::new([]);
        let reader = EnvelopeReader::new(cursor, &registry).unwrap();
        let (result, description) = reader.read().into_parts();

        assert_matches!(result, Err(PayloadError(PayloadErrorInner::JsonRead(_))));
        assert_eq!(description.header, header);
    }

    #[test]
    fn test_read_text_format() {
        let header = EnvelopeHeader {
            format: EnvelopeFormat::ModelTextWithExtensions,
            ..Default::default()
        };
        let mut cursor = Cursor::new(Vec::new());
        header.write(&mut cursor).unwrap();
        cursor.set_position(0);

        let registry = ExtensionRegistry::new([]);
        let reader = EnvelopeReader::new(cursor, &registry).unwrap();
        let (result, description) = reader.read().into_parts();

        assert_matches!(result, Err(PayloadError(PayloadErrorInner::ModelText(_))));
        assert_eq!(description.header, header);
    }

    #[test]
    fn test_partial_description_on_error() {
        let header = EnvelopeHeader {
            format: EnvelopeFormat::PackageJson,
            ..Default::default()
        };
        let mut cursor = Cursor::new(Vec::new());
        header.write(&mut cursor).unwrap();
        cursor.write_all(b"{\"modules\": [\"invalid\"]}").unwrap(); // Invalid module structure
        cursor.set_position(0);

        let registry = ExtensionRegistry::new([]);
        let reader = EnvelopeReader::new(cursor, &registry).unwrap();
        let desc_res = reader.read();

        assert_matches!(
            desc_res.as_ref(),
            Err(PayloadError(PayloadErrorInner::JsonRead(_)))
        );
        let description = desc_res.description();
        assert_eq!(description.header, header);
        assert_eq!(description.n_modules(), 0); // No valid modules should be set
    }
}

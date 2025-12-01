use std::io::{BufRead, Read};
use std::str::FromStr as _;

use hugr_model::v0::table;
use itertools::{Either, Itertools as _};

use crate::HugrView as _;
use crate::envelope::description::{ExtensionDesc, ModuleDesc, PackageDesc};
use crate::envelope::header::{EnvelopeFormat, HeaderError};
use crate::envelope::{EnvelopeHeader, ExtensionBreakingError, FormatUnsupportedError};
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

    fn header(&self) -> &EnvelopeHeader {
        &self.description.header
    }

    fn register_packaged(&mut self, extensions: &ExtensionRegistry) {
        self.registry.extend(extensions);
    }

    /// Handle extension resolution errors by recording missing extensions in the description.
    ///
    /// This function inspects the error and adds any missing extensions to the module description
    /// with a default version of 0.0.0.
    fn handle_resolution_error(desc: &mut ModuleDesc, err: &ExtensionResolutionError) {
        match err {
            ExtensionResolutionError::MissingOpExtension {
                missing_extension, ..
            }
            | ExtensionResolutionError::MissingTypeExtension {
                missing_extension, ..
            } => desc.extend_used_extensions_resolved([ExtensionDesc::new(
                missing_extension,
                crate::extension::Version::new(0, 0, 0),
            )]),
            ExtensionResolutionError::InvalidConstTypes {
                missing_extensions, ..
            } => desc.extend_used_extensions_resolved(
                missing_extensions
                    .iter()
                    .map(|ext| ExtensionDesc::new(ext, crate::extension::Version::new(0, 0, 0))),
            ),
            _ => {}
        }
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

            module
                .resolve_extension_defs(&self.registry)
                .inspect_err(|err| Self::handle_resolution_error(desc, err))?;

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
    pub(super) fn read(mut self) -> (PackageDesc, Result<Package, PayloadError>) {
        let res = self.read_impl();

        (self.description, res)
    }

    /// Read a Package in json format from an io reader.
    /// Returns package and the combined extension registry
    /// of the provided registry and the package extensions.
    fn decode_json(&mut self) -> Result<Package, PackageEncodingError> {
        let super::package_json::PackageDeser {
            modules,
            extensions: pkg_extensions,
        } = serde_json::from_reader(&mut self.reader)
            .map_err(PackageEncodingError::JsonEncoding)?;
        let modules = modules.into_iter().map(|h| h.0).collect_vec();
        let pkg_extensions = ExtensionRegistry::new_with_extension_resolution(
            pkg_extensions,
            &WeakExtensionRegistry::from(&self.registry),
        )
        .map_err(PackageEncodingError::ExtensionResolution)?;

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
        self.register_packaged(&packaged_extensions);

        let modules = package
            .modules
            .iter()
            .enumerate()
            .map(|(index, module)| {
                let (desc, result) = import_described_hugr(module, &self.registry);
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
pub(crate) enum PayloadErrorInner {
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

impl<T: Into<PayloadErrorInner>> From<T> for PayloadError {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) enum ModelTextReadError {
    ParseString(#[from] hugr_model::v0::ast::ParseError),
    Import(#[from] ImportError),
    ExtensionLoad(#[from] crate::extension::ExtensionRegistryLoadError),
    FormatUnsupported(#[from] FormatUnsupportedError),
    ExtensionDeserialize(#[from] serde_json::Error),
    StringRead(#[from] std::io::Error),
    ResolveError(#[from] hugr_model::v0::ast::ResolveError),
}

#[derive(Debug, Error)]
#[error(transparent)]
pub(crate) enum ModelBinaryReadError {
    ParseString(#[from] hugr_model::v0::ast::ParseError),
    ReadBinary(#[from] hugr_model::v0::binary::ReadError),
    Import(#[from] ImportError),
    Extensions(#[from] crate::extension::ExtensionRegistryLoadError),
    FormatUnsupported(#[from] FormatUnsupportedError),
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::extension::{ExtensionId, ExtensionRegistry};

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
        let (description, result) = reader.read();

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
        let (description, result) = reader.read();

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
        let (description, result) = reader.read();

        assert_matches!(result, Err(PayloadError(PayloadErrorInner::JsonRead(_))));
        assert_eq!(description.header, header);
        assert_eq!(description.n_modules(), 0); // No valid modules should be set
    }

    #[test]
    fn test_handle_resolution_error() {
        use crate::extension::ExtensionId;
        use crate::ops::{OpName, constant::ValueName};
        use crate::types::TypeName;

        let mut desc = ModuleDesc::default();
        let handle_error = |d: &mut ModuleDesc, err: &ExtensionResolutionError| {
            EnvelopeReader::<Cursor<Vec<u8>>>::handle_resolution_error(d, err)
        };
        let assert_extensions = |d: &ModuleDesc, expected_ids: &[&ExtensionId]| {
            let resolved = d.used_extensions_resolved.as_ref().unwrap();
            assert_eq!(resolved.len(), expected_ids.len());
            let names: Vec<_> = resolved.iter().map(|e| &e.name).collect();
            for ext_id in expected_ids {
                assert!(names.contains(&&ext_id.to_string()));
            }
            assert!(
                resolved
                    .iter()
                    .all(|e| e.version == crate::extension::Version::new(0, 0, 0))
            );
        };

        // Test MissingOpExtension
        let ext_id = ExtensionId::new("test.extension").unwrap();
        let error = ExtensionResolutionError::MissingOpExtension {
            node: None,
            op: OpName::new("test.op"),
            missing_extension: ext_id.clone(),
            available_extensions: vec![],
        };
        handle_error(&mut desc, &error);
        assert_extensions(&desc, &[&ext_id]);

        // Test MissingTypeExtension
        desc.used_extensions_resolved = None;
        let ext_id2 = ExtensionId::new("test.extension2").unwrap();
        let error = ExtensionResolutionError::MissingTypeExtension {
            node: None,
            ty: TypeName::new("test.type"),
            missing_extension: ext_id2.clone(),
            available_extensions: vec![],
        };
        handle_error(&mut desc, &error);
        assert_extensions(&desc, &[&ext_id2]);

        // Test InvalidConstTypes with multiple extensions
        desc.used_extensions_resolved = None;
        let ext_id3 = ExtensionId::new("test.extension3").unwrap();
        let ext_id4 = ExtensionId::new("test.extension4").unwrap();
        let mut missing_exts = crate::extension::ExtensionSet::new();
        missing_exts.insert(ext_id3.clone());
        missing_exts.insert(ext_id4.clone());

        let error = ExtensionResolutionError::InvalidConstTypes {
            value: ValueName::new("test.value"),
            missing_extensions: missing_exts,
        };
        handle_error(&mut desc, &error);
        assert_extensions(&desc, &[&ext_id3, &ext_id4]);

        // Test other error variant (should not add anything)
        desc.used_extensions_resolved = None;
        let error = ExtensionResolutionError::WrongTypeDefExtension {
            extension: ExtensionId::new("ext1").unwrap(),
            def: TypeName::new("def"),
            wrong_extension: ExtensionId::new("ext2").unwrap(),
        };
        handle_error(&mut desc, &error);
        assert!(desc.used_extensions_resolved.is_none());
    }

    #[test]
    fn test_decode_model_ast_with_packaged_extensions() {
        let mut simple_package = crate::builder::test::simple_package();
        // Create a simple extension to package
        let ext_id = ExtensionId::new("test.packaged.extension").unwrap();
        let extension = Extension::new(ext_id.clone(), crate::extension::Version::new(1, 0, 0));

        simple_package
            .extensions
            .register(std::sync::Arc::new(extension))
            .unwrap();

        let header = EnvelopeHeader {
            format: EnvelopeFormat::ModelTextWithExtensions,
            ..Default::default()
        };

        let mut cursor = Cursor::new(Vec::new());
        simple_package.store(&mut cursor, header.config()).unwrap();
        cursor.set_position(0);

        let registry = ExtensionRegistry::new([]);
        let mut reader = EnvelopeReader::new(cursor, &registry).unwrap();

        // Before decoding, the packaged extension should not be registered
        assert!(!reader.registry.contains(&ext_id));

        let result = reader.decode_model_ast();

        // After decoding, the packaged extension should be registered
        assert!(result.is_ok());
        assert!(reader.registry.contains(&ext_id));

        let package = result.unwrap();
        assert!(package.extensions.contains(&ext_id));
    }
}

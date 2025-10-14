use std::io::{BufRead, Read};
use std::str::FromStr as _;

use itertools::{Either, Itertools as _};

use crate::envelope::description::PackageDesc;
use crate::envelope::header::{EnvelopeFormat, HeaderError};
use crate::envelope::{EnvelopeError, EnvelopeHeader, ExtensionBreakingError};
use crate::extension::resolution::ExtensionResolutionError;
use crate::extension::{Extension, ExtensionRegistry};
use crate::import::import_package;
use crate::package::Package;

use super::{
    ModelBinaryReadError, ModelTextReadError, WithGenerator, check_breaking_extensions,
    check_model_version, package_json::PackageEncodingError,
};
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
pub struct EnvelopeReader<R> {
    pub(crate) description: PackageDesc,
    pub(crate) reader: MaybeZstdRead<R>,
    pub(crate) registry: ExtensionRegistry,
}

impl<R: BufRead> EnvelopeReader<R> {
    /// Create a new `EnvelopeReader` from a reader and an extension registry.
    ///
    /// # Errors
    ///
    /// - If the header is invalid.
    /// - If zstd decompression is requested but the `zstd` feature is not
    ///   enabled.
    pub fn new(mut reader: R, registry: &ExtensionRegistry) -> Result<Self, HeaderError> {
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
            EnvelopeFormat::PackageJson => self.from_json_reader()?,
            EnvelopeFormat::Model | EnvelopeFormat::ModelWithExtensions => self.decode_model()?,
            EnvelopeFormat::ModelText | EnvelopeFormat::ModelTextWithExtensions => {
                self.decode_model_ast()?
            }
        };

        for module in package.modules.iter_mut() {
            check_breaking_extensions(&module)?;
            module.resolve_extension_defs(&self.registry)?;
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
    pub fn read(mut self) -> (PackageDesc, Result<Package, PayloadError>) {
        let res = self.read_impl();

        (self.description, res)
    }

    /// Read a Package in json format from an io reader.
    /// Returns package and the combined extension registry
    /// of the provided registry and the package extensions.
    fn from_json_reader(&mut self) -> Result<Package, PackageEncodingError> {
        let val: serde_json::Value = serde_json::from_reader(&mut self.reader)?;

        let super::package_json::PackageDeser {
            modules,
            extensions: pkg_extensions,
        } = serde_json::from_value(val.clone())?;
        let modules = modules.into_iter().map(|h| h.0).collect_vec();
        let pkg_extensions = ExtensionRegistry::new_with_extension_resolution(
            pkg_extensions,
            &crate::extension::resolution::WeakExtensionRegistry::from(&self.registry),
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

        let package = import_package(&model_package, packaged_extensions, &self.registry)?;

        Ok(package)
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
        self.register_packaged(&packaged_extensions);

        // Read the package into a string, then parse it.
        //
        // Due to how `to_string` works, we cannot append extensions after the package.
        let mut buffer = String::new();
        self.reader.read_to_string(&mut buffer)?;
        let ast_package = hugr_model::v0::ast::Package::from_str(&buffer)?;

        let bump = Bump::default();
        let model_package = ast_package.resolve(&bump)?;

        let package = import_package(&model_package, packaged_extensions, &self.registry)?;

        Ok(package)
    }
}

#[derive(Error, Debug, derive_more::Display)]
#[non_exhaustive]
/// Error decoding an envelope payload.
#[display("Error reading the envelope payload. {_0}")]
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

impl From<PayloadError> for EnvelopeError {
    fn from(value: PayloadError) -> Self {
        match value.0 {
            PayloadErrorInner::JsonRead(e) => e.into(),
            PayloadErrorInner::ModelBinary(e) => e.into(),
            PayloadErrorInner::ModelText(e) => e.into(),
            PayloadErrorInner::ExtensionsBreaking(e) => WithGenerator {
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

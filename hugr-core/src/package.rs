//! Bundles of hugr modules along with the extension required to load them.

use derive_more::{Display, Error, From};
use std::path::Path;
use std::{fs, io};

use crate::extension::{ExtensionRegistry, ExtensionRegistryError};
use crate::hugr::ValidationError;
use crate::{Extension, Hugr};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Package of module HUGRs and extensions.
/// The HUGRs are validated against the extensions.
pub struct Package {
    /// Module HUGRs included in the package.
    pub modules: Vec<Hugr>,
    /// Extensions to validate against.
    pub extensions: Vec<Extension>,
}

impl Package {
    /// Create a new package.
    pub fn new(
        modules: impl IntoIterator<Item = Hugr>,
        extensions: impl IntoIterator<Item = Extension>,
    ) -> Self {
        Self {
            modules: modules.into_iter().collect(),
            extensions: extensions.into_iter().collect(),
        }
    }

    /// Validate the package against an extension registry.
    ///
    /// `reg` is updated with any new extensions.
    pub fn validate(&mut self, reg: &mut ExtensionRegistry) -> Result<(), PackageValidationError> {
        for ext in &self.extensions {
            reg.register_updated_ref(ext)?;
        }
        for hugr in self.modules.iter_mut() {
            hugr.update_validate(reg)?;
        }
        Ok(())
    }

    /// Read a Package in json format from an io reader.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn from_json_reader(reader: impl io::Read) -> Result<Self, PackageLoadError> {
        let val: serde_json::Value = serde_json::from_reader(reader)?;
        let pkg_load_err = match serde_json::from_value::<Package>(val.clone()) {
            Ok(p) => return Ok(p),
            Err(e) => e,
        };

        if let Ok(hugr) = serde_json::from_value::<Hugr>(val) {
            return Ok(Package::new([hugr], []));
        }

        // Return the original error from parsing the package.
        Err(PackageLoadError::JsonEncoding(pkg_load_err))
    }

    /// Read a Package from a json string.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn from_json(json: impl AsRef<str>) -> Result<Self, PackageLoadError> {
        Self::from_json_reader(json.as_ref().as_bytes())
    }

    /// Read a Package from a json file.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, PackageLoadError> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        Self::from_json_reader(reader)
    }
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum PackageLoadError {
    /// Error raised while parsing the package json.
    JsonEncoding(serde_json::Error),
    /// Error raised while reading from a file.
    IOError(io::Error),
}

/// Error raised while validating a package.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum PackageValidationError {
    /// Error raised while processing the package extensions.
    Extension(ExtensionRegistryError),
    /// Error raised while validating the package hugrs.
    Validation(ValidationError),
}

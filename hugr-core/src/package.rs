//! Bundles of hugr modules along with the extension required to load them.

use derive_more::{Display, Error, From};
use std::collections::HashMap;
use std::path::Path;
use std::{fs, io};

use crate::extension::{ExtensionRegistry, ExtensionRegistryError};
use crate::hugr::{HugrView, ValidationError};
use crate::ops::{NamedOp, OpType};
use crate::{Extension, Hugr};

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
/// Package of module HUGRs and extensions.
/// The HUGRs are validated against the extensions.
pub struct Package {
    /// Module HUGRs included in the package.
    pub modules: Vec<Hugr>,
    /// Extensions to validate against.
    pub extensions: Vec<Extension>,
}

impl Package {
    /// Create a new package from a list of hugrs and extensions.
    ///
    /// All the HUGRs must have a `Module` operation at the root.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the HUGRs does not have a `Module` root.
    pub fn new(
        modules: impl IntoIterator<Item = Hugr>,
        extensions: impl IntoIterator<Item = Extension>,
    ) -> Result<Self, PackageError> {
        let modules: Vec<Hugr> = modules.into_iter().collect();
        for (idx, module) in modules.iter().enumerate() {
            let root_op = module.get_optype(module.root());
            if !root_op.is_module() {
                return Err(PackageError::NonModuleHugr {
                    module_index: idx,
                    root_op: root_op.clone(),
                });
            }
        }
        Ok(Self {
            modules,
            extensions: extensions.into_iter().collect(),
        })
    }

    /// Create a new package from a list of hugrs and extensions.
    ///
    /// HUGRs that do not have a `Module` root will be wrapped in a new `Module` root,
    /// depending on the root optype.
    ///
    /// - Currently all non-module roots will raise [PackageError::CannotWrapHugr].
    ///
    /// # Errors
    ///
    /// Returns an error if any of the HUGRs cannot be wrapped in a module.
    pub fn from_hugrs(
        modules: impl IntoIterator<Item = Hugr>,
        extensions: impl IntoIterator<Item = Extension>,
    ) -> Result<Self, PackageError> {
        let modules: Vec<Hugr> = modules
            .into_iter()
            .map(to_module_hugr)
            .collect::<Result<_, PackageError>>()?;
        Ok(Self {
            modules,
            extensions: extensions.into_iter().collect(),
        })
    }

    /// Create a new package containing a single HUGR, and no extension definitions.
    ///
    /// If the Hugr is not a module, a new [OpType::Module] root will be added.
    /// This behaviours depends on the root optype.
    ///
    /// - Currently all non-module roots will raise [PackageError::CannotWrapHugr].
    ///
    /// # Errors
    ///
    /// Returns an error if the hugr cannot be wrapped in a module.
    pub fn from_hugr(hugr: Hugr) -> Result<Self, PackageError> {
        let mut package = Self::default();
        let module = to_module_hugr(hugr)?;
        package.modules.push(module);
        Ok(package)
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
    pub fn from_json_reader(reader: impl io::Read) -> Result<Self, PackageEncodingError> {
        let val: serde_json::Value = serde_json::from_reader(reader)?;
        let pkg_load_err = match serde_json::from_value::<Package>(val.clone()) {
            Ok(p) => return Ok(p),
            Err(e) => e,
        };

        if let Ok(hugr) = serde_json::from_value::<Hugr>(val) {
            return Ok(Package::from_hugr(hugr)?);
        }

        // Return the original error from parsing the package.
        Err(PackageEncodingError::JsonEncoding(pkg_load_err))
    }

    /// Read a Package from a json string.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn from_json(json: impl AsRef<str>) -> Result<Self, PackageEncodingError> {
        Self::from_json_reader(json.as_ref().as_bytes())
    }

    /// Read a Package from a json file.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, PackageEncodingError> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        Self::from_json_reader(reader)
    }

    /// Write the Package in json format into an io writer.
    pub fn to_json_writer(&self, writer: impl io::Write) -> Result<(), PackageEncodingError> {
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Write the Package into a json string.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    pub fn to_json(&self) -> Result<String, PackageEncodingError> {
        let json = serde_json::to_string(self)?;
        Ok(json)
    }

    /// Write the Package into a json file.
    pub fn to_json_file(&self, path: impl AsRef<Path>) -> Result<(), PackageEncodingError> {
        let file = fs::File::open(path)?;
        let writer = io::BufWriter::new(file);
        self.to_json_writer(writer)
    }
}

impl PartialEq for Package {
    fn eq(&self, other: &Self) -> bool {
        if self.modules != other.modules || self.extensions.len() != other.extensions.len() {
            return false;
        }
        // Extensions may be in different orders, so we compare them as sets.
        let exts = self
            .extensions
            .iter()
            .map(|e| (&e.name, e))
            .collect::<HashMap<_, _>>();
        other
            .extensions
            .iter()
            .all(|e| exts.get(&e.name).map_or(false, |&e2| e == e2))
    }
}

/// Alter an arbitrary hugr to contain an [OpType::Module] root.
///
/// The behaviour depends on the root optype. See [Package::from_hugr] for details.
///
/// # Returns
///
fn to_module_hugr(hugr: Hugr) -> Result<Hugr, PackageError> {
    let root_op = hugr.get_optype(hugr.root());
    match root_op {
        OpType::Module(_) => Ok(hugr),
        _ => Err(PackageError::CannotWrapHugr {
            root_op: root_op.clone(),
        }),
    }
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error)]
#[non_exhaustive]
pub enum PackageError {
    /// A hugr in the package does not have an [OpType::Module] root.
    #[display("Module {module_index} in the package does not have an OpType::Module root, but {}", root_op.name())]
    NonModuleHugr {
        /// The module index.
        module_index: usize,
        /// The invalid root operation.
        root_op: OpType,
    },
    /// Tried to initialize a package with a hugr that cannot be wrapped in a module.
    #[display("A hugr with optype {} cannot be wrapped in a module.", root_op.name())]
    CannotWrapHugr {
        /// The invalid root operation.
        root_op: OpType,
    },
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum PackageEncodingError {
    /// Error raised while parsing the package json.
    JsonEncoding(serde_json::Error),
    /// Error raised while reading from a file.
    IOError(io::Error),
    /// Improper package definition.
    Package(PackageError),
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

#[cfg(test)]
mod test {
    use crate::builder::test::{
        simple_cfg_hugr, simple_dfg_hugr, simple_funcdef_hugr, simple_module_hugr,
    };
    use crate::extension::ExtensionId;

    use super::*;
    use rstest::{fixture, rstest};
    use semver::Version;

    #[fixture]
    fn simple_package() -> Package {
        let hugr0 = simple_module_hugr();
        let hugr1 = simple_module_hugr();

        let ext_1_id = ExtensionId::new("ext1").unwrap();
        let ext_2_id = ExtensionId::new("ext2").unwrap();
        let ext1 = Extension::new(ext_1_id.clone(), Version::new(2, 4, 8));
        let ext2 = Extension::new(ext_2_id, Version::new(1, 0, 0));

        Package {
            modules: vec![hugr0, hugr1],
            extensions: vec![ext1, ext2],
        }
    }

    #[rstest]
    #[case::empty(Package::default())]
    #[case::simple(simple_package())]
    fn package_roundtrip(#[case] package: Package) {
        let json = package.to_json().unwrap();
        let new_package = Package::from_json(&json).unwrap();
        assert_eq!(package, new_package);
    }

    #[rstest]
    #[case::module(simple_module_hugr(), false)]
    #[case::dfg(simple_funcdef_hugr(), true)]
    #[case::dfg(simple_dfg_hugr(), true)]
    #[case::cfg(simple_cfg_hugr(), true)]
    fn hugr_to_package(#[case] hugr: Hugr, #[case] errors: bool) {
        match (Package::from_hugr(hugr), errors) {
            (Ok(package), false) => {
                assert_eq!(package.modules.len(), 1);
                let root_op = package.modules[0].get_optype(package.modules[0].root());
                assert!(root_op.is_module());
            }
            (Err(_), true) => {}
            (p, _) => panic!("Unexpected result {:?}", p),
        }
    }
}

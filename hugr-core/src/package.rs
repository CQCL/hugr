//! Bundles of hugr modules along with the extension required to load them.

use derive_more::{Display, Error, From};
use std::collections::HashMap;
use std::path::Path;
use std::{fs, io, mem};

use crate::builder::{Container, Dataflow, DataflowSubContainer, ModuleBuilder};
use crate::extension::{ExtensionRegistry, ExtensionRegistryError};
use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{HugrView, ValidationError};
use crate::ops::{Module, NamedOp, OpTag, OpTrait, OpType};
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
    pub fn update_validate(
        &mut self,
        reg: &mut ExtensionRegistry,
    ) -> Result<(), PackageValidationError> {
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

impl AsRef<[Hugr]> for Package {
    fn as_ref(&self) -> &[Hugr] {
        &self.modules
    }
}

/// Alter an arbitrary hugr to contain an [OpType::Module] root.
///
/// The behaviour depends on the root optype. See [Package::from_hugr] for details.
///
/// # Errors
///
/// Returns [PackageError::]
fn to_module_hugr(mut hugr: Hugr) -> Result<Hugr, PackageError> {
    let root = hugr.root();
    let root_op = hugr.get_optype(root);
    let tag = root_op.tag();

    // Modules can be returned as is.
    if root_op.is_module() {
        return Ok(hugr);
    }
    // If possible, wrap the hugr directly in a module.
    if OpTag::ModuleOp.is_superset(tag) {
        let new_root = hugr.add_node(Module::new().into());
        hugr.set_root(new_root);
        hugr.set_parent(root, new_root);
        return Ok(hugr);
    }
    // Wrap it in a function definition named "main" inside the module otherwise.
    if OpTag::DataflowChild.is_superset(tag) && !root_op.is_input() && !root_op.is_output() {
        let signature = root_op
            .dataflow_signature()
            .unwrap_or_else(|| panic!("Dataflow child {} without signature", root_op.name()));
        let mut new_hugr = ModuleBuilder::new();
        let mut func = new_hugr.define_function("main", signature).unwrap();
        let dataflow_node = func.add_hugr_with_wires(hugr, func.input_wires()).unwrap();
        func.finish_with_outputs(dataflow_node.outputs()).unwrap();
        return Ok(mem::take(new_hugr.hugr_mut()));
    }
    // Reject all other hugrs.
    Err(PackageError::CannotWrapHugr {
        root_op: root_op.clone(),
    })
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error, PartialEq)]
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
    use cool_asserts::assert_matches;

    use crate::builder::test::{
        simple_cfg_hugr, simple_dfg_hugr, simple_funcdef_hugr, simple_module_hugr,
    };
    use crate::extension::{ExtensionId, EMPTY_REG};
    use crate::ops::dataflow::IOTrait;
    use crate::ops::Input;

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

    #[fixture]
    fn simple_input_node() -> Hugr {
        Hugr::new(Input::new(vec![]))
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
    #[case::funcdef(simple_funcdef_hugr(), false)]
    #[case::dfg(simple_dfg_hugr(), false)]
    #[case::cfg(simple_cfg_hugr(), false)]
    #[case::unsupported_input(simple_input_node(), true)]
    fn hugr_to_package(#[case] hugr: Hugr, #[case] errors: bool) {
        match (&Package::from_hugr(hugr), errors) {
            (Ok(package), false) => {
                assert_eq!(package.modules.len(), 1);
                let root_op = package.modules[0].get_optype(package.modules[0].root());
                assert!(root_op.is_module());
            }
            (Err(_), true) => {}
            (p, _) => panic!("Unexpected result {:?}", p),
        }
    }

    #[rstest]
    fn package_properties() {
        let module = simple_module_hugr();
        let dfg = simple_dfg_hugr();

        assert_matches!(
            Package::new([module.clone(), dfg.clone()], []),
            Err(PackageError::NonModuleHugr {
                module_index: 1,
                root_op: OpType::DFG(_),
            })
        );

        let mut pkg = Package::from_hugrs([module, dfg], []).unwrap();
        let mut reg = EMPTY_REG.clone();
        pkg.validate(&mut reg).unwrap();

        assert_eq!(pkg.modules.len(), 2);
    }
}

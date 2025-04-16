//! Bundles of hugr modules along with the extension required to load them.

use derive_more::{Display, Error, From};
use itertools::Itertools;
use std::path::Path;
use std::{fs, io, mem};

use crate::builder::{Container, Dataflow, DataflowSubContainer, ModuleBuilder};
use crate::envelope::{read_envelope, write_envelope, EnvelopeConfig, EnvelopeError};
use crate::extension::resolution::ExtensionResolutionError;
use crate::extension::{ExtensionId, ExtensionRegistry, PRELUDE_REGISTRY};
use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{ExtensionError, HugrView, ValidationError};
use crate::ops::{FuncDefn, Module, NamedOp, OpTag, OpTrait, OpType};
use crate::{Extension, Hugr};

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Package of module HUGRs.
pub struct Package {
    /// Module HUGRs included in the package.
    pub modules: Vec<Hugr>,
    /// Extensions used in the modules.
    ///
    /// This is a superset of the extensions used in the modules.
    pub extensions: ExtensionRegistry,
}

impl Package {
    /// Create a new package from a list of hugrs.
    ///
    /// All the HUGRs must have a `Module` operation at the root.
    ///
    /// Collects the extensions used in the modules and stores them in top-level
    /// `extensions` attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the HUGRs does not have a `Module` root.
    pub fn new(modules: impl IntoIterator<Item = Hugr>) -> Result<Self, PackageError> {
        let modules: Vec<Hugr> = modules.into_iter().collect();
        let mut extensions = ExtensionRegistry::default();
        for (idx, module) in modules.iter().enumerate() {
            let root_op = module.get_optype(module.root());
            if !root_op.is_module() {
                return Err(PackageError::NonModuleHugr {
                    module_index: idx,
                    root_op: root_op.clone(),
                });
            }
            extensions.extend(module.extensions());
        }
        Ok(Self {
            modules,
            extensions,
        })
    }

    /// Create a new package from a list of hugrs.
    ///
    /// HUGRs that do not have a `Module` root will be wrapped in a new `Module` root,
    /// depending on the root optype.
    ///
    /// - Currently all non-module roots will raise [PackageError::CannotWrapHugr].
    ///
    /// # Errors
    ///
    /// Returns an error if any of the HUGRs cannot be wrapped in a module.
    pub fn from_hugrs(modules: impl IntoIterator<Item = Hugr>) -> Result<Self, PackageError> {
        let modules: Vec<Hugr> = modules
            .into_iter()
            .map(to_module_hugr)
            .collect::<Result<_, PackageError>>()?;

        let mut extensions = ExtensionRegistry::default();
        for module in &modules {
            extensions.extend(module.extensions());
        }

        Ok(Self {
            modules,
            extensions,
        })
    }

    /// Create a new package containing a single HUGR.
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
        package.extensions = module.extensions().clone();
        package.modules.push(module);
        Ok(package)
    }

    /// Validate the modules of the package.
    ///
    /// Ensures that the top-level extension list is a superset of the extensions used in the modules.
    pub fn validate(&self) -> Result<(), PackageValidationError> {
        for hugr in self.modules.iter() {
            hugr.validate()?;

            let missing_exts = hugr
                .extensions()
                .ids()
                .filter(|id| !self.extensions.contains(id))
                .cloned()
                .collect_vec();
            if !missing_exts.is_empty() {
                return Err(PackageValidationError::MissingExtension {
                    missing: missing_exts,
                    available: self.extensions.ids().cloned().collect(),
                });
            }
        }
        Ok(())
    }

    /// Read a Package from a HUGR envelope.
    pub fn load(
        reader: impl io::BufRead,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        let extensions = extensions.unwrap_or(&PRELUDE_REGISTRY);
        let (_, pkg) = read_envelope(reader, extensions)?;
        Ok(pkg)
    }

    /// Read a Package from a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::load` with a bytearray instead.
    pub fn load_str(
        envelope: impl AsRef<str>,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        Self::load(envelope.as_ref().as_bytes(), extensions)
    }

    /// Store the Package in a HUGR envelope.
    pub fn store(
        &self,
        writer: impl io::Write,
        config: EnvelopeConfig,
    ) -> Result<(), EnvelopeError> {
        write_envelope(writer, self, config)
    }

    /// Store the Package in a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::store` with a bytearray instead.
    /// See [EnvelopeFormat::ascii_printable][crate::envelope::EnvelopeFormat::ascii_printable].
    pub fn store_str(&self, config: EnvelopeConfig) -> Result<String, EnvelopeError> {
        if !config.format.ascii_printable() {
            return Err(EnvelopeError::NonASCIIFormat {
                format: config.format,
            });
        }

        let mut buf = Vec::new();
        self.store(&mut buf, config)?;
        Ok(String::from_utf8(buf).expect("Envelope is valid utf8"))
    }

    /// Read a Package in json format from an io reader.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    //
    // TODO: Make this a private method only used by the envelope reader, and remove the automatic HUGR fallback.
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::load` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_json_reader(
        reader: impl io::Read,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, PackageEncodingError> {
        let val: serde_json::Value = serde_json::from_reader(reader)?;

        // Try to load a package json.
        // Defers the extension registry loading so we can call [`ExtensionRegistry::load_json_value`] directly.
        #[derive(Debug, serde::Deserialize)]
        struct PackageDeser {
            pub modules: Vec<Hugr>,
            pub extensions: Vec<Extension>,
        }
        let loaded_pkg = serde_json::from_value::<PackageDeser>(val.clone());

        if let Ok(PackageDeser {
            mut modules,
            extensions: pkg_extensions,
        }) = loaded_pkg
        {
            let mut pkg_extensions = ExtensionRegistry::new_with_extension_resolution(
                pkg_extensions,
                &extension_registry.into(),
            )?;

            // Resolve the operations in the modules using the defined registries.
            let mut combined_registry = extension_registry.clone();
            combined_registry.extend(&pkg_extensions);

            for module in &mut modules {
                module.resolve_extension_defs(&combined_registry)?;
                pkg_extensions.extend(module.extensions());
            }

            return Ok(Package {
                modules,
                extensions: pkg_extensions,
            });
        };
        let pkg_load_err = loaded_pkg.unwrap_err();

        // As a fallback, try to load a hugr json.
        if let Ok(mut hugr) = serde_json::from_value::<Hugr>(val) {
            hugr.resolve_extension_defs(extension_registry)?;
            if cfg!(feature = "extension_inference") {
                hugr.infer_extensions(false)?;
            }
            return Ok(Package::from_hugr(hugr)?);
        }

        // Return the original error from parsing the package.
        Err(PackageEncodingError::JsonEncoding(pkg_load_err))
    }

    /// Read a Package from a json string.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::load_str` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_json(
        json: impl AsRef<str>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, PackageEncodingError> {
        #[allow(deprecated)]
        Self::from_json_reader(json.as_ref().as_bytes(), extension_registry)
    }

    /// Read a Package from a json file.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::load` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_json_file(
        path: impl AsRef<Path>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, PackageEncodingError> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        #[allow(deprecated)]
        Self::from_json_reader(reader, extension_registry)
    }

    /// Write the Package in json format into an io writer.
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::store` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn to_json_writer(&self, writer: impl io::Write) -> Result<(), PackageEncodingError> {
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Write the Package into a json string.
    ///
    /// If the json encodes a single [Hugr] instead, it will be inserted in a new [Package].
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::store_str` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn to_json(&self) -> Result<String, PackageEncodingError> {
        let json = serde_json::to_string(self)?;
        Ok(json)
    }

    /// Write the Package into a json file.
    #[deprecated(
        since = "0.14.5",
        note = "Json encoding of packages is deprecated. Use `Package::store` instead"
    )]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn to_json_file(&self, path: impl AsRef<Path>) -> Result<(), PackageEncodingError> {
        let file = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)?;
        let writer = io::BufWriter::new(file);
        #[allow(deprecated)]
        self.to_json_writer(writer)
    }
}

impl AsRef<[Hugr]> for Package {
    fn as_ref(&self) -> &[Hugr] {
        &self.modules
    }
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
    /// Could not resolve the extension needed to encode the hugr.
    ExtensionResolution(ExtensionResolutionError),
    /// Could not resolve the runtime extensions for the hugr.
    RuntimeExtensionResolution(ExtensionError),
}

/// Error raised while validating a package.
#[derive(Debug, Display, From, Error)]
#[non_exhaustive]
pub enum PackageValidationError {
    /// Error raised while processing the package extensions.
    #[display("The package modules use the extension{} {} not present in the defined set. The declared extensions are {}",
            if missing.len() > 1 {"s"} else {""},
            missing.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
            available.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
        )]
    MissingExtension {
        /// The missing extensions.
        missing: Vec<ExtensionId>,
        /// The available extensions.
        available: Vec<ExtensionId>,
    },
    /// Error raised while validating the package hugrs.
    Validation(ValidationError),
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::builder::test::{
        simple_cfg_hugr, simple_dfg_hugr, simple_funcdef_hugr, simple_module_hugr,
    };
    use crate::ops::dataflow::IOTrait;
    use crate::ops::Input;

    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn simple_input_node() -> Hugr {
        Hugr::new(Input::new(vec![]))
    }

    #[rstest]
    #[case::module("module", simple_module_hugr(), false)]
    #[case::funcdef("funcdef", simple_funcdef_hugr(), false)]
    #[case::dfg("dfg", simple_dfg_hugr(), false)]
    #[case::cfg("cfg", simple_cfg_hugr(), false)]
    #[case::unsupported_input("input", simple_input_node(), true)]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_to_package(#[case] test_name: &str, #[case] hugr: Hugr, #[case] errors: bool) {
        match (&Package::from_hugr(hugr), errors) {
            (Ok(package), false) => {
                assert_eq!(package.modules.len(), 1);
                let hugr = &package.modules[0];
                let root_op = hugr.get_optype(hugr.root());
                assert!(root_op.is_module());

                insta::assert_snapshot!(test_name, hugr.mermaid_string());
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
            Package::new([module.clone(), dfg.clone()]),
            Err(PackageError::NonModuleHugr {
                module_index: 1,
                root_op: OpType::DFG(_),
            })
        );

        let pkg = Package::from_hugrs([module, dfg]).unwrap();
        pkg.validate().unwrap();

        assert_eq!(pkg.modules.len(), 2);
    }
}

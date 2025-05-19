//! Declarative extension definitions.
//!
//! This module includes functions to dynamically load HUGR extensions defined in a YAML file.
//!
//! An extension file may define multiple extensions, each with a set of types and operations.
//!
//! See the [specification] for more details.
//!
//! ### Example
//!
//! ```yaml
#![doc = include_str!("../../examples/extension/declarative.yaml")]
//! ```
//!
//! The definition can be loaded into a registry using the [`load_extensions`] or [`load_extensions_file`] functions.
//! ```rust
//! # const DECLARATIVE_YAML: &str = include_str!("../../examples/extension/declarative.yaml");
//! # use hugr::extension::declarative::load_extensions;
//! // Required extensions must already be present in the registry.
//! let mut reg = hugr::std_extensions::STD_REG.clone();
//! load_extensions(DECLARATIVE_YAML, &mut reg).unwrap();
//! ```
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format

mod ops;
mod signature;
mod types;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::Extension;
use crate::extension::prelude::PRELUDE_ID;
use crate::ops::OpName;
use crate::types::TypeName;

use super::{
    ExtensionBuildError, ExtensionId, ExtensionRegistry, ExtensionRegistryError, ExtensionSet,
    PRELUDE,
};
use ops::OperationDeclaration;
use smol_str::SmolStr;
use types::TypeDeclaration;

use serde::{Deserialize, Serialize};

/// Load a set of extensions from a YAML string into a registry.
///
/// Any required extensions must already be present in the registry.
pub fn load_extensions(
    yaml: &str,
    registry: &mut ExtensionRegistry,
) -> Result<(), ExtensionDeclarationError> {
    let ext: ExtensionSetDeclaration = serde_yaml::from_str(yaml)?;
    ext.add_to_registry(registry)
}

/// Load a set of extensions from a file into a registry.
///
/// Any required extensions must already be present in the registry.
pub fn load_extensions_file(
    path: &Path,
    registry: &mut ExtensionRegistry,
) -> Result<(), ExtensionDeclarationError> {
    let file = File::open(path)?;
    let ext: ExtensionSetDeclaration = serde_yaml::from_reader(file)?;
    ext.add_to_registry(registry)
}

/// A set of declarative extension definitions with some metadata.
///
/// These are normally contained in a single YAML file.
//
// TODO: More metadata, "namespace"?
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ExtensionSetDeclaration {
    /// A set of extension definitions.
    //
    // TODO: allow qualified, and maybe locally-scoped?
    extensions: Vec<ExtensionDeclaration>,
    /// A list of extension IDs that this extension depends on.
    /// Optional.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    imports: ExtensionSet,
}

/// A declarative extension definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ExtensionDeclaration {
    // TODO add version
    /// The name of the extension.
    name: ExtensionId,
    /// A list of types that this extension provides.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    types: Vec<TypeDeclaration>,
    /// A list of operations that this extension provides.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    operations: Vec<OperationDeclaration>,
    // TODO: Values?
}

impl ExtensionSetDeclaration {
    /// Register this set of extensions with the given registry.
    pub fn add_to_registry(
        &self,
        registry: &mut ExtensionRegistry,
    ) -> Result<(), ExtensionDeclarationError> {
        // All dependencies must be present in the registry.
        for imp in self.imports.iter() {
            if !registry.contains(imp) {
                return Err(ExtensionDeclarationError::MissingExtension { ext: imp.clone() });
            }
        }

        // A set of extensions that are in scope for the definition. This is a
        // subset of `registry` that includes `self.imports` and the previous
        // extensions defined in the declaration.
        let mut scope = self.imports.clone();

        // The prelude is auto-imported.
        if !registry.contains(&PRELUDE_ID) {
            registry.register(PRELUDE.clone())?;
        }
        if !scope.contains(&PRELUDE_ID) {
            scope.insert(PRELUDE_ID);
        }

        // Registers extensions sequentially, adding them to the current scope.
        for decl in &self.extensions {
            let ctx = DeclarationContext {
                scope: &scope,
                registry,
            };
            let ext = decl.make_extension(&self.imports, ctx)?;
            scope.insert(ext.name().clone());
            registry.register(ext)?;
        }

        Ok(())
    }
}

impl ExtensionDeclaration {
    /// Create an [`Extension`] from this declaration.
    pub fn make_extension(
        &self,
        _imports: &ExtensionSet,
        ctx: DeclarationContext<'_>,
    ) -> Result<Arc<Extension>, ExtensionDeclarationError> {
        // TODO: The imports were previously used as runtime extension
        // requirements for the constructed extension. Now that runtime
        // extension requirements are removed, they are no longer recorded
        // anywhere in the `Extension`.

        Extension::try_new_arc(
            self.name.clone(),
            // TODO: Get the version as a parameter.
            crate::extension::Version::new(0, 0, 0),
            |ext, extension_ref| {
                for t in &self.types {
                    t.register(ext, ctx, extension_ref)?;
                }

                for o in &self.operations {
                    o.register(ext, ctx, extension_ref)?;
                }

                Ok(())
            },
        )
    }
}

/// Some context data used while translating a declarative extension definition.
#[derive(Debug, Copy, Clone)]
struct DeclarationContext<'a> {
    /// The set of extensions that are in scope for this extension.
    pub scope: &'a ExtensionSet,
    /// The registry to use for resolving dependencies.
    pub registry: &'a ExtensionRegistry,
}

/// Errors that can occur while loading an extension set.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExtensionDeclarationError {
    /// An error occurred while deserializing the extension set.
    #[error("Error while parsing the extension set yaml: {0}")]
    Deserialize(#[from] serde_yaml::Error),
    /// An error in registering the loaded extensions.
    #[error("Error registering the extensions.")]
    ExtensionRegistryError(#[from] ExtensionRegistryError),
    /// An error occurred while adding operations or types to the extension.
    #[error("Error while adding operations or types to the extension: {0}")]
    ExtensionBuildError(#[from] ExtensionBuildError),
    /// Invalid yaml file.
    #[error("Invalid yaml declaration file {0}")]
    InvalidFile(#[from] std::io::Error),
    /// A required extension is missing.
    #[error("Missing required extension {ext}")]
    MissingExtension {
        /// The missing imported extension.
        ext: ExtensionId,
    },
    /// Referenced an unknown type.
    #[error("Extension {ext} referenced an unknown type {ty}.")]
    MissingType {
        /// The extension that referenced the unknown type.
        ext: ExtensionId,
        /// The unknown type.
        ty: TypeName,
    },
    /// Parametric types are not currently supported as type parameters.
    ///
    /// TODO: Support this.
    #[error("Found a currently unsupported higher-order type parameter {ty} in extension {ext}")]
    ParametricTypeParameter {
        /// The extension that referenced the unsupported type parameter.
        ext: ExtensionId,
        /// The unsupported type parameter.
        ty: TypeName,
    },
    /// Parametric types are not currently supported as type parameters.
    ///
    /// TODO: Support this.
    #[error("Found a currently unsupported parametric operation {op} in extension {ext}")]
    ParametricOperation {
        /// The extension that referenced the unsupported op parameter.
        ext: ExtensionId,
        /// The operation.
        op: OpName,
    },
    /// Operation definitions with no signature are not currently supported.
    ///
    /// TODO: Support this.
    #[error("Operation {op} in extension {ext} has no signature. This is not currently supported.")]
    MissingSignature {
        /// The extension containing the operation.
        ext: ExtensionId,
        /// The operation with no signature.
        op: OpName,
    },
    /// An unknown type was specified in a signature.
    #[error("Type {ty} is not in scope. In extension {ext}.")]
    UnknownType {
        /// The extension that referenced the type.
        ext: ExtensionId,
        /// The unsupported type.
        ty: String,
    },
    /// Parametric port repetitions are not currently supported.
    ///
    /// TODO: Support this.
    #[error("Unsupported port repetition {parametric_repetition} in extension {ext}")]
    UnsupportedPortRepetition {
        /// The extension that referenced the type.
        ext: crate::hugr::IdentList,
        /// The repetition expression
        parametric_repetition: SmolStr,
    },
    /// Lowering definitions for an operation are not currently supported.
    ///
    /// TODO: Support this.
    #[error("Unsupported lowering definition for op {op} in extension {ext}")]
    LoweringNotSupported {
        /// The extension.
        ext: crate::hugr::IdentList,
        /// The operation with the lowering definition.
        op: OpName,
    },
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::sync::Arc;

    use crate::extension::PRELUDE_REGISTRY;
    use crate::std_extensions;

    use super::*;

    /// A yaml extension defining an empty extension.
    const EMPTY_YAML: &str = r#"
extensions:
- name: EmptyExt
"#;

    /// A yaml extension defining an extension with one type and two operations.
    const BASIC_YAML: &str = r#"
imports: [prelude]

extensions:
- name: SimpleExt
  types:
  - name: MyType
    description: A simple type with no parameters
    bound: Any
  operations:
  - name: MyOperation
    description: A simple operation with no inputs nor outputs
    signature:
      inputs: []
      outputs: []
  - name: AnotherOperation
    description: An operation from 3 qubits to 3 qubits
    signature:
        inputs: [MyType, Q, Q]
        outputs: [[MyType, 1], [Control, Q, 2]]
"#;

    /// A yaml extension with unsupported features.
    const UNSUPPORTED_YAML: &str = r#"
extensions:
- name: UnsupportedExt
  types:
  - name: MyType
    description: A simple type with no parameters
    # Parametric types are not currently supported.
    params: [String]
  operations:
  - name: UnsupportedOperation
    description: An operation from 3 qubits to 3 qubits
    params:
        # Parametric operations are not currently supported.
         param1: String
    signature:
        # Type declarations will have their own syntax.
        inputs: []
        outputs: ["Array<param1>[USize]"]
"#;

    /// The yaml used in the module documentation.
    const EXAMPLE_YAML_FILE: &str = "examples/extension/declarative.yaml";

    #[rstest]
    #[case(EMPTY_YAML, 1, 0, 0, &PRELUDE_REGISTRY)]
    #[case(BASIC_YAML, 1, 1, 2, &PRELUDE_REGISTRY)]
    fn test_decode(
        #[case] yaml: &str,
        #[case] num_declarations: usize,
        #[case] num_types: usize,
        #[case] num_operations: usize,
        #[case] dependencies: &ExtensionRegistry,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut reg = dependencies.clone();
        load_extensions(yaml, &mut reg)?;

        let new_exts = new_extensions(&reg, dependencies).collect_vec();

        assert_eq!(new_exts.len(), num_declarations);
        assert_eq!(new_exts.iter().flat_map(|e| e.types()).count(), num_types);
        assert_eq!(
            new_exts.iter().flat_map(|e| e.operations()).count(),
            num_operations
        );
        Ok(())
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[rstest]
    #[case(EXAMPLE_YAML_FILE, 1, 1, 3, &std_extensions::STD_REG)]
    fn test_decode_file(
        #[case] yaml_file: &str,
        #[case] num_declarations: usize,
        #[case] num_types: usize,
        #[case] num_operations: usize,
        #[case] dependencies: &ExtensionRegistry,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut reg = dependencies.clone();
        load_extensions_file(&PathBuf::from(yaml_file), &mut reg)?;

        let new_exts = new_extensions(&reg, dependencies).collect_vec();

        assert_eq!(new_exts.len(), num_declarations);
        assert_eq!(new_exts.iter().flat_map(|e| e.types()).count(), num_types);
        assert_eq!(
            new_exts.iter().flat_map(|e| e.operations()).count(),
            num_operations
        );
        Ok(())
    }

    #[rstest]
    #[case(UNSUPPORTED_YAML, &PRELUDE_REGISTRY)]
    fn test_unsupported(
        #[case] yaml: &str,
        #[case] dependencies: &ExtensionRegistry,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut reg = dependencies.clone();

        // The parsing should not fail.
        let ext: ExtensionSetDeclaration = serde_yaml::from_str(yaml)?;

        assert!(ext.add_to_registry(&mut reg).is_err());

        Ok(())
    }

    /// Returns a list of new extensions that have been defined in a register,
    /// comparing against a set of pre-included dependencies.
    fn new_extensions<'a>(
        reg: &'a ExtensionRegistry,
        dependencies: &'a ExtensionRegistry,
    ) -> impl Iterator<Item = &'a Arc<Extension>> {
        reg.iter()
            .filter(move |ext| !dependencies.contains(ext.name()) && ext.name() != &PRELUDE_ID)
    }
}

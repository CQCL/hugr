//! Declarative extension definitions.
//!
//! This module defines a YAML schema for defining extensions in a declarative way.
//!
//! See the [specification] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format

mod ops;
mod signature;
mod types;

use std::fs::File;

use crate::extension::prelude::PRELUDE_ID;
use crate::types::TypeName;
use crate::Extension;

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
    path: &std::path::Path,
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
            scope.insert(&PRELUDE_ID);
        }

        // Registers extensions sequentially, adding them to the current scope.
        for decl in &self.extensions {
            let ext = decl.make_extension(&self.imports, &scope, registry)?;
            let ext = registry.register(ext)?;
            scope.insert(ext.name())
        }

        Ok(())
    }
}

impl ExtensionDeclaration {
    /// Create an [`Extension`] from this declaration.
    ///
    /// Parameters:
    /// - `imports`: The set of extensions that this extension depends on.
    /// - `scope`: The set of extensions that are in scope for this extension.
    ///     This may include other extensions in the same file, in addition to `imports`.
    /// - `registry`: The registry to use for resolving dependencies.
    ///     Extensions not in `scope` will be ignored.
    pub fn make_extension(
        &self,
        imports: &ExtensionSet,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
    ) -> Result<Extension, ExtensionDeclarationError> {
        let mut ext = Extension::new_with_reqs(self.name.clone(), imports.clone());

        for t in &self.types {
            t.register(&mut ext, scope, registry)?;
        }

        for o in &self.operations {
            o.register(&mut ext, scope, registry)?;
        }

        Ok(ext)
    }
}

/// Errors that can occur while loading an extension set.
#[derive(Debug, thiserror::Error)]
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
        op: SmolStr,
    },
    /// Operation definitions with no signature are not currently supported.
    ///
    /// TODO: Support this.
    #[error(
        "Operation {op} in extension {ext} has no signature. This is not currently supported."
    )]
    MissingSignature {
        /// The extension containing the operation.
        ext: ExtensionId,
        /// The operation with no signature.
        op: SmolStr,
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
        op: SmolStr,
    },
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::extension::prelude::PRELUDE_ID;
    use crate::extension::PRELUDE_REGISTRY;

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
  operations:
  - name: MyOperation
    description: A simple operation with no inputs nor outputs
    signature:
      inputs: []
      outputs: []
  - name: AnotherOperation
    description: An operation from 2 qubits to 2 qubits
    signature:
        inputs: [["Target", Q], ["Control", Q, 1]]
        outputs: [[null, Q, 2]]
"#;

    #[rstest]
    #[case(EMPTY_YAML, 1, 0, 0)]
    #[case(BASIC_YAML, 1, 1, 2)]
    fn test_decode(
        #[case] yaml: &str,
        #[case] num_declarations: usize,
        #[case] num_types: usize,
        #[case] num_operations: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut reg = PRELUDE_REGISTRY.to_owned();
        load_extensions(yaml, &mut reg)?;

        let non_prelude_regs = || reg.iter().filter(|(id, _)| *id != &PRELUDE_ID);

        assert_eq!(non_prelude_regs().count(), num_declarations);
        assert_eq!(
            non_prelude_regs().flat_map(|(_, e)| e.types()).count(),
            num_types
        );
        assert_eq!(
            non_prelude_regs().flat_map(|(_, e)| e.operations()).count(),
            num_operations
        );
        Ok(())
    }
}

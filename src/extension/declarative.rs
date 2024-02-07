//! Declarative extension definitions.
//!
//! This module defines a YAML schema for defining extensions in a declarative way.
//!
//! See the [specification] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format

mod ops;
mod types;

use std::fs::File;

use crate::Extension;

use super::{ExtensionBuildError, ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError};
use ops::OperationDeclaration;
use types::TypeDeclaration;

use serde::{Deserialize, Serialize};

/// Load a set of extensions from a YAML string.
pub fn load_extensions(yaml: &str) -> Result<ExtensionRegistry, ExtensionDeclarationError> {
    let ext: ExtensionSetDeclaration = serde_yaml::from_str(yaml)?;
    ext.make_registry()
}

/// Load a set of extensions from a file.
pub fn load_extensions_file(
    path: &std::path::Path,
) -> Result<ExtensionRegistry, ExtensionDeclarationError> {
    let file = File::open(path)?;
    let ext: ExtensionSetDeclaration = serde_yaml::from_reader(file)?;
    ext.make_registry()
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
    pub fn make_registry(&self) -> Result<ExtensionRegistry, ExtensionDeclarationError> {
        let exts = self
            .extensions
            .iter()
            .map(|ext| ext.make_extension(&self.imports))
            .collect::<Result<Vec<Extension>, _>>()?;
        Ok(ExtensionRegistry::try_new(exts)?)
    }
}

impl ExtensionDeclaration {
    /// Create an [`Extension`] from this declaration.
    pub fn make_extension(
        &self,
        imports: &ExtensionSet,
    ) -> Result<Extension, ExtensionDeclarationError> {
        let mut ext = Extension::new_with_reqs(self.name.clone(), imports.clone());

        for t in &self.types {
            t.register(&mut ext)?;
        }

        for o in &self.operations {
            o.register(&mut ext)?;
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
    /// An error in the validation of the loaded extensions.
    #[error("Error validating extension {ext}: {err}")]
    ExtensionValidationError {
        /// The extension that failed validation.
        ext: ExtensionId,
        /// The error that occurred.
        err: SignatureError,
    },
    /// An error occurred while adding operations or types to the extension.
    #[error("Error while adding operations or types to the extension: {0}")]
    ExtensionBuildError(#[from] ExtensionBuildError),
    /// Invalid yaml file.
    #[error("Invalid yaml declaration file {0}")]
    InvalidFile(#[from] std::io::Error),
}

impl From<(ExtensionId, SignatureError)> for ExtensionDeclarationError {
    fn from((ext, err): (ExtensionId, SignatureError)) -> Self {
        ExtensionDeclarationError::ExtensionValidationError { ext, err }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use super::*;

    /// A yaml extension defining an empty extension.
    const EMPTY_YAML: &str = r#"
extensions:
- name: EmptyExt
"#;

    /// A yaml extension defining an extension with one type and two operations.
    const BASIC_YAML: &str = r#"
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
  - name: UnusableOperation
    description: An operation without a defined signature
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
        let ext: ExtensionRegistry = load_extensions(yaml)?;

        assert_eq!(ext.len(), num_declarations);
        assert_eq!(ext.iter().flat_map(|(_, e)| e.types()).count(), num_types);
        assert_eq!(
            ext.iter().flat_map(|(_, e)| e.operations()).count(),
            num_operations
        );
        Ok(())
    }
}

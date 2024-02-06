//! Declarative extension definitions.
//!
//! This module defines a YAML schema for defining extensions in a declarative way.
//!
//! See the [specification] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format

mod ops;
mod types;

use super::ExtensionId;
use ops::OperationDeclaration;
use types::TypeDeclaration;

use serde::{Deserialize, Serialize};

/// A set of declarative extension definitions with some metadata.
///
/// These are normally contained in a single YAML file.
//
// TODO: More metadata, "namespace"?
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
struct ExtensionSetDeclaration {
    /// A set of extension definitions.
    //
    // TODO: allow qualified, and maybe locally-scoped?
    extensions: Vec<ExtensionDeclaration>,
    /// A list of extension IDs that this extension depends on.
    /// Optional.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    imports: Vec<ExtensionId>,
}

/// A declarative extension definition.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
struct ExtensionDeclaration {
    /// The name of the extension.
    name: String,
    /// A list of types that this extension provides.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    types: Vec<TypeDeclaration>,
    /// A list of operations that this extension provides.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    operations: Vec<OperationDeclaration>,
}

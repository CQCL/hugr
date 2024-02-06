//! Declarative operation definitions.
//!
//! This module defines a YAML schema for defining operations in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use serde::{Deserialize, Serialize};

/// A declarative operation definition.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub(super) struct OperationDeclaration {}

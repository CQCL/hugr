//! Utilities for resolving operations and types present in a HUGR, and updating
//! the list of used extensions. See [`crate::Hugr::resolve_extension_defs`].
//!
//! When listing "used extensions" we only care about _definitional_ extension
//! requirements, i.e., the operations and types that are required to define the
//! HUGR nodes and wire types. This is computed from the union of all extension
//! required across the HUGR.
//!
//! This is distinct from _runtime_ extension requirements, which are defined
//! more granularly in each function signature by the `required_extensions`
//! field. See the `extension_inference` feature and related modules for that.
//!
//! Note: These procedures are only temporary until `hugr-model` is stabilized.
//! Once that happens, hugrs will no longer be directly deserialized using serde
//! but instead will be created by the methods in `crate::import`. As these
//! (will) automatically resolve extensions as the operations are created,
//! we will no longer require this post-facto resolution step.

mod ops;
mod types;

pub(crate) use ops::update_op_extensions;
pub(crate) use types::update_op_types_extensions;

use derive_more::{Display, Error, From};

use super::{Extension, ExtensionId, ExtensionRegistry};
use crate::ops::custom::OpaqueOpError;
use crate::ops::{NamedOp, OpName, OpType};
use crate::types::TypeName;
use crate::Node;

/// Errors that can occur during extension resolution.
#[derive(Debug, Display, Clone, Error, From, PartialEq)]
#[non_exhaustive]
pub enum ExtensionResolutionError {
    /// Could not resolve an opaque operation to an extension operation.
    #[display("Error resolving opaque operation: {_0}")]
    #[from]
    OpaqueOpError(OpaqueOpError),
    /// An operation requires an extension that is not in the given registry.
    #[display(
        "{op} ({node}) requires extension {missing_extension}, but it could not be found in the extension list used during resolution. The available extensions are: {}",
        available_extensions.join(", ")
    )]
    MissingOpExtension {
        /// The node that requires the extension.
        node: Node,
        /// The operation that requires the extension.
        op: OpName,
        /// The missing extension
        missing_extension: ExtensionId,
        /// A list of available extensions.
        available_extensions: Vec<ExtensionId>,
    },
    #[display(
        "Type {ty} in {node} requires extension {missing_extension}, but it could not be found in the extension list used during resolution. The available extensions are: {}",
        available_extensions.join(", ")
    )]
    /// A type references an extension that is not in the given registry.
    MissingTypeExtension {
        /// The node that requires the extension.
        node: Node,
        /// The type that requires the extension.
        ty: TypeName,
        /// The missing extension
        missing_extension: ExtensionId,
        /// A list of available extensions.
        available_extensions: Vec<ExtensionId>,
    },
}

impl ExtensionResolutionError {
    /// Create a new error for missing operation extensions.
    pub fn missing_op_extension(
        node: Node,
        op: &OpType,
        missing_extension: &ExtensionId,
        extensions: &ExtensionRegistry,
    ) -> Self {
        Self::MissingOpExtension {
            node,
            op: NamedOp::name(op),
            missing_extension: missing_extension.clone(),
            available_extensions: extensions.ids().cloned().collect(),
        }
    }

    /// Create a new error for missing type extensions.
    pub fn missing_type_extension(
        node: Node,
        ty: &TypeName,
        missing_extension: &ExtensionId,
        extensions: &ExtensionRegistry,
    ) -> Self {
        Self::MissingTypeExtension {
            node,
            ty: ty.clone(),
            missing_extension: missing_extension.clone(),
            available_extensions: extensions.ids().cloned().collect(),
        }
    }
}

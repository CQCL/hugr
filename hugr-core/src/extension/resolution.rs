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
//! (should) automatically resolve extensions as the operations are created,
//! we will no longer require this post-facto resolution step.

mod types;

pub(crate) use types::update_op_types_extensions;

use std::sync::Arc;

use derive_more::{Display, Error, From};

use super::{Extension, ExtensionRegistry};
use crate::ops::custom::OpaqueOpError;
use crate::ops::{DataflowOpTrait, ExtensionOp, NamedOp, OpType};
use crate::Node;

/// The result of resolving an operation.
#[derive(Debug, Clone, Default)]
pub(crate) struct OpResolutionResult<'e> {
    /// If `op` was an opaque operation that got resolved, this contains the new
    /// [`ExtensionOp`] it should be replaced with.
    pub replacement_op: Option<OpType>,
    /// If `op` was an opaque or extension operation, this contains the extension
    /// reference that should be added to the hugr's extension registry.
    pub used_extension: Option<&'e Arc<Extension>>,
}

/// Try to resolve an [`OpType::OpaqueOp`] into an [`OpType::ExtensionOp`] by
/// looking searching for the operation in the extension registries.
///
/// # Errors
/// If the serialized opaque resolves to a definition that conflicts with what
/// was serialized. Or if the operation is not found in the registry.
pub(crate) fn resolve_op_extensions<'e>(
    node: Node,
    op: &OpType,
    extensions: &'e ExtensionRegistry,
) -> Result<OpResolutionResult<'e>, ExtensionResolutionError> {
    let OpType::OpaqueOp(opaque) = op else {
        return Ok(OpResolutionResult {
            replacement_op: None,
            used_extension: operation_extension(node, op, extensions)?,
        });
    };

    // Fail if the Extension is not in the registry, or if the Extension was
    // found but did not have the expected operation.
    let extension =
        operation_extension(node, op, extensions)?.expect("Opaque ops always have an extension");
    let Some(def) = extension.get_op(opaque.op_name()) else {
        return Err(OpaqueOpError::OpNotFoundInExtension {
            node,
            op: opaque.name().clone(),
            extension: extension.name().clone(),
            available_ops: extension
                .operations()
                .map(|(name, _)| name.clone())
                .collect(),
        }
        .into());
    };

    let ext_op = ExtensionOp::new_with_cached(def.clone(), opaque.args(), opaque, extensions)
        .map_err(|e| OpaqueOpError::SignatureError {
            node,
            name: opaque.name().clone(),
            cause: e,
        })?;

    if opaque.signature() != ext_op.signature() {
        return Err(OpaqueOpError::SignatureMismatch {
            node,
            extension: opaque.extension().clone(),
            op: def.name().clone(),
            computed: ext_op.signature().clone(),
            stored: opaque.signature().clone(),
        }
        .into());
    };

    Ok(OpResolutionResult {
        replacement_op: Some(ext_op.into()),
        used_extension: Some(extension),
    })
}

// Returns the extension in the registry required by the operation.
//
// If the operation does not require an extension, returns `None`.
fn operation_extension<'e>(
    node: Node,
    op: &OpType,
    extensions: &'e ExtensionRegistry,
) -> Result<Option<&'e Arc<Extension>>, ExtensionResolutionError> {
    let err = |ext: &str| ExtensionResolutionError::MissingOpExtension {
        node,
        op: NamedOp::name(op).to_string(),
        missing_extension: ext.to_owned(),
        available_extensions: extensions.ids().map(|id| id.to_string()).collect(),
    };
    let extension = match op {
        OpType::OpaqueOp(opaque) => opaque.extension(),
        OpType::ExtensionOp(e) => e.def().extension_id(),
        _ => return Ok(None),
    };
    match extensions.get(extension) {
        Some(e) => Ok(Some(e)),
        None => Err(err(extension)),
    }
}

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
        op: String,
        /// The missing extension
        missing_extension: String,
        /// A list of available extensions.
        available_extensions: Vec<String>,
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
        ty: String,
        /// The missing extension
        missing_extension: String,
        /// A list of available extensions.
        available_extensions: Vec<String>,
    },
}

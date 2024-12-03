//! Resolve `OpaqueOp`s into `ExtensionOp`s and return an operation's required
//! extension.
//!
//! Contains both mutable ([`update_op_extensions`]) and immutable
//! ([`collect_operation_extension`]) methods to resolve operations and collect
//! the required extensions respectively.

use std::sync::Arc;

use super::{Extension, ExtensionCollectionError, ExtensionRegistry, ExtensionResolutionError};
use crate::ops::custom::OpaqueOpError;
use crate::ops::{DataflowOpTrait, ExtensionOp, NamedOp, OpType};
use crate::Node;

/// Returns the extension in the registry required by the operation.
///
/// If the operation does not require an extension, returns `None`.
///
/// [`ExtensionOp`]s store a [`Weak`] reference to their extension, which can be
/// invalidated if the original `Arc<Extension>` is dropped. On such cases, we
/// return an error with the missing extension names.
///
/// # Attributes
///
/// - `node`: The node where the operation is located, if available. This is
///   used to provide context in the error message.
/// - `op`: The operation to collect the extensions from.
pub(crate) fn collect_op_extensions(
    node: Option<Node>,
    op: &OpType,
) -> Result<Option<Arc<Extension>>, ExtensionCollectionError> {
    let OpType::ExtensionOp(ext_op) = op else {
        return Ok(None);
    };
    let ext = ext_op.def().extension();
    match ext.upgrade() {
        Some(e) => Ok(Some(e)),
        None => Err(ExtensionCollectionError::dropped_op_extension(
            node,
            op,
            [ext_op.def().extension_id().clone()],
        )),
    }
}

/// Compute the required extension for an operation.
///
/// If the op is a [`OpType::OpaqueOp`], replace it with a resolved
/// [`OpType::ExtensionOp`] by looking searching for the operation in the
/// extension registries.
///
/// If `op` was an opaque or extension operation, the result contains the
/// extension reference that should be added to the hugr's extension registry.
///
/// # Errors
///
/// If the serialized opaque resolves to a definition that conflicts with what
/// was serialized. Or if the operation is not found in the registry.
pub(crate) fn update_op_extensions<'e>(
    node: Node,
    op: &mut OpType,
    extensions: &'e ExtensionRegistry,
) -> Result<Option<&'e Arc<Extension>>, ExtensionResolutionError> {
    let extension = operation_extension(node, op, extensions)?;

    let OpType::OpaqueOp(opaque) = op else {
        return Ok(extension);
    };

    // Fail if the Extension is not in the registry, or if the Extension was
    // found but did not have the expected operation.
    let extension = extension.expect("OpaqueOp should have an extension");
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

    let ext_op =
        ExtensionOp::new_with_cached(def.clone(), opaque.args().to_vec(), opaque, extensions)
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

    // Replace the opaque operation with the resolved extension operation.
    *op = ext_op.into();

    Ok(Some(extension))
}

/// Returns the extension in the registry required by the operation.
///
/// If the operation does not require an extension, returns `None`.
fn operation_extension<'e>(
    node: Node,
    op: &OpType,
    extensions: &'e ExtensionRegistry,
) -> Result<Option<&'e Arc<Extension>>, ExtensionResolutionError> {
    let Some(ext) = op.extension_id() else {
        return Ok(None);
    };
    match extensions.get(ext) {
        Some(e) => Ok(Some(e)),
        None => Err(ExtensionResolutionError::missing_op_extension(
            node, op, ext, extensions,
        )),
    }
}

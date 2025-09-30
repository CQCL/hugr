//! Utilities for resolving operations and types present in a HUGR, and updating
//! the list of used extensions.
//!
//! The functionalities of this module can be called from the type methods
//! [`crate::ops::OpType::used_extensions`] and
//! [`crate::types::Signature::used_extensions`].
//!
//! When listing "used extensions" we only care about _definitional_ extension
//! requirements, i.e., the operations and types that are required to define the
//! HUGR nodes and wire types. This is computed from the union of all extension
//! required across the HUGR.
//!
//! Note: These procedures are only temporary until `hugr-model` is stabilized.
//! Once that happens, hugrs will no longer be directly deserialized using serde
//! but instead will be created by the methods in `crate::import`. As these
//! (will) automatically resolve extensions as the operations are created, we
//! will no longer require this post-facto resolution step.

mod extension;
mod ops;
mod types;
mod types_mut;
mod weak_registry;

pub use weak_registry::WeakExtensionRegistry;

pub(crate) use ops::{collect_op_extension, resolve_op_extensions};
pub(crate) use types::{collect_op_types_extensions, collect_signature_exts, collect_type_exts};
pub(crate) use types_mut::resolve_op_types_extensions;
use types_mut::{
    resolve_custom_type_exts, resolve_term_exts, resolve_type_exts, resolve_value_exts,
};

use derive_more::{Display, Error, From};

use super::{Extension, ExtensionId, ExtensionRegistry, ExtensionSet};
use crate::Node;
use crate::core::HugrNode;
use crate::ops::constant::ValueName;
use crate::ops::custom::OpaqueOpError;
use crate::ops::{NamedOp, OpName, OpType, Value};
use crate::types::{CustomType, FuncTypeBase, MaybeRV, TypeArg, TypeBase, TypeName};

/// Update all weak Extension pointers inside a type.
pub fn resolve_type_extensions<RV: MaybeRV>(
    typ: &mut TypeBase<RV>,
    extensions: &WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    let mut used_extensions = WeakExtensionRegistry::default();
    resolve_type_exts(None, typ, extensions, &mut used_extensions)
}

/// Update all weak Extension pointers in a custom type.
pub fn resolve_custom_type_extensions(
    typ: &mut CustomType,
    extensions: &WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    let mut used_extensions = WeakExtensionRegistry::default();
    resolve_custom_type_exts(None, typ, extensions, &mut used_extensions)
}

/// Update all weak Extension pointers inside a type argument.
pub fn resolve_typearg_extensions(
    arg: &mut TypeArg,
    extensions: &WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    let mut used_extensions = WeakExtensionRegistry::default();
    resolve_term_exts(None, arg, extensions, &mut used_extensions)
}

/// Update all weak Extension pointers inside a constant value.
pub fn resolve_value_extensions(
    value: &mut Value,
    extensions: &WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    let mut used_extensions = WeakExtensionRegistry::default();
    resolve_value_exts(None, value, extensions, &mut used_extensions)
}

/// Errors that can occur during extension resolution.
#[derive(Debug, Display, Clone, Error, From, PartialEq)]
#[non_exhaustive]
pub enum ExtensionResolutionError<N: HugrNode = Node> {
    /// Could not resolve an opaque operation to an extension operation.
    #[display("Error resolving opaque operation: {_0}")]
    #[from]
    OpaqueOpError(OpaqueOpError<N>),
    /// An operation requires an extension that is not in the given registry.
    #[display(
        "{op}{} requires extension {missing_extension}, but it could not be found in the extension list used during resolution. The available extensions are: {}",
        node.map(|n| format!(" in {n}")).unwrap_or_default(),
        available_extensions.join(", ")
    )]
    MissingOpExtension {
        /// The node that requires the extension.
        node: Option<N>,
        /// The operation that requires the extension.
        op: OpName,
        /// The missing extension
        missing_extension: ExtensionId,
        /// A list of available extensions.
        available_extensions: Vec<ExtensionId>,
    },
    /// A type references an extension that is not in the given registry.
    #[display(
        "Type {ty}{} requires extension {missing_extension}, but it could not be found in the extension list used during resolution. The available extensions are: {}",
        node.map(|n| format!(" in {n}")).unwrap_or_default(),
        available_extensions.join(", ")
    )]
    MissingTypeExtension {
        /// The node that requires the extension.
        node: Option<N>,
        /// The type that requires the extension.
        ty: TypeName,
        /// The missing extension
        missing_extension: ExtensionId,
        /// A list of available extensions.
        available_extensions: Vec<ExtensionId>,
    },
    /// A type definition's `extension_id` does not match the extension it is in.
    #[display(
        "Type definition {def} in extension {extension} declares it was defined in {wrong_extension} instead."
    )]
    WrongTypeDefExtension {
        /// The extension that defines the type.
        extension: ExtensionId,
        /// The type definition name.
        def: TypeName,
        /// The extension declared in the type definition's `extension_id`.
        wrong_extension: ExtensionId,
    },
    /// An operation definition's `extension_id` does not match the extension it is in.
    #[display(
        "Operation definition {def} in extension {extension} declares it was defined in {wrong_extension} instead."
    )]
    WrongOpDefExtension {
        /// The extension that defines the op.
        extension: ExtensionId,
        /// The op definition name.
        def: OpName,
        /// The extension declared in the op definition's `extension_id`.
        wrong_extension: ExtensionId,
    },
    /// The type of an `OpaqueValue` has types which do not reference their defining extensions.
    #[display(
        "The type of the opaque value '{value}' requires extensions {missing_extensions}, but does not reference their definition."
    )]
    InvalidConstTypes {
        /// The value that has invalid types.
        value: ValueName,
        /// The missing extension.
        missing_extensions: ExtensionSet,
    },
}

impl<N: HugrNode> ExtensionResolutionError<N> {
    /// Create a new error for missing operation extensions.
    pub fn missing_op_extension(
        node: Option<N>,
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
        node: Option<N>,
        ty: &TypeName,
        missing_extension: &ExtensionId,
        extensions: &WeakExtensionRegistry,
    ) -> Self {
        Self::MissingTypeExtension {
            node,
            ty: ty.clone(),
            missing_extension: missing_extension.clone(),
            available_extensions: extensions.ids().cloned().collect(),
        }
    }
}

/// Errors that can occur when collecting extension requirements.
#[derive(Debug, Display, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum ExtensionCollectionError<N: HugrNode = Node> {
    /// An operation requires an extension that is not in the given registry.
    #[display(
        "{op}{} contains custom types for which have lost the reference to their defining extensions. Dropped extensions: {}",
        if let Some(node) = node { format!(" ({node})") } else { String::new() },
        missing_extensions.join(", ")
    )]
    DroppedOpExtensions {
        /// The node that is missing extensions.
        node: Option<N>,
        /// The operation that is missing extensions.
        op: OpName,
        /// The missing extensions.
        missing_extensions: Vec<ExtensionId>,
    },
    /// A signature requires an extension that is not in the given registry.
    #[display(
        "Signature {signature} contains custom types for which have lost the reference to their defining extensions. Dropped extensions: {}",
        missing_extensions.join(", ")
    )]
    DroppedSignatureExtensions {
        /// The signature that is missing extensions.
        signature: String,
        /// The missing extensions.
        missing_extensions: Vec<ExtensionId>,
    },
    /// A signature requires an extension that is not in the given registry.
    #[display(
        "Type {typ} contains custom types which have lost the reference to their defining extensions. Dropped extensions: {}",
        missing_extensions.join(", ")
    )]
    DroppedTypeExtensions {
        /// The type that is missing extensions.
        typ: String,
        /// The missing extensions.
        missing_extensions: Vec<ExtensionId>,
    },
}

impl<N: HugrNode> ExtensionCollectionError<N> {
    /// Create a new error when operation extensions have been dropped.
    pub fn dropped_op_extension(
        node: Option<N>,
        op: &OpType,
        missing_extension: impl IntoIterator<Item = ExtensionId>,
    ) -> Self {
        Self::DroppedOpExtensions {
            node,
            op: NamedOp::name(op),
            missing_extensions: missing_extension.into_iter().collect(),
        }
    }

    /// Create a new error when signature extensions have been dropped.
    pub fn dropped_signature<RV: MaybeRV>(
        signature: &FuncTypeBase<RV>,
        missing_extension: impl IntoIterator<Item = ExtensionId>,
    ) -> Self {
        Self::DroppedSignatureExtensions {
            signature: format!("{signature}"),
            missing_extensions: missing_extension.into_iter().collect(),
        }
    }

    /// Create a new error when signature extensions have been dropped.
    pub fn dropped_type<RV: MaybeRV>(
        typ: &TypeBase<RV>,
        missing_extension: impl IntoIterator<Item = ExtensionId>,
    ) -> Self {
        Self::DroppedTypeExtensions {
            typ: format!("{typ}"),
            missing_extensions: missing_extension.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod test;

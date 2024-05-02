//! Extensible operations.

use std::sync::Arc;
use thiserror::Error;

use crate::extension::{ConstFoldResult, ExtensionId, ExtensionRegistry, OpDef, SignatureError};
use crate::hugr::hugrmut::sealed::HugrMutInternals;
use crate::hugr::{HugrView, NodeType};
use crate::types::EdgeKind;
use crate::types::{type_param::TypeArg, FunctionType};
use crate::{ops, Hugr, IncomingPort, Node};

use super::dataflow::DataflowOpTrait;
use super::tag::OpTag;
use super::{NamedOp, OpName, OpNameRef, OpTrait, OpType};

/// A user-defined operation defined in an extension.
///
/// Any custom operation can be encoded as a serializable [`OpaqueOp`]. If the
/// operation's extension is loaded in the current context, the operation can be
/// resolved into an [`ExtensionOp`] containing a reference to its definition.
///
///   [`OpaqueOp`]: crate::ops::custom::OpaqueOp
///   [`ExtensionOp`]: crate::ops::custom::ExtensionOp
#[derive(Clone, Debug, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "OpaqueOp", from = "OpaqueOp")]
pub enum CustomOp {
    /// When we've found (loaded) the [Extension] definition and identified the [OpDef]
    ///
    /// [Extension]: crate::Extension
    Extension(Box<ExtensionOp>),
    /// When we either haven't tried to identify the [Extension] or failed to find it.
    ///
    /// [Extension]: crate::Extension
    Opaque(Box<OpaqueOp>),
}

impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Extension(l0), Self::Extension(r0)) => l0 == r0,
            (Self::Opaque(l0), Self::Opaque(r0)) => l0 == r0,
            (Self::Extension(l0), Self::Opaque(r0)) => &l0.make_opaque() == r0.as_ref(),
            (Self::Opaque(l0), Self::Extension(r0)) => l0.as_ref() == &r0.make_opaque(),
        }
    }
}

impl CustomOp {
    /// Create a new CustomOp from an [ExtensionOp].
    pub fn new_extension(op: ExtensionOp) -> Self {
        Self::Extension(Box::new(op))
    }

    /// Create a new CustomOp from an [OpaqueOp].
    pub fn new_opaque(op: OpaqueOp) -> Self {
        Self::Opaque(Box::new(op))
    }

    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        match self {
            Self::Opaque(op) => op.args(),
            Self::Extension(op) => op.args(),
        }
    }

    /// Returns the extension ID of this operation.
    pub fn extension(&self) -> &ExtensionId {
        match self {
            Self::Opaque(op) => op.extension(),
            Self::Extension(op) => op.def.extension(),
        }
    }

    /// If the operation is an instance of [ExtensionOp], return a reference to it.
    /// If the operation is opaque, return None.
    pub fn as_extension_op(&self) -> Option<&ExtensionOp> {
        match self {
            Self::Extension(e) => Some(e),
            Self::Opaque(_) => None,
        }
    }

    /// Downgrades this opaque operation into an [`OpaqueOp`].
    pub fn into_opaque(self) -> OpaqueOp {
        match self {
            Self::Opaque(op) => *op,
            Self::Extension(op) => (*op).into(),
        }
    }

    /// Returns `true` if this operation is an instance of [`ExtensionOp`].
    pub fn is_extension_op(&self) -> bool {
        matches!(self, Self::Extension(_))
    }

    /// Returns `true` if this operation is an instance of [`OpaqueOp`].
    pub fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque(_))
    }
}

impl NamedOp for CustomOp {
    /// The name of the operation.
    fn name(&self) -> OpName {
        let (res_id, op_name) = match self {
            Self::Opaque(op) => (&op.extension, &op.op_name),
            Self::Extension(ext) => (ext.def.extension(), ext.def.name()),
        };
        qualify_name(res_id, op_name)
    }
}

impl DataflowOpTrait for CustomOp {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        match self {
            Self::Opaque(op) => DataflowOpTrait::description(op.as_ref()),
            Self::Extension(ext_op) => DataflowOpTrait::description(ext_op.as_ref()),
        }
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        match self {
            Self::Opaque(op) => op.signature.clone(),
            Self::Extension(ext_op) => ext_op.signature(),
        }
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

impl From<OpaqueOp> for CustomOp {
    fn from(op: OpaqueOp) -> Self {
        Self::new_opaque(op)
    }
}

impl From<CustomOp> for OpaqueOp {
    fn from(value: CustomOp) -> Self {
        value.into_opaque()
    }
}

impl From<ExtensionOp> for CustomOp {
    fn from(op: ExtensionOp) -> Self {
        Self::new_extension(op)
    }
}

/// An operation defined by an [OpDef] from a loaded [Extension].
///
/// Extension ops are not serializable. They must be downgraded into an [OpaqueOp] instead.
/// See [ExtensionOp::make_opaque].
///
/// [Extension]: crate::Extension
#[derive(Clone, Debug)]
pub struct ExtensionOp {
    def: Arc<OpDef>,
    args: Vec<TypeArg>,
    signature: FunctionType, // Cache
}

impl ExtensionOp {
    /// Create a new ExtensionOp given the type arguments and specified input extensions
    pub fn new(
        def: Arc<OpDef>,
        args: impl Into<Vec<TypeArg>>,
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let args = args.into();
        let signature = def.compute_signature(&args, exts)?;
        Ok(Self {
            def,
            args,
            signature,
        })
    }

    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns a reference to the [`OpDef`] of this [`ExtensionOp`].
    pub fn def(&self) -> &OpDef {
        self.def.as_ref()
    }

    /// Attempt to evaluate this operation. See [`OpDef::constant_fold`].
    pub fn constant_fold(&self, consts: &[(IncomingPort, ops::Value)]) -> ConstFoldResult {
        self.def().constant_fold(self.args(), consts)
    }

    /// Creates a new [`OpaqueOp`] as a downgraded version of this
    /// [`ExtensionOp`].
    ///
    /// Regenerating the [`ExtensionOp`] back from the [`OpaqueOp`] requires a
    /// registry with the appropriate extension. See [`resolve_opaque_op`].
    ///
    /// For a non-cloning version of this operation, use [`OpaqueOp::from`].
    pub fn make_opaque(&self) -> OpaqueOp {
        OpaqueOp {
            extension: self.def.extension().clone(),
            op_name: self.def.name().clone(),
            description: self.def.description().into(),
            args: self.args.clone(),
            signature: self.signature.clone(),
        }
    }
}

impl From<ExtensionOp> for OpaqueOp {
    fn from(op: ExtensionOp) -> Self {
        let ExtensionOp {
            def,
            args,
            signature,
        } = op;
        OpaqueOp {
            extension: def.extension().clone(),
            op_name: def.name().clone(),
            description: def.description().into(),
            args,
            signature,
        }
    }
}

impl From<ExtensionOp> for OpType {
    fn from(value: ExtensionOp) -> Self {
        OpType::CustomOp(value.into())
    }
}

impl PartialEq for ExtensionOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::<OpDef>::ptr_eq(&self.def, &other.def) && self.args == other.args
    }
}

impl Eq for ExtensionOp {}

impl DataflowOpTrait for ExtensionOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        self.def().description()
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
    }
}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    extension: ExtensionId,
    op_name: OpName,
    description: String, // cache in advance so description() can return &str
    args: Vec<TypeArg>,
    signature: FunctionType,
}

fn qualify_name(res_id: &ExtensionId, op_name: &OpNameRef) -> OpName {
    format!("{}.{}", res_id, op_name).into()
}

impl OpaqueOp {
    /// Creates a new OpaqueOp from all the fields we'd expect to serialize.
    pub fn new(
        extension: ExtensionId,
        op_name: impl Into<OpName>,
        description: String,
        args: impl Into<Vec<TypeArg>>,
        signature: FunctionType,
    ) -> Self {
        Self {
            extension,
            op_name: op_name.into(),
            description,
            args: args.into(),
            signature,
        }
    }
}

impl OpaqueOp {
    /// Unique name of the operation.
    pub fn name(&self) -> &OpName {
        &self.op_name
    }

    /// Type arguments.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Parent extension.
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }
}

impl From<OpaqueOp> for OpType {
    fn from(value: OpaqueOp) -> Self {
        OpType::CustomOp(value.into())
    }
}

impl DataflowOpTrait for OpaqueOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        &self.description
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
    }
}

/// Resolve serialized names of operations into concrete implementation (OpDefs) where possible
pub fn resolve_extension_ops(
    h: &mut Hugr,
    extension_registry: &ExtensionRegistry,
) -> Result<(), CustomOpError> {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::CustomOp(CustomOp::Opaque(opaque)) = h.get_optype(n) {
            if let Some(resolved) = resolve_opaque_op(n, opaque, extension_registry)? {
                replacements.push((n, resolved))
            }
        }
    }
    // Only now can we perform the replacements as the 'for' loop was borrowing 'h' preventing use from using it mutably
    for (n, op) in replacements {
        let node_type = NodeType::new(op, h.get_nodetype(n).input_extensions().cloned());
        debug_assert_eq!(h.get_optype(n).tag(), OpTag::Leaf);
        debug_assert_eq!(node_type.tag(), OpTag::Leaf);
        h.replace_op(n, node_type).unwrap();
    }
    Ok(())
}

/// Try to resolve a [`OpaqueOp`] to a [`ExtensionOp`] by looking the op up in
/// the registry.
///
/// # Return
/// Some if the serialized opaque resolves to an extension-defined op and all is
/// ok; None if the serialized opaque doesn't identify an extension
///
/// # Errors
/// If the serialized opaque resolves to a definition that conflicts with what
/// was serialized
pub fn resolve_opaque_op(
    _n: Node,
    opaque: &OpaqueOp,
    extension_registry: &ExtensionRegistry,
) -> Result<Option<ExtensionOp>, CustomOpError> {
    if let Some(r) = extension_registry.get(&opaque.extension) {
        // Fail if the Extension was found but did not have the expected operation
        let Some(def) = r.get_op(&opaque.op_name) else {
            return Err(CustomOpError::OpNotFoundInExtension(
                opaque.op_name.clone(),
                r.name().clone(),
            ));
        };
        let ext_op =
            ExtensionOp::new(def.clone(), opaque.args.clone(), extension_registry).unwrap();
        if opaque.signature != ext_op.signature {
            return Err(CustomOpError::SignatureMismatch {
                extension: opaque.extension.clone(),
                op: def.name().clone(),
                computed: ext_op.signature.clone(),
                stored: opaque.signature.clone(),
            });
        };
        Ok(Some(ext_op))
    } else {
        Ok(None)
    }
}

/// Errors that arise after loading a Hugr containing opaque ops (serialized just as their names)
/// when trying to resolve the serialized names against a registry of known Extensions.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum CustomOpError {
    /// The Extension was found but did not contain the expected OpDef
    #[error("Operation {0} not found in Extension {1}")]
    OpNotFoundInExtension(OpName, ExtensionId),
    /// Extension and OpDef found, but computed signature did not match stored
    #[error("Conflicting signature: resolved {op} in extension {extension} to a concrete implementation which computed {computed} but stored signature was {stored}")]
    #[allow(missing_docs)]
    SignatureMismatch {
        extension: ExtensionId,
        op: OpName,
        stored: FunctionType,
        computed: FunctionType,
    },
}

#[cfg(test)]
mod test {

    use crate::extension::prelude::{QB_T, USIZE_T};

    use super::*;

    #[test]
    fn new_opaque_op() {
        let sig = FunctionType::new_endo(vec![QB_T]);
        let op: CustomOp = OpaqueOp::new(
            "res".try_into().unwrap(),
            "op",
            "desc".into(),
            vec![TypeArg::Type { ty: USIZE_T }],
            sig.clone(),
        )
        .into();
        assert_eq!(op.name(), "res.op");
        assert_eq!(DataflowOpTrait::description(&op), "desc");
        assert_eq!(op.args(), &[TypeArg::Type { ty: USIZE_T }]);
        assert_eq!(op.signature(), sig);
        assert!(op.is_opaque());
        assert!(!op.is_extension_op());
    }
}

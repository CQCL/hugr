//! Extensible operations.

use smol_str::SmolStr;
use std::sync::Arc;
use thiserror::Error;

use crate::extension::{ConstFoldResult, ExtensionId, ExtensionRegistry, OpDef, SignatureError};
use crate::hugr::hugrmut::sealed::HugrMutInternals;
use crate::hugr::{HugrView, NodeType};
use crate::types::{type_param::TypeArg, FunctionType};
use crate::{ops, Hugr, IncomingPort, Node};

use super::dataflow::DataflowOpTrait;
use super::tag::OpTag;
use super::{LeafOp, OpTrait, OpType};

/// An instantiation of an operation (declared by a extension) with values for the type arguments
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "OpaqueOp", from = "OpaqueOp")]
pub enum ExternalOp {
    /// When we've found (loaded) the [Extension] definition and identified the [OpDef]
    ///
    /// [Extension]: crate::Extension
    Extension(ExtensionOp),
    /// When we either haven't tried to identify the [Extension] or failed to find it.
    ///
    /// [Extension]: crate::Extension
    Opaque(OpaqueOp),
}

impl ExternalOp {
    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        match self {
            Self::Opaque(op) => op.args(),
            Self::Extension(op) => op.args(),
        }
    }

    /// Name of the ExternalOp
    pub fn name(&self) -> SmolStr {
        let (res_id, op_name) = match self {
            Self::Opaque(op) => (&op.extension, &op.op_name),
            Self::Extension(ExtensionOp { def, .. }) => (def.extension(), def.name()),
        };
        qualify_name(res_id, op_name)
    }

    /// Downgrades this ExternalOp into an OpaqueOp
    pub fn as_opaque(self) -> OpaqueOp {
        match self {
            Self::Opaque(op) => op,
            Self::Extension(op) => op.into(),
        }
    }
}

impl From<ExternalOp> for OpaqueOp {
    fn from(value: ExternalOp) -> Self {
        match value {
            ExternalOp::Opaque(op) => op,
            ExternalOp::Extension(op) => op.into(),
        }
    }
}

impl From<OpaqueOp> for ExternalOp {
    fn from(op: OpaqueOp) -> Self {
        Self::Opaque(op)
    }
}

impl From<ExternalOp> for LeafOp {
    fn from(value: ExternalOp) -> Self {
        LeafOp::CustomOp(Box::new(value))
    }
}

impl From<ExternalOp> for OpType {
    fn from(value: ExternalOp) -> Self {
        let leaf: LeafOp = value.into();
        leaf.into()
    }
}

impl DataflowOpTrait for ExternalOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        match self {
            Self::Opaque(op) => DataflowOpTrait::description(op),
            Self::Extension(ext_op) => DataflowOpTrait::description(ext_op),
        }
    }

    fn signature(&self) -> FunctionType {
        match self {
            Self::Opaque(op) => op.signature.clone(),
            Self::Extension(ext_op) => ext_op.signature(),
        }
    }
}

/// An operation defined by an [OpDef] from a loaded [Extension].
/// Note *not* Serializable: container ([ExternalOp]) is serialized as an [OpaqueOp] instead.
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
    pub fn constant_fold(&self, consts: &[(IncomingPort, ops::Const)]) -> ConstFoldResult {
        self.def().constant_fold(self.args(), consts)
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

impl From<ExtensionOp> for LeafOp {
    fn from(value: ExtensionOp) -> Self {
        LeafOp::CustomOp(Box::new(ExternalOp::Extension(value)))
    }
}

impl From<ExtensionOp> for OpType {
    fn from(value: ExtensionOp) -> Self {
        let leaf: LeafOp = value.into();
        leaf.into()
    }
}

impl PartialEq for ExtensionOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::<OpDef>::ptr_eq(&self.def, &other.def) && self.args == other.args
    }
}

impl DataflowOpTrait for ExtensionOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        self.def().description()
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
    }
}

impl Eq for ExtensionOp {}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    extension: ExtensionId,
    op_name: SmolStr,
    description: String, // cache in advance so description() can return &str
    args: Vec<TypeArg>,
    signature: FunctionType,
}

fn qualify_name(res_id: &ExtensionId, op_name: &SmolStr) -> SmolStr {
    format!("{}.{}", res_id, op_name).into()
}

impl OpaqueOp {
    /// Creates a new OpaqueOp from all the fields we'd expect to serialize.
    pub fn new(
        extension: ExtensionId,
        op_name: impl Into<SmolStr>,
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
    pub fn name(&self) -> &SmolStr {
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

impl From<OpaqueOp> for LeafOp {
    fn from(value: OpaqueOp) -> Self {
        LeafOp::CustomOp(Box::new(ExternalOp::Opaque(value)))
    }
}

impl From<OpaqueOp> for OpType {
    fn from(value: OpaqueOp) -> Self {
        let leaf: LeafOp = value.into();
        leaf.into()
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
#[allow(dead_code)]
pub fn resolve_extension_ops(
    h: &mut Hugr,
    extension_registry: &ExtensionRegistry,
) -> Result<(), CustomOpError> {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::LeafOp(LeafOp::CustomOp(op)) = h.get_optype(n) {
            if let ExternalOp::Opaque(opaque) = op.as_ref() {
                if let Some(resolved) = resolve_opaque_op(n, opaque, extension_registry)? {
                    replacements.push((n, resolved))
                }
            }
        }
    }
    // Only now can we perform the replacements as the 'for' loop was borrowing 'h' preventing use from using it mutably
    for (n, op) in replacements {
        let leaf: LeafOp = op.into();
        let node_type = NodeType::new(leaf, h.get_nodetype(n).input_extensions().cloned());
        debug_assert_eq!(h.get_optype(n).tag(), OpTag::Leaf);
        debug_assert_eq!(node_type.tag(), OpTag::Leaf);
        h.replace_op(n, node_type).unwrap();
    }
    Ok(())
}

/// Try to resolve an [`ExternalOp::Opaque`] to a [`ExternalOp::Extension`]
///
/// # Return
/// Some if the serialized opaque resolves to an extension-defined op and all is ok;
/// None if the serialized opaque doesn't identify an extension
///
/// # Errors
/// If the serialized opaque resolves to a definition that conflicts with what was serialized
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
pub enum CustomOpError {
    /// The Extension was found but did not contain the expected OpDef
    #[error("Operation {0} not found in Extension {1}")]
    OpNotFoundInExtension(SmolStr, ExtensionId),
    /// Extension and OpDef found, but computed signature did not match stored
    #[error("Conflicting signature: resolved {op} in extension {extension} to a concrete implementation which computed {computed} but stored signature was {stored}")]
    #[allow(missing_docs)]
    SignatureMismatch {
        extension: ExtensionId,
        op: SmolStr,
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
        let op = OpaqueOp::new(
            "res".try_into().unwrap(),
            "op",
            "desc".into(),
            vec![TypeArg::Type { ty: USIZE_T }],
            sig.clone(),
        );
        let op: ExternalOp = op.into();
        assert_eq!(op.name(), "res.op");
        assert_eq!(DataflowOpTrait::description(&op), "desc");
        assert_eq!(op.args(), &[TypeArg::Type { ty: USIZE_T }]);
        assert_eq!(op.signature(), sig);
    }
}

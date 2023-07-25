//! Extensible operations.

use smol_str::SmolStr;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::hugr::{HugrMut, HugrView, NodeType};
use crate::resource::{OpDef, ResourceId, ResourceSet, SignatureError};
use crate::types::{type_param::TypeArg, AbstractSignature, SignatureDescription};
use crate::{Hugr, Node, Resource};

use super::tag::OpTag;
use super::{LeafOp, OpName, OpTrait, OpType};

/// An instantiation of an operation (declared by a resource) with values for the type arguments
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "OpaqueOp", from = "OpaqueOp")]
pub enum ExternalOp {
    /// When we've found (loaded) the [Resource] definition and identified the [OpDef]
    Resource(ResourceOp),
    /// When we either haven't tried to identify the [Resource] or failed to find it.
    Opaque(OpaqueOp),
}

impl ExternalOp {
    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        match self {
            Self::Opaque(op) => op.args(),
            Self::Resource(op) => op.args(),
        }
    }
}

impl From<ExternalOp> for OpaqueOp {
    fn from(value: ExternalOp) -> Self {
        match value {
            ExternalOp::Opaque(op) => op,
            ExternalOp::Resource(op) => op.into(),
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
        LeafOp::CustomOp(value)
    }
}

impl OpName for ExternalOp {
    fn name(&self) -> SmolStr {
        let (res_id, op_name) = match self {
            Self::Opaque(op) => (&op.resource, &op.op_name),
            Self::Resource(ResourceOp { def, .. }) => (&def.resource, &def.name),
        };
        qualify_name(res_id, op_name)
    }
}

impl OpTrait for ExternalOp {
    fn description(&self) -> &str {
        match self {
            Self::Opaque(op) => op.description.as_str(),
            Self::Resource(ResourceOp { def, .. }) => def.description.as_str(),
        }
    }

    fn signature_desc(&self) -> SignatureDescription {
        match self {
            Self::Opaque(_) => SignatureDescription::default(),
            Self::Resource(ResourceOp { def, args, .. }) => def.signature_desc(args),
        }
    }

    fn tag(&self) -> OpTag {
        OpTag::Leaf
    }

    /// Note the case of an OpaqueOp without a signature should already
    /// have been detected in [resolve_extension_ops]
    fn signature(&self) -> AbstractSignature {
        match self {
            Self::Opaque(op) => op.signature.clone().unwrap(),
            Self::Resource(ResourceOp { signature, .. }) => signature.clone(),
        }
    }
}

/// An operation defined by an [OpDef] from a loaded [Resource].
// Note *not* Serializable: container (ExternalOp) is serialized as an OpaqueOp instead.
#[derive(Clone, Debug)]
pub struct ResourceOp {
    def: Arc<OpDef>,
    args: Vec<TypeArg>,
    signature: AbstractSignature, // Cache
}

impl ResourceOp {
    /// Create a new ResourceOp given the type arguments and specified input resources
    pub fn new(
        def: Arc<OpDef>,
        args: &[TypeArg],
        resources_in: &ResourceSet,
    ) -> Result<Self, SignatureError> {
        let signature = def.compute_signature(args, resources_in)?;
        Ok(Self {
            def,
            args: args.to_vec(),
            signature,
        })
    }

    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }
}

impl From<ResourceOp> for OpaqueOp {
    fn from(op: ResourceOp) -> Self {
        let ResourceOp {
            def,
            args,
            signature,
        } = op;
        let opt_sig = if def.should_serialize_signature() {
            Some(signature)
        } else {
            None
        };
        OpaqueOp {
            resource: def.resource.clone(),
            op_name: def.name.clone(),
            description: def.description.clone(),
            args,
            signature: opt_sig,
        }
    }
}

impl PartialEq for ResourceOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::<OpDef>::ptr_eq(&self.def, &other.def) && self.args == other.args
    }
}

impl Eq for ResourceOp {}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    resource: ResourceId,
    op_name: SmolStr,
    description: String, // cache in advance so description() can return &str
    args: Vec<TypeArg>,
    signature: Option<AbstractSignature>,
}

fn qualify_name(res_id: &ResourceId, op_name: &SmolStr) -> SmolStr {
    format!("{}.{}", res_id, op_name).into()
}

impl OpaqueOp {
    /// Creates a new OpaqueOp from all the fields we'd expect to serialize.
    pub fn new(
        resource: ResourceId,
        op_name: impl Into<SmolStr>,
        description: String,
        args: impl Into<Vec<TypeArg>>,
        signature: Option<AbstractSignature>,
    ) -> Self {
        Self {
            resource,
            op_name: op_name.into(),
            description,
            args: args.into(),
            signature,
        }
    }

    /// Return the argument values for this operation.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }
}

/// Resolve serialized names of operations into concrete implementation (OpDefs) where possible
#[allow(dead_code)]
pub fn resolve_extension_ops(
    h: &mut Hugr,
    resource_registry: &HashMap<SmolStr, Resource>,
) -> Result<(), CustomOpError> {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::LeafOp(LeafOp::CustomOp(op @ ExternalOp::Opaque(opaque))) = h.get_optype(n) {
            if let Some(r) = resource_registry.get(&opaque.resource) {
                // Fail if the Resource was found but did not have the expected operation
                let Some(def) = r.operations().get(&opaque.op_name) else {
                    return Err(CustomOpError::OpNotFoundInResource(opaque.op_name.to_string(), r.name().to_string()));
                };
                // TODO input resources. From type checker, or just drop by storing only delta in Signature.
                let op = ExternalOp::Resource(
                    ResourceOp::new(def.clone(), &opaque.args, &ResourceSet::default()).unwrap(),
                );
                if let Some(sig) = &opaque.signature {
                    if sig != &op.signature() {
                        return Err(CustomOpError::SignatureMismatch(
                            def.name.to_string(),
                            op.signature(),
                            sig.clone(),
                        ));
                    };
                };
                replacements.push((n, op));
            } else if opaque.signature.is_none() {
                return Err(CustomOpError::NoStoredSignature(op.name(), n));
            }
        }
    }
    // Only now can we perform the replacements as the 'for' loop was borrowing 'h' preventing use from using it mutably
    for (n, op) in replacements {
        let node_type = NodeType::pure(Into::<LeafOp>::into(op));
        h.replace_op(n, node_type);
    }
    Ok(())
}

/// Errors that arise after loading a Hugr containing opaque ops (serialized just as their names)
/// when trying to resolve the serialized names against a registry of known Resources.
#[derive(Clone, Debug, Error)]
pub enum CustomOpError {
    /// Resource not found, and no signature
    #[error("Unable to resolve operation {0} for node {1:?} with no saved signature")]
    NoStoredSignature(SmolStr, Node),
    /// The Resource was found but did not contain the expected OpDef
    #[error("Operation {0} not found in Resource {1}")]
    OpNotFoundInResource(String, String),
    /// Resource and OpDef found, but computed signature did not match stored
    #[error("Resolved {0} to a concrete implementation which computed a conflicting signature: {1:?} vs stored {2:?}")]
    SignatureMismatch(String, AbstractSignature, AbstractSignature),
}

#[cfg(test)]
mod test {
    use crate::types::ClassicType;

    use super::*;

    #[test]
    fn new_opaque_op() {
        let op = OpaqueOp::new(
            "res".into(),
            "op",
            "desc".into(),
            vec![TypeArg::ClassicType(ClassicType::F64)],
            None,
        );
        let op: ExternalOp = op.into();
        assert_eq!(op.name(), "res.op");
        assert_eq!(op.description(), "desc");
        assert_eq!(op.args(), &[TypeArg::ClassicType(ClassicType::F64)]);
    }
}

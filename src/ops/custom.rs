//! Extensible operations.

use smol_str::SmolStr;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hugr::{HugrMut, HugrView};
use crate::resource::{OpDef, ResourceId, ResourceSet};
use crate::types::{type_param::TypeArg, Signature, SignatureDescription};
use crate::{Hugr, Resource};

use super::tag::OpTag;
use super::{LeafOp, OpName, OpTrait, OpType};

/// An instantiation of an [`OpDef`] with values for the type arguments
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(into = "OpaqueOp", from = "OpaqueOp")]
pub enum ExternalOp {
    Resource { def: Arc<OpDef>, args: Vec<TypeArg> },
    Opaque(OpaqueOp),
}

impl From<ExternalOp> for OpaqueOp {
    fn from(value: ExternalOp) -> Self {
        match value {
            ExternalOp::Opaque(op) => op,
            ExternalOp::Resource { def, args } => {
                // There's no way to report a panic here, but serde requires Into not TryInto. Eeeek!
                let signature = if def.should_serialize_signature() {
                    // TODO More issues with the input resource set here!
                    Some(def.signature(&args, &ResourceSet::default()).unwrap())
                } else {
                    None
                };
                OpaqueOp {
                    resource: def.resource.clone(),
                    op_name: def.name.clone(),
                    description: def.description.clone(),
                    args,
                    signature,
                }
            }
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

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    resource: ResourceId,
    op_name: SmolStr,
    #[serde(skip)]
    description: String, // cache in advance as description() returns &str
    args: Vec<TypeArg>,
    signature: Option<Signature>,
}

fn qualify_name(res_id: &ResourceId, op_name: &SmolStr) -> SmolStr {
    format!("{}/{}", res_id, op_name).into()
}

impl OpaqueOp {
    pub fn new(
        resource: ResourceId,
        op_name: impl Into<SmolStr>,
        args: impl Into<Vec<TypeArg>>,
        signature: Option<Signature>,
    ) -> Self {
        let op_name: SmolStr = op_name.into();
        let description = qualify_name(&resource, &op_name).into();
        Self {
            resource,
            op_name,
            description,
            args: args.into(),
            signature,
        }
    }
}

impl OpName for ExternalOp {
    fn name(&self) -> SmolStr {
        let (res_id, op_name) = match self {
            Self::Opaque(op) => (&op.resource, &op.op_name),
            Self::Resource { def, .. } => (&def.resource, &def.name),
        };
        qualify_name(res_id, op_name)
    }
}

impl OpTrait for ExternalOp {
    fn description(&self) -> &str {
        match self {
            Self::Opaque(op) => op.description.as_str(),
            Self::Resource { def, .. } => def.description.as_str(),
        }
    }

    fn signature_desc(&self) -> SignatureDescription {
        match self {
            Self::Opaque(_) => SignatureDescription::default(),
            Self::Resource { def, args } => def.signature_desc(args),
        }
    }

    fn tag(&self) -> OpTag {
        OpTag::DataflowChild
    }

    /// Note that there is no way to indicate failure here! We could fail in [resolve_extension_ops]?
    fn signature(&self) -> Signature {
        match self {
            Self::Opaque(op) => op.signature.clone().unwrap(),
            Self::Resource { def, args } => {
                // TODO the Resources in inputs and outputs appear in Signature here, so we need to get them from somewhere?
                // Do we store the input ResourceSet as another field in the CustomOp? If so, we should do the same for *all* ops?
                def.signature(args, &ResourceSet::default()).unwrap()
            }
        }
    }
}

impl PartialEq for ExternalOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Opaque(op1), Self::Opaque(op2)) => op1 == op2,
            (Self::Resource { def: d1, args: a1 }, Self::Resource { def: d2, args: a2 }) => {
                Arc::<OpDef>::ptr_eq(d1, d2) && a1 == a2
            }
            (_, _) => false,
        }
    }
}

impl Eq for ExternalOp {}

/// Resolve serialized names of operations into concrete implementation (OpDefs) where possible
#[allow(dead_code)]
pub fn resolve_extension_ops(h: &mut Hugr, rsrcs: &HashMap<SmolStr, Resource>) {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::LeafOp(LeafOp::CustomOp(ExternalOp::Opaque(opaque))) = h.get_optype(n) {
            if let Some(r) = rsrcs.get(&opaque.resource) {
                // Fail if the Resource was found but did not have the expected operation
                let Some(def) = r.operations().get(&opaque.op_name) else {
                    panic!("Conflicting declaration of Resource {}, did not find OpDef for {}", r.name(), opaque.op_name);
                };
                // Check Signature is correct if stored
                if let Some(sig) = &opaque.signature {
                    let computed_sig = def.signature(&opaque.args, &sig.input_resources).unwrap();
                    if sig != &computed_sig {
                        panic!("Resolved {} to a concrete implementation which computed a conflicting signature: {} vs stored {}", opaque.op_name, computed_sig, sig);
                    };
                    replacements.push((
                        n,
                        ExternalOp::Resource {
                            def: def.clone(),
                            args: opaque.args.clone(),
                        },
                    ));
                }
            } else if opaque.signature.is_none() {
                panic!(
                    "Loaded node with operation {} of unknown resource {} and no stored Signature",
                    opaque.op_name, opaque.resource
                );
            }
        }
    }
    // Only now can we perform the replacements as the 'for' loop was borrowing 'h' preventing use from using it mutably
    for (n, op) in replacements {
        h.replace_op(n, Into::<LeafOp>::into(op));
    }
}

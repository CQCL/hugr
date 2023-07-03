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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "OpaqueOp", from = "OpaqueOp")]
pub enum ExternalOp {
    Resource(ResourceOp), // When we've found the Resource definition
    Opaque(OpaqueOp),     // When we haven't
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
        OpTag::DataflowChild
    }

    /// Note that there is no way to indicate failure here! We could fail in [resolve_extension_ops]?
    fn signature(&self) -> Signature {
        match self {
            Self::Opaque(op) => op.signature.clone().unwrap(),
            Self::Resource(ResourceOp { signature, .. }) => signature.clone(),
        }
    }
}

// Note *not* Serializable: container (ExternalOp) should have serialized as an OpaqueOp instead.
#[derive(Clone, Debug)]
pub struct ResourceOp {
    def: Arc<OpDef>,
    args: Vec<TypeArg>,
    signature: Signature, // Cache
}

impl Into<OpaqueOp> for ResourceOp {
    fn into(self) -> OpaqueOp {
        let ResourceOp {
            def,
            args,
            signature,
        } = self;
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
                // Check Signature is correct if stored. TODO input_resources
                let computed_sig = def
                    .signature(&opaque.args, &ResourceSet::default())
                    .unwrap();
                if let Some(sig) = &opaque.signature {
                    if sig != &computed_sig {
                        panic!("Resolved {} to a concrete implementation which computed a conflicting signature: {} vs stored {}", opaque.op_name, computed_sig, sig);
                    };
                };
                replacements.push((
                    n,
                    ExternalOp::Resource(ResourceOp {
                        def: def.clone(),
                        args: opaque.args.clone(),
                        signature: computed_sig,
                    }),
                ));
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

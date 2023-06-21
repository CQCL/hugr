//! Extensible operations.

use smol_str::SmolStr;
use std::collections::HashMap;
use std::rc::Rc;

use crate::hugr::{HugrMut, HugrView};
use crate::resource::{OpDef, ResourceId, ResourceSet};
use crate::types::{type_arg::TypeArgValue, Signature, SignatureDescription};
use crate::{Hugr, Resource};

use super::tag::OpTag;
use super::{LeafOp, OpName, OpTrait, OpType};

/// An instantiation of an [`OpDef`] with values for the type arguments
#[derive(Clone, Debug)]
pub struct ExternalOp {
    def: Rc<OpDef>,
    args: Vec<TypeArgValue>,
}

impl Into<OpaqueOp> for &ExternalOp {
    fn into(self) -> OpaqueOp {
        // There's no way to report a panic here, but serde requires Into not TryInto. Eeeek!
        // Also, we don't necessarily need to store the signature for all extensions/stages...?
        let sig = self
            .def
            .signature(&self.args, &ResourceSet::default())
            .unwrap();
        OpaqueOp {
            resource: self.def.resource.clone(),
            op_name: self.def.name.clone(),
            args: self.args.clone(),
            signature: Some(sig),
        }
    }
}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    resource: ResourceId,
    op_name: SmolStr,
    args: Vec<TypeArgValue>,
    signature: Option<Signature>,
}

impl OpName for ExternalOp {
    fn name(&self) -> SmolStr {
        // TODO should we fully qualify?
        self.def.name.clone()
    }
}

impl OpName for OpaqueOp {
    fn name(&self) -> SmolStr {
        return self.op_name.clone();
    }
}

impl OpTrait for ExternalOp {
    fn description(&self) -> &str {
        self.def.description.as_str()
    }

    fn signature_desc(&self) -> SignatureDescription {
        self.def.signature_desc(&self.args)
    }

    fn tag(&self) -> OpTag {
        OpTag::DataflowChild
    }

    fn signature(&self) -> Signature {
        // TODO the Resources in inputs and outputs appear in Signature here, so we need to get them from somewhere?
        // Do we store the input ResourceSet as another field in the CustomOp? If so, we should do the same for *all* ops?
        // Also there is no way to indicate failure here...
        self.def
            .signature(&self.args, &ResourceSet::default())
            .unwrap()
    }
}

impl OpTrait for OpaqueOp {
    fn description(&self) -> &str {
        "<opaque op from unknown Resource>"
    }

    fn signature_desc(&self) -> SignatureDescription {
        SignatureDescription::default()
    }

    fn tag(&self) -> OpTag {
        OpTag::DataflowChild
    }

    fn signature(&self) -> Signature {
        self.signature.clone().unwrap()
    }
}

impl PartialEq for ExternalOp {
    fn eq(&self, other: &Self) -> bool {
        Rc::<OpDef>::ptr_eq(&self.def, &other.def) && self.args == other.args
    }
}

/// Resolve serialized names of operations into concrete implementation (OpDefs) where possible
#[allow(dead_code)]
pub fn resolve_extension_ops(h: &mut Hugr, rsrcs: &HashMap<SmolStr, Resource>) -> () {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::LeafOp(LeafOp::UnknownOp { opaque }) = h.get_optype(n) {
            if let Some(r) = rsrcs.get(&opaque.resource) {
                // Fail if the Resource was found but did not have the expected operation
                let Some(def) = r.operations().get(&opaque.op_name) else {
                    panic!("Conflicting declaration of Resource {}, did not find OpDef for {}", r.name(), opaque.op_name);
                };
                // Check Signature is correct if stored
                if let Some(sig) = &opaque.signature {
                    let computed_sig = def.signature(&opaque.args, &sig.input_resources).unwrap();
                    if sig != &computed_sig {
                        panic!("Resolved {} to a concrete implementation which computed a conflicting signature: {} vs stored {}", opaque.name(), computed_sig, sig);
                    };
                    replacements.push((
                        n,
                        ExternalOp {
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
        h.replace_op(n, LeafOp::CustomOp { ext: op });
    }
}

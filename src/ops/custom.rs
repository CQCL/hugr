//! Extensible operations.

use smol_str::SmolStr;
use std::rc::Rc;

use crate::resource::{OpDef, ResourceSet};
use crate::types::{type_arg::TypeArgValue, Signature, SignatureDescription};

use super::tag::OpTag;
use super::{OpName, OpTrait};

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
            name: self.def.name.clone(),
            args: self.args.clone(),
            signature: Some(sig),
        }
    }
}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    name: SmolStr, // This should be fully-qualified, somehow
    args: Vec<TypeArgValue>,
    signature: Option<Signature>,
}

impl OpName for ExternalOp {
    fn name(&self) -> SmolStr {
        self.def.name.clone()
    }
}

impl OpName for OpaqueOp {
    fn name(&self) -> SmolStr {
        return self.name.clone();
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

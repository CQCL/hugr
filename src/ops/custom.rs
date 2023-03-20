//! Extensible operations.
//!
//! TODO: The `OpDef` variant could be defined elsewhere, and just keep the `Box<dyn CustomOp>` in `LeafOp`.

use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::types::Signature;
use downcast_rs::{impl_downcast, Downcast};
use std::any::Any;

use super::Op;

pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Op + Any + Downcast {
    /// Try to convert the custom op to a graph definition.
    ///
    /// TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn try_to_hugr(&self) -> Option<Hugr> {
        None
    }

    /// Check if two custom ops are equal, by downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomOp) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);

/// Dynamically loaded operation definition.
///
/// TODO: How do we encode the operation? Could this be managed by a plugin
/// module, and just have the `Box<dyn CustomOp>` in LeafOp?
#[derive(Clone, Debug)]
pub struct OpDef {
    name: String,
    signature: Signature,
}

impl Op for OpDef {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> Signature {
        self.signature.clone()
    }
}

impl CustomOp for OpDef {
    fn eq(&self, other: &dyn CustomOp) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.name == other.name && self.signature == other.signature
        } else {
            false
        }
    }

    fn try_to_hugr(&self) -> Option<Hugr> {
        todo!()
    }
}

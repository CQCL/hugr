//! Extensible operations.

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;
use std::any::Any;
use std::ops::Deref;

use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::resource::ResourceSet;
use crate::types::{Signature, SignatureDescription};

/// A wrapped [`CustomOp`] with fast equality checks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    /// Operation name, cached for fast equality checks.
    id: SmolStr,

    /// The custom operation.
    op: Box<dyn CustomOp>,
}

impl OpaqueOp {
    /// The name of the operation, cached for fast equality checks.
    pub fn name(&self) -> SmolStr {
        self.id.clone()
    }
}

impl PartialEq for OpaqueOp {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for OpaqueOp {}

impl Deref for OpaqueOp {
    type Target = dyn CustomOp;

    fn deref(&self) -> &Self::Target {
        self.op.as_ref()
    }
}

impl<T: CustomOp> From<T> for OpaqueOp {
    fn from(op: T) -> Self {
        Self {
            id: op.name(),
            op: Box::new(op),
        }
    }
}

/// Custom definition for an operation.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
#[typetag::serde]
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Any + Downcast {
    /// Try to convert the custom op to a graph definition.
    ///
    /// TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn try_into_hugr(&self, resources: &ResourceSet) -> Option<Hugr> {
        let _ = resources;
        None
    }

    /// List the resources required to execute this operation.
    fn resources(&self) -> &ResourceSet;

    /// The name of the operation.
    fn name(&self) -> SmolStr;

    /// Optional description of the operation.
    fn description(&self) -> &str {
        ""
    }

    /// The signature of the operation.
    fn signature(&self) -> Signature;

    /// Optional descriptions of the ports in the signature.
    fn signature_desc(&self) -> SignatureDescription {
        Default::default()
    }
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);

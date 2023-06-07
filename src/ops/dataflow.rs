//! Dataflow operations.

use super::{impl_op_name, tag::OpTag, OpName, OpTrait};
use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, TypeRow};
/// An input node.
/// The outputs of this node are the inputs to the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Input {
    pub types: TypeRow,
}

impl_op_name!(Input);

trait DataflowOpTrait {
    fn description(&self) -> &str;
    fn tag(&self) -> OpTag;
    fn signature(&self) -> Signature;
    /// The edge kind for the inputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other input edges. Otherwise, all other input
    /// edges will be of that kind.
    fn other_inputs(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
    /// The edge kind for the outputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other output edges. Otherwise, all other
    /// output edges will be of that kind.
    fn other_outputs(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}
impl DataflowOpTrait for Input {
    fn description(&self) -> &str {
        "The input node for this dataflow subgraph"
    }

    fn tag(&self) -> super::tag::OpTag {
        OpTag::Input
    }

    fn other_inputs(&self) -> Option<EdgeKind> {
        None
    }

    fn signature(&self) -> Signature {
        Signature::new_df(TypeRow::new(), self.types.clone())
    }
}

/// An output node. The inputs are the outputs of the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Output {
    pub types: TypeRow,
}

impl_op_name!(Output);

impl DataflowOpTrait for Output {
    fn description(&self) -> &str {
        "The output node for this dataflow subgraph"
    }

    fn tag(&self) -> super::tag::OpTag {
        OpTag::Output
    }

    fn signature(&self) -> Signature {
        Signature::new_df(self.types.clone(), TypeRow::new())
    }

    fn other_outputs(&self) -> Option<EdgeKind> {
        None
    }
}

impl<T: DataflowOpTrait> OpTrait for T {
    fn description(&self) -> &str {
        DataflowOpTrait::description(self)
    }

    fn tag(&self) -> OpTag {
        DataflowOpTrait::tag(self)
    }
    fn signature(&self) -> Signature {
        DataflowOpTrait::signature(self)
    }
}

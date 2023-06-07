//! Dataflow operations.

use super::{impl_op_name, tag::OpTag, OpName, OpTrait};
use smol_str::SmolStr;

use crate::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};
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

/// Call a function directly.
///
/// The first ports correspond to the signature of the function being
/// called. Immediately following those ports, the first input port is
/// connected to the def/declare block with a `ConstE<Graph>` edge.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Call {
    pub signature: Signature,
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    fn description(&self) -> &str {
        "Call a function directly"
    }

    fn tag(&self) -> OpTag {
        OpTag::FnCall
    }

    fn signature(&self) -> Signature {
        Signature {
            const_input: vec![ClassicType::graph_from_sig(self.signature.clone()).into()].into(),
            ..self.signature.clone()
        }
    }
}

/// Call a function indirectly. Like call, but the first input is a standard dataflow graph type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CallIndirect {
    pub signature: Signature,
}
impl_op_name!(CallIndirect);

impl DataflowOpTrait for CallIndirect {
    fn description(&self) -> &str {
        "Call a function indirectly"
    }

    fn tag(&self) -> OpTag {
        OpTag::FnCall
    }

    fn signature(&self) -> Signature {
        let mut s = self.signature.clone();
        s.input.to_mut().insert(
            0,
            ClassicType::graph_from_sig(self.signature.clone()).into(),
        );
        s
    }
}

/// Load a static constant in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadConstant {
    pub datatype: ClassicType,
}
impl_op_name!(LoadConstant);
impl DataflowOpTrait for LoadConstant {
    fn description(&self) -> &str {
        "Load a static constant in to the local dataflow graph"
    }

    fn tag(&self) -> OpTag {
        OpTag::LoadConst
    }

    fn signature(&self) -> Signature {
        Signature::new(
            TypeRow::new(),
            vec![SimpleType::Classic(self.datatype.clone())],
            vec![SimpleType::Classic(self.datatype.clone())],
        )
    }
}

/// A simply nested dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DFG {
    pub signature: Signature,
}

impl_op_name!(DFG);
impl DataflowOpTrait for DFG {
    fn description(&self) -> &str {
        "A simply nested dataflow graph"
    }

    fn tag(&self) -> OpTag {
        OpTag::Dfg
    }

    fn signature(&self) -> Signature {
        self.signature.clone()
    }
}

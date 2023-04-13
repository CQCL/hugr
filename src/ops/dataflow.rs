use smol_str::SmolStr;

use super::{controlflow::ControlFlowOp, LeafOp, Op};
use crate::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DataflowOp {
    /// An input node.
    /// The outputs of this node are the inputs to the function.
    Input { types: TypeRow },
    /// An output node. The inputs are the outputs of the function.
    Output { types: TypeRow },
    /// Call a function directly.
    ///
    /// The first port is connected to the def/declare of the function being
    /// called directly, with a `ConstE<Graph>` edge. The signature of the
    /// remaining ports matches the function being called.
    Call { signature: Signature },
    /// Call a function indirectly. Like call, but the first input is a standard dataflow graph type
    CallIndirect { signature: Signature },
    /// Load a static constant in to the local dataflow graph
    LoadConstant { datatype: ClassicType },
    /// Simple operation that has only value inputs+outputs and (potentially) StateOrder edges.
    Leaf { op: LeafOp },
    /// δ (delta): a simply nested dataflow graph
    Nested { signature: Signature },
    /// Operation related to control flow
    ControlFlow { op: ControlFlowOp },
}

impl DataflowOp {
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        if let DataflowOp::Input { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }

    pub fn other_outputs(&self) -> Option<EdgeKind> {
        if let DataflowOp::Output { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }
}

impl Default for DataflowOp {
    fn default() -> Self {
        Self::Leaf {
            op: LeafOp::default(),
        }
    }
}

impl Op for DataflowOp {
    fn name(&self) -> SmolStr {
        match self {
            DataflowOp::Input { .. } => "input",
            DataflowOp::Output { .. } => "output",
            DataflowOp::Call { .. } => "call",
            DataflowOp::CallIndirect { .. } => "call_indirect",
            DataflowOp::LoadConstant { .. } => "load",
            DataflowOp::Leaf { op } => return op.name(),
            DataflowOp::Nested { .. } => "nested",
            DataflowOp::ControlFlow { op } => return op.name(),
        }
        .into()
    }

    fn signature(&self) -> Signature {
        match self {
            DataflowOp::Input { types } => Signature::new_df(TypeRow::new(), types.clone()),
            DataflowOp::Output { types } => Signature::new_df(types.clone(), TypeRow::new()),
            DataflowOp::Call { signature } => Signature {
                const_input: ClassicType::graph_from_sig(signature.clone()).into(),
                ..signature.clone()
            },
            DataflowOp::CallIndirect { signature } => {
                let mut s = signature.clone();
                s.input
                    .to_mut()
                    .insert(0, ClassicType::graph_from_sig(signature.clone()).into());
                s
            }
            DataflowOp::LoadConstant { datatype } => Signature {
                const_input: Some(datatype.clone()),
                ..Signature::new_df(TypeRow::new(), vec![SimpleType::Classic(datatype.clone())])
            },
            DataflowOp::Leaf { op } => op.signature(),
            DataflowOp::Nested { signature } => signature.clone(),
            DataflowOp::ControlFlow { op } => op.signature(),
        }
    }
}

impl From<LeafOp> for DataflowOp {
    fn from(op: LeafOp) -> Self {
        Self::Leaf { op }
    }
}

impl From<ControlFlowOp> for DataflowOp {
    fn from(op: ControlFlowOp) -> Self {
        Self::ControlFlow { op }
    }
}

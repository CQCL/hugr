use smol_str::SmolStr;

use super::{controlflow::ControlFlowOp, LeafOp, Op};
use crate::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FunctionOp {
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

impl FunctionOp {
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        if let FunctionOp::Input { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }

    pub fn other_outputs(&self) -> Option<EdgeKind> {
        if let FunctionOp::Output { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }
}

impl Default for FunctionOp {
    fn default() -> Self {
        Self::Leaf {
            op: LeafOp::default(),
        }
    }
}

impl Op for FunctionOp {
    fn name(&self) -> SmolStr {
        match self {
            FunctionOp::Input { .. } => "input",
            FunctionOp::Output { .. } => "output",
            FunctionOp::Call { .. } => "call",
            FunctionOp::CallIndirect { .. } => "call_indirect",
            FunctionOp::LoadConstant { .. } => "load",
            FunctionOp::Leaf { op } => return op.name(),
            FunctionOp::Nested { .. } => "nested",
            FunctionOp::ControlFlow { op } => return op.name(),
        }
        .into()
    }

    fn signature(&self) -> Signature {
        match self {
            FunctionOp::Input { types } => Signature::new_df(TypeRow::new(), types.clone()),
            FunctionOp::Output { types } => Signature::new_df(types.clone(), TypeRow::new()),
            FunctionOp::Call { signature } => Signature {
                const_input: ClassicType::graph_from_sig(signature.clone()).into(),
                ..signature.clone()
            },
            FunctionOp::CallIndirect { signature } => {
                let mut s = signature.clone();
                s.input
                    .to_mut()
                    .insert(0, ClassicType::graph_from_sig(signature.clone()).into());
                s
            }
            FunctionOp::LoadConstant { datatype } => Signature {
                const_input: Some(datatype.clone()),
                ..Signature::new_df(TypeRow::new(), vec![SimpleType::Classic(datatype.clone())])
            },
            FunctionOp::Leaf { op } => op.signature(),
            FunctionOp::Nested { signature } => signature.clone(),
            FunctionOp::ControlFlow { op } => op.signature(),
        }
    }
}

impl From<LeafOp> for FunctionOp {
    fn from(op: LeafOp) -> Self {
        Self::Leaf { op }
    }
}

impl From<ControlFlowOp> for FunctionOp {
    fn from(op: ControlFlowOp) -> Self {
        Self::ControlFlow { op }
    }
}

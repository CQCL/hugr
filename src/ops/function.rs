use smol_str::SmolStr;

use super::{LeafOp, Op};
use crate::types::{ClassicType, Signature, SimpleType, TypeRow};

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
    LoadConstant { datatype: SimpleType },
    /// Simple operation that has only value inputs+outputs and (potentially) StateOrder edges.
    Leaf { op: LeafOp },
    /// δ (delta): a simply nested dataflow graph
    Nested { signature: Signature },
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
        }
        .into()
    }

    fn signature(&self) -> Signature {
        match self {
            FunctionOp::Input { types } => Signature::new_df(TypeRow::new(), types.clone()),
            FunctionOp::Output { types } => Signature::new_df(types.clone(), TypeRow::new()),
            FunctionOp::Call { signature } => {
                let mut s = signature.clone();
                s.const_input.to_mut().insert(
                    0,
                    ClassicType::Graph(Box::new((Default::default(), signature.clone()))).into(),
                );
                s
            }
            FunctionOp::CallIndirect { signature } => {
                let mut s = signature.clone();
                s.input.to_mut().insert(
                    0,
                    ClassicType::Graph(Box::new((Default::default(), signature.clone()))).into(),
                );
                s
            }
            FunctionOp::LoadConstant { datatype } => Signature::new(
                vec![datatype.clone()],
                TypeRow::new(),
                TypeRow::new(),
                vec![datatype.clone()],
            ),
            FunctionOp::Leaf { op } => op.signature(),
            FunctionOp::Nested { signature } => signature.clone(),
        }
    }
}

impl From<LeafOp> for FunctionOp {
    fn from(op: LeafOp) -> Self {
        Self::Leaf { op }
    }
}

use super::Op;
use crate::types::{AngleValue, DataType, Quat, Signature};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionOp {
    /// An input/output node.
    ///
    /// The outputs of this node are the inputs to the function, and the inputs
    /// are the outputs of the function. In a data dependency subgraph, a valid
    /// ordering of operations can be achieved by topologically sorting the
    /// nodes starting from the outputs of this node.
    IO { signature: Signature },
    /// Call a function directly.
    ///
    /// The first port is connected to the def/declare of the function being
    /// called directly, with a ConstE<Graph> edge. The signature of the
    /// remaining ports matches the function being called.
    Call { signature: Signature },
    /// Call a function indirectly. Like call, but the first input is a standard dataflow graph type
    CallIndirect { signature: Signature },
    /// Load a static constant in to the local dataflow graph
    LoadConstant { datatype: DataType },
    /// δ (delta): a simply nested dataflow graph
    Nested { signature: Signature },
}

impl Default for FunctionOp {
    fn default() -> Self {
        FunctionOp::IO {
            signature: Default::default(),
        }
    }
}

impl Op for FunctionOp {
    fn name(&self) -> &str {
        match self {
            FunctionOp::IO { .. } => "io",
            FunctionOp::Call { .. } => "call",
            FunctionOp::CallIndirect { .. } => "call_indirect",
            FunctionOp::LoadConstant { .. } => "load",
            FunctionOp::Nested { .. } => "nested",
        }
    }

    fn signature(&self) -> Signature {
        match self {
            FunctionOp::IO { signature } => signature.clone(),
            FunctionOp::Call { signature } => {
                let mut s = signature.clone();
                s.const_input.0.insert(
                    0,
                    DataType::Graph {
                        resources: Default::default(),
                        signature: signature.clone(),
                    },
                );
                s
            }
            FunctionOp::CallIndirect { signature } => {
                let mut s = signature.clone();
                s.input.0.insert(
                    0,
                    DataType::Graph {
                        resources: Default::default(),
                        signature: signature.clone(),
                    },
                );
                s
            }
            FunctionOp::LoadConstant { datatype } => {
                Signature::new([datatype.clone()], [], [], [datatype.clone()])
            }
            FunctionOp::Nested { signature } => signature.clone(),
        }
    }
}

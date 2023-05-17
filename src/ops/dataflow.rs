//! Dataflow operations.

use smol_str::SmolStr;

use super::{controlflow::ControlFlowOp, tag::OpTag, LeafOp};
use crate::types::{ClassicType, EdgeKind, Signature, SignatureDescription, SimpleType, TypeRow};

/// A dataflow operation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
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
    /// Call a function indirectly. Like call, but the first input is a standard dataflow graph type.
    CallIndirect { signature: Signature },
    /// Load a static constant in to the local dataflow graph.
    LoadConstant { datatype: ClassicType },
    /// Simple operation that has only value inputs+outputs and (potentially) StateOrder edges.
    Leaf { op: LeafOp },
    /// A simply nested dataflow graph.
    DFG { signature: Signature },
    /// Operation related to control flow.
    ControlFlow { op: ControlFlowOp },
}

impl DataflowOp {
    /// The edge kind for the inputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other input edges. Otherwise, all other input
    /// edges will be of that kind.
    pub fn other_inputs(&self) -> Option<EdgeKind> {
        if let DataflowOp::Input { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }

    /// The edge kind for the outputs of the operation not described by the
    /// signature.
    ///
    /// If None, there will be no other output edges. Otherwise, all other
    /// output edges will be of that kind.
    pub fn other_outputs(&self) -> Option<EdgeKind> {
        if let DataflowOp::Output { .. } = self {
            None
        } else {
            Some(EdgeKind::StateOrder)
        }
    }

    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            DataflowOp::Input { .. } => "input",
            DataflowOp::Output { .. } => "output",
            DataflowOp::Call { .. } => "call",
            DataflowOp::CallIndirect { .. } => "call_indirect",
            DataflowOp::LoadConstant { .. } => "load",
            DataflowOp::Leaf { op } => return op.name(),
            DataflowOp::DFG { .. } => "nested",
            DataflowOp::ControlFlow { op } => return op.name(),
        }
        .into()
    }

    /// A human-readable description of the operation.
    pub fn description(&self) -> &str {
        match self {
            DataflowOp::Input { .. } => "The input node for this dataflow subgraph",
            DataflowOp::Output { .. } => "The output node for this dataflow subgraph",
            DataflowOp::Call { .. } => "Call a function directly",
            DataflowOp::CallIndirect { .. } => "Call a function indirectly",
            DataflowOp::LoadConstant { .. } => {
                "Load a static constant in to the local dataflow graph"
            }
            DataflowOp::Leaf { op } => return op.description(),
            DataflowOp::DFG { .. } => "A simply nested dataflow graph",
            DataflowOp::ControlFlow { op } => return op.description(),
        }
    }

    /// Tag identifying the operation.
    pub fn tag(&self) -> OpTag {
        match self {
            DataflowOp::Input { .. } => OpTag::Input,
            DataflowOp::Output { .. } => OpTag::Output,
            DataflowOp::Call { .. } | DataflowOp::CallIndirect { .. } => OpTag::FnCall,
            DataflowOp::LoadConstant { .. } => OpTag::LoadConst,
            DataflowOp::Leaf { .. } => OpTag::Leaf,
            DataflowOp::DFG { .. } => OpTag::Dfg,
            DataflowOp::ControlFlow { op } => op.tag(),
        }
    }

    /// The signature of the operation.
    pub fn signature(&self) -> Signature {
        match self {
            DataflowOp::Input { types } => Signature::new_df(TypeRow::new(), types.clone()),
            DataflowOp::Output { types } => Signature::new_df(types.clone(), TypeRow::new()),
            DataflowOp::Call { signature } => Signature {
                const_input: vec![ClassicType::graph_from_sig(signature.clone()).into()].into(),
                ..signature.clone()
            },
            DataflowOp::CallIndirect { signature } => {
                let mut s = signature.clone();
                s.input
                    .to_mut()
                    .insert(0, ClassicType::graph_from_sig(signature.clone()).into());
                s
            }
            DataflowOp::LoadConstant { datatype } => Signature::new(
                TypeRow::new(),
                vec![SimpleType::Classic(datatype.clone())],
                vec![SimpleType::Classic(datatype.clone())],
            ),
            DataflowOp::Leaf { op } => op.signature(),
            DataflowOp::DFG { signature } => signature.clone(),
            DataflowOp::ControlFlow { op } => op.signature(),
        }
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        match self {
            DataflowOp::Leaf { op } => op.signature_desc(),
            DataflowOp::ControlFlow { op } => op.signature_desc(),
            // TODO: add port descriptions for other ops
            _ => SignatureDescription::default(),
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

use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, TypeRow};

use super::Op;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlFlowOp {
    /// ɣ (gamma) node: conditional operation
    Conditional { inputs: TypeRow, outputs: TypeRow },
    /// θ (theta) node: tail-controlled loop. Here we assume the same inputs + outputs variant.
    Loop { vars: TypeRow },
    /// β (beta): a CFG basic block node. The signature is that of the internal Dataflow graph.
    BasicBlock { inputs: TypeRow, outputs: TypeRow },
    /// 𝛋 (kappa): a dataflow node which is defined by a child CFG
    CFG { inputs: TypeRow, outputs: TypeRow },
}

impl ControlFlowOp {
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(if let ControlFlowOp::BasicBlock { .. } = self {
            EdgeKind::ControlFlow
        } else {
            EdgeKind::StateOrder
        })
    }
}

impl Op for ControlFlowOp {
    fn name(&self) -> SmolStr {
        match self {
            ControlFlowOp::Conditional { .. } => "ɣ",
            ControlFlowOp::Loop { .. } => "θ",
            ControlFlowOp::BasicBlock { .. } => "β",
            ControlFlowOp::CFG { .. } => "𝛋",
        }
        .into()
    }

    fn signature(&self) -> Signature {
        match self {
            ControlFlowOp::Conditional { inputs, outputs } => {
                Signature::new_df(inputs.clone(), outputs.clone())
            }
            ControlFlowOp::Loop { vars } => Signature::new_linear(vars.clone()),
            ControlFlowOp::BasicBlock { .. } => Signature::default(),
            ControlFlowOp::CFG { inputs, outputs } => {
                Signature::new_df(inputs.clone(), outputs.clone())
            }
        }
    }
}

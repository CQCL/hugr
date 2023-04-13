use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, TypeRow};

use super::Op;

/// Dataflow operations that are (informally) related to control flow.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlFlowOp {
    /// ɣ (gamma) node: conditional operation
    Conditional { inputs: TypeRow, outputs: TypeRow },
    /// θ (theta) node: tail-controlled loop. Here we assume the same inputs + outputs variant.
    Loop { vars: TypeRow },
    /// 𝛋 (kappa): a dataflow node which is defined by a child CFG
    CFG { inputs: TypeRow, outputs: TypeRow },
}

impl Op for ControlFlowOp {
    fn name(&self) -> SmolStr {
        match self {
            ControlFlowOp::Conditional { .. } => "ɣ",
            ControlFlowOp::Loop { .. } => "θ",
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
            ControlFlowOp::CFG { inputs, outputs } => {
                Signature::new_df(inputs.clone(), outputs.clone())
            }
        }
    }
}

/// β (beta): a CFG basic block node. The signature is that of the internal Dataflow graph.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BasicBlockOp {
    inputs: TypeRow,
    outputs: TypeRow,
}

impl BasicBlockOp {
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }
}

impl Op for BasicBlockOp {
    fn name(&self) -> SmolStr {
        "β".into()
    }

    fn signature(&self) -> Signature {
        // The value edges into/out of the beta-node itself
        Signature::default()
    }
}

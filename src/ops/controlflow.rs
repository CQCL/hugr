use crate::types::Signature;

use super::Op;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlFlowOp {
    /// ɣ (gamma) node: conditional operation
    Conditional { signature: Signature },
    /// θ (theta) node: tail-controlled loop
    Loop { signature: Signature },
    /// β (beta): a CFG basic block node
    BasicBlock { signature: Signature },
    /// 𝛋 (kappa): a dataflow node which is defined by a child CFG
    CFG { signature: Signature },
}

impl Op for ControlFlowOp {
    fn name(&self) -> &str {
        match self {
            ControlFlowOp::Conditional { .. } => "ɣ",
            ControlFlowOp::Loop { .. } => "θ",
            ControlFlowOp::BasicBlock { .. } => "β",
            ControlFlowOp::CFG { .. } => "𝛋",
        }
    }

    fn signature(&self) -> Signature {
        match self {
            ControlFlowOp::Conditional { signature } => signature.clone(),
            ControlFlowOp::Loop { signature } => signature.clone(),
            ControlFlowOp::BasicBlock { signature } => signature.clone(),
            ControlFlowOp::CFG { signature } => signature.clone(),
        }
    }
}

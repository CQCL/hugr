use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, SignatureDescription, TypeRow};

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

impl ControlFlowOp {
    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        match self {
            ControlFlowOp::Conditional { .. } => "ɣ",
            ControlFlowOp::Loop { .. } => "θ",
            ControlFlowOp::CFG { .. } => "𝛋",
        }
        .into()
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        match self {
            ControlFlowOp::Conditional { .. } => "HUGR conditional operation",
            ControlFlowOp::Loop { .. } => "A tail-controlled loop",
            ControlFlowOp::CFG { .. } => "A dataflow node defined by a child CFG",
        }
    }

    /// The signature of the operation
    pub fn signature(&self) -> Signature {
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

    /// Optional description of the ports in the signature
    pub fn signature_desc(&self) -> SignatureDescription {
        // TODO: add descriptions
        Default::default()
    }
}

/// β (beta): a CFG basic block node. The signature is that of the internal Dataflow graph.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BasicBlockOp {
    pub inputs: TypeRow,
    pub outputs: TypeRow,
}

impl BasicBlockOp {
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        "β".into()
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        "A CFG basic block node"
    }
}

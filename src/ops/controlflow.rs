use smol_str::SmolStr;

use crate::{
    ops::DataflowOp,
    types::{EdgeKind, Signature, SignatureDescription, TypeRow},
};

use super::{OpType, OpTypeValidator};

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

impl OpTypeValidator for ControlFlowOp {
    // TODO: CFG nodes require checking the internal signature of pairs of
    // BasicBlocks connected by ControlFlow edges. This is not currently
    // implemented, and should probably go outside of the OpTypeValidator trait.

    fn is_valid_parent(&self, parent: &OpType) -> bool {
        // Note: This method is never used. `DataflowOp::is_valid_parent` calls
        // `is_df_container` directly.
        parent.is_df_container()
    }

    fn is_container(&self) -> bool {
        true
    }

    fn is_df_container(&self) -> bool {
        matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        )
    }

    fn first_child_valid(&self, child: OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Input { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }

    fn last_child_valid(&self, child: OpType) -> bool {
        // TODO: check signatures
        match self {
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. } => {
                matches!(child, OpType::Function(DataflowOp::Output { .. }))
            }
            ControlFlowOp::CFG { .. } => matches!(child, OpType::BasicBlock(_)),
        }
    }

    fn require_dag(&self) -> bool {
        matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        )
    }

    fn require_dominators(&self) -> bool {
        // TODO: Should we require the CFGs entry(exit) to be the single source(sink)?
        matches!(
            self,
            ControlFlowOp::Conditional { .. } | ControlFlowOp::Loop { .. }
        )
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

    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        "β".into()
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        "A CFG basic block node"
    }
}

impl OpTypeValidator for BasicBlockOp {
    fn is_valid_parent(&self, parent: &OpType) -> bool {
        matches!(
            parent,
            OpType::Function(DataflowOp::ControlFlow {
                op: ControlFlowOp::CFG { .. }
            })
        )
    }

    fn is_container(&self) -> bool {
        true
    }

    fn is_df_container(&self) -> bool {
        true
    }

    fn require_dag(&self) -> bool {
        true
    }

    fn require_dominators(&self) -> bool {
        true
    }
}

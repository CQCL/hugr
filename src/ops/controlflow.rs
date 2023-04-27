use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, SignatureDescription, SimpleType, TypeRow};

/// Dataflow operations that are (informally) related to control flow.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlFlowOp {
    /// ɣ (gamma) node: conditional operation
    Conditional {
        /// The branch predicate. It's len is equal to the number of branches.
        predicate_inputs: TypeRow,
        /// Other inputs passed to all branches.
        inputs: TypeRow,
        /// Common output of all branches.
        outputs: TypeRow,
    },
    /// θ (theta) node: tail-controlled loop.
    Loop { inputs: TypeRow, outputs: TypeRow },
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
            ControlFlowOp::Conditional {
                predicate_inputs,
                inputs,
                outputs,
            } => {
                let predicate = SimpleType::new_sum(predicate_inputs.clone());
                let mut sig_in = vec![predicate];
                sig_in.extend_from_slice(inputs);
                Signature::new_df(sig_in, outputs.clone())
            }
            ControlFlowOp::Loop { inputs, outputs } => {
                Signature::new_df(inputs.clone(), outputs.clone())
            }
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Basic block ops - nodes valid in control flow graphs
pub enum BasicBlockOp {
    /// β (beta): a CFG basic block node. The signature is that of the internal Dataflow graph.
    Beta { inputs: TypeRow, outputs: TypeRow },
    /// β_e (beta exit): the single exit node of the CFG,
    /// stores the types of the CFG node output
    Exit { cfg_outputs: TypeRow },
}

impl BasicBlockOp {
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        match self {
            BasicBlockOp::Beta { .. } => "β".into(),
            BasicBlockOp::Exit { .. } => "β_e".into(),
        }
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        match self {
            BasicBlockOp::Beta { .. } => "A CFG basic block node",
            BasicBlockOp::Exit { .. } => "A CFG exit block node",
        }
    }
}

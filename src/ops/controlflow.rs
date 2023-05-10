use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, SignatureDescription, SimpleType, TypeRow};

/// Dataflow operations that are (informally) related to control flow.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlFlowOp {
    /// Conditional operation, defined by child `Case` nodes for each branch
    Conditional {
        /// The branch predicate. It's len is equal to the number of cases.
        predicate_inputs: TypeRow,
        /// Other inputs passed to all cases.
        inputs: TypeRow,
        /// Common output of all cases.
        outputs: TypeRow,
    },
    /// Tail-controlled loop.
    TailLoop { inputs: TypeRow, outputs: TypeRow },
    /// A dataflow node which is defined by a child CFG
    CFG { inputs: TypeRow, outputs: TypeRow },
}

impl ControlFlowOp {
    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        match self {
            ControlFlowOp::Conditional { .. } => "Conditional",
            ControlFlowOp::TailLoop { .. } => "TailLoop",
            ControlFlowOp::CFG { .. } => "CFG",
        }
        .into()
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        match self {
            ControlFlowOp::Conditional { .. } => "HUGR conditional operation",
            ControlFlowOp::TailLoop { .. } => "A tail-controlled loop",
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
            ControlFlowOp::TailLoop { inputs, outputs } => {
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
    /// A CFG basic block node. The signature is that of the internal Dataflow graph.
    Block {
        inputs: TypeRow,
        outputs: TypeRow,
        n_cases: usize,
    },
    /// The single exit node of the CFG, has no children,
    /// stores the types of the CFG node output
    Exit { cfg_outputs: TypeRow },
}

impl BasicBlockOp {
    /// Non dataflow edge types allowed for this node
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        match self {
            BasicBlockOp::Block { .. } => "BasicBlock".into(),
            BasicBlockOp::Exit { .. } => "ExitBlock".into(),
        }
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        match self {
            BasicBlockOp::Block { .. } => "A CFG basic block node",
            BasicBlockOp::Exit { .. } => "A CFG exit block node",
        }
    }

    /// The input signature of the contained dataflow graph
    pub fn dataflow_input(&self) -> &TypeRow {
        match self {
            BasicBlockOp::Block { inputs, .. } => inputs,
            BasicBlockOp::Exit { cfg_outputs } => cfg_outputs,
        }
    }

    /// The output signature of the contained dataflow graph
    pub fn dataflow_output(&self) -> &TypeRow {
        match self {
            BasicBlockOp::Block { outputs, .. } => outputs,
            BasicBlockOp::Exit { cfg_outputs } => cfg_outputs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Case ops - nodes valid inside Conditional nodes
pub struct CaseOp {
    pub signature: Signature,
}

impl CaseOp {
    /// Non dataflow edge types allowed for this node
    pub fn other_edges(&self) -> Option<EdgeKind> {
        None
    }

    /// The name of the operation
    pub fn name(&self) -> SmolStr {
        "Case".into()
    }

    /// The description of the operation
    pub fn description(&self) -> &str {
        "A case node inside a conditional"
    }

    /// The input signature of the contained dataflow graph
    pub fn dataflow_input(&self) -> &TypeRow {
        &self.signature.input
    }

    /// The output signature of the contained dataflow graph
    pub fn dataflow_output(&self) -> &TypeRow {
        &self.signature.output
    }
}

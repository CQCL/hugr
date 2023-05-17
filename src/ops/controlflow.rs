//! Control flow operations.

use smol_str::SmolStr;

use crate::types::{EdgeKind, Signature, SignatureDescription, SimpleType, TypeRow};

use super::tag::OpTag;

/// Type rows defining the inner and outer signatures of a [`ControlFlowOp::TailLoop`]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailLoopSignature {
    /// Types that are only input
    pub just_inputs: TypeRow,
    /// Types that are only output
    pub just_outputs: TypeRow,
    /// Types that are appended to both input and output
    pub rest: TypeRow,
}

impl From<TailLoopSignature> for ControlFlowOp {
    fn from(value: TailLoopSignature) -> Self {
        ControlFlowOp::TailLoop(value)
    }
}

// Implement conversion to standard signature
impl From<TailLoopSignature> for Signature {
    fn from(tail_sig: TailLoopSignature) -> Self {
        let [inputs, outputs] = [tail_sig.just_inputs, tail_sig.just_outputs].map(|mut row| {
            row.to_mut().extend(tail_sig.rest.iter().cloned());
            row
        });
        Signature::new_df(inputs, outputs)
    }
}
impl TailLoopSignature {
    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn loop_output_row(&self) -> TypeRow {
        let predicate =
            SimpleType::new_predicate([self.just_inputs.clone(), self.just_outputs.clone()]);
        let mut outputs = self.rest.clone();
        outputs.to_mut().insert(0, predicate);
        outputs
    }

    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn loop_input_row(&self) -> TypeRow {
        let mut inputs = self.just_inputs.clone();
        inputs.to_mut().extend_from_slice(&self.rest);
        inputs
    }
}

/// Dataflow operations that are (informally) related to control flow.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControlFlowOp {
    /// Conditional operation, defined by child `Case` nodes for each branch.
    Conditional {
        /// The branch predicate. It's len is equal to the number of cases.
        predicate_inputs: TypeRow,
        /// Other inputs passed to all cases.
        inputs: TypeRow,
        /// Common output of all cases.
        outputs: TypeRow,
    },
    /// Tail-controlled loop.
    #[allow(missing_docs)]
    TailLoop(TailLoopSignature),
    /// A dataflow node which is defined by a child CFG.
    #[allow(missing_docs)]
    CFG { inputs: TypeRow, outputs: TypeRow },
}

impl ControlFlowOp {
    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            ControlFlowOp::Conditional { .. } => "Conditional",
            ControlFlowOp::TailLoop { .. } => "TailLoop",
            ControlFlowOp::CFG { .. } => "CFG",
        }
        .into()
    }

    /// The description of the operation.
    pub fn description(&self) -> &str {
        match self {
            ControlFlowOp::Conditional { .. } => "HUGR conditional operation",
            ControlFlowOp::TailLoop { .. } => "A tail-controlled loop",
            ControlFlowOp::CFG { .. } => "A dataflow node defined by a child CFG",
        }
    }

    /// Tag identifying the operation.
    pub fn tag(&self) -> OpTag {
        match self {
            ControlFlowOp::Conditional { .. } => OpTag::Conditional,
            ControlFlowOp::TailLoop { .. } => OpTag::TailLoop,
            ControlFlowOp::CFG { .. } => OpTag::Cfg,
        }
    }

    /// The signature of the operation.
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
            ControlFlowOp::TailLoop(tail_op_sig) => tail_op_sig.clone().into(),
            ControlFlowOp::CFG { inputs, outputs } => {
                Signature::new_df(inputs.clone(), outputs.clone())
            }
        }
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self) -> SignatureDescription {
        // TODO: add descriptions
        Default::default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Basic block ops - nodes valid in control flow graphs.
#[allow(missing_docs)]
pub enum BasicBlockOp {
    /// A CFG basic block node. The signature is that of the internal Dataflow graph.
    Block {
        inputs: TypeRow,
        other_outputs: TypeRow,
        predicate_variants: Vec<TypeRow>,
    },
    /// The single exit node of the CFG, has no children,
    /// stores the types of the CFG node output.
    Exit { cfg_outputs: TypeRow },
}

impl BasicBlockOp {
    /// The edge kind for the inputs and outputs of the operation not described
    /// by the signature.
    pub fn other_edges(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        match self {
            BasicBlockOp::Block { .. } => "BasicBlock".into(),
            BasicBlockOp::Exit { .. } => "ExitBlock".into(),
        }
    }

    /// The description of the operation.
    pub fn description(&self) -> &str {
        match self {
            BasicBlockOp::Block { .. } => "A CFG basic block node",
            BasicBlockOp::Exit { .. } => "A CFG exit block node",
        }
    }

    /// Tag identifying the operation.
    pub fn tag(&self) -> OpTag {
        match self {
            BasicBlockOp::Block { .. } => OpTag::BasicBlock,
            BasicBlockOp::Exit { .. } => OpTag::BasicBlockExit,
        }
    }

    /// The input signature of the contained dataflow graph.
    pub fn dataflow_input(&self) -> &TypeRow {
        match self {
            BasicBlockOp::Block { inputs, .. } => inputs,
            BasicBlockOp::Exit { cfg_outputs } => cfg_outputs,
        }
    }

    /// The correct inputs of any successors. Returns None if successor is not a
    /// valid index.
    pub fn successor_input(&self, successor: usize) -> Option<TypeRow> {
        match self {
            BasicBlockOp::Block {
                other_outputs: outputs,
                predicate_variants,
                ..
            } => {
                let mut row = predicate_variants.get(successor)?.clone();
                row.to_mut().extend_from_slice(outputs);
                Some(row)
            }
            BasicBlockOp::Exit { .. } => panic!("Exit should have no successors"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Case ops - nodes valid inside Conditional nodes.
pub struct CaseOp {
    /// The signature of the contained dataflow graph.
    pub signature: Signature,
}

impl CaseOp {
    /// The edge kind for the inputs and outputs of the operation not described
    /// by the signature.
    pub fn other_edges(&self) -> Option<EdgeKind> {
        None
    }

    /// The name of the operation.
    pub fn name(&self) -> SmolStr {
        "Case".into()
    }

    /// The description of the operation.
    pub fn description(&self) -> &str {
        "A case node inside a conditional"
    }

    /// Tag identifying the operation.
    pub fn tag(&self) -> OpTag {
        OpTag::Case
    }

    /// The input signature of the contained dataflow graph.
    pub fn dataflow_input(&self) -> &TypeRow {
        &self.signature.input
    }

    /// The output signature of the contained dataflow graph.
    pub fn dataflow_output(&self) -> &TypeRow {
        &self.signature.output
    }
}

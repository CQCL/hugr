//! Control flow operations.

use smol_str::SmolStr;

use crate::types::{ClassicType, EdgeKind, Signature, SimpleType, TypeRow};

use super::dataflow::DataflowOpTrait;
use super::tag::OpTag;
use super::{impl_op_name, OpName, OpTrait};

/// Tail-controlled loop.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailLoop {
    /// Types that are only input
    pub just_inputs: TypeRow<ClassicType>,
    /// Types that are only output
    pub just_outputs: TypeRow<ClassicType>,
    /// Types that are appended to both input and output
    pub rest: TypeRow<SimpleType>,
}

impl_op_name!(TailLoop);

impl DataflowOpTrait for TailLoop {
    fn description(&self) -> &str {
        "A tail-controlled loop"
    }

    fn tag(&self) -> OpTag {
        OpTag::TailLoop
    }

    fn signature(&self) -> Signature {
        let [inputs, outputs] =
            [&self.just_inputs, &self.just_outputs].map(|row| predicate_first(row, &self.rest));
        Signature::new_df(inputs, outputs)
    }
}

impl TailLoop {
    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_output_row(&self) -> TypeRow<SimpleType> {
        let predicate =
            SimpleType::new_predicate([self.just_inputs.clone(), self.just_outputs.clone()]);
        let mut outputs = vec![predicate];
        outputs.extend_from_slice(&self.rest);
        outputs.into()
    }

    /// Build the input TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_input_row(&self) -> TypeRow<SimpleType> {
        predicate_first(&self.just_inputs, &self.rest)
    }
}

/// Conditional operation, defined by child `Case` nodes for each branch.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Conditional {
    /// The possible rows of the predicate input
    pub predicate_inputs: Vec<TypeRow<ClassicType>>,
    /// Remaining input types
    pub other_inputs: TypeRow<SimpleType>,
    /// Output types
    pub outputs: TypeRow<SimpleType>,
}
impl_op_name!(Conditional);

impl DataflowOpTrait for Conditional {
    fn description(&self) -> &str {
        "HUGR conditional operation"
    }

    fn tag(&self) -> OpTag {
        OpTag::Conditional
    }

    fn signature(&self) -> Signature {
        let mut inputs = self.other_inputs.clone();
        inputs.to_mut().insert(
            0,
            SimpleType::new_predicate(self.predicate_inputs.clone().into_iter()),
        );
        Signature::new_df(inputs, self.outputs.clone())
    }
}

impl Conditional {
    /// Build the input TypeRow of the nth child graph of a Conditional node.
    pub(crate) fn case_input_row(&self, case: usize) -> Option<TypeRow<SimpleType>> {
        Some(predicate_first(
            self.predicate_inputs.get(case)?,
            &self.other_inputs,
        ))
    }
}

/// A dataflow node which is defined by a child CFG.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
pub struct CFG {
    pub inputs: TypeRow<SimpleType>,
    pub outputs: TypeRow<SimpleType>,
}

impl_op_name!(CFG);

impl DataflowOpTrait for CFG {
    fn description(&self) -> &str {
        "A dataflow node defined by a child CFG"
    }

    fn tag(&self) -> OpTag {
        OpTag::Cfg
    }

    fn signature(&self) -> Signature {
        Signature::new_df(self.inputs.clone(), self.outputs.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "block")]
/// Basic block ops - nodes valid in control flow graphs.
#[allow(missing_docs)]
pub enum BasicBlock {
    /// A CFG basic block node. The signature is that of the internal Dataflow graph.
    DFB {
        inputs: TypeRow<SimpleType>,
        other_outputs: TypeRow<SimpleType>,
        predicate_variants: Vec<TypeRow<ClassicType>>,
    },
    /// The single exit node of the CFG, has no children,
    /// stores the types of the CFG node output.
    Exit { cfg_outputs: TypeRow<SimpleType> },
}

impl OpName for BasicBlock {
    /// The name of the operation.
    fn name(&self) -> SmolStr {
        match self {
            BasicBlock::DFB { .. } => "DFB".into(),
            BasicBlock::Exit { .. } => "Exit".into(),
        }
    }
}

impl OpTrait for BasicBlock {
    /// The description of the operation.
    fn description(&self) -> &str {
        match self {
            BasicBlock::DFB { .. } => "A CFG basic block node",
            BasicBlock::Exit { .. } => "A CFG exit block node",
        }
    }
    /// Tag identifying the operation.
    fn tag(&self) -> OpTag {
        match self {
            BasicBlock::DFB { .. } => OpTag::BasicBlock,
            BasicBlock::Exit { .. } => OpTag::BasicBlockExit,
        }
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }
}

impl BasicBlock {
    /// The input signature of the contained dataflow graph.
    pub fn dataflow_input(&self) -> &TypeRow<SimpleType> {
        match self {
            BasicBlock::DFB { inputs, .. } => inputs,
            BasicBlock::Exit { cfg_outputs } => cfg_outputs,
        }
    }

    /// The correct inputs of any successors. Returns None if successor is not a
    /// valid index.
    pub fn successor_input(&self, successor: usize) -> Option<TypeRow<SimpleType>> {
        match self {
            BasicBlock::DFB {
                predicate_variants,
                other_outputs: outputs,
                ..
            } => Some(predicate_first(predicate_variants.get(successor)?, outputs)),
            BasicBlock::Exit { .. } => panic!("Exit should have no successors"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Case ops - nodes valid inside Conditional nodes.
pub struct Case {
    /// The signature of the contained dataflow graph.
    pub signature: Signature,
}

impl_op_name!(Case);

impl OpTrait for Case {
    fn description(&self) -> &str {
        "A case node inside a conditional"
    }

    fn tag(&self) -> OpTag {
        OpTag::Case
    }
}

impl Case {
    /// The input signature of the contained dataflow graph.
    pub fn dataflow_input(&self) -> &TypeRow<SimpleType> {
        &self.signature.input
    }

    /// The output signature of the contained dataflow graph.
    pub fn dataflow_output(&self) -> &TypeRow<SimpleType> {
        &self.signature.output
    }
}

fn predicate_first(pred: &TypeRow<ClassicType>, rest: &TypeRow<SimpleType>) -> TypeRow<SimpleType> {
    TypeRow::from(
        pred.iter()
            .cloned()
            .map(SimpleType::Classic)
            .chain(rest.iter().cloned())
            .collect::<Vec<_>>(),
    )
}

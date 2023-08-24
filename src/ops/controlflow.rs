//! Control flow operations.

use smol_str::SmolStr;

use crate::types::{EdgeKind, FunctionType, Type, TypeRow};

use super::dataflow::DataflowOpTrait;
use super::OpTag;
use super::{impl_op_name, OpName, OpTrait, StaticTag};

/// Tail-controlled loop.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailLoop {
    /// Types that are only input
    pub just_inputs: TypeRow,
    /// Types that are only output
    pub just_outputs: TypeRow,
    /// Types that are appended to both input and output
    pub rest: TypeRow,
}

impl_op_name!(TailLoop);

impl DataflowOpTrait for TailLoop {
    const TAG: OpTag = OpTag::TailLoop;

    fn description(&self) -> &str {
        "A tail-controlled loop"
    }

    fn signature(&self) -> FunctionType {
        let [inputs, outputs] =
            [&self.just_inputs, &self.just_outputs].map(|row| predicate_first(row, &self.rest));
        FunctionType::new(inputs, outputs)
    }
}

impl TailLoop {
    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_output_row(&self) -> TypeRow {
        let predicate = Type::new_predicate([self.just_inputs.clone(), self.just_outputs.clone()]);
        let mut outputs = vec![predicate];
        outputs.extend_from_slice(&self.rest);
        outputs.into()
    }

    /// Build the input TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_input_row(&self) -> TypeRow {
        predicate_first(&self.just_inputs, &self.rest)
    }
}

/// Conditional operation, defined by child `Case` nodes for each branch.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Conditional {
    /// The possible rows of the predicate input
    pub predicate_inputs: Vec<TypeRow>,
    /// Remaining input types
    pub other_inputs: TypeRow,
    /// Output types
    pub outputs: TypeRow,
}
impl_op_name!(Conditional);

impl DataflowOpTrait for Conditional {
    const TAG: OpTag = OpTag::Conditional;

    fn description(&self) -> &str {
        "HUGR conditional operation"
    }

    fn signature(&self) -> FunctionType {
        let mut inputs = self.other_inputs.clone();
        inputs
            .to_mut()
            .insert(0, Type::new_predicate(self.predicate_inputs.clone()));
        FunctionType::new(inputs, self.outputs.clone())
    }
}

impl Conditional {
    /// Build the input TypeRow of the nth child graph of a Conditional node.
    pub(crate) fn case_input_row(&self, case: usize) -> Option<TypeRow> {
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
    pub inputs: TypeRow,
    pub outputs: TypeRow,
}

impl_op_name!(CFG);

impl DataflowOpTrait for CFG {
    const TAG: OpTag = OpTag::Cfg;

    fn description(&self) -> &str {
        "A dataflow node defined by a child CFG"
    }

    fn signature(&self) -> FunctionType {
        FunctionType::new(self.inputs.clone(), self.outputs.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "block")]
/// Basic block ops - nodes valid in control flow graphs.
#[allow(missing_docs)]
pub enum BasicBlock {
    /// A CFG basic block node. The signature is that of the internal Dataflow graph.
    DFB {
        inputs: TypeRow,
        other_outputs: TypeRow,
        predicate_variants: Vec<TypeRow>,
    },
    /// The single exit node of the CFG, has no children,
    /// stores the types of the CFG node output.
    Exit { cfg_outputs: TypeRow },
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

impl StaticTag for BasicBlock {
    const TAG: OpTag = OpTag::BasicBlock;
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
    pub fn dataflow_input(&self) -> &TypeRow {
        match self {
            BasicBlock::DFB { inputs, .. } => inputs,
            BasicBlock::Exit { cfg_outputs } => cfg_outputs,
        }
    }

    /// The correct inputs of any successors. Returns None if successor is not a
    /// valid index.
    pub fn successor_input(&self, successor: usize) -> Option<TypeRow> {
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
    pub signature: FunctionType,
}

impl_op_name!(Case);

impl StaticTag for Case {
    const TAG: OpTag = OpTag::Case;
}

impl OpTrait for Case {
    fn description(&self) -> &str {
        "A case node inside a conditional"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }
}

impl Case {
    /// The input signature of the contained dataflow graph.
    pub fn dataflow_input(&self) -> &TypeRow {
        &self.signature.input
    }

    /// The output signature of the contained dataflow graph.
    pub fn dataflow_output(&self) -> &TypeRow {
        &self.signature.output
    }
}

fn predicate_first(pred: &TypeRow, rest: &TypeRow) -> TypeRow {
    TypeRow::from(
        pred.iter()
            .cloned()
            .chain(rest.iter().cloned())
            .collect::<Vec<_>>(),
    )
}

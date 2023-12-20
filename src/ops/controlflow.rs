//! Control flow operations.

use smol_str::SmolStr;

use crate::extension::ExtensionSet;
use crate::types::{EdgeKind, FunctionType, Type, TypeRow};
use crate::{type_row, Direction};

use super::dataflow::{DataflowOpTrait, DataflowParent};
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
            [&self.just_inputs, &self.just_outputs].map(|row| tuple_sum_first(row, &self.rest));
        FunctionType::new(inputs, outputs)
    }
}

impl TailLoop {
    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_output_row(&self) -> TypeRow {
        let tuple_sum_type =
            Type::new_tuple_sum([self.just_inputs.clone(), self.just_outputs.clone()]);
        let mut outputs = vec![tuple_sum_type];
        outputs.extend_from_slice(&self.rest);
        outputs.into()
    }

    /// Build the input TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_input_row(&self) -> TypeRow {
        tuple_sum_first(&self.just_inputs, &self.rest)
    }
}

/// Conditional operation, defined by child `Case` nodes for each branch.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Conditional {
    /// The possible rows of the TupleSum input
    pub tuple_sum_rows: Vec<TypeRow>,
    /// Remaining input types
    pub other_inputs: TypeRow,
    /// Output types
    pub outputs: TypeRow,
    /// Extensions used to produce the outputs
    pub extension_delta: ExtensionSet,
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
            .insert(0, Type::new_tuple_sum(self.tuple_sum_rows.clone()));
        FunctionType::new(inputs, self.outputs.clone()).with_extension_delta(&self.extension_delta)
    }
}

impl Conditional {
    /// Build the input TypeRow of the nth child graph of a Conditional node.
    pub(crate) fn case_input_row(&self, case: usize) -> Option<TypeRow> {
        Some(tuple_sum_first(
            self.tuple_sum_rows.get(case)?,
            &self.other_inputs,
        ))
    }
}

/// A dataflow node which is defined by a child CFG.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
pub struct CFG {
    pub signature: FunctionType,
}

impl_op_name!(CFG);

impl DataflowOpTrait for CFG {
    const TAG: OpTag = OpTag::Cfg;

    fn description(&self) -> &str {
        "A dataflow node defined by a child CFG"
    }

    fn signature(&self) -> FunctionType {
        self.signature.clone()
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
        tuple_sum_rows: Vec<TypeRow>,
        extension_delta: ExtensionSet,
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

impl DataflowParent for BasicBlock {
    fn inner_signature(&self) -> FunctionType {
        match self {
            BasicBlock::DFB {
                inputs,
                other_outputs,
                tuple_sum_rows,
                ..
            } => {
                // The node outputs a TupleSum before the data outputs of the block node
                let tuple_sum_type = Type::new_tuple_sum(tuple_sum_rows.clone());
                let mut node_outputs = vec![tuple_sum_type];
                node_outputs.extend_from_slice(other_outputs);
                FunctionType::new(inputs.clone(), TypeRow::from(node_outputs))
            }
            BasicBlock::Exit { cfg_outputs } => FunctionType::new(type_row![], cfg_outputs.clone()),
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

    fn dataflow_signature(&self) -> Option<FunctionType> {
        Some(match self {
            BasicBlock::DFB {
                extension_delta, ..
            } => FunctionType::new(type_row![], type_row![]).with_extension_delta(extension_delta),
            BasicBlock::Exit { .. } => FunctionType::new(type_row![], type_row![]),
        })
    }

    fn non_df_port_count(&self, dir: Direction) -> usize {
        match self {
            Self::DFB { tuple_sum_rows, .. } if dir == Direction::Outgoing => tuple_sum_rows.len(),
            Self::Exit { .. } if dir == Direction::Outgoing => 0,
            _ => 1,
        }
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
                tuple_sum_rows,
                other_outputs: outputs,
                ..
            } => Some(tuple_sum_first(tuple_sum_rows.get(successor)?, outputs)),
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

impl DataflowParent for Case {
    fn inner_signature(&self) -> FunctionType {
        self.signature.clone()
    }
}

impl OpTrait for Case {
    fn description(&self) -> &str {
        "A case node inside a conditional"
    }

    fn extension_delta(&self) -> ExtensionSet {
        self.signature.extension_reqs.clone()
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

fn tuple_sum_first(tuple_sum_row: &TypeRow, rest: &TypeRow) -> TypeRow {
    TypeRow::from(
        tuple_sum_row
            .iter()
            .cloned()
            .chain(rest.iter().cloned())
            .collect::<Vec<_>>(),
    )
}

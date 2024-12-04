//! Control flow operations.

use crate::extension::ExtensionSet;
use crate::types::{EdgeKind, Signature, Substitution, Type, TypeRow};
use crate::Direction;

use super::dataflow::{DataflowOpTrait, DataflowParent};
use super::{impl_op_name, NamedOp, OpTrait, StaticTag};
use super::{OpName, OpTag};

/// Tail-controlled loop.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct TailLoop {
    /// Types that are only input
    pub just_inputs: TypeRow,
    /// Types that are only output
    pub just_outputs: TypeRow,
    /// Types that are appended to both input and output
    pub rest: TypeRow,
    /// Extension requirements to execute the body
    pub extension_delta: ExtensionSet,
}

impl_op_name!(TailLoop);

impl DataflowOpTrait for TailLoop {
    const TAG: OpTag = OpTag::TailLoop;

    fn description(&self) -> &str {
        "A tail-controlled loop"
    }

    fn signature(&self) -> Signature {
        let [inputs, outputs] =
            [&self.just_inputs, &self.just_outputs].map(|row| row.extend(self.rest.iter()));
        Signature::new(inputs, outputs).with_extension_delta(self.extension_delta.clone())
    }

    fn subst_mut(&mut self, subst: &Substitution) {
        self.just_inputs = self.just_inputs.substitute(subst);
        self.just_outputs = self.just_outputs.substitute(subst);
        self.rest = self.rest.substitute(subst);
        self.extension_delta = self.extension_delta.substitute(subst);
    }
}

impl TailLoop {
    /// The [tag] for a loop body output to indicate the loop should iterate again.
    ///
    /// [tag]: crate::ops::constant::Sum::tag
    pub const CONTINUE_TAG: usize = 0;

    /// The [tag] for a loop body output to indicate the loop should exit with the supplied values.
    ///
    /// [tag]: crate::ops::constant::Sum::tag
    pub const BREAK_TAG: usize = 1;

    /// Build the output TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_output_row(&self) -> TypeRow {
        let sum_type = Type::new_sum([self.just_inputs.clone(), self.just_outputs.clone()]);
        let mut outputs = vec![sum_type];
        outputs.extend_from_slice(&self.rest);
        outputs.into()
    }

    /// Build the input TypeRow of the child graph of a TailLoop node.
    pub(crate) fn body_input_row(&self) -> TypeRow {
        self.just_inputs.extend(self.rest.iter())
    }
}

impl DataflowParent for TailLoop {
    fn inner_signature(&self) -> Signature {
        Signature::new(self.body_input_row(), self.body_output_row())
            .with_extension_delta(self.extension_delta.clone())
    }
}

/// Conditional operation, defined by child `Case` nodes for each branch.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Conditional {
    /// The possible rows of the Sum input
    pub sum_rows: Vec<TypeRow>,
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

    fn signature(&self) -> Signature {
        let mut inputs = self.other_inputs.clone();
        inputs
            .to_mut()
            .insert(0, Type::new_sum(self.sum_rows.clone()));
        Signature::new(inputs, self.outputs.clone())
            .with_extension_delta(self.extension_delta.clone())
    }

    fn subst_mut(&mut self, subst: &Substitution) {
        for row in self.sum_rows.iter_mut() {
            *row = row.substitute(subst)
        }
        self.other_inputs = self.other_inputs.substitute(subst);
        self.outputs = self.outputs.substitute(subst);
        self.extension_delta = self.extension_delta.substitute(subst);
    }
}

impl Conditional {
    /// Build the input TypeRow of the nth child graph of a Conditional node.
    pub(crate) fn case_input_row(&self, case: usize) -> Option<TypeRow> {
        Some(self.sum_rows.get(case)?.extend(self.other_inputs.iter()))
    }
}

/// A dataflow node which is defined by a child CFG.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct CFG {
    pub signature: Signature,
}

impl_op_name!(CFG);

impl DataflowOpTrait for CFG {
    const TAG: OpTag = OpTag::Cfg;

    fn description(&self) -> &str {
        "A dataflow node defined by a child CFG"
    }

    fn signature(&self) -> Signature {
        self.signature.clone()
    }

    fn subst_mut(&mut self, subst: &Substitution) {
        self.signature = self.signature.substitute(subst);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
/// A CFG basic block node. The signature is that of the internal Dataflow graph.
#[allow(missing_docs)]
pub struct DataflowBlock {
    pub inputs: TypeRow,
    pub other_outputs: TypeRow,
    pub sum_rows: Vec<TypeRow>,
    pub extension_delta: ExtensionSet,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
/// The single exit node of the CFG. Has no children,
/// stores the types of the CFG node output.
pub struct ExitBlock {
    /// Output type row of the CFG.
    pub cfg_outputs: TypeRow,
}

impl NamedOp for DataflowBlock {
    fn name(&self) -> OpName {
        "DataflowBlock".into()
    }
}

impl NamedOp for ExitBlock {
    fn name(&self) -> OpName {
        "ExitBlock".into()
    }
}

impl StaticTag for DataflowBlock {
    const TAG: OpTag = OpTag::DataflowBlock;
}

impl StaticTag for ExitBlock {
    const TAG: OpTag = OpTag::BasicBlockExit;
}

impl DataflowParent for DataflowBlock {
    fn inner_signature(&self) -> Signature {
        // The node outputs a Sum before the data outputs of the block node
        let sum_type = Type::new_sum(self.sum_rows.clone());
        let mut node_outputs = vec![sum_type];
        node_outputs.extend_from_slice(&self.other_outputs);
        Signature::new(self.inputs.clone(), TypeRow::from(node_outputs))
            .with_extension_delta(self.extension_delta.clone())
    }
}

impl OpTrait for DataflowBlock {
    fn description(&self) -> &str {
        "A CFG basic block node"
    }
    /// Tag identifying the operation.
    fn tag(&self) -> OpTag {
        Self::TAG
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    fn extension_delta(&self) -> ExtensionSet {
        self.extension_delta.clone()
    }

    fn non_df_port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => 1,
            Direction::Outgoing => self.sum_rows.len(),
        }
    }

    fn subst_mut(&mut self, subst: &Substitution) {
        self.inputs = self.inputs.substitute(subst);
        self.other_outputs = self.other_outputs.substitute(subst);
        for r in self.sum_rows.iter_mut() {
            *r = r.substitute(subst);
        }
        self.extension_delta = self.extension_delta.substitute(subst);
    }
}

impl OpTrait for ExitBlock {
    fn description(&self) -> &str {
        "A CFG exit block node"
    }
    /// Tag identifying the operation.
    fn tag(&self) -> OpTag {
        Self::TAG
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::ControlFlow)
    }

    fn non_df_port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => 1,
            Direction::Outgoing => 0,
        }
    }

    fn subst_mut(&mut self, subst: &Substitution) {
        self.cfg_outputs = self.cfg_outputs.substitute(subst);
    }
}

/// Functionality shared by DataflowBlock and Exit CFG block types.
pub trait BasicBlock {
    /// The input dataflow signature of the CFG block.
    fn dataflow_input(&self) -> &TypeRow;
}

impl BasicBlock for DataflowBlock {
    fn dataflow_input(&self) -> &TypeRow {
        &self.inputs
    }
}
impl DataflowBlock {
    /// The correct inputs of any successors. Returns None if successor is not a
    /// valid index.
    pub fn successor_input(&self, successor: usize) -> Option<TypeRow> {
        Some(
            self.sum_rows
                .get(successor)?
                .extend(self.other_outputs.iter()),
        )
    }
}

impl BasicBlock for ExitBlock {
    fn dataflow_input(&self) -> &TypeRow {
        &self.cfg_outputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
/// Case ops - nodes valid inside Conditional nodes.
pub struct Case {
    /// The signature of the contained dataflow graph.
    pub signature: Signature,
}

impl_op_name!(Case);

impl StaticTag for Case {
    const TAG: OpTag = OpTag::Case;
}

impl DataflowParent for Case {
    fn inner_signature(&self) -> Signature {
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

    fn subst_mut(&mut self, subst: &Substitution) {
        self.signature = self.signature.substitute(subst);
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

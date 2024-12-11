//! Control flow operations.

use std::borrow::Cow;

use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::types::{EdgeKind, Signature, Type, TypeRow};
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

    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        let [inputs, outputs] =
            [&self.just_inputs, &self.just_outputs].map(|row| row.extend(self.rest.iter()));
        Cow::Owned(
            Signature::new(inputs, outputs).with_extension_delta(self.extension_delta.clone()),
        )
    }

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            just_inputs: self.just_inputs.substitute(subst, reg),
            just_outputs: self.just_outputs.substitute(subst, reg),
            rest: self.rest.substitute(subst, reg),
            extension_delta: self.extension_delta.substitute(subst),
        }
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
    fn inner_signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        Cow::Owned(
            Signature::new(self.body_input_row(), self.body_output_row())
                .with_extension_delta(self.extension_delta.clone()),
        )
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

    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        let mut inputs = self.other_inputs.clone();
        inputs
            .to_mut()
            .insert(0, Type::new_sum(self.sum_rows.clone()));
        Cow::Owned(
            Signature::new(inputs, self.outputs.clone())
                .with_extension_delta(self.extension_delta.clone()),
        )
    }

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            sum_rows: self.sum_rows.iter().map(|r| r.substitute(subst, reg)).collect(),
            other_inputs: self.other_inputs.substitute(subst, reg),
            outputs: self.outputs.substitute(subst, reg),
            extension_delta: self.extension_delta.substitute(subst),
        }
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

    fn signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.signature)
    }

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            signature: self.signature.substitute(subst, reg),
        }
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
    fn inner_signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        // The node outputs a Sum before the data outputs of the block node
        let sum_type = Type::new_sum(self.sum_rows.clone());
        let mut node_outputs = vec![sum_type];
        node_outputs.extend_from_slice(&self.other_outputs);
        Cow::Owned(
            Signature::new(self.inputs.clone(), TypeRow::from(node_outputs))
                .with_extension_delta(self.extension_delta.clone()),
        )
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

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            inputs: self.inputs.substitute(subst, reg),
            other_outputs: self.other_outputs.substitute(subst, reg),
            sum_rows: self.sum_rows.iter().map(|r| r.substitute(subst, reg)).collect(),
            extension_delta: self.extension_delta.substitute(subst),
        }
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

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            cfg_outputs: self.cfg_outputs.substitute(subst, reg),
        }
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
    fn inner_signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.signature)
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

    fn substitute(&self, subst: &crate::types::Substitution, reg: &ExtensionRegistry) -> Self {
        Self {
            signature: self.signature.substitute(subst, reg),
        }
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

#[cfg(test)]
mod test {
    use std::borrow::Borrow;

    use crate::{
        extension::{
            prelude::{qb_t, usize_t, PRELUDE_ID},
            ExtensionSet, PRELUDE_REGISTRY,
        },
        ops::{Conditional, DataflowOpTrait, DataflowParent},
        types::{Signature, Substitution, Type, TypeArg, TypeBound, TypeRV},
    };

    use super::{DataflowBlock, TailLoop};

    #[test]
    fn test_subst_dataflow_block() {
        use crate::ops::OpTrait;
        let tv0 = Type::new_var_use(0, TypeBound::Any);
        let dfb = DataflowBlock {
            inputs: vec![usize_t(), tv0.clone()].into(),
            other_outputs: vec![tv0.clone()].into(),
            sum_rows: vec![usize_t().into(), vec![qb_t(), tv0.clone()].into()],
            extension_delta: ExtensionSet::type_var(1),
        };
        let dfb2 = dfb.substitute(&Substitution::new(
            &[
                qb_t().into(),
                TypeArg::Extensions {
                    es: PRELUDE_ID.into(),
                },
            ]),
            &PRELUDE_REGISTRY,
        );
        let st = Type::new_sum(vec![vec![usize_t()], vec![qb_t(); 2]]);
        assert_eq!(
            dfb2.inner_signature(),
            Signature::new(vec![usize_t(), qb_t()], vec![st, qb_t()])
                .with_extension_delta(PRELUDE_ID)
        );
    }

    #[test]
    fn test_subst_conditional() {
        let tv1 = Type::new_var_use(1, TypeBound::Any);
        let cond = Conditional {
            sum_rows: vec![usize_t().into(), tv1.clone().into()],
            other_inputs: vec![Type::new_tuple(TypeRV::new_row_var_use(0, TypeBound::Any))].into(),
            outputs: vec![usize_t(), tv1].into(),
            extension_delta: ExtensionSet::new(),
        };
        let cond2 = cond.substitute(&Substitution::new(
            &[
                TypeArg::Sequence {
                    elems: vec![usize_t().into(); 3],
                },
                qb_t().into(),
            ]),
            &PRELUDE_REGISTRY,
        );
        let st = Type::new_sum(vec![usize_t(), qb_t()]); //both single-element variants
        assert_eq!(
            cond2.signature(),
            Signature::new(
                vec![st, Type::new_tuple(vec![usize_t(); 3])],
                vec![usize_t(), qb_t()]
            )
        );
    }

    #[test]
    fn test_tail_loop() {
        let tv0 = Type::new_var_use(0, TypeBound::Copyable);
        let tail_loop = TailLoop {
            just_inputs: vec![qb_t(), tv0.clone()].into(),
            just_outputs: vec![tv0.clone(), qb_t()].into(),
            rest: vec![tv0.clone()].into(),
            extension_delta: ExtensionSet::type_var(1),
        };
        let tail2 = tail_loop.substitute(&Substitution::new(
            &[
                usize_t().into(),
                TypeArg::Extensions {
                    es: PRELUDE_ID.into(),
                },
            ]),
            &PRELUDE_REGISTRY,
        );
        assert_eq!(
            tail2.signature(),
            ow r::new(
                vec![qb_t(), usize_t(), usize_t()],
                vec![usize_t(), qb_t(), usize_t()]
            )
            .with_extension_delta(PRELUDE_ID)
        );
    }
}

use crate::types::{Signature, TypeRow};

use crate::ops::{controlflow::ControlFlowOp, CaseOp, OpType};

use super::nodehandle::BuildHandle;
use super::{
    build_traits::Container,
    dataflow::{DFGBuilder, DFGWrapper},
    nodehandle::CaseID,
    BuildError, ConditionalID,
};

use crate::{hugr::HugrMut, Hugr};

use std::collections::HashSet;

use itertools::Itertools;
use portgraph::NodeIndex;
use thiserror::Error;

/// Builder for a [`CaseOp`] child graph.
pub type CaseBuilder<'b> = DFGWrapper<'b, BuildHandle<CaseID>>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConditionalBuildError {
    /// Case already built.
    #[error("Case {case} of Conditional node {conditional:?} has already been built.")]
    CaseBuilt { conditional: NodeIndex, case: usize },
    /// Case already built.
    #[error("Conditional node {conditional:?} has no case with index {case}.")]
    NotCase { conditional: NodeIndex, case: usize },
    /// Not all cases of Conditional built.
    #[error("Cases {cases:?} of Conditional node {conditional:?} have not been built.")]
    NotAllCasesBuilt {
        conditional: NodeIndex,
        cases: HashSet<usize>,
    },
}

/// Builder for a [`ControlFlowOp::Conditional`] node's children.
pub struct ConditionalBuilder<'f> {
    pub(super) base: &'f mut HugrMut,
    pub(super) conditional_node: NodeIndex,
    pub(super) n_out_wires: usize,
    pub(super) case_nodes: Vec<Option<NodeIndex>>,
}

impl<'f> Container for ConditionalBuilder<'f> {
    type ContainerHandle = Result<BuildHandle<ConditionalID>, ConditionalBuildError>;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.conditional_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.hugr()
    }

    fn finish(self) -> Self::ContainerHandle {
        let cases: HashSet<usize> = self
            .case_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| if node.is_none() { Some(i) } else { None })
            .collect();
        if !cases.is_empty() {
            return Err(ConditionalBuildError::NotAllCasesBuilt {
                conditional: self.conditional_node,
                cases,
            });
        }
        Ok((self.conditional_node, self.n_out_wires).into())
    }
}

impl<'f> ConditionalBuilder<'f> {
    /// Return a builder the Case node with index `case`.
    ///
    /// # Panics
    ///
    /// Panics if the parent node is not of type [`ControlFlowOp::Conditional`].
    ///
    /// # Errors
    ///
    /// This function will return an error if the case has already been built,
    /// `case` is not a valid index or if there is an error adding nodes.
    pub fn case_builder<'a: 'b, 'b>(
        &'a mut self,
        case: usize,
    ) -> Result<CaseBuilder<'b>, BuildError> {
        let conditional = self.conditional_node;
        let control_op: Result<ControlFlowOp, ()> = self
            .hugr()
            .get_optype(self.conditional_node)
            .clone()
            .try_into();

        let Ok(ControlFlowOp::Conditional {
            predicate_inputs,
            inputs,
            outputs,
        }) = control_op else {panic!("Parent node does not have Conditional optype.")};
        let sum_input = predicate_inputs
            .get(case)
            .ok_or(ConditionalBuildError::NotCase { conditional, case })?
            .clone();

        if self.case_nodes.get(case).unwrap().is_some() {
            return Err(ConditionalBuildError::CaseBuilt { conditional, case }.into());
        }

        let inputs: TypeRow = [vec![sum_input], inputs.iter().cloned().collect_vec()]
            .concat()
            .into();

        let bb_op = OpType::Case(CaseOp {
            signature: Signature::new_df(inputs.clone(), outputs.clone()),
        });
        let case_node =
            // add case before any existing subsequent cases
            if let Some(&sibling_node) = self.case_nodes[case + 1..].iter().flatten().next() {
                self.base().add_op_before(sibling_node, bb_op)?
            } else {
                self.add_child_op(bb_op)?
            };

        self.case_nodes[case] = Some(case_node);

        let dfg_builder = DFGBuilder::create_with_io(self.base(), case_node, inputs, outputs)?;

        Ok(CaseBuilder::new(dfg_builder))
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            module_builder::ModuleBuilder,
            test::{n_identity, NAT},
            Dataflow,
        },
        ops::ConstValue,
        type_row,
        types::SimpleType,
    };

    use super::*;

    #[test]
    fn basic_conditional() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder
                .declare("main", Signature::new_df(type_row![NAT], type_row![NAT]))?;
            let tru_const = module_builder.constant(ConstValue::predicate(1, 2))?;
            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;

                let const_wire = fbuild.load_const(&tru_const)?;
                let [int] = fbuild.input_wires_arr();
                let conditional_id = {
                    let predicate_inputs = vec![SimpleType::new_unit(); 2].into();
                    let other_inputs = vec![(NAT, int)];
                    let outputs = vec![SimpleType::new_unit(), NAT].into();
                    let mut conditional_b = fbuild.conditional_builder(
                        (predicate_inputs, const_wire),
                        other_inputs,
                        outputs,
                    )?;

                    n_identity(conditional_b.case_builder(0)?)?;
                    n_identity(conditional_b.case_builder(1)?)?;

                    conditional_b.finish()?
                };
                let [unit, int] = conditional_id.outputs_arr();
                fbuild.discard(unit)?;
                fbuild.finish_with_outputs([int])?
            };
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }
}

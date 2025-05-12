use crate::hugr::views::HugrView;
use crate::types::{Signature, TypeRow};

use crate::ops::handle::{CaseID, NodeHandle};
use crate::ops::{self};

use super::HugrBuilder;
use super::build_traits::SubContainer;
use super::handle::BuildHandle;
use super::{
    BuildError, ConditionalID,
    build_traits::Container,
    dataflow::{DFGBuilder, DFGWrapper},
};

use crate::Node;
use crate::{Hugr, hugr::HugrMut};

use std::collections::HashSet;

use thiserror::Error;

/// Builder for a [`ops::Case`] child graph.
pub type CaseBuilder<B> = DFGWrapper<B, BuildHandle<CaseID>>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ConditionalBuildError {
    /// Case already built.
    #[error("Case {case} of Conditional node {conditional} has already been built.")]
    CaseBuilt { conditional: Node, case: usize },
    /// Case already built.
    #[error("Conditional node {conditional} has no case with index {case}.")]
    NotCase { conditional: Node, case: usize },
    /// Not all cases of Conditional built.
    #[error("Cases {cases:?} of Conditional node {conditional} have not been built.")]
    NotAllCasesBuilt {
        conditional: Node,
        cases: HashSet<usize>,
    },
}

/// Builder for a [`ops::Conditional`] node's children.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalBuilder<T> {
    pub(super) base: T,
    pub(super) conditional_node: Node,
    pub(super) n_out_wires: usize,
    pub(super) case_nodes: Vec<Option<Node>>,
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> Container for ConditionalBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.conditional_node
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.base.as_mut()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.as_ref()
    }
}

impl<H: AsMut<Hugr> + AsRef<Hugr>> SubContainer for ConditionalBuilder<H> {
    type ContainerHandle = BuildHandle<ConditionalID>;

    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
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
            }
            .into());
        }
        Ok((self.conditional_node, self.n_out_wires).into())
    }
}
impl<B: AsMut<Hugr> + AsRef<Hugr>> ConditionalBuilder<B> {
    /// Return a builder the Case node with index `case`.
    ///
    /// # Panics
    ///
    /// Panics if the parent node is not of type [`ops::Conditional`].
    ///
    /// # Errors
    ///
    /// This function will return an error if the case has already been built,
    /// `case` is not a valid index or if there is an error adding nodes.
    pub fn case_builder(&mut self, case: usize) -> Result<CaseBuilder<&mut Hugr>, BuildError> {
        let conditional = self.conditional_node;
        let control_op = self.hugr().get_optype(self.conditional_node);

        let cond: ops::Conditional = control_op
            .clone()
            .try_into()
            .expect("Parent node does not have Conditional optype.");
        let inputs = cond
            .case_input_row(case)
            .ok_or(ConditionalBuildError::NotCase { conditional, case })?;

        if self.case_nodes.get(case).unwrap().is_some() {
            return Err(ConditionalBuildError::CaseBuilt { conditional, case }.into());
        }

        let outputs = cond.outputs;
        let case_op = ops::Case {
            signature: Signature::new(inputs.clone(), outputs.clone()),
        };
        let case_node =
            // add case before any existing subsequent cases
            if let Some(&sibling_node) = self.case_nodes[case + 1..].iter().flatten().next() {
                self.hugr_mut().add_node_before(sibling_node, case_op)
            } else {
                self.add_child_node(case_op)
            };

        self.case_nodes[case] = Some(case_node);

        let dfg_builder = DFGBuilder::create_with_io(
            self.hugr_mut(),
            case_node,
            Signature::new(inputs, outputs),
        )?;

        Ok(CaseBuilder::from_dfg_builder(dfg_builder))
    }
}

impl HugrBuilder for ConditionalBuilder<Hugr> {
    fn finish_hugr(self) -> Result<Hugr, crate::hugr::ValidationError<Node>> {
        self.base.validate()?;
        Ok(self.base)
    }
}

impl ConditionalBuilder<Hugr> {
    /// Initialize a Conditional rooted HUGR builder.
    pub fn new(
        sum_rows: impl IntoIterator<Item = TypeRow>,
        other_inputs: impl Into<TypeRow>,
        outputs: impl Into<TypeRow>,
    ) -> Result<Self, BuildError> {
        let sum_rows: Vec<_> = sum_rows.into_iter().collect();
        let other_inputs = other_inputs.into();
        let outputs: TypeRow = outputs.into();

        let n_out_wires = outputs.len();
        let n_cases = sum_rows.len();

        let op = ops::Conditional {
            sum_rows,
            other_inputs,
            outputs,
        };
        let base = Hugr::new_with_entrypoint(op).expect("Conditional entrypoint should be valid");
        let conditional_node = base.entrypoint();

        Ok(ConditionalBuilder {
            base,
            conditional_node,
            n_out_wires,
            case_nodes: vec![None; n_cases],
        })
    }
}

impl CaseBuilder<Hugr> {
    /// Initialize a Case rooted HUGR
    pub fn new(signature: Signature) -> Result<Self, BuildError> {
        // Start by building a conditional with a single case
        let mut conditional =
            ConditionalBuilder::new([signature.input.clone()], vec![], signature.output.clone())?;
        let case = conditional.case_builder(0)?.finish_sub_container()?.node();

        // Extract the half-finished hugr, and wrap it in an owned case builder
        let mut base = std::mem::take(conditional.hugr_mut());
        base.set_entrypoint(case);
        let dfg_builder = DFGBuilder::create(base, case)?;
        Ok(CaseBuilder::from_dfg_builder(dfg_builder))
    }
}
#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::builder::{DataflowSubContainer, ModuleBuilder};

    use crate::extension::prelude::usize_t;
    use crate::{
        builder::{Dataflow, test::n_identity},
        ops::Value,
        type_row,
    };

    use super::*;

    #[test]
    fn basic_conditional_case() -> Result<(), BuildError> {
        let case_b = CaseBuilder::new(Signature::new_endo(vec![usize_t(), usize_t()]))?;
        let [in0, in1] = case_b.input_wires_arr();
        case_b.finish_with_outputs([in0, in1])?;
        Ok(())
    }

    #[test]
    fn basic_conditional() -> Result<(), BuildError> {
        let mut conditional_b =
            ConditionalBuilder::new([type_row![], type_row![]], vec![usize_t()], vec![usize_t()])?;

        n_identity(conditional_b.case_builder(1)?)?;
        n_identity(conditional_b.case_builder(0)?)?;
        Ok(())
    }

    #[test]
    fn basic_conditional_module() -> Result<(), BuildError> {
        let build_result: Result<Hugr, BuildError> = {
            let mut module_builder = ModuleBuilder::new();
            let mut fbuild = module_builder
                .define_function("main", Signature::new(vec![usize_t()], vec![usize_t()]))?;
            let tru_const = fbuild.add_constant(Value::true_val());
            let _fdef = {
                let const_wire = fbuild.load_const(&tru_const);
                let [int] = fbuild.input_wires_arr();
                let conditional_id = {
                    let other_inputs = vec![(usize_t(), int)];
                    let outputs = vec![usize_t()].into();
                    let mut conditional_b = fbuild.conditional_builder(
                        ([type_row![], type_row![]], const_wire),
                        other_inputs,
                        outputs,
                    )?;

                    n_identity(conditional_b.case_builder(0)?)?;
                    n_identity(conditional_b.case_builder(1)?)?;

                    conditional_b.finish_sub_container()?
                };
                let [int] = conditional_id.outputs_arr();
                fbuild.finish_with_outputs([int])?
            };
            Ok(module_builder.finish_hugr()?)
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    fn test_not_all_cases() -> Result<(), BuildError> {
        let mut builder =
            ConditionalBuilder::new([type_row![], type_row![]], type_row![], type_row![])?;
        n_identity(builder.case_builder(0)?)?;
        assert_matches!(
            builder.finish_sub_container().map(|_| ()),
            Err(BuildError::ConditionalError(
                ConditionalBuildError::NotAllCasesBuilt { .. }
            ))
        );
        Ok(())
    }

    #[test]
    fn test_case_already_built() -> Result<(), BuildError> {
        let mut builder =
            ConditionalBuilder::new([type_row![], type_row![]], type_row![], type_row![])?;
        n_identity(builder.case_builder(0)?)?;
        assert_matches!(
            builder.case_builder(0).map(|_| ()),
            Err(BuildError::ConditionalError(
                ConditionalBuildError::CaseBuilt { .. }
            ))
        );
        Ok(())
    }
}

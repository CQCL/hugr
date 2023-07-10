use crate::hugr::view::HugrView;
use crate::types::{ClassicType, Signature, SimpleType, TypeRow};

use crate::ops;
use crate::ops::handle::CaseID;

use super::build_traits::SubContainer;
use super::handle::BuildHandle;
use super::HugrBuilder;
use super::{
    build_traits::Container,
    dataflow::{DFGBuilder, DFGWrapper},
    BuildError, ConditionalID,
};

use crate::Node;
use crate::{hugr::HugrMut, Hugr};

use std::collections::HashSet;

use thiserror::Error;

/// Builder for a [`ops::Case`] child graph.
pub type CaseBuilder<B> = DFGWrapper<B, BuildHandle<CaseID>>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConditionalBuildError {
    /// Case already built.
    #[error("Case {case} of Conditional node {conditional:?} has already been built.")]
    CaseBuilt { conditional: Node, case: usize },
    /// Case already built.
    #[error("Conditional node {conditional:?} has no case with index {case}.")]
    NotCase { conditional: Node, case: usize },
    /// Not all cases of Conditional built.
    #[error("Cases {cases:?} of Conditional node {conditional:?} have not been built.")]
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
            signature: Signature::new_df(inputs.clone(), outputs.clone()),
        };
        let case_node =
            // add case before any existing subsequent cases
            if let Some(&sibling_node) = self.case_nodes[case + 1..].iter().flatten().next() {
                self.hugr_mut().add_op_before(sibling_node, case_op)?
            } else {
                self.add_child_op(case_op)?
            };

        self.case_nodes[case] = Some(case_node);

        let dfg_builder = DFGBuilder::create_with_io(
            self.hugr_mut(),
            case_node,
            Signature::new_df(inputs, outputs),
        )?;

        Ok(CaseBuilder::from_dfg_builder(dfg_builder))
    }
}

impl HugrBuilder for ConditionalBuilder<Hugr> {
    fn finish_hugr(self) -> Result<Hugr, crate::hugr::ValidationError> {
        self.base.validate()?;
        Ok(self.base)
    }
}

impl ConditionalBuilder<Hugr> {
    /// Initialize a Conditional rooted HUGR builder
    pub fn new(
        predicate_inputs: impl IntoIterator<Item = TypeRow<ClassicType>>,
        other_inputs: impl Into<TypeRow<SimpleType>>,
        outputs: impl Into<TypeRow<SimpleType>>,
    ) -> Result<Self, BuildError> {
        let predicate_inputs: Vec<_> = predicate_inputs.into_iter().collect();
        let other_inputs = other_inputs.into();
        let outputs = outputs.into();

        let n_out_wires = outputs.len();
        let n_cases = predicate_inputs.len();

        let op = ops::Conditional {
            predicate_inputs,
            other_inputs,
            outputs,
        };
        let base = Hugr::new(op);
        let conditional_node = base.root();

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
    pub fn new(
        input: impl Into<TypeRow<SimpleType>>,
        output: impl Into<TypeRow<SimpleType>>,
    ) -> Result<Self, BuildError> {
        let input = input.into();
        let output = output.into();
        let signature = Signature::new_df(input, output);
        let op = ops::Case {
            signature: signature.clone(),
        };
        let base = Hugr::new(op);
        let root = base.root();
        let dfg_builder = DFGBuilder::create_with_io(base, root, signature)?;

        Ok(CaseBuilder::from_dfg_builder(dfg_builder))
    }
}
#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::builder::{DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::{
        builder::{
            test::{n_identity, NAT},
            Dataflow,
        },
        ops::ConstValue,
        type_row,
    };

    use super::*;

    #[test]
    fn basic_conditional() -> Result<(), BuildError> {
        let predicate_inputs = vec![type_row![]; 2];
        let mut conditional_b =
            ConditionalBuilder::new(predicate_inputs, type_row![NAT], type_row![NAT])?;

        n_identity(conditional_b.case_builder(0)?)?;
        n_identity(conditional_b.case_builder(1)?)?;

        Ok(())
    }

    #[test]
    fn basic_conditional_module() -> Result<(), BuildError> {
        let build_result: Result<Hugr, BuildError> = {
            let mut module_builder = ModuleBuilder::new();
            let mut fbuild = module_builder
                .define_function("main", Signature::new_df(type_row![NAT], type_row![NAT]))?;
            let tru_const = fbuild.add_constant(ConstValue::true_val())?;
            let _fdef = {
                let const_wire = fbuild.load_const(&tru_const)?;
                let [int] = fbuild.input_wires_arr();
                let conditional_id = {
                    let predicate_inputs = vec![type_row![]; 2];
                    let other_inputs = vec![(NAT, int)];
                    let outputs = vec![NAT].into();
                    let mut conditional_b = fbuild.conditional_builder(
                        (predicate_inputs, const_wire),
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
}

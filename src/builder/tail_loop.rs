use crate::hugr::HugrMut;
use crate::ops::controlflow::TailLoopSignature;
use crate::ops::{controlflow::ControlFlowOp, DataflowOp, OpType};

use crate::hugr::view::HugrView;
use crate::types::{Signature, TypeRow};
use crate::Node;

use super::build_traits::SubContainer;
use super::handle::BuildHandle;
use super::HugrMutRef;
use super::{
    dataflow::{DFGBuilder, DFGWrapper},
    BuildError, Container, Dataflow, TailLoopID, Wire,
};

/// Builder for a [`crate::ops::controlflow::ControlFlowOp::TailLoop`] node.
pub type TailLoopBuilder<B> = DFGWrapper<B, BuildHandle<TailLoopID>>;

impl<B: HugrMutRef> TailLoopBuilder<B> {
    pub(super) fn create_with_io(
        base: B,
        loop_node: Node,
        tail_loop_sig: &TailLoopSignature,
    ) -> Result<Self, BuildError> {
        let signature = Signature::new_df(
            tail_loop_sig.body_input_row(),
            tail_loop_sig.body_output_row(),
        );
        let dfg_build = DFGBuilder::create_with_io(base, loop_node, signature)?;

        Ok(TailLoopBuilder::from_dfg_builder(dfg_build))
    }
    /// Set the outputs of the [`ControlFlowOp::TailLoop`], with `out_variant` as the value of the
    /// termination predicate, and `rest` being the remaining outputs
    pub fn set_outputs(
        &mut self,
        out_variant: Wire,
        rest: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [out_variant].into_iter().chain(rest.into_iter()))
    }

    /// Get a reference to the [`crate::ops::controlflow::TailLoopSignature`]
    /// that defines the signature of the [`ControlFlowOp::TailLoop`]
    pub fn loop_signature(&self) -> Result<&TailLoopSignature, BuildError> {
        if let OpType::Dataflow(DataflowOp::ControlFlow {
            op: ControlFlowOp::TailLoop(tail_sig),
        }) = self.hugr().get_optype(self.container_node())
        {
            Ok(tail_sig)
        } else {
            Err(BuildError::UnexpectedType {
                node: self.container_node(),
                op_desc: "ControlFlowOp::TailLoop",
            })
        }
    }

    /// The output types of the child graph, including the predicate as the first.
    pub fn internal_output_row(&self) -> Result<TypeRow, BuildError> {
        self.loop_signature()
            .map(TailLoopSignature::body_output_row)
    }
}

impl TailLoopBuilder<&mut HugrMut> {
    /// Set outputs and finish, see [`TailLoopBuilder::set_outputs`]
    pub fn finish_with_outputs(
        mut self,
        out_variant: Wire,
        rest: impl IntoIterator<Item = Wire>,
    ) -> Result<<Self as SubContainer>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(out_variant, rest)?;
        self.finish_sub_container()
    }
}

impl TailLoopBuilder<HugrMut> {
    /// Initialize new builder for a [`ControlFlowOp::TailLoop`] rooted HUGR
    pub fn new(
        just_inputs: impl Into<TypeRow>,
        inputs_outputs: impl Into<TypeRow>,
        just_outputs: impl Into<TypeRow>,
    ) -> Result<Self, BuildError> {
        let tail_loop_sig = TailLoopSignature {
            just_inputs: just_inputs.into(),
            just_outputs: just_outputs.into(),
            rest: inputs_outputs.into(),
        };
        let op = ControlFlowOp::TailLoop(tail_loop_sig.clone());
        let base = HugrMut::new(op);
        let root = base.hugr().root();
        Self::create_with_io(base, root, &tail_loop_sig)
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{BIT, NAT},
            DataflowSubContainer, HugrBuilder, ModuleBuilder,
        },
        hugr::ValidationError,
        ops::ConstValue,
        type_row,
        types::Signature,
        Hugr,
    };

    use super::*;
    #[test]
    fn basic_loop() -> Result<(), BuildError> {
        let build_result: Result<Hugr, ValidationError> = {
            let mut loop_b = TailLoopBuilder::new(vec![], vec![BIT], type_row![NAT])?;
            let [i1] = loop_b.input_wires_arr();
            let const_wire = loop_b.add_load_const(ConstValue::i64(1))?;

            let break_wire = loop_b.make_break(loop_b.loop_signature()?.clone(), [const_wire])?;
            loop_b.set_outputs(break_wire, [i1])?;
            loop_b.finish_hugr()
        };

        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn loop_with_conditional() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder
                .declare("main", Signature::new_df(type_row![BIT], type_row![NAT]))?;

            let s2 = module_builder.add_constant(ConstValue::i64(2))?;
            let tru_const = module_builder.add_constant(ConstValue::true_val())?;

            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;
                let [b1] = fbuild.input_wires_arr();
                let loop_id = {
                    let mut loop_b =
                        fbuild.tail_loop_builder(vec![(BIT, b1)], vec![], type_row![NAT])?;
                    let signature = loop_b.loop_signature()?.clone();
                    let const_wire = loop_b.load_const(&tru_const)?;
                    let [b1] = loop_b.input_wires_arr();
                    let conditional_id = {
                        let predicate_inputs = vec![type_row![]; 2];
                        let output_row = loop_b.internal_output_row()?;
                        let mut conditional_b = loop_b.conditional_builder(
                            (predicate_inputs, const_wire),
                            vec![(BIT, b1)],
                            output_row,
                        )?;

                        let mut branch_0 = conditional_b.case_builder(0)?;
                        let [b1] = branch_0.input_wires_arr();

                        let continue_wire = branch_0.make_continue(signature.clone(), [b1])?;
                        branch_0.finish_with_outputs([continue_wire])?;

                        let mut branch_1 = conditional_b.case_builder(1)?;
                        let [_b1] = branch_1.input_wires_arr();

                        let wire = branch_1.load_const(&s2)?;
                        let break_wire = branch_1.make_break(signature, [wire])?;
                        branch_1.finish_with_outputs([break_wire])?;

                        conditional_b.finish_sub_container()?
                    };

                    loop_b.finish_with_outputs(conditional_id.out_wire(0), [])?
                };

                fbuild.finish_with_outputs(loop_id.outputs())?
            };
            module_builder.finish_hugr()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }
}

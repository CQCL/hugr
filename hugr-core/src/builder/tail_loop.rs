use crate::extension::{ExtensionSet, TO_BE_INFERRED};
use crate::ops::{self, DataflowOpTrait};

use crate::hugr::views::HugrView;
use crate::types::{Signature, TypeRow};
use crate::{Hugr, Node};

use super::handle::BuildHandle;
use super::{
    dataflow::{DFGBuilder, DFGWrapper},
    BuildError, Container, Dataflow, TailLoopID, Wire,
};

/// Builder for a [`ops::TailLoop`] node.
pub type TailLoopBuilder<B> = DFGWrapper<B, BuildHandle<TailLoopID>>;

impl<B: AsMut<Hugr> + AsRef<Hugr>> TailLoopBuilder<B> {
    pub(super) fn create_with_io(
        base: B,
        loop_node: Node,
        tail_loop: &ops::TailLoop,
    ) -> Result<Self, BuildError> {
        let signature = Signature::new(tail_loop.body_input_row(), tail_loop.body_output_row());
        let dfg_build = DFGBuilder::create_with_io(base, loop_node, signature)?;

        Ok(TailLoopBuilder::from_dfg_builder(dfg_build))
    }
    /// Set the outputs of the [`ops::TailLoop`], with `out_variant` as the value of the
    /// termination Sum, and `rest` being the remaining outputs.
    pub fn set_outputs(
        &mut self,
        out_variant: Wire,
        rest: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [out_variant].into_iter().chain(rest))
    }

    /// Get a reference to the [`ops::TailLoop`]
    /// that defines the signature of the [`ops::TailLoop`]
    pub fn loop_signature(&self) -> Result<&ops::TailLoop, BuildError> {
        self.hugr()
            .get_optype(self.container_node())
            .as_tail_loop()
            .ok_or(BuildError::UnexpectedType {
                node: self.container_node(),
                op_desc: "crate::ops::TailLoop",
            })
    }

    /// The output types of the child graph, including the Sum as the first.
    pub fn internal_output_row(&self) -> Result<TypeRow, BuildError> {
        self.loop_signature().map(ops::TailLoop::body_output_row)
    }

    /// Set outputs and finish, see [`TailLoopBuilder::set_outputs`]
    pub fn finish_with_outputs(
        mut self,
        out_variant: Wire,
        rest: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<TailLoopID>, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(out_variant, rest)?;
        Ok((
            self.container_node(),
            self.loop_signature()?.signature().output_count(),
        )
            .into())
    }
}

impl TailLoopBuilder<Hugr> {
    /// Initialize new builder for a [`ops::TailLoop`] rooted HUGR.
    /// Extension delta will be inferred.
    pub fn new(
        just_inputs: impl Into<TypeRow>,
        inputs_outputs: impl Into<TypeRow>,
        just_outputs: impl Into<TypeRow>,
    ) -> Result<Self, BuildError> {
        Self::new_exts(just_inputs, inputs_outputs, just_outputs, TO_BE_INFERRED)
    }

    /// Initialize new builder for a [`ops::TailLoop`] rooted HUGR.
    /// `extension_delta` is explicitly specified; alternatively, [new](Self::new)
    /// may be used to infer it.
    pub fn new_exts(
        just_inputs: impl Into<TypeRow>,
        inputs_outputs: impl Into<TypeRow>,
        just_outputs: impl Into<TypeRow>,
        extension_delta: impl Into<ExtensionSet>,
    ) -> Result<Self, BuildError> {
        let tail_loop = ops::TailLoop {
            just_inputs: just_inputs.into(),
            just_outputs: just_outputs.into(),
            rest: inputs_outputs.into(),
            extension_delta: extension_delta.into(),
        };
        let base = Hugr::new(tail_loop.clone());
        let root = base.root();
        Self::create_with_io(base, root, &tail_loop)
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{BIT, NAT},
            DataflowSubContainer, HugrBuilder, ModuleBuilder, SubContainer,
        },
        extension::prelude::{leaf::Lift, ConstUsize, PRELUDE_ID, USIZE_T},
        hugr::ValidationError,
        ops::Value,
        type_row,
        types::Signature,
    };

    use super::*;
    #[test]
    fn basic_loop() -> Result<(), BuildError> {
        let build_result: Result<Hugr, ValidationError> = {
            let mut loop_b =
                TailLoopBuilder::new_exts(vec![], vec![BIT], vec![USIZE_T], PRELUDE_ID)?;
            let [i1] = loop_b.input_wires_arr();
            let const_wire = loop_b.add_load_value(ConstUsize::new(1));

            let break_wire = loop_b.make_break(loop_b.loop_signature()?.clone(), [const_wire])?;
            loop_b.set_outputs(break_wire, [i1])?;
            loop_b.finish_prelude_hugr()
        };

        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn loop_with_conditional() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let mut fbuild = module_builder.define_function(
                "main",
                Signature::new(type_row![BIT], type_row![NAT]).with_prelude(),
            )?;
            let _fdef = {
                let [b1] = fbuild
                    .add_dataflow_op(Lift::new(type_row![BIT], PRELUDE_ID), fbuild.input_wires())?
                    .outputs_arr();
                let loop_id = {
                    let mut loop_b =
                        fbuild.tail_loop_builder(vec![(BIT, b1)], vec![], type_row![NAT])?;
                    let signature = loop_b.loop_signature()?.clone();
                    let const_val = Value::true_val();
                    let const_wire = loop_b.add_load_const(Value::true_val());
                    let lift_node = loop_b.add_dataflow_op(
                        Lift::new(vec![const_val.get_type().clone()].into(), PRELUDE_ID),
                        [const_wire],
                    )?;
                    let [const_wire] = lift_node.outputs_arr();
                    let [b1] = loop_b.input_wires_arr();
                    let conditional_id = {
                        let output_row = loop_b.internal_output_row()?;
                        let mut conditional_b = loop_b.conditional_builder(
                            ([type_row![], type_row![]], const_wire),
                            vec![(BIT, b1)],
                            output_row,
                        )?;

                        let mut branch_0 = conditional_b.case_builder(0)?;
                        let [b1] = branch_0.input_wires_arr();

                        let continue_wire = branch_0.make_continue(signature.clone(), [b1])?;
                        branch_0.finish_with_outputs([continue_wire])?;

                        let mut branch_1 = conditional_b.case_builder(1)?;
                        let [_b1] = branch_1.input_wires_arr();

                        let wire = branch_1.add_load_value(ConstUsize::new(2));
                        let break_wire = branch_1.make_break(signature, [wire])?;
                        branch_1.finish_with_outputs([break_wire])?;

                        conditional_b.finish_sub_container()?
                    };
                    loop_b.finish_with_outputs(conditional_id.out_wire(0), [])?
                };
                fbuild.finish_with_outputs(loop_id.outputs())?
            };
            module_builder.finish_prelude_hugr()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    // fixed: issue 1257: When building a TailLoop, calling outputs_arr, you are given an OrderEdge "output wire"
    fn tailloop_output_arr() {
        let mut builder = TailLoopBuilder::new(type_row![], type_row![], type_row![]).unwrap();
        let control = builder.add_load_value(Value::false_val());
        let tailloop = builder.finish_with_outputs(control, []).unwrap();
        let [] = tailloop.outputs_arr();
    }
}

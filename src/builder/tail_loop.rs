use crate::ops::{controlflow::ControlFlowOp, DataflowOp, OpType};

use crate::types::{Signature, SimpleType, TypeRow};

use super::{
    dataflow::{DFGBuilder, DFGWrapper},
    BuildError, Container, Dataflow, TailLoopID, Wire,
};

use portgraph::NodeIndex;

use crate::hugr::HugrMut;

/// Builder for a [`crate::ops::controlflow::ControlFlowOp::TailLoop`] node.
pub type TailLoopBuilder<'b> = DFGWrapper<'b, TailLoopID>;

impl<'b> TailLoopBuilder<'b> {
    pub(super) fn create_with_io(
        base: &'b mut HugrMut,
        loop_node: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, BuildError> {
        let dfg_build = DFGBuilder::create_with_io(
            base,
            loop_node,
            inputs.clone(),
            loop_output_row(inputs, outputs),
        )?;

        Ok(TailLoopBuilder::new(dfg_build))
    }
    pub fn set_outputs(&mut self, out_variant: Wire) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [out_variant])
    }

    pub fn finish_with_outputs(
        mut self,
        out_variant: Wire,
    ) -> Result<<TailLoopBuilder<'b> as Container>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(out_variant)?;
        Ok(self.finish())
    }

    pub fn loop_signature(&self) -> Result<Signature, BuildError> {
        let hugr = self.hugr();

        if let OpType::Dataflow(DataflowOp::ControlFlow {
            op: ControlFlowOp::TailLoop { inputs, outputs },
        }) = hugr.get_optype(self.container_node())
        {
            Ok(Signature::new_df(inputs.clone(), outputs.clone()))
        } else {
            Err(BuildError::UnexpectedType {
                node: self.container_node(),
                op_desc: "ControlFlowOp::TailLoop",
            })
        }
    }

    pub fn internal_output_row(&self) -> Result<TypeRow, BuildError> {
        let Signature { input, output, .. } = self.loop_signature()?;

        Ok(loop_output_row(input, output))
    }
}

/// Build the output TypeRow of the child graph of a TailLoop node.
pub(super) fn loop_output_row(input: TypeRow, output: TypeRow) -> TypeRow {
    vec![SimpleType::new_sum(loop_sum_variants(input, output))].into()
}

/// Build the row of variants for the single Sum output of a TailLoop child graph.
pub(super) fn loop_sum_variants(input: TypeRow, output: TypeRow) -> TypeRow {
    vec![SimpleType::new_tuple(input), SimpleType::new_tuple(output)].into()
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            module_builder::ModuleBuilder,
            test::{BIT, NAT},
            BuildHandle, ConditionalID,
        },
        ops::ConstValue,
        type_row,
    };

    use super::*;
    #[test]
    fn basic_loop() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main =
                module_builder.declare("main", Signature::new_df(type_row![], type_row![NAT]))?;
            let s1 = module_builder.constant(ConstValue::Int(1))?;
            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;

                let loop_id: TailLoopID = {
                    let mut loop_b = fbuild.tail_loop_builder(vec![], type_row![NAT])?;

                    let const_wire = loop_b.load_const(&s1)?;

                    let break_wire = loop_b.make_break(loop_b.loop_signature()?, [const_wire])?;

                    loop_b.finish_with_outputs(break_wire)?
                };

                fbuild.finish_with_outputs(loop_id.outputs())?
            };
            module_builder.finish()
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

            let s2 = module_builder.constant(ConstValue::Int(2))?;
            let tru_const = module_builder.constant(ConstValue::predicate(1, 2))?;

            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;
                let [b1] = fbuild.input_wires_arr();
                let loop_id: TailLoopID = {
                    let mut loop_b = fbuild.tail_loop_builder(vec![(BIT, b1)], type_row![NAT])?;
                    let signature = loop_b.loop_signature()?;
                    let const_wire = loop_b.load_const(&tru_const)?;
                    let [b1] = loop_b.input_wires_arr();
                    let conditional_id: ConditionalID = {
                        let predicate_inputs = vec![SimpleType::new_unit(); 2].into();
                        let output_row = loop_b.internal_output_row()?;
                        let mut conditional_b = loop_b.conditional_builder(
                            (predicate_inputs, const_wire),
                            vec![(BIT, b1)],
                            output_row,
                        )?;

                        let mut branch_0 = conditional_b.case_builder(0)?;
                        let [pred, b1] = branch_0.input_wires_arr();
                        branch_0.discard(pred)?;

                        let continue_wire = branch_0.make_continue(signature.clone(), [b1])?;
                        branch_0.finish_with_outputs([continue_wire])?;

                        let mut branch_1 = conditional_b.case_builder(1)?;
                        let [pred, b1] = branch_1.input_wires_arr();

                        branch_1.discard(pred)?;
                        branch_1.discard(b1)?;

                        let wire = branch_1.load_const(&s2)?;
                        let break_wire = branch_1.make_break(signature, [wire])?;
                        branch_1.finish_with_outputs([break_wire])?;

                        conditional_b.finish()?
                    };

                    loop_b.finish_with_outputs(conditional_id.out_wire(0))?
                };

                fbuild.finish_with_outputs(loop_id.outputs())?
            };
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }
}

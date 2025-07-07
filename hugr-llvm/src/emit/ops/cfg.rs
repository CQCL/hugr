use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use hugr_core::{
    HugrView, Node, NodeIndex,
    ops::{CFG, DataflowBlock, ExitBlock, OpType},
    types::SumType,
};
use inkwell::{basic_block::BasicBlock, values::BasicValueEnum};
use itertools::Itertools as _;

use crate::{
    emit::{
        EmitOpArgs,
        func::{EmitFuncContext, RowMailBox, RowPromise},
    },
    sum::LLVMSumValue,
    utils::fat::FatNode,
};

use super::emit_dataflow_parent;

pub struct CfgEmitter<'c, 'hugr, H> {
    bbs: BTreeMap<FatNode<'hugr, OpType, H>, (BasicBlock<'c>, RowMailBox<'c>)>,
    inputs: Option<Vec<BasicValueEnum<'c>>>,
    outputs: Option<RowPromise<'c>>,
    node: FatNode<'hugr, CFG, H>,
    entry_node: FatNode<'hugr, DataflowBlock, H>,
    exit_node: FatNode<'hugr, ExitBlock, H>,
}

impl<'c, 'hugr, H: HugrView<Node = Node>> CfgEmitter<'c, 'hugr, H> {
    // Constructs a new CfgEmitter. Creates a basic block for each of
    // the children in the llvm function. Note that this does not move the
    // position of the builder.
    pub fn new<'d>(
        context: &'d mut EmitFuncContext<'c, '_, H>,
        args: EmitOpArgs<'c, 'hugr, CFG, H>,
    ) -> Result<Self>
    where
        'c: 'd,
    {
        let node = args.node();
        let (inputs, outputs) = (Some(args.inputs), Some(args.outputs));

        // create this now so that it will be the last block and we can use it
        // to crate the other blocks immediately before it. This is just for
        // nice block ordering.
        let exit_block = context.new_basic_block("", None);
        let mut bbs = BTreeMap::new();
        for child in node.children() {
            if child.is_exit_block() {
                let output_row = {
                    let out_types = node.out_value_types().map(|x| x.1).collect_vec();
                    context.new_row_mail_box(out_types.iter(), "")?
                };
                bbs.insert(child, (exit_block, output_row));
            } else if child.is_dataflow_block() {
                let bb = context.new_basic_block("", Some(exit_block));
                let (i, _) = child.get_io().unwrap();
                bbs.insert(child, (bb, context.node_outs_rmb(i)?));
            }
        }
        let (entry_node, exit_node) = node.get_entry_exit();
        Ok(CfgEmitter {
            bbs,
            inputs,
            outputs,
            node,
            entry_node,
            exit_node,
        })
    }

    fn take_inputs(&mut self) -> Result<Vec<BasicValueEnum<'c>>> {
        self.inputs.take().ok_or(anyhow!("Couldn't take inputs"))
    }

    fn take_outputs(&mut self) -> Result<RowPromise<'c>> {
        self.outputs.take().ok_or(anyhow!("Couldn't take inputs"))
    }

    fn get_block_data<OT: 'hugr>(
        &self,
        node: &FatNode<'hugr, OT, H>,
    ) -> Result<(BasicBlock<'c>, RowMailBox<'c>)>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        self.bbs
            .get(&node.generalise())
            .ok_or(anyhow!("Couldn't get block data for: {}", node.index()))
            .cloned()
    }

    /// Consume the emitter by emitting each child of the node.
    /// After returning the builder will be at the end of the exit block.
    pub fn emit_children(mut self, context: &mut EmitFuncContext<'c, '_, H>) -> Result<()> {
        // write the inputs of the cfg node into the inputs of the entry
        // dataflowblock node, and then branch to the basic block of that entry
        // node.
        let inputs = self.take_inputs()?;
        let (entry_bb, inputs_rmb) = self.get_block_data(&self.entry_node)?;
        let builder = context.builder();
        inputs_rmb.write(builder, inputs)?;
        builder.build_unconditional_branch(entry_bb)?;

        // emit each child by delegating to the `impl EmitOp<_>` of self.
        for child_node in self.node.children() {
            let (inputs, outputs) = (vec![], RowMailBox::new_empty().promise());
            match child_node.as_ref() {
                OpType::DataflowBlock(dfb) => self.emit_dataflow_block(
                    context,
                    EmitOpArgs {
                        node: child_node.into_ot(dfb),
                        inputs,
                        outputs,
                    },
                ),
                OpType::ExitBlock(eb) => self.emit_exit_block(
                    context,
                    EmitOpArgs {
                        node: child_node.into_ot(eb),
                        inputs,
                        outputs,
                    },
                ),

                // Const is allowed, but requires no work here. FuncDecl is
                // technically not allowed, but there is no harm in allowing it.
                OpType::Const(_) => Ok(()),
                OpType::FuncDecl(_) => Ok(()),
                OpType::FuncDefn(fd) => {
                    context.push_todo_func(child_node.into_ot(fd));
                    Ok(())
                }

                ot => Err(anyhow!("unknown optype: {ot:?}")),
            }?;
        }

        // move the builder to the end of the exit block
        let (exit_bb, _) = self.get_block_data(&self.exit_node)?;
        context.builder().position_at_end(exit_bb);
        Ok(())
    }
    fn emit_dataflow_block(
        &mut self,
        context: &mut EmitFuncContext<'c, '_, H>,
        EmitOpArgs {
            node,
            inputs: _,
            outputs: _,
        }: EmitOpArgs<'c, 'hugr, DataflowBlock, H>,
    ) -> Result<()> {
        // our entry basic block and our input RowMailBox
        let (bb, inputs_rmb) = self.get_block_data(&node)?;
        // the basic block and mailbox of each of our successors
        let successor_data = node
            .output_neighbours()
            .map(|succ| self.get_block_data(&succ))
            .collect::<Result<Vec<_>>>()?;

        context.build_positioned(bb, |context| {
            let (_, o) = node.get_io().unwrap();
            // get the rowmailbox for our output node
            let outputs_rmb = context.node_ins_rmb(o)?;
            // read the values from our input node
            let inputs = inputs_rmb.read_vec(context.builder(), [])?;

            // emit all our children and read the values from the rowmailbox of our output node
            emit_dataflow_parent(
                context,
                EmitOpArgs {
                    node,
                    inputs,
                    outputs: outputs_rmb.promise(),
                },
            )?;
            let outputs = outputs_rmb.read_vec(context.builder(), [])?;

            // We create a helper block per-tag. We switch to the helper block,
            // where we then store the input args for the successor block, then
            // unconditionally branch.
            //
            // We use switch even when we have 1 or 2 successors, where we could
            // use unconditional branch or conditional branch, to simplify the
            // code here at the expense of messier generated code. We expect the
            // simplify-cfg pass to clean this up without issue.
            let branch_sum_type = SumType::new(node.sum_rows.clone());
            let sum_input =
                LLVMSumValue::try_new(outputs[0], context.llvm_sum_type(branch_sum_type)?)?;

            sum_input.build_destructure(context.builder(), |builder, tag, mut values| {
                let (target_bb, target_rmb) = &successor_data[tag];
                values.extend(&outputs[1..]);
                target_rmb.write(builder, values)?;
                builder.build_unconditional_branch(*target_bb)?;
                Ok(())
            })
        })
    }

    fn emit_exit_block(
        &mut self,
        context: &mut EmitFuncContext<'c, '_, H>,
        args: EmitOpArgs<'c, 'hugr, ExitBlock, H>,
    ) -> Result<()> {
        let outputs = self.take_outputs()?;
        let (bb, inputs_rmb) = self.get_block_data(&args.node())?;
        context.build_positioned(bb, |context| {
            let builder = context.builder();
            outputs.finish(builder, inputs_rmb.read_vec(builder, [])?)
        })
    }
}

#[cfg(test)]
mod test {
    use hugr_core::builder::{Dataflow, DataflowHugr, SubContainer};
    use hugr_core::extension::ExtensionRegistry;
    use hugr_core::extension::prelude::{self, bool_t};
    use hugr_core::ops::Value;
    use hugr_core::std_extensions::arithmetic::int_types::{self, INT_TYPES};
    use hugr_core::type_row;

    use itertools::Itertools as _;
    use rstest::rstest;

    use crate::custom::CodegenExtsBuilder;
    use crate::emit::test::SimpleHugrConfig;
    use crate::test::{TestContext, llvm_ctx};

    use crate::check_emission;
    use crate::types::HugrType;

    #[rstest]
    fn diverse_outputs(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);
        let t1 = INT_TYPES[0].clone();
        let t2 = INT_TYPES[1].clone();
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![t1.clone(), t2.clone()])
            .with_outs(t2.clone())
            .with_extensions(ExtensionRegistry::new([
                int_types::EXTENSION.to_owned(),
                prelude::PRELUDE.to_owned(),
            ]))
            .finish(|mut builder| {
                let [in1, in2] = builder.input_wires_arr();
                let mut cfg_builder = builder
                    .cfg_builder([(t1.clone(), in1), (t2.clone(), in2)], t2.clone().into())
                    .unwrap();

                // entry block takes (t1,t2) and unconditionally branches to b1 with no other outputs
                let mut entry_builder = cfg_builder
                    .entry_builder([vec![t1.clone(), t2.clone()].into()], type_row![])
                    .unwrap();
                let [entry_in1, entry_in2] = entry_builder.input_wires_arr();
                let r = entry_builder.make_tuple([entry_in1, entry_in2]).unwrap();
                let entry_block = entry_builder.finish_with_outputs(r, []).unwrap();

                // b1 takes (t1,t2) and branches to either entry or exit, with sum type [(t1) + ()] and other outputs [t2]
                let variants = vec![t1.clone().into(), type_row![]];
                let mut b1_builder = cfg_builder
                    .block_builder(
                        vec![t1.clone(), t2.clone()].into(),
                        variants.clone(),
                        t2.clone().into(),
                    )
                    .unwrap();
                let [b1_in1, b1_in2] = b1_builder.input_wires_arr();
                let r = b1_builder.make_sum(0, variants, [b1_in1]).unwrap();
                let b1 = b1_builder.finish_with_outputs(r, [b1_in2]).unwrap();

                let exit_block = cfg_builder.exit_block();
                cfg_builder.branch(&entry_block, 0, &b1).unwrap();
                cfg_builder.branch(&b1, 0, &entry_block).unwrap();
                cfg_builder.branch(&b1, 1, &exit_block).unwrap();
                let cfg = cfg_builder.finish_sub_container().unwrap();
                let [cfg_out] = cfg.outputs_arr();
                builder.finish_hugr_with_outputs([cfg_out]).unwrap()
            });
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn nested(llvm_ctx: TestContext) {
        let t1 = HugrType::new_unit_sum(3);
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![t1.clone(), bool_t()])
            .with_outs(bool_t())
            .finish(|mut builder| {
                let [in1, in2] = builder.input_wires_arr();
                let unit_val = builder.add_load_value(Value::unit());
                let [outer_cfg_out] = {
                    let mut outer_cfg_builder = builder
                        .cfg_builder([(t1.clone(), in1), (bool_t(), in2)], bool_t().into())
                        .unwrap();

                    let outer_entry_block = {
                        let mut outer_entry_builder = outer_cfg_builder
                            .entry_builder([type_row![], type_row![]], type_row![])
                            .unwrap();
                        let [outer_entry_in1, outer_entry_in2] =
                            outer_entry_builder.input_wires_arr();
                        let [outer_entry_out] = {
                            let mut inner_cfg_builder = outer_entry_builder
                                .cfg_builder([], bool_t().into())
                                .unwrap();
                            let inner_exit_block = inner_cfg_builder.exit_block();
                            let inner_entry_block = {
                                let inner_entry_builder = inner_cfg_builder
                                    .entry_builder(
                                        [type_row![], type_row![], type_row![]],
                                        type_row![],
                                    )
                                    .unwrap();
                                // non-local edge
                                inner_entry_builder
                                    .finish_with_outputs(outer_entry_in1, [])
                                    .unwrap()
                            };
                            let [b1, b2, b3] = (0..3)
                                .map(|i| {
                                    let mut b_builder = inner_cfg_builder
                                        .block_builder(
                                            type_row![],
                                            vec![type_row![]],
                                            bool_t().into(),
                                        )
                                        .unwrap();
                                    let output = match i {
                                        0 => b_builder.add_load_value(Value::true_val()),
                                        1 => b_builder.add_load_value(Value::false_val()),
                                        2 => outer_entry_in2,
                                        _ => unreachable!(),
                                    };
                                    b_builder.finish_with_outputs(unit_val, [output]).unwrap()
                                })
                                .collect_vec()
                                .try_into()
                                .unwrap();
                            inner_cfg_builder
                                .branch(&inner_entry_block, 0, &b1)
                                .unwrap();
                            inner_cfg_builder
                                .branch(&inner_entry_block, 1, &b2)
                                .unwrap();
                            inner_cfg_builder
                                .branch(&inner_entry_block, 2, &b3)
                                .unwrap();
                            inner_cfg_builder.branch(&b1, 0, &inner_exit_block).unwrap();
                            inner_cfg_builder.branch(&b2, 0, &inner_exit_block).unwrap();
                            inner_cfg_builder.branch(&b3, 0, &inner_exit_block).unwrap();
                            inner_cfg_builder
                                .finish_sub_container()
                                .unwrap()
                                .outputs_arr()
                        };

                        outer_entry_builder
                            .finish_with_outputs(outer_entry_out, [])
                            .unwrap()
                    };

                    let [b1, b2] = (0..2)
                        .map(|i| {
                            let mut b_builder = outer_cfg_builder
                                .block_builder(type_row![], vec![type_row![]], bool_t().into())
                                .unwrap();
                            let output = match i {
                                0 => b_builder.add_load_value(Value::true_val()),
                                1 => b_builder.add_load_value(Value::false_val()),
                                _ => unreachable!(),
                            };
                            b_builder.finish_with_outputs(unit_val, [output]).unwrap()
                        })
                        .collect_vec()
                        .try_into()
                        .unwrap();

                    let exit_block = outer_cfg_builder.exit_block();
                    outer_cfg_builder
                        .branch(&outer_entry_block, 0, &b1)
                        .unwrap();
                    outer_cfg_builder
                        .branch(&outer_entry_block, 1, &b2)
                        .unwrap();
                    outer_cfg_builder.branch(&b1, 0, &exit_block).unwrap();
                    outer_cfg_builder.branch(&b2, 0, &exit_block).unwrap();
                    outer_cfg_builder
                        .finish_sub_container()
                        .unwrap()
                        .outputs_arr()
                };
                builder.finish_hugr_with_outputs([outer_cfg_out]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }
}

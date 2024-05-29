use anyhow::{anyhow, Result};
use hugr::{
    hugr::views::SiblingGraph,
    ops::{
        Call, Case, Conditional, Const, Input, LoadConstant, MakeTuple, NamedOp, OpTag, OpTrait,
        OpType, Output, Tag, UnpackTuple, Value,
    },
    types::{SumType, Type, TypeEnum},
    HugrView, NodeIndex,
};
use inkwell::{builder::Builder, types::BasicType, values::BasicValueEnum};
use itertools::Itertools;
use petgraph::visit::Walker;

use crate::fat::FatExt as _;
use crate::{fat::FatNode, types::LLVMSumType};

use super::{
    func::{EmitFuncContext, RowPromise},
    EmitOp, EmitOpArgs,
};

struct SumOpEmitter<'c, 'd, H: HugrView>(&'d mut EmitFuncContext<'c, H>, LLVMSumType<'c>);

impl<'c, 'd, H: HugrView> SumOpEmitter<'c, 'd, H> {
    pub fn new(context: &'d mut EmitFuncContext<'c, H>, st: LLVMSumType<'c>) -> Self {
        Self(context, st)
    }
    pub fn try_new(
        context: &'d mut EmitFuncContext<'c, H>,
        ts: impl IntoIterator<Item = Type>,
    ) -> Result<Self> {
        let llvm_sum_type = context.llvm_sum_type(get_exactly_one_sum_type(ts)?)?;
        Ok(Self::new(context, llvm_sum_type))
    }
}

impl<'c, H: HugrView> EmitOp<'c, MakeTuple, H> for SumOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, MakeTuple, H>) -> Result<()> {
        let builder = self.0.builder();
        args.outputs
            .finish(builder, [self.1.build_tag(builder, 0, args.inputs)?])
    }
}

impl<'c, H: HugrView> EmitOp<'c, UnpackTuple, H> for SumOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, UnpackTuple, H>) -> Result<()> {
        let builder = self.0.builder();
        let input = args
            .inputs
            .into_iter()
            .exactly_one()
            .map_err(|_| anyhow!("unpacktuple expected exactly one input"))?;
        args.outputs
            .finish(builder, self.1.build_untag(builder, 0, input)?)
    }
}

impl<'c, H: HugrView> EmitOp<'c, Tag, H> for SumOpEmitter<'c, '_, H> {
    fn emit(&mut self, args: EmitOpArgs<'c, Tag, H>) -> Result<()> {
        let builder = self.0.builder();
        args.outputs.finish(
            builder,
            [self
                .1
                .build_tag(builder, args.node.tag as u32, args.inputs)?],
        )
    }
}

struct DataflowParentEmitter<'c, 'd, OT, H: HugrView> {
    context: &'d mut EmitFuncContext<'c, H>,
    node: FatNode<'c, OT, H>,
    inputs: Option<Vec<BasicValueEnum<'c>>>,
    outputs: Option<RowPromise<'c>>,
}

impl<'c, 'd, OT: OpTrait + 'c, H: HugrView> DataflowParentEmitter<'c, 'd, OT, H>
where
    &'c OpType: TryInto<&'c OT>,
    // &'c OpType: TryInto<&'c OT>,
    // <&'c OpType as TryInto<&'c OT>>::Error: std::fmt::Debug,
{
    pub fn new(context: &'d mut EmitFuncContext<'c, H>, args: EmitOpArgs<'c, OT, H>) -> Self {
        Self {
            context,
            node: args.node,
            inputs: Some(args.inputs),
            outputs: Some(args.outputs),
        }
    }

    /// safe because we are guarenteed only one input or output node
    fn take_input(&mut self) -> Result<Vec<BasicValueEnum<'c>>> {
        self.inputs
            .take()
            .ok_or(anyhow!("DataflowParentEmitter: Input taken twice"))
    }

    fn take_output(&mut self) -> Result<RowPromise<'c>> {
        self.outputs
            .take()
            .ok_or(anyhow!("DataflowParentEmitter: Output taken twice"))
    }

    pub fn builder(&mut self) -> &Builder<'c> {
        self.context.builder()
    }

    pub fn emit_children(mut self) -> Result<()> {
        use hugr::hugr::views::HierarchyView;
        use petgraph::visit::Topo;
        let node = self.node.clone();
        if !OpTag::DataflowParent.is_superset(OpTrait::tag(node.get())) {
            Err(anyhow!("Not a dataflow parent"))?
        };

        let (i, o): (FatNode<Input, H>, FatNode<Output, H>) = node
            .get_io()
            .ok_or(anyhow!("emit_dataflow_parent: no io nodes"))?;
        debug_assert!(i.out_value_types().count() == self.inputs.as_ref().unwrap().len());
        debug_assert!(o.in_value_types().count() == self.outputs.as_ref().unwrap().len());

        let region: SiblingGraph = SiblingGraph::try_new(node.hugr(), node.node()).unwrap();
        Topo::new(&region.as_petgraph())
            .iter(&region.as_petgraph())
            .filter(|x| (*x != node.node()))
            .map(|x| node.hugr().fat_optype(x))
            .try_for_each(|node| {
                let inputs_rmb = self.context.node_ins_rmb(node.clone())?;
                let inputs = inputs_rmb.read(self.builder(), [])?;
                let outputs = self.context.node_outs_rmb(node.clone())?.promise();
                self.emit(EmitOpArgs {
                    node,
                    inputs,
                    outputs,
                })
            })
    }
}

impl<'c, OT: OpTrait + 'c, H: HugrView> EmitOp<'c, OpType, H>
    for DataflowParentEmitter<'c, '_, OT, H>
where
    &'c OpType: TryInto<&'c OT>,
{
    fn emit(&mut self, args: EmitOpArgs<'c, OpType, H>) -> Result<()> {
        if !OpTag::DataflowChild.is_superset(args.node().tag()) {
            Err(anyhow!("Not a dataflow child"))?
        };

        match args.node().get() {
            OpType::Input(_) => {
                let i = self.take_input()?;
                args.outputs.finish(self.builder(), i)
            }
            OpType::Output(_) => {
                let o = self.take_output()?;
                o.finish(self.builder(), args.inputs)
            }
            _ => emit_optype(self.context, args),
        }
    }
}

struct ConditionalEmitter<'c, 'd, H: HugrView>(&'d mut EmitFuncContext<'c, H>);

impl<'c, H: HugrView> EmitOp<'c, Conditional, H> for ConditionalEmitter<'c, '_, H> {
    fn emit(
        &mut self,
        EmitOpArgs {
            node,
            inputs,
            outputs,
        }: EmitOpArgs<'c, Conditional, H>,
    ) -> Result<()> {
        let context = &mut self.0;
        let exit_rmb = context
            .new_row_mail_box(node.dataflow_signature().unwrap().output.iter(), "exit_rmb")?;
        let exit_block = context.build_positioned_new_block(
            format!("cond_exit_{}", node.node().index()),
            None,
            |context, bb| {
                let builder = context.builder();
                outputs.finish(builder, exit_rmb.read_vec(builder, [])?)?;
                Ok::<_, anyhow::Error>(bb)
            },
        )?;

        let case_values_rmbs_blocks = node
            .children()
            .enumerate()
            .map(|(i, n)| {
                let label = format!("cond_{}_case_{}", node.node().index(), i);
                let node = n.try_into_ot::<Case>().ok_or(anyhow!("not a case node"))?;
                let rmb =
                    context.new_row_mail_box(node.get_io().unwrap().0.types.iter(), &label)?;
                context.build_positioned_new_block(&label, Some(exit_block), |context, bb| {
                    let inputs = rmb.read_vec(context.builder(), [])?;
                    emit_dataflow_parent(
                        context,
                        EmitOpArgs {
                            node,
                            inputs,
                            outputs: exit_rmb.promise(),
                        },
                    )?;
                    context.builder().build_unconditional_branch(exit_block)?;
                    Ok((i, rmb, bb))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let sum_type = get_exactly_one_sum_type(node.in_value_types().next().map(|x| x.1))?;
        let llvm_sum_type = context.llvm_sum_type(sum_type)?;
        debug_assert!(inputs[0].get_type() == llvm_sum_type.as_basic_type_enum());

        let sum_input = inputs[0].into_struct_value();
        let builder = context.builder();
        let tag = llvm_sum_type.build_get_tag(builder, sum_input)?;
        let switches = case_values_rmbs_blocks
            .into_iter()
            .map(|(i, rmb, bb)| {
                let mut vs = llvm_sum_type.build_untag(builder, i as u32, sum_input)?;
                vs.extend(&inputs[1..]);
                rmb.write(builder, vs)?;
                Ok((llvm_sum_type.get_tag_type().const_int(i as u64, false), bb))
            })
            .collect::<Result<Vec<_>>>()?;

        builder.build_switch(tag.into_int_value(), switches[0].1, &switches[1..])?;
        builder.position_at_end(exit_block);
        Ok(())
    }
}

fn get_exactly_one_sum_type(ts: impl IntoIterator<Item = Type>) -> Result<SumType> {
    let Some(TypeEnum::Sum(sum_type)) = ts
        .into_iter()
        .map(|t| t.as_type_enum().clone())
        .exactly_one()
        .ok()
    else {
        Err(anyhow!("Not exactly one SumType"))?
    };
    Ok(sum_type)
}

fn emit_value<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    v: &Value,
) -> Result<BasicValueEnum<'c>> {
    match v {
        Value::Extension { e } => {
            let exts = context.extensions();
            exts.load_constant(context, e.value())
        }
        Value::Function { .. } => todo!(),
        Value::Tuple { vs } => {
            let tys = vs.iter().map(|x| x.get_type()).collect_vec();
            let llvm_st = LLVMSumType::try_new(&context.typing_session(), SumType::new([tys]))?;
            let llvm_vs = vs
                .iter()
                .map(|x| emit_value(context, x))
                .collect::<Result<Vec<_>>>()?;
            llvm_st.build_tag(context.builder(), 0, llvm_vs)
        }
        Value::Sum {
            tag,
            values,
            sum_type,
        } => {
            let llvm_st = LLVMSumType::try_new(&context.typing_session(), sum_type.clone())?;
            let vs = values
                .iter()
                .map(|x| emit_value(context, x))
                .collect::<Result<Vec<_>>>()?;
            llvm_st.build_tag(context.builder(), *tag as u32, vs)
        }
    }
}

pub(crate) fn emit_dataflow_parent<'c, OT: OpTrait + 'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, OT, H>,
) -> Result<()>
where
    &'c OpType: TryInto<&'c OT>,
{
    DataflowParentEmitter::new(context, args).emit_children()
}

fn emit_make_tuple<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, MakeTuple, H>,
) -> Result<()> {
    SumOpEmitter::try_new(context, args.node.out_value_types().map(|x| x.1))?.emit(args)
}

fn emit_unpack_tuple<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, UnpackTuple, H>,
) -> Result<()> {
    SumOpEmitter::try_new(context, args.node.in_value_types().map(|x| x.1))?.emit(args)
}

fn emit_tag<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, Tag, H>,
) -> Result<()> {
    SumOpEmitter::try_new(context, args.node.out_value_types().map(|x| x.1))?.emit(args)
}

fn emit_conditional<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, Conditional, H>,
) -> Result<()> {
    ConditionalEmitter(context).emit(args)
}

fn emit_load_constant<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, LoadConstant, H>,
) -> Result<()> {
    let konst_node = args
        .node
        .single_linked_output(0.into())
        .unwrap()
        .0
        .try_into_ot::<Const>()
        .unwrap();
    let r = emit_value(context, konst_node.value())?;
    args.outputs.finish(context.builder(), [r])
}

fn emit_call<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, Call, H>,
) -> Result<()> {
    if !args.node.called_function_type().params().is_empty() {
        todo!("Call of generic function");
    }
    let (func_node, _) = args
        .node
        .single_linked_output(args.node.called_function_port())
        .unwrap();
    let func = match func_node.get() {
        OpType::FuncDecl(_) => context.get_func_decl(func_node.try_into_ot().unwrap()),
        OpType::FuncDefn(_) => context.get_func_defn(func_node.try_into_ot().unwrap()),
        _ => Err(anyhow!("emit_call: Not a Decl or Defn")),
    };
    let inputs: Vec<_> = args.inputs.iter().map(|&x| x.into()).collect();
    let call = context
        .builder()
        .build_call(func?, inputs.as_slice(), "")?
        .try_as_basic_value();
    let rets = match args.outputs.len() {
        0 => {
            call.expect_right("void");
            vec![]
        }
        1 => vec![call.expect_left("non-void")],
        n => call
            .expect_left("non-void")
            .into_struct_value()
            .get_fields()
            // For some reason `get_fields()` returns an extra field at the end with the type of
            // a pointer to the struct??? Just take the first n fields until we figure out what's
            // going on...
            .take(n)
            .collect(),
    };
    args.outputs.finish(context.builder(), rets)
}

fn emit_optype<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, OpType, H>,
) -> Result<()> {
    let node = args.node();
    match node.get() {
        OpType::MakeTuple(ref mt) => emit_make_tuple(context, args.into_ot(mt)),
        OpType::UnpackTuple(ref ut) => emit_unpack_tuple(context, args.into_ot(ut)),
        OpType::Tag(ref tag) => emit_tag(context, args.into_ot(tag)),
        OpType::DFG(_) => emit_dataflow_parent(context, args),

        // TODO Test cases
        OpType::CustomOp(ref co) => {
            let extensions = context.extensions();
            extensions.emit(context, args.into_ot(co))
        }
        OpType::Const(_) => Ok(()),
        OpType::LoadConstant(ref lc) => emit_load_constant(context, args.into_ot(lc)),
        OpType::Call(ref cl) => emit_call(context, args.into_ot(cl)),
        OpType::Conditional(ref co) => emit_conditional(context, args.into_ot(co)),

        // OpType::FuncDefn(fd) => self.emit(ot.into_ot(fd), context, inputs, outputs),
        _ => todo!("Unimplemented OpTypeEmitter: {}", args.node().name()),
    }
}

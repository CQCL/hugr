use anyhow::{anyhow, bail, Result};
use hugr_core::ops::{
    constant::Sum, Call, CallIndirect, Case, Conditional, Const, ExtensionOp, Input, LoadConstant,
    LoadFunction, OpTag, OpTrait, OpType, Output, Tag, Value, CFG,
};
use hugr_core::{
    hugr::views::SiblingGraph,
    types::{SumType, Type, TypeEnum},
    HugrView, NodeIndex,
};
use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicValueEnum, CallableValue};
use itertools::{zip_eq, Itertools};
use petgraph::visit::Walker;

use crate::types::LLVMSumType;
use crate::{
    sum::LLVMSumValue,
    utils::fat::{FatExt as _, FatNode},
};

use super::{
    deaggregate_call_result,
    func::{EmitFuncContext, RowPromise},
    EmitOpArgs,
};

mod cfg;

struct DataflowParentEmitter<'c, 'hugr, OT, H> {
    node: FatNode<'hugr, OT, H>,
    inputs: Option<Vec<BasicValueEnum<'c>>>,
    outputs: Option<RowPromise<'c>>,
}

impl<'c, 'hugr, OT: OpTrait, H: HugrView> DataflowParentEmitter<'c, 'hugr, OT, H>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    pub fn new(args: EmitOpArgs<'c, 'hugr, OT, H>) -> Self {
        Self {
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

    pub fn emit_children(mut self, context: &mut EmitFuncContext<'c, '_, H>) -> Result<()> {
        use petgraph::visit::Topo;
        let node = self.node;
        if !OpTag::DataflowParent.is_superset(node.tag()) {
            Err(anyhow!("Not a dataflow parent"))?
        };

        let (i, o): (FatNode<Input, H>, FatNode<Output, H>) = node
            .get_io()
            .ok_or(anyhow!("emit_dataflow_parent: no io nodes"))?;
        debug_assert!(i.out_value_types().count() == self.inputs.as_ref().unwrap().len());
        debug_assert!(o.in_value_types().count() == self.outputs.as_ref().unwrap().len());

        let region: SiblingGraph = node.try_new_hierarchy_view().unwrap();
        Topo::new(&region.as_petgraph())
            .iter(&region.as_petgraph())
            .filter(|x| (*x != node.node()))
            .map(|x| node.hugr().fat_optype(x))
            .try_for_each(|node| {
                let inputs_rmb = context.node_ins_rmb(node)?;
                let inputs = inputs_rmb.read(context.builder(), [])?;
                let outputs = context.node_outs_rmb(node)?.promise();
                match node.as_ref() {
                    OpType::Input(_) => {
                        let i = self.take_input()?;
                        outputs.finish(context.builder(), i)
                    }
                    OpType::Output(_) => {
                        let o = self.take_output()?;
                        o.finish(context.builder(), inputs)
                    }
                    _ => emit_optype(
                        context,
                        EmitOpArgs {
                            node,
                            inputs,
                            outputs,
                        },
                    ),
                }
            })
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

pub fn emit_value<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    v: &Value,
) -> Result<BasicValueEnum<'c>> {
    match v {
        Value::Extension { e } => context.emit_custom_const(e.value()),
        Value::Function { .. } => bail!(
            "Value::Function Const nodes are not supported. \
            Ensure you eliminate these from the HUGR before lowering to LLVM. \
            `hugr_llvm::utils::inline_constant_functions` is provided for this purpose."
        ),
        Value::Sum(Sum {
            tag,
            values,
            sum_type,
        }) => {
            let llvm_st = LLVMSumType::try_new(&context.typing_session(), sum_type.clone())?;
            let vs = values
                .iter()
                .map(|x| emit_value(context, x))
                .collect::<Result<Vec<_>>>()?;
            llvm_st.build_tag(context.builder(), *tag, vs)
        }
    }
}

pub(crate) fn emit_dataflow_parent<'c, 'hugr, OT: OpTrait, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, 'hugr, OT, H>,
) -> Result<()>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    DataflowParentEmitter::new(args).emit_children(context)
}

fn emit_tag<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, Tag, H>,
) -> Result<()> {
    let st = context.llvm_sum_type(get_exactly_one_sum_type(
        args.node.out_value_types().map(|x| x.1),
    )?)?;
    let builder = context.builder();
    args.outputs.finish(
        builder,
        [st.build_tag(builder, args.node.tag, args.inputs)?],
    )
}

fn emit_conditional<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    EmitOpArgs {
        node,
        inputs,
        outputs,
    }: EmitOpArgs<'c, '_, Conditional, H>,
) -> Result<()> {
    let exit_rmb =
        context.new_row_mail_box(node.dataflow_signature().unwrap().output.iter(), "exit_rmb")?;
    let exit_block = context.build_positioned_new_block(
        format!("cond_exit_{}", node.node().index()),
        None,
        |context, bb| {
            let builder = context.builder();
            outputs.finish(builder, exit_rmb.read_vec(builder, [])?)?;
            Ok::<_, anyhow::Error>(bb)
        },
    )?;

    let rmbs_blocks = node
        .children()
        .enumerate()
        .map(|(i, n)| {
            let label = format!("cond_{}_case_{}", node.node().index(), i);
            let node = n.try_into_ot::<Case>().ok_or(anyhow!("not a case node"))?;
            let rmb = context.new_row_mail_box(node.get_io().unwrap().0.types.iter(), &label)?;
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
                Ok((rmb, bb))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let sum_type = get_exactly_one_sum_type(node.in_value_types().next().map(|x| x.1))?;
    let sum_input = LLVMSumValue::try_new(inputs[0], context.llvm_sum_type(sum_type)?)?;
    let builder = context.builder();
    sum_input.build_destructure(builder, |builder, tag, mut vs| {
        let (rmb, bb) = &rmbs_blocks[tag];
        vs.extend(&inputs[1..]);
        rmb.write(builder, vs)?;
        builder.build_unconditional_branch(*bb)?;
        Ok(())
    })?;
    builder.position_at_end(exit_block);
    Ok(())
}

fn emit_load_constant<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, LoadConstant, H>,
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
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, Call, H>,
) -> Result<()> {
    if !args.node.called_function_type().params().is_empty() {
        return Err(anyhow!("Call of generic function"));
    }
    let (func_node, _) = args
        .node
        .single_linked_output(args.node.called_function_port())
        .unwrap();
    let func = match func_node.as_ref() {
        OpType::FuncDecl(_) => context.get_func_decl(func_node.try_into_ot().unwrap()),
        OpType::FuncDefn(_) => context.get_func_defn(func_node.try_into_ot().unwrap()),
        _ => Err(anyhow!("emit_call: Not a Decl or Defn")),
    };
    let inputs = args.inputs.into_iter().map_into().collect_vec();
    let builder = context.builder();
    let call = builder.build_call(func?, inputs.as_slice(), "")?;
    let call_results = deaggregate_call_result(builder, call, args.outputs.len())?;
    args.outputs.finish(builder, call_results)
}

fn emit_call_indirect<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, CallIndirect, H>,
) -> Result<()> {
    let func_ptr = match args.inputs[0] {
        BasicValueEnum::PointerValue(v) => Ok(v),
        _ => Err(anyhow!("emit_call_indirect: Not a pointer")),
    }?;
    let func =
        CallableValue::try_from(func_ptr).expect("emit_call_indirect: Not a function pointer");
    let inputs = args.inputs.into_iter().skip(1).map_into().collect_vec();
    let builder = context.builder();
    let call = builder.build_call(func, inputs.as_slice(), "")?;
    let call_results = deaggregate_call_result(builder, call, args.outputs.len())?;
    args.outputs.finish(builder, call_results)
}

fn emit_load_function<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, LoadFunction, H>,
) -> Result<()> {
    if !args.node.func_sig.params().is_empty() {
        return Err(anyhow!("Load of generic function"));
    }
    let (func_node, _) = args
        .node
        .single_linked_output(args.node.function_port())
        .unwrap();

    let func = match func_node.as_ref() {
        OpType::FuncDecl(_) => context.get_func_decl(func_node.try_into_ot().unwrap()),
        OpType::FuncDefn(_) => context.get_func_defn(func_node.try_into_ot().unwrap()),
        _ => Err(anyhow!("emit_call: Not a Decl or Defn")),
    }?;
    args.outputs.finish(
        context.builder(),
        [func.as_global_value().as_pointer_value().into()],
    )
}

fn emit_cfg<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, CFG, H>,
) -> Result<()> {
    cfg::CfgEmitter::new(context, args)?.emit_children(context)
}

fn emit_optype<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, OpType, H>,
) -> Result<()> {
    let node = args.node();
    match node.as_ref() {
        OpType::Tag(ref tag) => emit_tag(context, args.into_ot(tag)),
        OpType::DFG(_) => emit_dataflow_parent(context, args),

        OpType::ExtensionOp(ref co) => context.emit_extension_op(args.into_ot(co)),
        OpType::LoadConstant(ref lc) => emit_load_constant(context, args.into_ot(lc)),
        OpType::Call(ref cl) => emit_call(context, args.into_ot(cl)),
        OpType::CallIndirect(ref cl) => emit_call_indirect(context, args.into_ot(cl)),
        OpType::LoadFunction(ref lf) => emit_load_function(context, args.into_ot(lf)),
        OpType::Conditional(ref co) => emit_conditional(context, args.into_ot(co)),
        OpType::CFG(ref cfg) => emit_cfg(context, args.into_ot(cfg)),
        // Const is allowed, but requires no work here. FuncDecl is technically
        // not allowed, but there is no harm in allowing it.
        OpType::Const(_) => Ok(()),
        OpType::FuncDecl(_) => Ok(()),
        OpType::FuncDefn(ref fd) => {
            context.push_todo_func(node.into_ot(fd));
            Ok(())
        }

        _ => Err(anyhow!("Invalid child for Dataflow Parent: {node}")),
    }
}

/// Emit a custom operation with a single input.
///
/// # Arguments
///
/// * `context` - The context in which to emit the operation.
/// * `args` - The arguments to the operation.
/// * `go` - The operation to build the result given a [`Builder`], the input,
///   and an iterator over the expected output types.
pub(crate) fn emit_custom_unary_op<'c, 'hugr, H, F>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, 'hugr, ExtensionOp, H>,
    go: F,
) -> Result<()>
where
    H: HugrView,
    F: FnOnce(
        &mut EmitFuncContext<'c, '_, H>,
        BasicValueEnum<'c>,
        &[BasicTypeEnum<'c>],
    ) -> Result<Vec<BasicValueEnum<'c>>>,
{
    let [inp] = TryInto::<[_; 1]>::try_into(args.inputs).map_err(|v| {
        anyhow!(
            "emit_custom_unary_op: expected exactly one input, got {}",
            v.len()
        )
    })?;
    let out_types = args.outputs.get_types().collect_vec();
    let res = go(context, inp, &out_types)?;
    if res.len() != args.outputs.len()
        || zip_eq(res.iter(), out_types).any(|(a, b)| a.get_type() != b)
    {
        return Err(anyhow!(
            "emit_custom_unary_op: expected outputs of types {:?}, got {:?}",
            args.outputs.get_types().collect_vec(),
            res.iter().map(BasicValueEnum::get_type).collect_vec()
        ));
    }
    args.outputs.finish(context.builder(), res)
}

/// Emit a custom operation with two inputs of the same type.
///
/// # Arguments
///
/// * `context` - The context in which to emit the operation.
/// * `args` - The arguments to the operation.
/// * `go` - The operation to build the result given a [`Builder`], the two
///   inputs, and an iterator over the expected output types.
pub(crate) fn emit_custom_binary_op<'c, 'hugr, H, F>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, 'hugr, ExtensionOp, H>,
    go: F,
) -> Result<()>
where
    H: HugrView,
    F: FnOnce(
        &mut EmitFuncContext<'c, '_, H>,
        (BasicValueEnum<'c>, BasicValueEnum<'c>),
        &[BasicTypeEnum<'c>],
    ) -> Result<Vec<BasicValueEnum<'c>>>,
{
    let [lhs, rhs] = TryInto::<[_; 2]>::try_into(args.inputs).map_err(|v| {
        anyhow!(
            "emit_custom_binary_op: expected exactly 2 inputs, got {}",
            v.len()
        )
    })?;
    if lhs.get_type() != rhs.get_type() {
        return Err(anyhow!(
            "emit_custom_binary_op: expected inputs of the same type, got {} and {}",
            lhs.get_type(),
            rhs.get_type()
        ));
    }
    let out_types = args.outputs.get_types().collect_vec();
    let res = go(context, (lhs, rhs), &out_types)?;
    if res.len() != out_types.len() || zip_eq(res.iter(), out_types).any(|(a, b)| a.get_type() != b)
    {
        return Err(anyhow!(
            "emit_custom_binary_op: expected outputs of types {:?}, got {:?}",
            args.outputs.get_types().collect_vec(),
            res.iter().map(BasicValueEnum::get_type).collect_vec()
        ));
    }
    args.outputs.finish(context.builder(), res)
}
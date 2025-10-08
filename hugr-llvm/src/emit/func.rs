use std::{collections::BTreeMap, rc::Rc};

use anyhow::{Result, anyhow};
use hugr_core::{
    HugrView, Node, NodeIndex, PortIndex, Wire,
    extension::prelude::{either_type, option_type},
    ops::{ExtensionOp, FuncDecl, FuncDefn, constant::CustomConst},
    types::Type,
};
use inkwell::{
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    module::{Linkage, Module},
    types::{BasicType, BasicTypeEnum, FunctionType},
    values::{BasicValue, BasicValueEnum, FunctionValue, GlobalValue, IntValue},
};
use itertools::{Itertools, zip_eq};

use crate::types::{HugrFuncType, HugrSumType, HugrType, TypingSession};
use crate::{custom::CodegenExtsMap, types::LLVMSumType, utils::fat::FatNode};
use delegate::delegate;

use self::mailbox::ValueMailBox;

use super::{EmissionSet, EmitModuleContext, EmitOpArgs};

mod mailbox;
pub use mailbox::{RowMailBox, RowPromise};

/// A context for emitting an LLVM function.
///
/// One of the primary interfaces for implementing codegen extensions.
/// We have methods for:
///   * Converting from hugr [Type]s to LLVM [Type](BasicTypeEnum)s;
///   * Maintaining [`MailBox`](RowMailBox) for each [Wire] in the [`FuncDefn`];
///   * Accessing the [`CodegenExtsMap`];
///   * Accessing an in internal [Builder].
///
/// The internal [Builder] must always be positioned at the end of a
/// [`BasicBlock`]. This invariant is not checked when the builder is accessed
/// through [`EmitFuncContext::builder`].
///
/// [`MailBox`](RowMailBox)es are stack allocations that are `alloca`ed in the
/// first basic block of the function, read from to get the input values of each
/// node, and written to with the output values of each node.
pub struct EmitFuncContext<'c, 'a, H>
where
    'a: 'c,
{
    emit_context: EmitModuleContext<'c, 'a, H>,
    todo: EmissionSet,
    func: FunctionValue<'c>,
    env: BTreeMap<Wire, ValueMailBox<'c>>,
    builder: Builder<'c>,
    prologue_bb: BasicBlock<'c>,
    launch_bb: BasicBlock<'c>,
}

impl<'c, 'a, H: HugrView<Node = Node>> EmitFuncContext<'c, 'a, H> {
    delegate! {
        to self.emit_context {
            /// Returns the inkwell [Context].
            pub fn iw_context(&self) ->  &'c Context;
            /// Returns the internal [CodegenExtsMap] .
            pub fn extensions(&self) ->  Rc<CodegenExtsMap<'a,H>>;
            /// Returns a new [TypingSession].
            pub fn typing_session(&self) -> TypingSession<'c, 'a>;
            /// Convert hugr [HugrType] into an LLVM [Type](BasicTypeEnum).
            pub fn llvm_type(&self, hugr_type: &HugrType) -> Result<BasicTypeEnum<'c> >;
            /// Convert a [HugrFuncType] into an LLVM [FunctionType].
            pub fn llvm_func_type(&self, hugr_type: &HugrFuncType) -> Result<FunctionType<'c> >;
            /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
            pub fn llvm_sum_type(&self, sum_type: HugrSumType) -> Result<LLVMSumType<'c>>;
            /// Adds or gets the [FunctionValue] in the [inkwell::module::Module] corresponding to the given [FuncDefn].
            ///
            /// The name of the result may have been mangled.
            pub fn get_func_defn(&self, node: FatNode<FuncDefn, H>) -> Result<FunctionValue<'c>>;
            /// Adds or gets the [FunctionValue] in the [inkwell::module::Module] corresponding to the given [FuncDecl].
            ///
            /// The name of the result may have been mangled.
            pub fn get_func_decl(&self, node: FatNode<FuncDecl, H>) -> Result<FunctionValue<'c>>;
            /// Adds or get the [FunctionValue] in the [inkwell::module::Module] with the given symbol
            /// and function type.
            ///
            /// The name undergoes no mangling. The [FunctionValue] will have
            /// [inkwell::module::Linkage::External].
            ///
            /// If this function is called multiple times with the same arguments it
            /// will return the same [FunctionValue].
            ///
            /// If a function with the given name exists but the type does not match
            /// then an Error is returned.
            pub fn get_extern_func(&self, symbol: impl AsRef<str>, typ: FunctionType<'c>,) -> Result<FunctionValue<'c>>;
            /// Adds or gets the [GlobalValue] in the [inkwell::module::Module] corresponding to the
            /// given symbol and LLVM type.
            ///
            /// The name will not be mangled.
            ///
            /// If a global with the given name exists but the type or constant-ness
            /// does not match then an error will be returned.
            pub fn get_global(&self, symbol: impl AsRef<str>, typ: impl BasicType<'c>, constant: bool) -> Result<GlobalValue<'c>>;
        }
    }

    /// Used when emitters encounter a scoped definition. `node` will be
    /// returned from [`EmitFuncContext::finish`].
    pub fn push_todo_func(&mut self, node: FatNode<'_, FuncDefn, H>) {
        self.todo.insert(node.node());
    }

    /// Returns the current [FunctionValue] being emitted.
    pub fn func(&self) -> FunctionValue<'c> {
        self.func
    }

    /// Returns the internal [Builder]. Callers must ensure that it is
    /// positioned at the end of a basic block. This invariant is not checked(it
    /// doesn't seem possible to check it).
    pub fn builder(&self) -> &Builder<'c> {
        &self.builder
    }

    /// Create a new basic block. When `before` is `Some` the block will be
    /// created immediately before that block, otherwise at the end of the
    /// function.
    pub(crate) fn new_basic_block(
        &mut self,
        name: impl AsRef<str>,
        before: Option<BasicBlock<'c>>,
    ) -> BasicBlock<'c> {
        if let Some(before) = before {
            self.iw_context().prepend_basic_block(before, name.as_ref())
        } else {
            self.iw_context()
                .append_basic_block(self.func, name.as_ref())
        }
    }

    fn prologue_block(&self) -> BasicBlock<'c> {
        // guaranteed to exist because we create it in new
        self.func.get_first_basic_block().unwrap()
    }

    /// Creates a new `EmitFuncContext` for `func`, taking ownership of
    /// `emit_context`. `emit_context` will be returned in
    /// [`EmitFuncContext::finish`].
    ///
    /// If `func` has any existing [`BasicBlock`]s we will fail.
    ///
    /// TODO on failure return `emit_context`
    pub fn new(
        emit_context: EmitModuleContext<'c, 'a, H>,
        func: FunctionValue<'c>,
    ) -> Result<EmitFuncContext<'c, 'a, H>> {
        if func.get_first_basic_block().is_some() {
            Err(anyhow!(
                "EmitContext::new: Function already has a basic block: {:?}",
                func.get_name()
            ))?;
        }
        let prologue_bb = emit_context
            .iw_context()
            .append_basic_block(func, "alloca_block");
        let launch_bb = emit_context
            .iw_context()
            .append_basic_block(func, "entry_block");
        let builder = emit_context.iw_context().create_builder();
        builder.position_at_end(launch_bb);
        Ok(Self {
            emit_context,
            todo: Default::default(),
            func,
            env: Default::default(),
            builder,
            prologue_bb,
            launch_bb,
        })
    }

    fn new_value_mail_box(&mut self, t: &Type, name: impl AsRef<str>) -> Result<ValueMailBox<'c>> {
        let bte = self.llvm_type(t)?;
        let ptr = self.build_prologue(|builder| builder.build_alloca(bte, name.as_ref()))?;
        Ok(ValueMailBox::new(bte, ptr, Some(name.as_ref().into())))
    }

    /// Create a new anonymous [`RowMailBox`]. This mailbox is not mapped to any
    /// [Wire]s, and so will not interact with any mailboxes returned from
    /// [`EmitFuncContext::node_ins_rmb`] or [`EmitFuncContext::node_outs_rmb`].
    pub fn new_row_mail_box<'t>(
        &mut self,
        ts: impl IntoIterator<Item = &'t Type>,
        name: impl AsRef<str>,
    ) -> Result<RowMailBox<'c>> {
        Ok(RowMailBox::new(
            ts.into_iter()
                .enumerate()
                .map(|(i, t)| self.new_value_mail_box(t, format!("{i}")))
                .collect::<Result<Vec<_>>>()?,
            Some(name.as_ref().into()),
        ))
    }

    fn build_prologue<T>(&mut self, f: impl FnOnce(&Builder<'c>) -> T) -> T {
        let b = self.prologue_block();
        self.build_positioned(b, |x| f(&x.builder))
    }

    /// Creates a new [`BasicBlock`] and calls `f` with the internal builder
    /// positioned at the start of that [`BasicBlock`]. The builder will be
    /// repositioned back to it's location before `f` before this function
    /// returns.
    pub fn build_positioned_new_block<T>(
        &mut self,
        name: impl AsRef<str>,
        before: Option<BasicBlock<'c>>,
        f: impl FnOnce(&mut Self, BasicBlock<'c>) -> T,
    ) -> T {
        let bb = self.new_basic_block(name, before);
        self.build_positioned(bb, |s| f(s, bb))
    }

    /// Positions the internal builder at the end of `block` and calls `f`.  The
    /// builder will be repositioned back to it's location before `f` before
    /// this function returns.
    pub fn build_positioned<T>(
        &mut self,
        block: BasicBlock<'c>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        // safe because our builder is always positioned
        let current = self.builder.get_insert_block().unwrap();
        self.builder.position_at_end(block);
        let r = f(self);
        self.builder.position_at_end(current);
        r
    }

    /// Returns a [`RowMailBox`] mapped to the input wires of `node`. When emitting a node
    /// input values are from this mailbox.
    pub fn node_ins_rmb<'hugr, OT: 'hugr>(
        &mut self,
        node: FatNode<'hugr, OT, H>,
    ) -> Result<RowMailBox<'c>> {
        let r = node
            .in_value_types()
            .map(|(p, t)| {
                let (slo_n, slo_p) = node
                    .single_linked_output(p)
                    .ok_or(anyhow!("No single linked output"))?;
                self.map_wire(slo_n, slo_p, &t)
            })
            .collect::<Result<RowMailBox>>()?;

        debug_assert!(
            zip_eq(node.in_value_types(), r.get_types())
                .all(|((_, t), lt)| self.llvm_type(&t).unwrap() == lt)
        );
        Ok(r)
    }

    /// Returns a [`RowMailBox`] mapped to the output wires of `node`. When emitting a node
    /// output values are written to this mailbox.
    pub fn node_outs_rmb<'hugr, OT: 'hugr>(
        &mut self,
        node: FatNode<'hugr, OT, H>,
    ) -> Result<RowMailBox<'c>> {
        let r = node
            .out_value_types()
            .map(|(port, hugr_type)| self.map_wire(node, port, &hugr_type))
            .collect::<Result<RowMailBox>>()?;
        debug_assert!(
            zip_eq(node.out_value_types(), r.get_types())
                .all(|((_, t), lt)| self.llvm_type(&t).unwrap() == lt)
        );
        Ok(r)
    }

    fn map_wire<'hugr, OT>(
        &mut self,
        node: FatNode<'hugr, OT, H>,
        port: hugr_core::OutgoingPort,
        hugr_type: &Type,
    ) -> Result<ValueMailBox<'c>> {
        let wire = Wire::new(node.node(), port);
        if let Some(mb) = self.env.get(&wire) {
            debug_assert_eq!(self.llvm_type(hugr_type).unwrap(), mb.get_type());
            return Ok(mb.clone());
        }
        let mb = self.new_value_mail_box(
            hugr_type,
            format!("{}_{}", node.node().index(), port.index()),
        )?;
        self.env.insert(wire, mb.clone());
        Ok(mb)
    }

    pub fn get_current_module(&self) -> &Module<'c> {
        self.emit_context.module()
    }

    pub(crate) fn emit_custom_const(&mut self, v: &dyn CustomConst) -> Result<BasicValueEnum<'c>> {
        let exts = self.extensions();
        exts.as_ref()
            .load_constant_handlers
            .emit_load_constant(self, v)
    }

    pub(crate) fn emit_extension_op(
        &mut self,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let exts = self.extensions();
        exts.as_ref()
            .extension_op_handlers
            .emit_extension_op(self, args)
    }

    /// Consumes the `EmitFuncContext` and returns both the inner
    /// [`EmitModuleContext`] and the scoped [`FuncDefn`]s that were encountered.
    pub fn finish(self) -> Result<(EmitModuleContext<'c, 'a, H>, EmissionSet)> {
        self.builder.position_at_end(self.prologue_bb);
        self.builder.build_unconditional_branch(self.launch_bb)?;
        Ok((self.emit_context, self.todo))
    }
}

/// Builds an optional value wrapping `some_value` conditioned on the provided `is_some` flag.
pub fn build_option<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    is_some: IntValue<'c>,
    some_value: BasicValueEnum<'c>,
    hugr_ty: HugrType,
) -> Result<BasicValueEnum<'c>> {
    let option_ty = ctx.llvm_sum_type(option_type(hugr_ty))?;
    let builder = ctx.builder();
    let some = option_ty.build_tag(builder, 1, vec![some_value])?;
    let none = option_ty.build_tag(builder, 0, vec![])?;
    let option = builder.build_select(is_some, some, none, "")?;
    Ok(option)
}

/// Builds a result value wrapping either `ok_value` or `else_value` depending on the provided
/// `is_ok` flag.
pub fn build_ok_or_else<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    is_ok: IntValue<'c>,
    ok_value: BasicValueEnum<'c>,
    ok_hugr_ty: HugrType,
    else_value: BasicValueEnum<'c>,
    else_hugr_ty: HugrType,
) -> Result<BasicValueEnum<'c>> {
    let either_ty = ctx.llvm_sum_type(either_type(else_hugr_ty, ok_hugr_ty))?;
    let builder = ctx.builder();
    let left = either_ty.build_tag(builder, 0, vec![else_value])?;
    let right = either_ty.build_tag(builder, 1, vec![ok_value])?;
    let either = builder.build_select(is_ok, right, left, "")?;
    Ok(either)
}
/// Helper to outline LLVM IR into a function call instead of inlining it every time.
///
/// The first time this helper is called with a given function name, a function is built
/// using the provided closure. Future invocations with the same name will just emit calls
/// to this function.
///
/// The return type is specified by `ret_type`, and if `Some` then the closure must return
/// a value of that type, which will be returned from the function. Otherwise, the function
/// will return void.
pub fn get_or_make_function<'c, H: HugrView<Node = Node>, const N: usize>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    func_name: &str,
    args: [BasicValueEnum<'c>; N],
    ret_type: Option<BasicTypeEnum<'c>>,
    go: impl FnOnce(
        &mut EmitFuncContext<'c, '_, H>,
        [BasicValueEnum<'c>; N],
    ) -> Result<Option<BasicValueEnum<'c>>>,
) -> Result<Option<BasicValueEnum<'c>>> {
    let func = match ctx.get_current_module().get_function(func_name) {
        Some(func) => func,
        None => {
            let arg_tys = args.iter().map(|v| v.get_type().into()).collect_vec();
            let sig = match ret_type {
                Some(ret_ty) => ret_ty.fn_type(&arg_tys, false),
                None => ctx.iw_context().void_type().fn_type(&arg_tys, false),
            };
            let func =
                ctx.get_current_module()
                    .add_function(func_name, sig, Some(Linkage::Internal));
            let bb = ctx.iw_context().append_basic_block(func, "");
            let args = (0..N)
                .map(|i| func.get_nth_param(i as u32).unwrap())
                .collect_array()
                .unwrap();

            let curr_bb = ctx.builder().get_insert_block().unwrap();
            let curr_func = ctx.func;

            ctx.builder().position_at_end(bb);
            ctx.func = func;
            let ret_val = go(ctx, args)?;
            if ctx
                .builder()
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                ctx.builder()
                    .build_return(ret_val.as_ref().map::<&dyn BasicValue, _>(|v| v))?;
            }

            ctx.builder().position_at_end(curr_bb);
            ctx.func = curr_func;
            func
        }
    };
    let call_site =
        ctx.builder()
            .build_call(func, &args.iter().map(|&a| a.into()).collect_vec(), "")?;
    let result = call_site.try_as_basic_value().left();
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_func_getter() {
        // Use TestContext for consistent test setup
        let test_ctx = crate::test::test_ctx(-1);
        let emit_context = test_ctx.get_emit_module_context();
        let func_type = emit_context.iw_context().void_type().fn_type(&[], false);
        let function = emit_context
            .module()
            .add_function("test_func", func_type, None);
        let func_context = super::EmitFuncContext::new(emit_context, function).unwrap();

        // Assert the getter returns the correct function
        assert_eq!(func_context.func(), function);
    }
}

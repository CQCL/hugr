//! Lower shared pointer operations with a complete storage and ownership backend.
//!
//! Register [`PtrCodegenExtension`] with a [`PtrCodegen`] implementation that owns
//! creation, duplication, final extraction and storage reclamation. The backend
//! may emit reference-count arithmetic or delegate ownership to a runtime.
//! [`DefaultPtrCodegen`] uses configurable array heap operations, an LLVM-managed count
//! and checked, non-atomic lock state. Concurrent targets must choose a backend
//! that supplies synchronization.
//!
//! `Read`, `Write`, `Swap` and `Map` lock the handle, project only the stored value,
//! then unlock it after accesses end. `Map` holds the lock throughout its callback.
//! Reentrant access through another handle panics in the default backend. Custom
//! backends must also reject reentrant access rather than expose the linear value
//! already owned by the callback.
//! A callback that does not return normally does not unlock the cell.
//!
//! `New`, `Dup` and `Free` delegate their complete ownership transitions to the
//! backend, including any synchronization. After `Free` consumes a handle, the
//! lowering never projects, accesses or unlocks it. `Eq` directly compares opaque
//! handles without invoking any hooks. Threaded handles order operations; separate
//! handles have unspecified relative order unless another dependency orders them.

use anyhow::{Result, anyhow, bail};
use hugr_core::{
    HugrView, Node,
    extension::prelude::ConstError,
    extension::{prelude::option_type, simple_op::MakeExtensionOp},
    ops::Value,
    std_extensions::ptr::{self, PtrOp, PtrOpDef},
    types::Signature,
};
use inkwell::{
    IntPredicate,
    context::Context,
    types::{BasicType, BasicTypeEnum, StructType},
    values::{BasicValueEnum, IntValue, PointerValue},
};

use crate::{
    CodegenExtension, CodegenExtsBuilder,
    emit::{EmitFuncContext, RowPromise, deaggregate_call_result, emit_value},
    extension::{PreludeCodegen, collections::array::ArrayCodegen},
};

/// A complete backend for pointer storage, ownership and synchronization.
///
/// Implement every method together. The backend owns its count and storage layout;
/// the operation lowering assumes only a typed stored value. Handles to the same
/// live cell have the same address, and distinct live cells have distinct addresses.
/// Every successful creation or duplication supplies one linear owner.
///
/// Lifecycle hooks are called without the payload lock held and own any required
/// synchronization. Access hooks keep a live owner and provide mutual exclusion
/// and acquire/release synchronization for concurrent accesses. Hooks must leave
/// the builder at the end of an unterminated basic block. Runtime failure must
/// terminate execution rather than return a consumed or invalid handle.
///
/// Use [`DefaultPtrCodegen`] for the concrete counted-storage backend.
pub trait PtrCodegen: Clone {
    /// Create one owner and transfer `value` into live, correctly aligned storage.
    /// Initialize any ownership metadata and synchronization before returning.
    /// The backend determines its allocation layout and handles allocation failure.
    fn emit_new<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>>;

    /// Retain one additional owner of `ptr` without copying the stored value.
    /// `value_ty` is the LLVM type of that value, not the backend's storage layout.
    /// Both output handles have the same identity as `ptr`.
    fn emit_dup<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<()>;

    /// Consume one owner and return an `i1` indicating whether it was the last.
    ///
    /// On the last owner, move the stored value to `destination` before reclaiming
    /// storage and any synchronization resources. Otherwise leave the destination
    /// untouched. It is disjoint, writable storage aligned for `value_ty`, including
    /// a non-null buffer for zero-sized values. No payload destructor is requested.
    ///
    /// The caller holds no lock and will never use this owner or unlock it after
    /// the call. Complete any locking, extraction and teardown inside this hook.
    /// The caller loads the destination only on the true branch.
    fn emit_free<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
        destination: PointerValue<'c>,
    ) -> Result<IntValue<'c>>;

    /// Project live, aligned storage for only the value of type `value_ty`.
    /// Called with the payload lock held. The pointer is used only before unlock;
    /// no reference-count field or backend metadata layout is assumed.
    fn emit_get_ptr<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<PointerValue<'c>>;

    /// Acquire exclusive access to the stored value through a live owner.
    /// Implementations must panic on reentry through any alias of the same cell.
    fn emit_lock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()>;

    /// End protected access and publish writes through the still-live owner.
    /// Implementations must panic if the cell is not locked by this execution.
    fn emit_unlock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()>;
}

/// Counted-storage backend using supplied heap operations and panic lowering.
///
/// Storage holds a count, checked lock state and the value. The supplied array
/// codegen controls allocation/free, and must provide the value's alignment.
/// Allocation failure, count overflow and invalid lock transitions panic through
/// the supplied prelude codegen. Lock state detects reentrant access; it provides
/// no synchronization, so this backend requires sequential execution per cell.
#[derive(Clone, Debug)]
pub struct DefaultPtrCodegen<PCG, ACG> {
    prelude: PCG,
    heap: ACG,
    allocation_error: ConstError,
    refcount_overflow_error: ConstError,
    already_locked_error: ConstError,
    not_locked_error: ConstError,
}
impl<PCG: PreludeCodegen, ACG: ArrayCodegen> DefaultPtrCodegen<PCG, ACG> {
    /// Default allocation-failure signal, preserving the prelude error convention.
    pub const ALLOCATION_FAILURE_SIGNAL: u32 = 2;
    /// Default reference-count overflow signal.
    pub const REFCOUNT_OVERFLOW_SIGNAL: u32 = 2;
    /// Default reentrant-lock signal.
    pub const ALREADY_LOCKED_SIGNAL: u32 = 2;
    /// Default unmatched-unlock signal.
    pub const NOT_LOCKED_SIGNAL: u32 = 2;

    /// Replace the error emitted when heap allocation returns null.
    pub fn with_allocation_error(mut self, error: ConstError) -> Self {
        self.allocation_error = error;
        self
    }
    /// Replace the error emitted when retaining would overflow the count.
    pub fn with_refcount_overflow_error(mut self, error: ConstError) -> Self {
        self.refcount_overflow_error = error;
        self
    }
    /// Replace the error emitted on reentrant locking.
    pub fn with_already_locked_error(mut self, error: ConstError) -> Self {
        self.already_locked_error = error;
        self
    }
    /// Replace the error emitted when unlocking an unlocked cell.
    pub fn with_not_locked_error(mut self, error: ConstError) -> Self {
        self.not_locked_error = error;
        self
    }

    /// Use the same heap and panic implementations selected for other extensions.
    pub fn new(prelude: PCG, heap: ACG) -> Self {
        Self {
            prelude,
            heap,
            allocation_error: ConstError::new(
                Self::ALLOCATION_FAILURE_SIGNAL,
                "Pointer allocation failed",
            ),
            refcount_overflow_error: ConstError::new(
                Self::REFCOUNT_OVERFLOW_SIGNAL,
                "Pointer reference count overflow",
            ),
            already_locked_error: ConstError::new(
                Self::ALREADY_LOCKED_SIGNAL,
                "Pointer cell is already locked",
            ),
            not_locked_error: ConstError::new(
                Self::NOT_LOCKED_SIGNAL,
                "Pointer cell is not locked",
            ),
        }
    }
}
impl<PCG: PreludeCodegen + Default, ACG: ArrayCodegen + Default> Default
    for DefaultPtrCodegen<PCG, ACG>
{
    fn default() -> Self {
        Self::new(PCG::default(), ACG::default())
    }
}
impl<PCG: PreludeCodegen, ACG: ArrayCodegen> PtrCodegen for DefaultPtrCodegen<PCG, ACG> {
    fn emit_new<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let layout = counted_layout(ctx.iw_context(), value.get_type());
        let ptr = emit_alloc_checked(
            ctx,
            &self.heap,
            &self.prelude,
            &self.allocation_error,
            layout,
        )?;
        emit_counted_init(ctx, ptr, value)?;
        Ok(ptr)
    }

    fn emit_dup<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<()> {
        self.emit_lock(ctx, ptr)?;
        let (count, _) = counted_fields(ctx, ptr, value_ty)?;
        emit_counted_dup(ctx, &self.prelude, &self.refcount_overflow_error, count)?;
        self.emit_unlock(ctx, ptr)
    }

    fn emit_free<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
        destination: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        self.emit_lock(ctx, ptr)?;
        let (count, value) = counted_fields(ctx, ptr, value_ty)?;
        let last = emit_counted_take(ctx, count, value, value_ty, destination)?;
        self.emit_unlock(ctx, ptr)?;
        emit_free_if_last(ctx, &self.heap, ptr, last)?;
        Ok(last)
    }

    fn emit_get_ptr<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        Ok(counted_fields(ctx, ptr, value_ty)?.1)
    }

    fn emit_lock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        let flag = lock_field(ctx, ptr)?;
        let locked = ctx
            .builder()
            .build_load(ctx.iw_context().bool_type(), flag, "ptr.locked")?
            .into_int_value();
        panic_if(ctx, &self.prelude, locked, &self.already_locked_error)?;
        ctx.builder()
            .build_store(flag, ctx.iw_context().bool_type().const_int(1, false))?;
        Ok(())
    }
    fn emit_unlock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        let flag = lock_field(ctx, ptr)?;
        let locked = ctx
            .builder()
            .build_load(ctx.iw_context().bool_type(), flag, "ptr.locked")?
            .into_int_value();
        let unlocked = ctx.builder().build_not(locked, "")?;
        panic_if(ctx, &self.prelude, unlocked, &self.not_locked_error)?;
        ctx.builder()
            .build_store(flag, ctx.iw_context().bool_type().const_zero())?;
        Ok(())
    }
}

fn lock_field<'c, H: HugrView<Node = Node>>(
    ctx: &EmitFuncContext<'c, '_, H>,
    ptr: PointerValue<'c>,
) -> Result<PointerValue<'c>> {
    // The lock field has a fixed offset independent of the stored value type.
    let header = ctx.iw_context().struct_type(
        &[
            ctx.iw_context().i64_type().into(),
            ctx.iw_context().bool_type().into(),
        ],
        false,
    );
    Ok(ctx.builder().build_struct_gep(header, ptr, 1, "ptr.lock")?)
}

// Counted-storage helpers belong to the concrete backend, not emit_ptr_op.
fn counted_layout<'c>(ctx: &'c Context, value_ty: BasicTypeEnum<'c>) -> StructType<'c> {
    ctx.struct_type(
        &[ctx.i64_type().into(), ctx.bool_type().into(), value_ty],
        false,
    )
}

fn emit_alloc_checked<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    heap: &impl ArrayCodegen,
    prelude: &impl PreludeCodegen,
    error: &ConstError,
    layout: impl BasicType<'c>,
) -> Result<PointerValue<'c>> {
    let size = layout
        .size_of()
        .ok_or_else(|| anyhow!("Unsized pointer storage"))?;
    let ptr = heap.emit_allocate_array(ctx, size)?;
    let failed = ctx.builder().build_is_null(ptr, "ptr.alloc.failed")?;
    panic_if(ctx, prelude, failed, error)?;
    Ok(ptr)
}

fn counted_fields<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    payload: PointerValue<'c>,
    value_ty: BasicTypeEnum<'c>,
) -> Result<(PointerValue<'c>, PointerValue<'c>)> {
    let layout = counted_layout(ctx.iw_context(), value_ty);
    Ok((
        ctx.builder()
            .build_struct_gep(layout, payload, 0, "ptr.count")?,
        ctx.builder()
            .build_struct_gep(layout, payload, 2, "ptr.value")?,
    ))
}

fn emit_counted_init<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    payload: PointerValue<'c>,
    value: BasicValueEnum<'c>,
) -> Result<()> {
    let (count, value_ptr) = counted_fields(ctx, payload, value.get_type())?;
    ctx.builder()
        .build_store(count, ctx.iw_context().i64_type().const_int(1, false))?;
    let flag = lock_field(ctx, payload)?;
    ctx.builder()
        .build_store(flag, ctx.iw_context().bool_type().const_zero())?;
    ctx.builder().build_store(value_ptr, value)?;
    Ok(())
}

fn emit_counted_dup<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    prelude: &impl PreludeCodegen,
    error: &ConstError,
    count_ptr: PointerValue<'c>,
) -> Result<()> {
    let ty = ctx.iw_context().i64_type();
    let count = ctx
        .builder()
        .build_load(ty, count_ptr, "")?
        .into_int_value();
    let overflow = ctx.builder().build_int_compare(
        IntPredicate::EQ,
        count,
        ty.const_all_ones(),
        "ptr.refcount.overflow",
    )?;
    panic_if(ctx, prelude, overflow, error)?;
    let count = ctx
        .builder()
        .build_int_add(count, ty.const_int(1, false), "")?;
    ctx.builder().build_store(count_ptr, count)?;
    Ok(())
}

fn emit_counted_take<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    count_ptr: PointerValue<'c>,
    value_ptr: PointerValue<'c>,
    value_ty: BasicTypeEnum<'c>,
    destination: PointerValue<'c>,
) -> Result<IntValue<'c>> {
    let ty = ctx.iw_context().i64_type();
    let count = ctx
        .builder()
        .build_load(ty, count_ptr, "")?
        .into_int_value();
    let count = ctx
        .builder()
        .build_int_sub(count, ty.const_int(1, false), "")?;
    ctx.builder().build_store(count_ptr, count)?;
    let last =
        ctx.builder()
            .build_int_compare(IntPredicate::EQ, count, ty.const_zero(), "ptr.last")?;
    let take = ctx.new_basic_block("ptr.take", None);
    let exit = ctx.new_basic_block("ptr.taken", None);
    ctx.builder().build_conditional_branch(last, take, exit)?;
    ctx.builder().position_at_end(take);
    let value = ctx
        .builder()
        .build_load(value_ty, value_ptr, "ptr.take.value")?;
    ctx.builder().build_store(destination, value)?;
    ctx.builder().build_unconditional_branch(exit)?;
    ctx.builder().position_at_end(exit);
    Ok(last)
}

fn emit_free_if_last<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    heap: &impl ArrayCodegen,
    ptr: PointerValue<'c>,
    last: IntValue<'c>,
) -> Result<()> {
    let free = ctx.new_basic_block("ptr.destroy", None);
    let exit = ctx.new_basic_block("ptr.released", None);
    ctx.builder().build_conditional_branch(last, free, exit)?;
    ctx.builder().position_at_end(free);
    heap.emit_free_array(ctx, ptr)?;
    ctx.builder().build_unconditional_branch(exit)?;
    ctx.builder().position_at_end(exit);
    Ok(())
}

fn value_buffer<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ty: BasicTypeEnum<'c>,
    name: &str,
) -> Result<PointerValue<'c>> {
    // Allocate once in the function prologue, including when Free is in a loop.
    let entry = ctx
        .builder()
        .get_insert_block()
        .unwrap()
        .get_parent()
        .unwrap()
        .get_first_basic_block()
        .unwrap();
    Ok(ctx.build_positioned(entry, |ctx| ctx.builder().build_alloca(ty, name))?)
}

/// Registers the pointer type and all pointer operations.
#[derive(Clone, Debug, Default)]
pub struct PtrCodegenExtension<CCG>(CCG);
impl<CCG: PtrCodegen> PtrCodegenExtension<CCG> {
    /// Use the supplied runtime hooks.
    pub fn new(ccg: CCG) -> Self {
        Self(ccg)
    }
}
impl<CCG: PtrCodegen> From<CCG> for PtrCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}
impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Register pointer lowering using the supplied runtime hooks.
    pub fn add_ptr_extensions(self, ccg: impl PtrCodegen + 'a) -> Self {
        self.add_extension(PtrCodegenExtension::new(ccg))
    }
    /// Register pointer lowering with [`DefaultPtrCodegen`].
    #[must_use]
    pub fn add_default_ptr_extensions(
        self,
        prelude: impl PreludeCodegen + 'a,
        heap: impl ArrayCodegen + 'a,
    ) -> Self {
        self.add_ptr_extensions(DefaultPtrCodegen::new(prelude, heap))
    }
}
impl<CCG: PtrCodegen> CodegenExtension for PtrCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type((ptr::EXTENSION_ID, ptr::PTR_TYPE_ID), |ts, _| {
                Ok(ts.llvm_ptr_type().into())
            })
            .simple_extension_op::<PtrOpDef>(move |ctx, args, _| {
                let op = PtrOp::from_extension_op(args.node().as_ref())?;
                emit_ptr_op(&self.0, ctx, op, args.inputs, args.outputs)
            })
    }
}

fn panic_if<H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<H>,
    prelude: &impl PreludeCodegen,
    cond: IntValue,
    error: &ConstError,
) -> Result<()> {
    let failed = ctx.new_basic_block("ptr.panic", None);
    let success = ctx.new_basic_block("ptr.continue", None);
    ctx.builder()
        .build_conditional_branch(cond, failed, success)?;
    ctx.builder().position_at_end(failed);
    let err = prelude.emit_const_error(ctx, error)?;
    prelude.emit_panic(ctx, err)?;
    ctx.builder().build_unreachable()?;
    ctx.builder().position_at_end(success);
    Ok(())
}

fn emit_ptr_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl PtrCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: PtrOp,
    inputs: Vec<BasicValueEnum<'c>>,
    outputs: RowPromise<'c>,
) -> Result<()> {
    if op.def == PtrOpDef::Eq {
        let equal = ctx.builder().build_int_compare(
            IntPredicate::EQ,
            inputs[0].into_pointer_value(),
            inputs[1].into_pointer_value(),
            "ptr.eq",
        )?;
        let true_val = emit_value(ctx, &Value::true_val())?;
        let false_val = emit_value(ctx, &Value::false_val())?;
        let equal = ctx.builder().build_select(equal, true_val, false_val, "")?;
        return outputs.finish(ctx.builder(), [inputs[0], inputs[1], equal]);
    }
    let value_ty = ctx.llvm_type(&op.ty)?;
    if op.def == PtrOpDef::New {
        let ptr = ccg.emit_new(ctx, inputs[0])?;
        return outputs.finish(ctx.builder(), [ptr.into()]);
    }
    let cell = inputs[0].into_pointer_value();
    if op.def == PtrOpDef::Dup {
        ccg.emit_dup(ctx, cell, value_ty)?;
        return outputs.finish(ctx.builder(), [cell.into(), cell.into()]);
    }
    if op.def == PtrOpDef::Free {
        let destination = value_buffer(ctx, value_ty, "ptr.free.value")?;
        let last = ccg.emit_free(ctx, cell, value_ty, destination)?;
        let some_bb = ctx.new_basic_block("ptr.free.some", None);
        let none_bb = ctx.new_basic_block("ptr.free.none", None);
        let exit = ctx.new_basic_block("ptr.free.exit", None);
        let option = option_type([op.ty]);
        let sum_ty = ctx.llvm_sum_type(option.clone())?;
        let mailbox = ctx.new_row_mail_box([&option.into()], "ptr.free.result")?;
        ctx.builder()
            .build_conditional_branch(last, some_bb, none_bb)?;
        ctx.builder().position_at_end(some_bb);
        let value = ctx
            .builder()
            .build_load(value_ty, destination, "ptr.free.extracted")?;
        let some = sum_ty.build_tag(ctx.builder(), 1, vec![value])?;
        mailbox.write(ctx.builder(), [some.into()])?;
        ctx.builder().build_unconditional_branch(exit)?;
        ctx.builder().position_at_end(none_bb);
        let none = sum_ty.build_tag(ctx.builder(), 0, vec![])?;
        mailbox.write(ctx.builder(), [none.into()])?;
        ctx.builder().build_unconditional_branch(exit)?;
        ctx.builder().position_at_end(exit);
        return outputs.finish(ctx.builder(), mailbox.read_vec(ctx.builder(), [])?);
    }
    ccg.emit_lock(ctx, cell)?;
    let value_ptr = ccg.emit_get_ptr(ctx, cell, value_ty)?;
    let results = match op.def {
        PtrOpDef::Read => vec![
            cell.into(),
            ctx.builder().build_load(value_ty, value_ptr, "ptr.read")?,
        ],
        PtrOpDef::Write => {
            ctx.builder().build_store(value_ptr, inputs[1])?;
            vec![cell.into()]
        }
        PtrOpDef::Swap => {
            let old = ctx.builder().build_load(value_ty, value_ptr, "ptr.old")?;
            ctx.builder().build_store(value_ptr, inputs[1])?;
            vec![cell.into(), old]
        }
        PtrOpDef::Map => {
            let mut in_row = vec![op.ty.clone()];
            in_row.extend(op.map_signature.input().iter().cloned());
            let mut out_row = vec![op.ty];
            out_row.extend(op.map_signature.output().iter().cloned());
            let callback_ty = ctx.llvm_func_type(&Signature::new(in_row, out_row))?;
            let value = ctx
                .builder()
                .build_load(value_ty, value_ptr, "ptr.map.input")?;
            let mut args = vec![value.into()];
            args.extend(
                inputs[2..]
                    .iter()
                    .copied()
                    .map(inkwell::values::BasicMetadataValueEnum::from),
            );
            let call = ctx.builder().build_indirect_call(
                callback_ty,
                inputs[1].into_pointer_value(),
                &args,
                "ptr.map",
            )?;
            let mut values =
                deaggregate_call_result(ctx.builder(), call, 1 + op.map_signature.output().len())?;
            ctx.builder().build_store(value_ptr, values[0])?;
            values[0] = cell.into();
            values
        }
        _ => bail!("Unsupported pointer operation: {:?}", op.def),
    };
    ccg.emit_unlock(ctx, cell)?;
    outputs.finish(ctx.builder(), results)
}

#[cfg(test)]
mod test;

//! Codegen for prelude array operations.
//!
//! An `array<n, T>` is now lowered to a fat pointer `{ptr, usize}` that is allocated
//! to at least `n * sizeof(T)` bytes. The extra `usize` is an offset pointing to the
//! first element, i.e. the first element is at address `ptr + offset * sizeof(T)`.
//!
//! The rational behind the additional offset is the `pop_left` operation which bumps
//! the offset instead of mutating the pointer. This way, we can still free the original
//! pointer when the array is discarded after a pop.
//!
//! We provide utility functions [`array_fat_pointer_ty`], [`build_array_fat_pointer`], and
//! [`decompose_array_fat_pointer`] to work with array fat pointers.
//!
//! The [`DefaultArrayCodegen`] extension allocates all arrays on the heap using the
//! standard libc `malloc` and `free` functions. This behaviour can be customised
//! by providing a different implementation for [`ArrayCodegen::emit_allocate_array`]
//! and [`ArrayCodegen::emit_free_array`].
use std::iter;

use anyhow::{Ok, Result, anyhow};
use hugr_core::extension::prelude::{option_type, usize_t};
use hugr_core::extension::simple_op::{MakeExtensionOp, MakeRegisteredOp};
use hugr_core::ops::DataflowOpTrait;
use hugr_core::std_extensions::collections::array::{
    self, ArrayClone, ArrayDiscard, ArrayOp, ArrayOpDef, ArrayRepeat, ArrayScan, array_type,
};
use hugr_core::types::{TypeArg, TypeEnum};
use hugr_core::{HugrView, Node};
use inkwell::builder::Builder;
use inkwell::intrinsics::Intrinsic;
use inkwell::types::{BasicType, BasicTypeEnum, IntType, StructType};
use inkwell::values::{
    BasicValue as _, BasicValueEnum, CallableValue, IntValue, PointerValue, StructValue,
};
use inkwell::{AddressSpace, IntPredicate};
use itertools::Itertools;

use crate::emit::emit_value;
use crate::emit::libc::{emit_libc_free, emit_libc_malloc};
use crate::{CodegenExtension, CodegenExtsBuilder};
use crate::{
    emit::{EmitFuncContext, RowPromise, deaggregate_call_result},
    types::{HugrType, TypingSession},
};

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [`ArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using `ccg`
    /// as the implementation.
    #[must_use]
    pub fn add_default_array_extensions(self) -> Self {
        self.add_array_extensions(DefaultArrayCodegen)
    }

    /// Add a [`ArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using
    /// [`DefaultArrayCodegen`] as the implementation.
    pub fn add_array_extensions(self, ccg: impl ArrayCodegen + 'a) -> Self {
        self.add_extension(ArrayCodegenExtension::from(ccg))
    }
}

/// A helper trait for customising the lowering of [`hugr_core::std_extensions::collections::array`]
/// types, [`hugr_core::ops::constant::CustomConst`]s, and ops.
///
/// An `array<n, T>` is now lowered to a fat pointer `{ptr, usize}` that is allocated
/// to at least `n * sizeof(T)` bytes. The extra `usize` is an offset pointing to the
/// first element, i.e. the first element is at address `ptr + offset * sizeof(T)`.
///
/// The rational behind the additional offset is the `pop_left` operation which bumps
/// the offset instead of mutating the pointer. This way, we can still free the original
/// pointer when the array is discarded after a pop.
///
/// By default, all arrays are allocated on the heap using the standard libc `malloc`
/// and `free` functions. This behaviour can be customised by providing a different
/// implementation for [`ArrayCodegen::emit_allocate_array`] and
/// [`ArrayCodegen::emit_free_array`].
pub trait ArrayCodegen: Clone {
    /// Emit an allocation of `size` bytes and return the corresponding pointer.
    ///
    /// The default implementation allocates on the heap by emitting a call to the
    /// standard libc `malloc` function.
    fn emit_allocate_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        size: IntValue<'c>,
    ) -> Result<PointerValue<'c>> {
        let ptr = emit_libc_malloc(ctx, size.into())?;
        Ok(ptr.into_pointer_value())
    }

    /// Emit an deallocation of a pointer.
    ///
    /// The default implementation emits a call to the standard libc `free` function.
    fn emit_free_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        emit_libc_free(ctx, ptr.into())
    }

    /// Return the llvm type of [`hugr_core::std_extensions::collections::array::ARRAY_TYPENAME`].
    fn array_type<'c>(
        &self,
        session: &TypingSession<'c, '_>,
        elem_ty: BasicTypeEnum<'c>,
        _size: u64,
    ) -> impl BasicType<'c> {
        array_fat_pointer_ty(session, elem_ty)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayValue`].
    fn emit_array_value<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: &array::ArrayValue,
    ) -> Result<BasicValueEnum<'c>> {
        emit_array_value(self, ctx, value)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayOp`].
    fn emit_array_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: ArrayOp,
        inputs: Vec<BasicValueEnum<'c>>,
        outputs: RowPromise<'c>,
    ) -> Result<()> {
        emit_array_op(self, ctx, op, inputs, outputs)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayClone`] operation.
    fn emit_array_clone<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: ArrayClone,
        array_v: BasicValueEnum<'c>,
    ) -> Result<(BasicValueEnum<'c>, BasicValueEnum<'c>)> {
        emit_clone_op(self, ctx, op, array_v)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayDiscard`] operation.
    fn emit_array_discard<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: ArrayDiscard,
        array_v: BasicValueEnum<'c>,
    ) -> Result<()> {
        emit_array_discard(self, ctx, op, array_v)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayRepeat`] op.
    fn emit_array_repeat<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: ArrayRepeat,
        func: BasicValueEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        emit_repeat_op(self, ctx, op, func)
    }

    /// Emit a [`hugr_core::std_extensions::collections::array::ArrayScan`] op.
    ///
    /// Returns the resulting array and the final values of the accumulators.
    fn emit_array_scan<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: ArrayScan,
        src_array: BasicValueEnum<'c>,
        func: BasicValueEnum<'c>,
        initial_accs: &[BasicValueEnum<'c>],
    ) -> Result<(BasicValueEnum<'c>, Vec<BasicValueEnum<'c>>)> {
        emit_scan_op(
            self,
            ctx,
            op,
            src_array.into_struct_value(),
            func,
            initial_accs,
        )
    }
}

/// A trivial implementation of [`ArrayCodegen`] which passes all methods
/// through to their default implementations.
#[derive(Default, Clone)]
pub struct DefaultArrayCodegen;

impl ArrayCodegen for DefaultArrayCodegen {}

#[derive(Clone, Debug, Default)]
pub struct ArrayCodegenExtension<CCG>(CCG);

impl<CCG: ArrayCodegen> ArrayCodegenExtension<CCG> {
    pub fn new(ccg: CCG) -> Self {
        Self(ccg)
    }
}

impl<CCG: ArrayCodegen> From<CCG> for ArrayCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}

impl<CCG: ArrayCodegen> CodegenExtension for ArrayCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type((array::EXTENSION_ID, array::ARRAY_TYPENAME), {
                let ccg = self.0.clone();
                move |ts, hugr_type| {
                    let [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] = hugr_type.args() else {
                        return Err(anyhow!("Invalid type args for array type"));
                    };
                    let elem_ty = ts.llvm_type(ty)?;
                    Ok(ccg.array_type(&ts, elem_ty, *n).as_basic_type_enum())
                }
            })
            .custom_const::<array::ArrayValue>({
                let ccg = self.0.clone();
                move |context, k| ccg.emit_array_value(context, k)
            })
            .simple_extension_op::<ArrayOpDef>({
                let ccg = self.0.clone();
                move |context, args, _| {
                    ccg.emit_array_op(
                        context,
                        ArrayOp::from_extension_op(args.node().as_ref())?,
                        args.inputs,
                        args.outputs,
                    )
                }
            })
            .extension_op(array::EXTENSION_ID, array::ARRAY_CLONE_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let arr = args.inputs[0];
                    let op = ArrayClone::from_extension_op(args.node().as_ref())?;
                    let (arr1, arr2) = ccg.emit_array_clone(context, op, arr)?;
                    args.outputs.finish(context.builder(), [arr1, arr2])
                }
            })
            .extension_op(array::EXTENSION_ID, array::ARRAY_DISCARD_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let arr = args.inputs[0];
                    let op = ArrayDiscard::from_extension_op(args.node().as_ref())?;
                    ccg.emit_array_discard(context, op, arr)?;
                    args.outputs.finish(context.builder(), [])
                }
            })
            .extension_op(array::EXTENSION_ID, array::ARRAY_REPEAT_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let func = args.inputs[0];
                    let op = ArrayRepeat::from_extension_op(args.node().as_ref())?;
                    let arr = ccg.emit_array_repeat(context, op, func)?;
                    args.outputs.finish(context.builder(), [arr])
                }
            })
            .extension_op(array::EXTENSION_ID, array::ARRAY_SCAN_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let src_array = args.inputs[0];
                    let func = args.inputs[1];
                    let initial_accs = &args.inputs[2..];
                    let op = ArrayScan::from_extension_op(args.node().as_ref())?;
                    let (tgt_array, final_accs) =
                        ccg.emit_array_scan(context, op, src_array, func, initial_accs)?;
                    args.outputs
                        .finish(context.builder(), iter::once(tgt_array).chain(final_accs))
                }
            })
    }
}

fn usize_ty<'c>(ts: &TypingSession<'c, '_>) -> IntType<'c> {
    ts.llvm_type(&usize_t())
        .expect("Prelude codegen is registered")
        .into_int_type()
}

/// Returns the LLVM representation of an array value as a fat pointer.
#[must_use]
pub fn array_fat_pointer_ty<'c>(
    session: &TypingSession<'c, '_>,
    elem_ty: BasicTypeEnum<'c>,
) -> StructType<'c> {
    let iw_ctx = session.iw_context();
    iw_ctx.struct_type(
        &[
            elem_ty.ptr_type(AddressSpace::default()).into(),
            usize_ty(session).into(),
        ],
        false,
    )
}

/// Constructs an array fat pointer value.
pub fn build_array_fat_pointer<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ptr: PointerValue<'c>,
    offset: IntValue<'c>,
) -> Result<StructValue<'c>> {
    let array_ty = array_fat_pointer_ty(
        &ctx.typing_session(),
        ptr.get_type().get_element_type().try_into().unwrap(),
    );
    let array_v = array_ty.get_poison();
    let array_v = ctx
        .builder()
        .build_insert_value(array_v, ptr.as_basic_value_enum(), 0, "")?;
    let array_v = ctx
        .builder()
        .build_insert_value(array_v, offset.as_basic_value_enum(), 1, "")?;
    Ok(array_v.into_struct_value())
}

/// Returns the underlying pointer and offset stored in a fat array pointer.
pub fn decompose_array_fat_pointer<'c>(
    builder: &Builder<'c>,
    array_v: BasicValueEnum<'c>,
) -> Result<(PointerValue<'c>, IntValue<'c>)> {
    let array_v = array_v.into_struct_value();
    let array_ptr = builder.build_extract_value(array_v, 0, "array_ptr")?;
    let array_offset = builder.build_extract_value(array_v, 1, "array_offset")?;
    Ok((
        array_ptr.into_pointer_value(),
        array_offset.into_int_value(),
    ))
}

/// Helper function to allocate a fat array pointer.
///
/// Returns a pointer and a struct: The pointer points to the first element of the array (i.e. it
/// is of type `elem_ty.ptr_type()`). The struct is the fat pointer of the that stores an additional
/// offset (initialised to be 0).
pub fn build_array_alloc<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &impl ArrayCodegen,
    elem_ty: BasicTypeEnum<'c>,
    size: u64,
) -> Result<(PointerValue<'c>, StructValue<'c>)> {
    let usize_t = usize_ty(&ctx.typing_session());
    let length = usize_t.const_int(size, false);
    let size_value = ctx
        .builder()
        .build_int_mul(length, elem_ty.size_of().unwrap(), "")?;
    let ptr = ccg.emit_allocate_array(ctx, size_value)?;
    let elem_ptr = ctx
        .builder()
        .build_bit_cast(ptr, elem_ty.ptr_type(AddressSpace::default()), "")?
        .into_pointer_value();
    let offset = usize_t.const_zero();
    let array_v = build_array_fat_pointer(ctx, elem_ptr, offset)?;
    Ok((elem_ptr, array_v))
}

/// Helper function to build a loop that repeats for a given number of iterations.
///
/// The provided closure is called to build the loop body. Afterwards, the builder is positioned at
/// the end of the loop exit block.
fn build_loop<'c, T, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    iters: IntValue<'c>,
    go: impl FnOnce(&mut EmitFuncContext<'c, '_, H>, IntValue<'c>) -> Result<T>,
) -> Result<T> {
    let builder = ctx.builder();
    let idx_ty = usize_ty(&ctx.typing_session());
    let idx_ptr = builder.build_alloca(idx_ty, "")?;
    builder.build_store(idx_ptr, idx_ty.const_zero())?;

    let exit_block = ctx.new_basic_block("", None);

    let (body_block, val) = ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
        let idx = ctx.builder().build_load(idx_ptr, "")?.into_int_value();
        let val = go(ctx, idx)?;
        let builder = ctx.builder();
        let inc_idx = builder.build_int_add(idx, idx_ty.const_int(1, false), "")?;
        builder.build_store(idx_ptr, inc_idx)?;
        // Branch to the head is built later
        Ok((bb, val))
    })?;

    let head_block = ctx.build_positioned_new_block("", Some(body_block), |ctx, bb| {
        let builder = ctx.builder();
        let idx = builder.build_load(idx_ptr, "")?.into_int_value();
        let cmp = builder.build_int_compare(IntPredicate::ULT, idx, iters, "")?;
        builder.build_conditional_branch(cmp, body_block, exit_block)?;
        Ok(bb)
    })?;

    let builder = ctx.builder();
    builder.build_unconditional_branch(head_block)?;
    builder.position_at_end(body_block);
    builder.build_unconditional_branch(head_block)?;
    ctx.builder().position_at_end(exit_block);
    Ok(val)
}

/// Emits an [`array::ArrayValue`].
pub fn emit_array_value<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    value: &array::ArrayValue,
) -> Result<BasicValueEnum<'c>> {
    let ts = ctx.typing_session();
    let elem_ty = ts.llvm_type(value.get_element_type())?;
    let (elem_ptr, array_v) =
        build_array_alloc(ctx, ccg, elem_ty, value.get_contents().len() as u64)?;
    for (i, v) in value.get_contents().iter().enumerate() {
        let llvm_v = emit_value(ctx, v)?;
        let idx = ts.iw_context().i32_type().const_int(i as u64, true);
        let elem_addr = unsafe { ctx.builder().build_in_bounds_gep(elem_ptr, &[idx], "")? };
        ctx.builder().build_store(elem_addr, llvm_v)?;
    }
    Ok(array_v.into())
}

/// Emits an [`ArrayOp`].
pub fn emit_array_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayOp,
    inputs: Vec<BasicValueEnum<'c>>,
    outputs: RowPromise<'c>,
) -> Result<()> {
    let builder = ctx.builder();
    let ts = ctx.typing_session();
    let sig = op
        .clone()
        .to_extension_op()
        .unwrap()
        .signature()
        .into_owned();
    let ArrayOp {
        def,
        elem_ty: ref hugr_elem_ty,
        size,
    } = op;
    let elem_ty = ts.llvm_type(hugr_elem_ty)?;
    match def {
        ArrayOpDef::new_array => {
            let (elem_ptr, array_v) = build_array_alloc(ctx, ccg, elem_ty, size)?;
            let usize_t = usize_ty(&ctx.typing_session());
            for (i, v) in inputs.into_iter().enumerate() {
                let idx = usize_t.const_int(i as u64, true);
                let elem_addr = unsafe { ctx.builder().build_in_bounds_gep(elem_ptr, &[idx], "")? };
                ctx.builder().build_store(elem_addr, v)?;
            }
            outputs.finish(ctx.builder(), [array_v.into()])
        }
        ArrayOpDef::unpack => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::unpack expects one argument"))?;
            let (array_ptr, array_offset) = decompose_array_fat_pointer(builder, array_v)?;

            let mut result = Vec::with_capacity(size as usize);
            let usize_t = usize_ty(&ctx.typing_session());

            for i in 0..size {
                let idx = builder.build_int_add(array_offset, usize_t.const_int(i, false), "")?;
                let elem_addr = unsafe { builder.build_in_bounds_gep(array_ptr, &[idx], "")? };
                let elem_v = builder.build_load(elem_addr, "")?;
                result.push(elem_v);
            }

            outputs.finish(ctx.builder(), result)
        }
        ArrayOpDef::get => {
            let [array_v, index_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::get expects two arguments"))?;
            let (array_ptr, array_offset) = decompose_array_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();
            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("ArrayOp::get has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("ArrayOp::get output is not a sum type"))?
                };
                ts.llvm_sum_type(st.clone())?
            };

            let exit_rmb = ctx.new_row_mail_box(sig.output.iter(), "")?;

            let exit_block = ctx.build_positioned_new_block("", None, |ctx, bb| {
                outputs.finish(ctx.builder(), exit_rmb.read_vec(ctx.builder(), [])?)?;
                Ok(bb)
            })?;

            let success_block =
                ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
                    let builder = ctx.builder();
                    // inside `success_block` we know `index_v` to be in bounds
                    let index_v = builder.build_int_add(index_v, array_offset, "")?;
                    let elem_addr =
                        unsafe { builder.build_in_bounds_gep(array_ptr, &[index_v], "")? };
                    let elem_v = builder.build_load(elem_addr, "")?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![elem_v])?;
                    exit_rmb.write(ctx.builder(), [success_v.into(), array_v])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v = res_sum_ty.build_tag(builder, 0, vec![])?;
                    exit_rmb.write(ctx.builder(), [failure_v.into(), array_v])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let builder = ctx.builder();
            let is_success = builder.build_int_compare(
                IntPredicate::ULT,
                index_v,
                index_v.get_type().const_int(size, false),
                "",
            )?;

            builder.build_conditional_branch(is_success, success_block, failure_block)?;
            builder.position_at_end(exit_block);
            Ok(())
        }
        ArrayOpDef::set => {
            let [array_v, index_v, value_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::set expects three arguments"))?;
            let (array_ptr, array_offset) = decompose_array_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();

            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("ArrayOp::set has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("ArrayOp::set output is not a sum type"))?
                };
                ts.llvm_sum_type(st.clone())?
            };

            let exit_rmb = ctx.new_row_mail_box([res_hugr_ty], "")?;

            let exit_block = ctx.build_positioned_new_block("", None, |ctx, bb| {
                outputs.finish(ctx.builder(), exit_rmb.read_vec(ctx.builder(), [])?)?;
                Ok(bb)
            })?;

            let success_block =
                ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
                    let builder = ctx.builder();
                    // inside `success_block` we know `index_v` to be in bounds.
                    let index_v = builder.build_int_add(index_v, array_offset, "")?;
                    let elem_addr =
                        unsafe { builder.build_in_bounds_gep(array_ptr, &[index_v], "")? };
                    let elem_v = builder.build_load(elem_addr, "")?;
                    builder.build_store(elem_addr, value_v)?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![elem_v, array_v])?;
                    exit_rmb.write(ctx.builder(), [success_v.into()])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v = res_sum_ty.build_tag(builder, 0, vec![value_v, array_v])?;
                    exit_rmb.write(ctx.builder(), [failure_v.into()])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let builder = ctx.builder();
            let is_success = builder.build_int_compare(
                IntPredicate::ULT,
                index_v,
                index_v.get_type().const_int(size, false),
                "",
            )?;
            builder.build_conditional_branch(is_success, success_block, failure_block)?;
            builder.position_at_end(exit_block);
            Ok(())
        }
        ArrayOpDef::swap => {
            let [array_v, index1_v, index2_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::swap expects three arguments"))?;
            let (array_ptr, array_offset) = decompose_array_fat_pointer(builder, array_v)?;
            let index1_v = index1_v.into_int_value();
            let index2_v = index2_v.into_int_value();

            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("ArrayOp::swap has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("ArrayOp::swap output is not a sum type"))?
                };
                ts.llvm_sum_type(st.clone())?
            };

            let exit_rmb = ctx.new_row_mail_box([res_hugr_ty], "")?;

            let exit_block = ctx.build_positioned_new_block("", None, |ctx, bb| {
                outputs.finish(ctx.builder(), exit_rmb.read_vec(ctx.builder(), [])?)?;
                Ok(bb)
            })?;

            let success_block =
                ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
                    // if `index1_v` == `index2_v` then the following is a no-op.
                    // We could check for this: either with a select instruction
                    // here, or by branching to another case in earlier.
                    // Doing so would generate better code in cases where the
                    // optimiser can determine that the indices are the same, at
                    // the cost of worse code in cases where it cannot.
                    // For now we choose the simpler option of omitting the check.
                    let builder = ctx.builder();
                    // inside `success_block` we know `index1_v` and `index2_v`
                    // to be in bounds.
                    let index1_v = builder.build_int_add(index1_v, array_offset, "")?;
                    let index2_v = builder.build_int_add(index2_v, array_offset, "")?;
                    let elem1_addr =
                        unsafe { builder.build_in_bounds_gep(array_ptr, &[index1_v], "")? };
                    let elem1_v = builder.build_load(elem1_addr, "")?;
                    let elem2_addr =
                        unsafe { builder.build_in_bounds_gep(array_ptr, &[index2_v], "")? };
                    let elem2_v = builder.build_load(elem2_addr, "")?;
                    builder.build_store(elem1_addr, elem2_v)?;
                    builder.build_store(elem2_addr, elem1_v)?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![array_v])?;
                    exit_rmb.write(ctx.builder(), [success_v.into()])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v = res_sum_ty.build_tag(builder, 0, vec![array_v])?;
                    exit_rmb.write(ctx.builder(), [failure_v.into()])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let builder = ctx.builder();
            let is_success = {
                let index1_ok = builder.build_int_compare(
                    IntPredicate::ULT,
                    index1_v,
                    index1_v.get_type().const_int(size, false),
                    "",
                )?;
                let index2_ok = builder.build_int_compare(
                    IntPredicate::ULT,
                    index2_v,
                    index2_v.get_type().const_int(size, false),
                    "",
                )?;
                builder.build_and(index1_ok, index2_ok, "")?
            };
            builder.build_conditional_branch(is_success, success_block, failure_block)?;
            builder.position_at_end(exit_block);
            Ok(())
        }
        ArrayOpDef::pop_left => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::pop_left expects one argument"))?;
            let r = emit_pop_op(
                ctx,
                hugr_elem_ty.clone(),
                size,
                array_v.into_struct_value(),
                true,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        ArrayOpDef::pop_right => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::pop_right expects one argument"))?;
            let r = emit_pop_op(
                ctx,
                hugr_elem_ty.clone(),
                size,
                array_v.into_struct_value(),
                false,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        ArrayOpDef::discard_empty => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::discard_empty expects one argument"))?;
            let (ptr, _) = decompose_array_fat_pointer(builder, array_v)?;
            ccg.emit_free_array(ctx, ptr)?;
            outputs.finish(ctx.builder(), [])
        }
        _ => todo!(),
    }
}

/// Emits an [`ArrayClone`] op.
pub fn emit_clone_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayClone,
    array_v: BasicValueEnum<'c>,
) -> Result<(BasicValueEnum<'c>, BasicValueEnum<'c>)> {
    let elem_ty = ctx.llvm_type(&op.elem_ty)?;
    let (array_ptr, array_offset) = decompose_array_fat_pointer(ctx.builder(), array_v)?;
    let (other_ptr, other_array_v) = build_array_alloc(ctx, ccg, elem_ty, op.size)?;
    let src_ptr = unsafe {
        ctx.builder()
            .build_in_bounds_gep(array_ptr, &[array_offset], "")?
    };
    let length = usize_ty(&ctx.typing_session()).const_int(op.size, false);
    let size_value = ctx
        .builder()
        .build_int_mul(length, elem_ty.size_of().unwrap(), "")?;
    let is_volatile = ctx.iw_context().bool_type().const_zero();

    let memcpy_intrinsic = Intrinsic::find("llvm.memcpy").unwrap();
    let memcpy = memcpy_intrinsic
        .get_declaration(
            ctx.get_current_module(),
            &[
                other_ptr.get_type().into(),
                src_ptr.get_type().into(),
                size_value.get_type().into(),
            ],
        )
        .unwrap();
    ctx.builder().build_call(
        memcpy,
        &[
            other_ptr.into(),
            src_ptr.into(),
            size_value.into(),
            is_volatile.into(),
        ],
        "",
    )?;
    Ok((array_v, other_array_v.into()))
}

/// Emits an [`ArrayDiscard`] op.
pub fn emit_array_discard<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    _op: ArrayDiscard,
    array_v: BasicValueEnum<'c>,
) -> Result<()> {
    let array_ptr =
        ctx.builder()
            .build_extract_value(array_v.into_struct_value(), 0, "array_ptr")?;
    ccg.emit_free_array(ctx, array_ptr.into_pointer_value())?;
    Ok(())
}

/// Emits the [`ArrayOpDef::pop_left`] and [`ArrayOpDef::pop_right`] operations.
fn emit_pop_op<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    elem_ty: HugrType,
    size: u64,
    array_v: StructValue<'c>,
    pop_left: bool,
) -> Result<BasicValueEnum<'c>> {
    let ts = ctx.typing_session();
    let builder = ctx.builder();
    let (array_ptr, array_offset) = decompose_array_fat_pointer(builder, array_v.into())?;
    let ret_ty = ts.llvm_sum_type(option_type(vec![
        elem_ty.clone(),
        array_type(size.saturating_add_signed(-1), elem_ty),
    ]))?;
    if size == 0 {
        return Ok(ret_ty.build_tag(builder, 0, vec![])?.into());
    }
    let (elem_ptr, new_array_offset) = {
        if pop_left {
            let new_array_offset = builder.build_int_add(
                array_offset,
                usize_ty(&ts).const_int(1, false),
                "new_offset",
            )?;
            let elem_ptr = unsafe { builder.build_in_bounds_gep(array_ptr, &[array_offset], "") }?;
            (elem_ptr, new_array_offset)
        } else {
            let idx = builder.build_int_add(
                array_offset,
                usize_ty(&ts).const_int(size - 1, false),
                "",
            )?;
            let elem_ptr = unsafe { builder.build_in_bounds_gep(array_ptr, &[idx], "") }?;
            (elem_ptr, array_offset)
        }
    };
    let elem_v = builder.build_load(elem_ptr, "")?;
    let new_array_v = build_array_fat_pointer(ctx, array_ptr, new_array_offset)?;

    Ok(ret_ty
        .build_tag(ctx.builder(), 1, vec![elem_v, new_array_v.into()])?
        .into())
}

/// Emits an [`ArrayRepeat`] op.
pub fn emit_repeat_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayRepeat,
    func: BasicValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let elem_ty = ctx.llvm_type(&op.elem_ty)?;
    let (ptr, array_v) = build_array_alloc(ctx, ccg, elem_ty, op.size)?;
    let array_len = usize_ty(&ctx.typing_session()).const_int(op.size, false);
    build_loop(ctx, array_len, |ctx, idx| {
        let builder = ctx.builder();
        let func_ptr = CallableValue::try_from(func.into_pointer_value())
            .map_err(|()| anyhow!("ArrayOpDef::repeat expects a function pointer"))?;
        let v = builder
            .build_call(func_ptr, &[], "")?
            .try_as_basic_value()
            .left()
            .ok_or(anyhow!("ArrayOpDef::repeat function must return a value"))?;
        let elem_addr = unsafe { builder.build_in_bounds_gep(ptr, &[idx], "")? };
        builder.build_store(elem_addr, v)?;
        Ok(())
    })?;
    Ok(array_v.into())
}

/// Emits an [`ArrayScan`] op.
///
/// Returns the resulting array and the final values of the accumulators.
pub fn emit_scan_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl ArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayScan,
    src_array_v: StructValue<'c>,
    func: BasicValueEnum<'c>,
    initial_accs: &[BasicValueEnum<'c>],
) -> Result<(BasicValueEnum<'c>, Vec<BasicValueEnum<'c>>)> {
    let (src_ptr, src_offset) = decompose_array_fat_pointer(ctx.builder(), src_array_v.into())?;
    let tgt_elem_ty = ctx.llvm_type(&op.tgt_ty)?;
    // TODO: If `sizeof(op.src_ty) >= sizeof(op.tgt_ty)`, we could reuse the memory
    // from `src` instead of allocating a fresh array
    let (tgt_ptr, tgt_array_v) = build_array_alloc(ctx, ccg, tgt_elem_ty, op.size)?;
    let array_len = usize_ty(&ctx.typing_session()).const_int(op.size, false);
    let acc_tys: Vec<_> = op
        .acc_tys
        .iter()
        .map(|ty| ctx.llvm_type(ty))
        .try_collect()?;
    let builder = ctx.builder();
    let acc_ptrs: Vec<_> = acc_tys
        .iter()
        .map(|ty| builder.build_alloca(*ty, ""))
        .try_collect()?;
    for (ptr, initial_val) in acc_ptrs.iter().zip(initial_accs) {
        builder.build_store(*ptr, *initial_val)?;
    }

    build_loop(ctx, array_len, |ctx, idx| {
        let builder = ctx.builder();
        let func_ptr = CallableValue::try_from(func.into_pointer_value())
            .map_err(|()| anyhow!("ArrayOpDef::scan expects a function pointer"))?;
        let src_idx = builder.build_int_add(idx, src_offset, "")?;
        let src_elem_addr = unsafe { builder.build_in_bounds_gep(src_ptr, &[src_idx], "")? };
        let src_elem = builder.build_load(src_elem_addr, "")?;
        let mut args = vec![src_elem.into()];
        for ptr in &acc_ptrs {
            args.push(builder.build_load(*ptr, "")?.into());
        }
        let call = builder.build_call(func_ptr, args.as_slice(), "")?;
        let call_results = deaggregate_call_result(builder, call, 1 + acc_tys.len())?;
        let tgt_elem_addr = unsafe { builder.build_in_bounds_gep(tgt_ptr, &[idx], "")? };
        builder.build_store(tgt_elem_addr, call_results[0])?;
        for (ptr, next_act) in acc_ptrs.iter().zip(call_results[1..].iter()) {
            builder.build_store(*ptr, *next_act)?;
        }
        Ok(())
    })?;

    ccg.emit_free_array(ctx, src_ptr)?;
    let builder = ctx.builder();
    let final_accs = acc_ptrs
        .into_iter()
        .map(|ptr| builder.build_load(ptr, ""))
        .try_collect()?;
    Ok((tgt_array_v.into(), final_accs))
}

#[cfg(test)]
mod test {
    use hugr_core::builder::{DataflowHugr, HugrBuilder};
    use hugr_core::extension::prelude::either_type;
    use hugr_core::ops::Tag;
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::std_extensions::collections::array::op_builder::build_all_array_ops;
    use hugr_core::std_extensions::collections::array::{
        self, ArrayOpBuilder, ArrayRepeat, ArrayScan, array_type,
    };
    use hugr_core::types::Type;
    use hugr_core::{
        builder::{Dataflow, DataflowSubContainer, SubContainer},
        extension::{
            ExtensionRegistry,
            prelude::{self, ConstUsize, UnwrapBuilder as _, bool_t, option_type, usize_t},
        },
        ops::Value,
        std_extensions::{
            arithmetic::{
                int_ops::{self},
                int_types::{self, ConstInt, int_type},
            },
            logic,
        },
        type_row,
        types::Signature,
    };
    use itertools::Itertools as _;
    use rstest::rstest;

    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        test::{TestContext, exec_ctx, llvm_ctx},
        utils::{IntOpBuilder, LogicOpBuilder},
    };

    #[rstest]
    fn emit_all_ops(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                build_all_array_ops(builder.dfg_builder_endo([]).unwrap())
                    .finish_sub_container()
                    .unwrap();
                builder.finish_hugr().unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
        });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_get(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let us2 = builder.add_load_value(ConstUsize::new(2));
                let arr = builder.add_new_array(usize_t(), [us1, us2]).unwrap();
                let (_, arr) = builder.add_array_get(usize_t(), 2, arr, us1).unwrap();
                builder.add_array_discard(usize_t(), 2, arr).unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
        });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_clone(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let us2 = builder.add_load_value(ConstUsize::new(2));
                let arr = builder.add_new_array(usize_t(), [us1, us2]).unwrap();
                let (arr1, arr2) = builder.add_array_clone(usize_t(), 2, arr).unwrap();
                builder.add_array_discard(usize_t(), 2, arr1).unwrap();
                builder.add_array_discard(usize_t(), 2, arr2).unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
        });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_array_value(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(STD_REG.to_owned())
            .with_outs(vec![array_type(2, usize_t())])
            .finish(|mut builder| {
                let vs = vec![ConstUsize::new(1).into(), ConstUsize::new(2).into()];
                let arr = builder.add_load_value(array::ArrayValue::new(usize_t(), vs));
                builder.finish_hugr_with_outputs([arr]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
        });
        check_emission!(hugr, llvm_ctx);
    }

    // #[rstest]
    // #[case(1, 2, 3)]
    // #[case(0, 0, 0)]
    // #[case(10, 20, 30)]
    // fn exec_unpack_and_sum(mut exec_ctx: TestContext, #[case] a: u64, #[case] b: u64, #[case] expected: u64) {
    //     let hugr = SimpleHugrConfig::new()
    //         .with_extensions(exec_registry())
    //         .with_outs(vec![usize_t()])
    //         .finish(|mut builder| {
    //             // Create an array with the test values
    //             let values = vec![ConstUsize::new(a).into(), ConstUsize::new(b).into()];
    //             let arr = builder.add_load_value(array::ArrayValue::new(usize_t(), values));

    //             // Unpack the array
    //             let [val_a, val_b] = builder.add_array_unpack(usize_t(), 2, arr).unwrap().try_into().unwrap();

    //             // Add the values
    //             let sum = {
    //                 let int_ty = int_type(6);
    //                 let a_int = builder.cast(val_a, int_ty.clone()).unwrap();
    //                 let b_int = builder.cast(val_b, int_ty.clone()).unwrap();
    //                 let sum_int = builder.add_iadd(6, a_int, b_int).unwrap();
    //                 builder.cast(sum_int, usize_t()).unwrap()
    //             };

    //             builder.finish_hugr_with_outputs([sum]).unwrap()
    //         });
    //     exec_ctx.add_extensions(|cge| {
    //         cge.add_default_prelude_extensions()
    //             .add_default_array_extensions()
    //             .add_default_int_extensions()
    //     });
    //     assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    // }

    fn exec_registry() -> ExtensionRegistry {
        ExtensionRegistry::new([
            int_types::EXTENSION.to_owned(),
            int_ops::EXTENSION.to_owned(),
            logic::EXTENSION.to_owned(),
            prelude::PRELUDE.to_owned(),
            array::EXTENSION.to_owned(),
        ])
    }

    #[rstest]
    #[case(0, 1)]
    #[case(1, 2)]
    #[case(3, 0)]
    #[case(999999, 0)]
    fn exec_get(mut exec_ctx: TestContext, #[case] index: u64, #[case] expected: u64) {
        // We build a HUGR that:
        // - Creates an array of [1,2]
        // - Gets the element at the given index
        // - Returns the element if the index is in bounds, otherwise 0
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let us0 = builder.add_load_value(ConstUsize::new(0));
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let us2 = builder.add_load_value(ConstUsize::new(2));
                let arr = builder.add_new_array(usize_t(), [us1, us2]).unwrap();
                let i = builder.add_load_value(ConstUsize::new(index));
                let (get_r, arr) = builder.add_array_get(usize_t(), 2, arr, i).unwrap();
                builder.add_array_discard(usize_t(), 2, arr).unwrap();
                let r = {
                    let ot = option_type(usize_t());
                    let variants = (0..ot.num_variants())
                        .map(|i| ot.get_variant(i).cloned().unwrap().try_into().unwrap())
                        .collect_vec();
                    let mut builder = builder
                        .conditional_builder((variants, get_r), [], usize_t().into())
                        .unwrap();
                    {
                        let failure_case = builder.case_builder(0).unwrap();
                        failure_case.finish_with_outputs([us0]).unwrap();
                    }
                    {
                        let success_case = builder.case_builder(1).unwrap();
                        let inputs = success_case.input_wires();
                        success_case.finish_with_outputs(inputs).unwrap();
                    }
                    builder.finish_sub_container().unwrap().out_wire(0)
                };
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
        });
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(0, 3, 1, [3,2])]
    #[case(1, 3, 2, [1,3])]
    #[case(2, 3, 3, [1,2])]
    #[case(999999, 3, 3, [1,2])]
    fn exec_set(
        mut exec_ctx: TestContext,
        #[case] index: u64,
        #[case] value: u64,
        #[case] expected_elem: u64,
        #[case] expected_arr: [u64; 2],
    ) {
        // We build a HUGR that
        // - Creates an array: [1,2]
        // - Sets the element at the given index to the given value
        // - Checks the following, returning 1 iff they are all true:
        //   - The element returned from set is `expected_elem`
        //   - The Oth element of the resulting array is `expected_arr_0`

        use hugr_core::extension::prelude::either_type;
        let int_ty = int_type(3);
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let us0 = builder.add_load_value(ConstUsize::new(0));
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let i1 = builder.add_load_value(ConstInt::new_u(3, 1).unwrap());
                let i2 = builder.add_load_value(ConstInt::new_u(3, 2).unwrap());
                let arr = builder.add_new_array(int_ty.clone(), [i1, i2]).unwrap();
                let index = builder.add_load_value(ConstUsize::new(index));
                let value = builder.add_load_value(ConstInt::new_u(3, value).unwrap());
                let get_r = builder
                    .add_array_set(int_ty.clone(), 2, arr, index, value)
                    .unwrap();
                let r = {
                    let res_sum_ty = {
                        let row = vec![int_ty.clone(), array_type(2, int_ty.clone())];
                        either_type(row.clone(), row)
                    };
                    let variants = (0..res_sum_ty.num_variants())
                        .map(|i| {
                            res_sum_ty
                                .get_variant(i)
                                .cloned()
                                .unwrap()
                                .try_into()
                                .unwrap()
                        })
                        .collect_vec();
                    let mut builder = builder
                        .conditional_builder((variants, get_r), [], bool_t().into())
                        .unwrap();
                    for i in 0..2 {
                        let mut builder = builder.case_builder(i).unwrap();
                        let [elem, arr] = builder.input_wires_arr();
                        let expected_elem =
                            builder.add_load_value(ConstInt::new_u(3, expected_elem).unwrap());
                        let expected_arr_0 =
                            builder.add_load_value(ConstInt::new_u(3, expected_arr[0]).unwrap());
                        let expected_arr_1 =
                            builder.add_load_value(ConstInt::new_u(3, expected_arr[1]).unwrap());
                        let (r, arr) = builder.add_array_get(int_ty.clone(), 2, arr, us0).unwrap();
                        let [arr_0] = builder
                            .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                            .unwrap();
                        let (r, arr) = builder.add_array_get(int_ty.clone(), 2, arr, us1).unwrap();
                        let [arr_1] = builder
                            .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                            .unwrap();
                        let elem_eq = builder.add_ieq(3, elem, expected_elem).unwrap();
                        let arr_0_eq = builder.add_ieq(3, arr_0, expected_arr_0).unwrap();
                        let arr_1_eq = builder.add_ieq(3, arr_1, expected_arr_1).unwrap();
                        let r = builder.add_and(elem_eq, arr_0_eq).unwrap();
                        let r = builder.add_and(r, arr_1_eq).unwrap();
                        builder.add_array_discard(int_ty.clone(), 2, arr).unwrap();
                        builder.finish_with_outputs([r]).unwrap();
                    }
                    builder.finish_sub_container().unwrap().out_wire(0)
                };
                let r = {
                    let mut conditional = builder
                        .conditional_builder(([type_row![], type_row![]], r), [], usize_t().into())
                        .unwrap();
                    conditional
                        .case_builder(0)
                        .unwrap()
                        .finish_with_outputs([us0])
                        .unwrap();
                    conditional
                        .case_builder(1)
                        .unwrap()
                        .finish_with_outputs([us1])
                        .unwrap();
                    conditional.finish_sub_container().unwrap().out_wire(0)
                };
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
                .add_logic_extensions()
        });
        assert_eq!(1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(0, 1, [2,1], true)]
    #[case(0, 0, [1,2], true)]
    #[case(0, 2, [1,2], false)]
    #[case(2, 0, [1,2], false)]
    #[case(9999999, 0, [1,2], false)]
    #[case(0, 9999999, [1,2], false)]
    fn exec_swap(
        mut exec_ctx: TestContext,
        #[case] index1: u64,
        #[case] index2: u64,
        #[case] expected_arr: [u64; 2],
        #[case] expected_succeeded: bool,
    ) {
        // We build a HUGR that:
        // - Creates an array: [1 ,2]
        // - Swaps the elements at the given indices
        // - Checks the following, returning 1 iff the following are all true:
        //  - The element at index 0 is `expected_elem_0`
        //  - The swap operation succeeded iff `expected_succeeded`

        let int_ty = int_type(3);
        let arr_ty = array_type(2, int_ty.clone());
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let us0 = builder.add_load_value(ConstUsize::new(0));
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let i1 = builder.add_load_value(ConstInt::new_u(3, 1).unwrap());
                let i2 = builder.add_load_value(ConstInt::new_u(3, 2).unwrap());
                let arr = builder.add_new_array(int_ty.clone(), [i1, i2]).unwrap();

                let index1 = builder.add_load_value(ConstUsize::new(index1));
                let index2 = builder.add_load_value(ConstUsize::new(index2));
                let r = builder
                    .add_array_swap(int_ty.clone(), 2, arr, index1, index2)
                    .unwrap();
                let [arr, was_expected_success] = {
                    let mut conditional = builder
                        .conditional_builder(
                            (
                                [vec![arr_ty.clone()].into(), vec![arr_ty.clone()].into()],
                                r,
                            ),
                            [],
                            vec![arr_ty, bool_t()].into(),
                        )
                        .unwrap();
                    for i in 0..2 {
                        let mut case = conditional.case_builder(i).unwrap();
                        let [arr] = case.input_wires_arr();
                        let was_expected_success =
                            case.add_load_value(if (i == 1) == expected_succeeded {
                                Value::true_val()
                            } else {
                                Value::false_val()
                            });
                        case.finish_with_outputs([arr, was_expected_success])
                            .unwrap();
                    }
                    conditional.finish_sub_container().unwrap().outputs_arr()
                };
                let (r, arr) = builder.add_array_get(int_ty.clone(), 2, arr, us0).unwrap();
                let elem_0 = builder
                    .build_unwrap_sum::<1>(1, option_type(int_ty.clone()), r)
                    .unwrap()[0];
                let (r, arr) = builder.add_array_get(int_ty.clone(), 2, arr, us1).unwrap();
                let elem_1 = builder
                    .build_unwrap_sum::<1>(1, option_type(int_ty.clone()), r)
                    .unwrap()[0];
                let expected_elem_0 =
                    builder.add_load_value(ConstInt::new_u(3, expected_arr[0]).unwrap());
                let elem_0_ok = builder.add_ieq(3, elem_0, expected_elem_0).unwrap();
                let expected_elem_1 =
                    builder.add_load_value(ConstInt::new_u(3, expected_arr[1]).unwrap());
                let elem_1_ok = builder.add_ieq(3, elem_1, expected_elem_1).unwrap();
                let r = builder.add_and(was_expected_success, elem_0_ok).unwrap();
                let r = builder.add_and(r, elem_1_ok).unwrap();
                let r = {
                    let mut conditional = builder
                        .conditional_builder(([type_row![], type_row![]], r), [], usize_t().into())
                        .unwrap();
                    conditional
                        .case_builder(0)
                        .unwrap()
                        .finish_with_outputs([us0])
                        .unwrap();
                    conditional
                        .case_builder(1)
                        .unwrap()
                        .finish_with_outputs([us1])
                        .unwrap();
                    conditional.finish_sub_container().unwrap().out_wire(0)
                };
                builder.add_array_discard(int_ty.clone(), 2, arr).unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
                .add_logic_extensions()
        });
        assert_eq!(1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(0, 5)]
    #[case(1, 5)]
    fn exec_clone(mut exec_ctx: TestContext, #[case] index: u64, #[case] new_v: u64) {
        // We build a HUGR that:
        // - Creates an array: [1, 2]
        // - Clones the array
        // - Mutates the original at the given index
        // - Returns the unchanged element of the cloned array

        let int_ty = int_type(3);
        let arr_ty = array_type(2, int_ty.clone());
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let idx = builder.add_load_value(ConstUsize::new(index));
                let i1 = builder.add_load_value(ConstInt::new_u(3, 1).unwrap());
                let i2 = builder.add_load_value(ConstInt::new_u(3, 2).unwrap());
                let inew = builder.add_load_value(ConstInt::new_u(3, new_v).unwrap());
                let arr = builder.add_new_array(int_ty.clone(), [i1, i2]).unwrap();

                let (arr, arr_clone) = builder.add_array_clone(int_ty.clone(), 2, arr).unwrap();
                let r = builder
                    .add_array_set(int_ty.clone(), 2, arr, idx, inew)
                    .unwrap();
                let [_, arr] = builder
                    .build_unwrap_sum(
                        1,
                        either_type(
                            vec![int_ty.clone(), arr_ty.clone()],
                            vec![int_ty.clone(), arr_ty.clone()],
                        ),
                        r,
                    )
                    .unwrap();
                let (r, arr_clone) = builder
                    .add_array_get(int_ty.clone(), 2, arr_clone, idx)
                    .unwrap();
                let [elem] = builder
                    .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                    .unwrap();
                builder.add_array_discard(int_ty.clone(), 2, arr).unwrap();
                builder
                    .add_array_discard(int_ty.clone(), 2, arr_clone)
                    .unwrap();
                builder.finish_hugr_with_outputs([elem]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
                .add_logic_extensions()
        });
        assert_eq!([1, 2][index as usize], exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(&[], 0)]
    #[case(&[true], 1)]
    #[case(&[false], 4)]
    #[case(&[true, true], 3)]
    #[case(&[false, false], 6)]
    #[case(&[true, false, true], 7)]
    #[case(&[false, true, false], 7)]
    fn exec_pop(mut exec_ctx: TestContext, #[case] from_left: &[bool], #[case] expected: u64) {
        // We build a HUGR that:
        // - Creates an array: [1,2,4]
        // - Pops `num` elements from the left or right
        // - Returns the sum of the popped elements

        let array_contents = [1, 2, 4];
        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let new_array_args = array_contents
                    .iter()
                    .map(|&i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let mut arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();
                for (i, left) in from_left.iter().enumerate() {
                    let array_size = (array_contents.len() - i) as u64;
                    let pop_res = if *left {
                        builder
                            .add_array_pop_left(int_ty.clone(), array_size, arr)
                            .unwrap()
                    } else {
                        builder
                            .add_array_pop_right(int_ty.clone(), array_size, arr)
                            .unwrap()
                    };
                    let [elem, new_arr] = builder
                        .build_unwrap_sum(
                            1,
                            option_type(vec![
                                int_ty.clone(),
                                array_type(array_size - 1, int_ty.clone()),
                            ]),
                            pop_res,
                        )
                        .unwrap();
                    arr = new_arr;
                    r = builder.add_iadd(6, r, elem).unwrap();
                }
                builder
                    .add_array_discard(
                        int_ty.clone(),
                        (array_contents.len() - from_left.len()) as u64,
                        arr,
                    )
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(&[], 0)]
    #[case(&[1, 2], 3)]
    #[case(&[6, 6, 6], 18)]
    fn exec_unpack(
        mut exec_ctx: TestContext,
        #[case] array_contents: &[u64],
        #[case] expected: u64,
    ) {
        // We build a HUGR that:
        // - Loads an array with the given contents
        // - Unpacks all the elements
        // - Returns the sum of the elements

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let array = array::ArrayValue::new(
                    int_ty.clone(),
                    array_contents
                        .iter()
                        .map(|&i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let array = builder.add_load_value(array);
                let unpacked = builder
                    .add_array_unpack(int_ty.clone(), array_contents.len() as u64, array)
                    .unwrap();
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for elem in unpacked {
                    r = builder.add_iadd(6, r, elem).unwrap();
                }

                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(5, 42, 0)]
    #[case(5, 42, 1)]
    #[case(5, 42, 2)]
    #[case(5, 42, 3)]
    #[case(5, 42, 4)]
    fn exec_repeat(
        mut exec_ctx: TestContext,
        #[case] size: u64,
        #[case] value: u64,
        #[case] idx: u64,
    ) {
        // We build a HUGR that:
        // - Contains a nested function that returns `value`
        // - Creates an array of length `size` populated via this function
        // - Looks up the value at `idx` and returns it

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut mb = builder.module_root_builder();
                let mut func = mb
                    .define_function("foo", Signature::new(vec![], vec![int_ty.clone()]))
                    .unwrap();
                let v = func.add_load_value(ConstInt::new_u(6, value).unwrap());
                let func_id = func.finish_with_outputs(vec![v]).unwrap();
                let func_v = builder.load_func(func_id.handle(), &[]).unwrap();
                let repeat = ArrayRepeat::new(int_ty.clone(), size);
                let arr = builder
                    .add_dataflow_op(repeat, vec![func_v])
                    .unwrap()
                    .out_wire(0);
                let idx_v = builder.add_load_value(ConstUsize::new(idx));
                let (get_res, arr) = builder
                    .add_array_get(int_ty.clone(), size, arr, idx_v)
                    .unwrap();
                let [elem] = builder
                    .build_unwrap_sum(1, option_type(vec![int_ty.clone()]), get_res)
                    .unwrap();
                builder
                    .add_array_discard(int_ty.clone(), size, arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([elem]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(value, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(10, 1)]
    #[case(10, 2)]
    #[case(0, 1)]
    fn exec_scan_map(mut exec_ctx: TestContext, #[case] size: u64, #[case] inc: u64) {
        // We build a HUGR that:
        // - Creates an array [1, 2, 3, ..., size]
        // - Maps a function that increments each element by `inc`
        // - Returns the sum of the array elements

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let new_array_args = (0..size)
                    .map(|i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();

                let mut mb = builder.module_root_builder();
                let mut func = mb
                    .define_function(
                        "foo",
                        Signature::new(vec![int_ty.clone()], vec![int_ty.clone()]),
                    )
                    .unwrap();
                let [elem] = func.input_wires_arr();
                let delta = func.add_load_value(ConstInt::new_u(6, inc).unwrap());
                let out = func.add_iadd(6, elem, delta).unwrap();
                let func_id = func.finish_with_outputs(vec![out]).unwrap();
                let func_v = builder.load_func(func_id.handle(), &[]).unwrap();
                let scan = ArrayScan::new(int_ty.clone(), int_ty.clone(), vec![], size);
                let mut arr = builder
                    .add_dataflow_op(scan, [arr, func_v])
                    .unwrap()
                    .out_wire(0);

                for i in 0..size {
                    let array_size = size - i;
                    let pop_res = builder
                        .add_array_pop_left(int_ty.clone(), array_size, arr)
                        .unwrap();
                    let [elem, new_arr] = builder
                        .build_unwrap_sum(
                            1,
                            option_type(vec![
                                int_ty.clone(),
                                array_type(array_size - 1, int_ty.clone()),
                            ]),
                            pop_res,
                        )
                        .unwrap();
                    arr = new_arr;
                    r = builder.add_iadd(6, r, elem).unwrap();
                }
                builder
                    .add_array_discard_empty(int_ty.clone(), arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        let expected: u64 = (inc..size + inc).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(10)]
    fn exec_scan_fold(mut exec_ctx: TestContext, #[case] size: u64) {
        // We build a HUGR that:
        // - Creates an array [1, 2, 3, ..., size]
        // - Sums up the elements of the array using a scan and returns that sum

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let new_array_args = (0..size)
                    .map(|i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();

                let mut mb = builder.module_root_builder();
                let mut func = mb
                    .define_function(
                        "foo",
                        Signature::new(
                            vec![int_ty.clone(), int_ty.clone()],
                            vec![Type::UNIT, int_ty.clone()],
                        ),
                    )
                    .unwrap();
                let [elem, acc] = func.input_wires_arr();
                let acc = func.add_iadd(6, elem, acc).unwrap();
                let unit = func
                    .add_dataflow_op(Tag::new(0, vec![type_row![]]), [])
                    .unwrap()
                    .out_wire(0);
                let func_id = func.finish_with_outputs(vec![unit, acc]).unwrap();
                let func_v = builder.load_func(func_id.handle(), &[]).unwrap();
                let scan = ArrayScan::new(int_ty.clone(), Type::UNIT, vec![int_ty.clone()], size);
                let zero = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let [arr, sum] = builder
                    .add_dataflow_op(scan, [arr, func_v, zero])
                    .unwrap()
                    .outputs_arr();
                builder.add_array_discard(Type::UNIT, size, arr).unwrap();
                builder.finish_hugr_with_outputs([sum]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        let expected: u64 = (0..size).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }
}

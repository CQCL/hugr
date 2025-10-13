//! Codegen for prelude borrow array operations.
//!
//! A `borrow_array<n, T>` is lowered to a fat pointer `{ptr, mask_ptr, usize}` that is
//! allocated to at least `n * sizeof(T)` bytes. The second pointer is a bit-packed mask
//! storing which array elements have been borrowed (1=borrowed, 0=available). It should
//! be allocated to at least `ceil(n / sizeof(usize)) * sizeof(usize)` bytes. The extra
//! `usize` is an offset pointing to the first element, i.e. the first element is at address
//! `ptr + offset * sizeof(T)`.
//!
//! The rationale behind the additional offset is the `pop_left` operation which bumps
//! the offset instead of mutating the pointer. This way, we can still free the original
//! pointer when the array is discarded after a pop.
//!
//! We provide utility functions [`barray_fat_pointer_ty`], [`build_barray_fat_pointer`], and
//! [`decompose_barray_fat_pointer`] to work with borrow-array fat pointers.
//!
//! The [`DefaultBorrowArrayCodegen`] extension allocates all arrays on the heap using the
//! standard libc `malloc` and `free` functions. This behaviour can be customised
//! by providing a different implementation for [`BorrowArrayCodegen::emit_allocate_array`]
//! and [`BorrowArrayCodegen::emit_free_array`].
use std::iter;
use std::sync::LazyLock;

use anyhow::{Ok, Result, anyhow};
use hugr_core::extension::prelude::{ConstError, option_type, usize_t};
use hugr_core::extension::simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp};
use hugr_core::ops::DataflowOpTrait;
use hugr_core::std_extensions::collections::array;
use hugr_core::std_extensions::collections::borrow_array::{
    self, BArrayClone, BArrayDiscard, BArrayFromArray, BArrayFromArrayDef, BArrayOp, BArrayOpDef,
    BArrayRepeat, BArrayScan, BArrayToArray, BArrayToArrayDef, BArrayUnsafeOp, BArrayUnsafeOpDef,
    borrow_array_type,
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
use crate::emit::func::get_or_make_function;
use crate::emit::libc::{emit_libc_free, emit_libc_malloc};
use crate::extension::PreludeCodegen;
use crate::extension::collections::array::{build_array_fat_pointer, decompose_array_fat_pointer};
use crate::{CodegenExtension, CodegenExtsBuilder};
use crate::{
    emit::{EmitFuncContext, RowPromise, deaggregate_call_result},
    types::{HugrType, TypingSession},
};

static ERR_ALREADY_BORROWED: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Array element is already borrowed".to_string(),
});

static ERR_NOT_BORROWED: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Array already contains an element at this index".to_string(),
});

static ERR_OUT_OF_BOUNDS: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Index out of bounds".to_string(),
});

static ERR_NOT_ALL_BORROWED: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Array contains non-borrowed elements and cannot be discarded".to_string(),
});

static ERR_SOME_BORROWED: LazyLock<ConstError> = LazyLock::new(|| ConstError {
    signal: 2,
    message: "Some array elements have been borrowed".to_string(),
});

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [`BorrowArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using
    /// [`DefaultBorrowArrayCodegen`] as the implementation.
    #[must_use]
    pub fn add_default_borrow_array_extensions(self, pcg: impl PreludeCodegen + 'a) -> Self {
        self.add_borrow_array_extensions(DefaultBorrowArrayCodegen(pcg))
    }

    /// Add a [`BorrowArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using `ccg`
    /// as the implementation.
    pub fn add_borrow_array_extensions(self, ccg: impl BorrowArrayCodegen + 'a) -> Self {
        self.add_extension(BorrowArrayCodegenExtension::from(ccg))
    }
}

/// A helper trait for customising the lowering of [`borrow_array`], including its
/// types, [`hugr_core::ops::constant::CustomConst`]s, and ops.
///
/// By default, all arrays are allocated on the heap using the standard libc `malloc`
/// and `free` functions. This behaviour can be customised by providing a different
/// implementation for [`BorrowArrayCodegen::emit_allocate_array`] and
/// [`BorrowArrayCodegen::emit_free_array`].
///
/// See [`crate::extension::collections::borrow_array`] for details.
pub trait BorrowArrayCodegen: Clone {
    /// Emit instructions to halt execution with the error `err`.
    ///
    /// This should be consistent with the panic implementation from the [PreludeCodegen].
    fn emit_panic<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()>;

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

    /// Return the llvm type of [`hugr_core::std_extensions::collections::borrow_array::BORROW_ARRAY_TYPENAME`].
    fn array_type<'c>(
        &self,
        session: &TypingSession<'c, '_>,
        elem_ty: BasicTypeEnum<'c>,
        _size: u64,
    ) -> impl BasicType<'c> {
        barray_fat_pointer_ty(session, elem_ty)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayValue`].
    fn emit_array_value<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: &borrow_array::BArrayValue,
    ) -> Result<BasicValueEnum<'c>> {
        emit_barray_value(self, ctx, value)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayOp`].
    fn emit_array_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayOp,
        inputs: Vec<BasicValueEnum<'c>>,
        outputs: RowPromise<'c>,
    ) -> Result<()> {
        emit_barray_op(self, ctx, op, inputs, outputs)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayUnsafeOp`].
    fn emit_array_unsafe_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayUnsafeOp,
        inputs: Vec<BasicValueEnum<'c>>,
        outputs: RowPromise<'c>,
    ) -> Result<()> {
        emit_barray_unsafe_op(self, ctx, op, inputs, outputs)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayClone`] operation.
    fn emit_array_clone<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayClone,
        array_v: BasicValueEnum<'c>,
    ) -> Result<(BasicValueEnum<'c>, BasicValueEnum<'c>)> {
        emit_clone_op(self, ctx, op, array_v)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayDiscard`] operation.
    fn emit_array_discard<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayDiscard,
        array_v: BasicValueEnum<'c>,
    ) -> Result<()> {
        emit_barray_discard(self, ctx, op, array_v)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayRepeat`] op.
    fn emit_array_repeat<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayRepeat,
        func: BasicValueEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        emit_repeat_op(self, ctx, op, func)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayScan`] op.
    ///
    /// Returns the resulting array and the final values of the accumulators.
    fn emit_array_scan<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayScan,
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

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayToArray`].
    fn emit_to_array_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayToArray,
        barray_v: BasicValueEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        emit_to_array_op(self, ctx, op, barray_v)
    }

    /// Emit a [`hugr_core::std_extensions::collections::borrow_array::BArrayFromArray`].
    fn emit_from_array_op<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        op: BArrayFromArray,
        array_v: BasicValueEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        emit_from_array_op(self, ctx, op, array_v)
    }
}

/// A trivial implementation of [`BorrowArrayCodegen`] which passes all methods
/// through to their default implementations.
#[derive(Default, Clone)]
pub struct DefaultBorrowArrayCodegen<PCG>(PCG);

impl<PCG: PreludeCodegen> BorrowArrayCodegen for DefaultBorrowArrayCodegen<PCG> {
    fn emit_panic<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        self.0.emit_panic(ctx, err)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BorrowArrayCodegenExtension<CCG>(CCG);

impl<CCG: BorrowArrayCodegen> BorrowArrayCodegenExtension<CCG> {
    pub fn new(ccg: CCG) -> Self {
        Self(ccg)
    }
}

impl<CCG: BorrowArrayCodegen> From<CCG> for BorrowArrayCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}

impl<CCG: BorrowArrayCodegen> CodegenExtension for BorrowArrayCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type(
                (
                    borrow_array::EXTENSION_ID,
                    borrow_array::BORROW_ARRAY_TYPENAME,
                ),
                {
                    let ccg = self.0.clone();
                    move |ts, hugr_type| {
                        let [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] = hugr_type.args()
                        else {
                            return Err(anyhow!("Invalid type args for array type"));
                        };
                        let elem_ty = ts.llvm_type(ty)?;
                        Ok(ccg.array_type(&ts, elem_ty, *n).as_basic_type_enum())
                    }
                },
            )
            .custom_const::<borrow_array::BArrayValue>({
                let ccg = self.0.clone();
                move |context, k| ccg.emit_array_value(context, k)
            })
            .simple_extension_op::<BArrayOpDef>({
                let ccg = self.0.clone();
                move |context, args, _| {
                    ccg.emit_array_op(
                        context,
                        BArrayOp::from_extension_op(args.node().as_ref())?,
                        args.inputs,
                        args.outputs,
                    )
                }
            })
            .simple_extension_op::<BArrayUnsafeOpDef>({
                let ccg = self.0.clone();
                move |context, args, _| {
                    ccg.emit_array_unsafe_op(
                        context,
                        BArrayUnsafeOp::from_extension_op(args.node().as_ref())?,
                        args.inputs,
                        args.outputs,
                    )
                }
            })
            .extension_op(borrow_array::EXTENSION_ID, array::ARRAY_CLONE_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let arr = args.inputs[0];
                    let op = BArrayClone::from_extension_op(args.node().as_ref())?;
                    let (arr1, arr2) = ccg.emit_array_clone(context, op, arr)?;
                    args.outputs.finish(context.builder(), [arr1, arr2])
                }
            })
            .extension_op(borrow_array::EXTENSION_ID, array::ARRAY_DISCARD_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let arr = args.inputs[0];
                    let op = BArrayDiscard::from_extension_op(args.node().as_ref())?;
                    ccg.emit_array_discard(context, op, arr)?;
                    args.outputs.finish(context.builder(), [])
                }
            })
            .extension_op(borrow_array::EXTENSION_ID, array::ARRAY_REPEAT_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let func = args.inputs[0];
                    let op = BArrayRepeat::from_extension_op(args.node().as_ref())?;
                    let arr = ccg.emit_array_repeat(context, op, func)?;
                    args.outputs.finish(context.builder(), [arr])
                }
            })
            .extension_op(borrow_array::EXTENSION_ID, array::ARRAY_SCAN_OP_ID, {
                let ccg = self.0.clone();
                move |context, args| {
                    let src_array = args.inputs[0];
                    let func = args.inputs[1];
                    let initial_accs = &args.inputs[2..];
                    let op = BArrayScan::from_extension_op(args.node().as_ref())?;
                    let (tgt_array, final_accs) =
                        ccg.emit_array_scan(context, op, src_array, func, initial_accs)?;
                    args.outputs
                        .finish(context.builder(), iter::once(tgt_array).chain(final_accs))
                }
            })
            .extension_op(
                borrow_array::EXTENSION_ID,
                BArrayToArrayDef::new().opdef_id(),
                {
                    let ccg = self.0.clone();
                    move |context, args| {
                        let barray = args.inputs[0];
                        let op = BArrayToArray::from_extension_op(args.node().as_ref())?;
                        let array = ccg.emit_to_array_op(context, op, barray)?;
                        args.outputs.finish(context.builder(), [array])
                    }
                },
            )
            .extension_op(
                borrow_array::EXTENSION_ID,
                BArrayFromArrayDef::new().opdef_id(),
                {
                    let ccg = self.0.clone();
                    move |context, args| {
                        let array = args.inputs[0];
                        let op = BArrayFromArray::from_extension_op(args.node().as_ref())?;
                        let barray = ccg.emit_from_array_op(context, op, array)?;
                        args.outputs.finish(context.builder(), [barray])
                    }
                },
            )
    }
}

fn usize_ty<'c>(ts: &TypingSession<'c, '_>) -> IntType<'c> {
    ts.llvm_type(&usize_t())
        .expect("Prelude codegen is registered")
        .into_int_type()
}

/// Returns the LLVM representation of a borrow array value as a fat pointer.
#[must_use]
pub fn barray_fat_pointer_ty<'c>(
    session: &TypingSession<'c, '_>,
    elem_ty: BasicTypeEnum<'c>,
) -> StructType<'c> {
    let iw_ctx = session.iw_context();
    let usize_t = usize_ty(session);
    iw_ctx.struct_type(
        &[
            // Pointer to the first array element
            elem_ty.ptr_type(AddressSpace::default()).into(),
            // Pointer to the bitarray mask storing whether values have been borrowed
            usize_t.ptr_type(AddressSpace::default()).into(),
            // Offset
            usize_t.into(),
        ],
        false,
    )
}

/// Constructs a borrow array fat pointer value.
pub fn build_barray_fat_pointer<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    BArrayFatPtrComponents {
        elems_ptr,
        mask_ptr,
        offset,
    }: BArrayFatPtrComponents<'c>,
) -> Result<StructValue<'c>> {
    let array_ty = barray_fat_pointer_ty(
        &ctx.typing_session(),
        elems_ptr.get_type().get_element_type().try_into().unwrap(),
    );
    let array_v = array_ty.get_poison();
    let array_v =
        ctx.builder()
            .build_insert_value(array_v, elems_ptr.as_basic_value_enum(), 0, "")?;
    let array_v =
        ctx.builder()
            .build_insert_value(array_v, mask_ptr.as_basic_value_enum(), 1, "")?;
    let array_v = ctx
        .builder()
        .build_insert_value(array_v, offset.as_basic_value_enum(), 2, "")?;
    Ok(array_v.into_struct_value())
}

pub struct BArrayFatPtrComponents<'a> {
    pub elems_ptr: PointerValue<'a>,
    pub mask_ptr: PointerValue<'a>,
    pub offset: IntValue<'a>,
}

/// Returns the underlying pointer, mask and offset stored in a fat borrow array pointer.
pub fn decompose_barray_fat_pointer<'c>(
    builder: &Builder<'c>,
    array_v: BasicValueEnum<'c>,
) -> Result<BArrayFatPtrComponents<'c>> {
    let array_v = array_v.into_struct_value();
    let elems_ptr = builder
        .build_extract_value(array_v, 0, "array_ptr")?
        .into_pointer_value();
    let mask_ptr = builder
        .build_extract_value(array_v, 1, "array_mask_ptr")?
        .into_pointer_value();
    let offset = builder
        .build_extract_value(array_v, 2, "array_offset")?
        .into_int_value();
    Ok(BArrayFatPtrComponents {
        elems_ptr,
        mask_ptr,
        offset,
    })
}

/// Helper function to allocate a typed array of `num_elems` elements of type `elem_ty`.
/// Returns the pointer to the first element and the size in memory of the allocation.
fn alloc_typed_array<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    elem_ty: BasicTypeEnum<'c>,
    num_elems: IntValue<'c>,
) -> Result<(PointerValue<'c>, IntValue<'c>)> {
    let size = ctx
        .builder()
        .build_int_mul(num_elems, elem_ty.size_of().unwrap(), "array_size")?;
    let ptr = ccg.emit_allocate_array(ctx, size)?;
    Ok((
        ctx.builder()
            .build_bit_cast(ptr, elem_ty.ptr_type(AddressSpace::default()), "")?
            .into_pointer_value(),
        size,
    ))
}

/// Helper function to allocate a fat borrow array pointer.
///
/// Returns a pointer and a struct:
/// * The pointer points to the first element of the array (i.e. it is of type `elem_ty.ptr_type()`).
/// * The struct is the fat pointer that stores also the pointer to the mask and an additional offset (initialised to 0).
pub fn build_barray_alloc<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &impl BorrowArrayCodegen,
    elem_ty: BasicTypeEnum<'c>,
    size: u64,
    set_borrowed: bool,
) -> Result<(PointerValue<'c>, StructValue<'c>)> {
    let usize_t = usize_ty(&ctx.typing_session());
    let length = usize_t.const_int(size, false);
    let (elems_ptr, _) = alloc_typed_array(ccg, ctx, elem_ty, length)?;

    // Mask is bit-packed into an array of values of type usize
    let mask_length = usize_t.const_int(size.div_ceil(usize_t.get_bit_width() as u64), false);
    let (mask_ptr, mask_size_value) = alloc_typed_array(ccg, ctx, usize_t.into(), mask_length)?;
    fill_mask(ctx, mask_ptr, mask_size_value, set_borrowed)?;

    let offset = usize_t.const_zero();
    let array_v = build_barray_fat_pointer(
        ctx,
        BArrayFatPtrComponents {
            elems_ptr,
            mask_ptr,
            offset,
        },
    )?;
    Ok((elems_ptr, array_v))
}

/// Emits instructions to fill the entire mask with a bit value.
fn fill_mask<H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<H>,
    mask_ptr: PointerValue,
    size: IntValue,
    value: bool,
) -> Result<()> {
    let memset = Intrinsic::find("llvm.memset")
        .unwrap()
        .get_declaration(
            ctx.get_current_module(),
            &[mask_ptr.get_type().into(), size.get_type().into()],
        )
        .unwrap();
    let i8t = ctx.iw_context().i8_type(); // Value to fill with is always this size
    let val = if value {
        i8t.const_all_ones()
    } else {
        i8t.const_zero()
    };
    let volatile = ctx.iw_context().bool_type().const_zero().into();
    ctx.builder().build_call(
        memset,
        &[mask_ptr.into(), val.into(), size.into(), volatile],
        "",
    )?;
    Ok(())
}

/// Enum for mask operations that can be performed on a single bit of the mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskCheck {
    /// Check the element is borrowed, panicking if it isnt; then mark as returned.
    Return,
    /// Check the element is not borrowed, panicking if it is. (Do not change the bit.)
    CheckNotBorrowed,
    /// Check the element is not borrowed, panicking if it is; then mark as borrowed.
    Borrow,
}

impl MaskCheck {
    fn func_name(&self) -> &'static str {
        match self {
            MaskCheck::Return => "__barray_mask_return",
            MaskCheck::CheckNotBorrowed => "__barray_mask_check_not_borrowed",
            MaskCheck::Borrow => "__barray_mask_borrow",
        }
    }

    /// Generate code to perform the check on the specified bit of the mask.
    /// (Does not check the index is in bounds.)
    fn emit<'c, H: HugrView<Node = Node>>(
        &self,
        ccg: &impl BorrowArrayCodegen,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        mask_ptr: PointerValue<'c>,
        idx: IntValue<'c>,
    ) -> Result<()> {
        get_or_make_function(
            ctx,
            self.func_name(),
            [mask_ptr.into(), idx.into()],
            None,
            |ctx, [mask_ptr, idx]| {
                // Compute mask bitarray block index via `idx // BLOCK_SIZE`
                let usize_t = usize_ty(&ctx.typing_session());
                let (
                    BlockData {
                        block_ptr,
                        block,
                        idx_in_block,
                    },
                    bit,
                ) = inspect_mask_idx_bit(ctx, mask_ptr, idx)?;

                let panic_bb = ctx.build_positioned_new_block("panic", None, |ctx, panic_bb| {
                    let err: &ConstError = match self {
                        MaskCheck::CheckNotBorrowed | MaskCheck::Borrow => &ERR_ALREADY_BORROWED,
                        MaskCheck::Return => &ERR_NOT_BORROWED,
                    };
                    let err_val = ctx.emit_custom_const(err).unwrap();
                    ccg.emit_panic(ctx, err_val)?;
                    ctx.builder().build_unreachable()?;
                    Ok(panic_bb)
                })?;
                let ok_bb = ctx.build_positioned_new_block("ok", None, |ctx, ok_bb| {
                    if let MaskCheck::Return | MaskCheck::Borrow = self {
                        // Update the mask to mark the element as borrowed or free
                        let builder = ctx.builder();
                        let update = builder.build_left_shift(
                            usize_t.const_int(1, false),
                            idx_in_block,
                            "",
                        )?;
                        let block = builder.build_xor(block, update, "")?;
                        builder.build_store(block_ptr, block)?;
                    }
                    ctx.builder().build_return(None)?;
                    Ok(ok_bb)
                })?;
                let (if_borrowed, if_present) = match self {
                    MaskCheck::CheckNotBorrowed | MaskCheck::Borrow => (panic_bb, ok_bb),
                    MaskCheck::Return => (ok_bb, panic_bb),
                };
                ctx.builder()
                    .build_conditional_branch(bit, if_borrowed, if_present)?;
                Ok(None)
            },
        )?;
        Ok(())
    }
}

struct BlockData<'c> {
    block_ptr: PointerValue<'c>,
    block: IntValue<'c>,
    idx_in_block: IntValue<'c>,
}

fn inspect_mask_idx_bit<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_ptr: BasicValueEnum<'c>,
    idx: BasicValueEnum<'c>,
) -> Result<(BlockData<'c>, IntValue<'c>)> {
    let usize_t = usize_ty(&ctx.typing_session());
    let mask_ptr = mask_ptr.into_pointer_value();
    let idx = idx.into_int_value();
    let block_size = usize_t.const_int(usize_t.get_bit_width() as u64, false);
    let builder = ctx.builder();
    let block_idx = builder.build_int_unsigned_div(idx, block_size, "")?;
    let block_ptr = unsafe { builder.build_in_bounds_gep(mask_ptr, &[block_idx], "")? };
    let block = builder.build_load(block_ptr, "")?.into_int_value();
    let idx_in_block = builder.build_int_unsigned_rem(idx, block_size, "")?;
    let block_shifted = builder.build_right_shift(block, idx_in_block, false, "")?;
    let bit = builder.build_int_truncate(block_shifted, ctx.iw_context().bool_type(), "")?;
    Ok((
        BlockData {
            block_ptr,
            block,
            idx_in_block,
        },
        bit,
    ))
}

struct MaskInfo<'a> {
    mask_ptr: PointerValue<'a>,
    offset: IntValue<'a>,
    size: IntValue<'a>,
}

/// Emits instructions to check all blocks of the borrowed mask are equal to `expected_val`
/// or else panic with the specified error.
fn check_all_mask_eq<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_info: &MaskInfo<'c>,
    expected: bool,
    err: &ConstError,
) -> Result<()> {
    build_mask_padding1d(
        ctx,
        mask_info.mask_ptr,
        mask_info.offset,
        BitsToPad::Before,
        expected,
    )?;
    let usize_t = usize_ty(&ctx.typing_session());
    let builder = ctx.builder();
    let last_idx = builder.build_int_sub(
        builder.build_int_add(mask_info.offset, mask_info.size, "")?,
        usize_t.const_int(1, false),
        "last_valid",
    )?;
    build_mask_padding1d(
        ctx,
        mask_info.mask_ptr,
        last_idx,
        BitsToPad::After,
        expected,
    )?;

    let builder = ctx.builder();
    let expected_val = if expected {
        usize_t.const_all_ones()
    } else {
        usize_t.const_zero()
    };
    let block_size = usize_t.const_int(usize_t.get_bit_width() as u64, false);
    let start_block = builder.build_int_unsigned_div(mask_info.offset, block_size, "")?;
    let end_block = builder.build_int_unsigned_div(last_idx, block_size, "")?;

    let iters = builder.build_int_sub(end_block, start_block, "")?;
    let iters = builder.build_int_add(iters, usize_t.const_int(1, false), "")?;
    build_loop(ctx, iters, |ctx, idx| {
        let builder = ctx.builder();
        let block_idx = builder.build_int_add(idx, start_block, "")?;
        let block_addr =
            unsafe { builder.build_in_bounds_gep(mask_info.mask_ptr, &[block_idx], "")? };
        let block = builder.build_load(block_addr, "")?.into_int_value();
        let err_bb = ctx.build_positioned_new_block("mask_block_err", None, |ctx, bb| {
            let err_val = ctx.emit_custom_const(err).unwrap();
            ccg.emit_panic(ctx, err_val)?;
            ctx.builder().build_unreachable()?;
            Ok(bb)
        })?;
        let ok_bb = ctx.build_positioned_new_block("mask_block_ok", None, |_, bb| bb);
        let builder = ctx.builder();
        let cond = builder.build_int_compare(IntPredicate::EQ, block, expected_val, "")?;
        builder.build_conditional_branch(cond, ok_bb, err_bb)?;
        builder.position_at_end(ok_bb);
        Ok(())
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BitsToPad {
    Before,
    After,
}

/// Emits instructions to update the mask, overwriting unused bits with a value.
fn build_mask_padding1d<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_ptr: PointerValue<'c>,
    idx: IntValue<'c>,
    end: BitsToPad,
    value: bool,
) -> Result<()> {
    let builder = ctx.builder();
    let usize_t = usize_ty(&ctx.typing_session());
    let block_size = usize_t.const_int(usize_t.get_bit_width() as u64, false);

    // Find the first block that contain some used bits
    let block_idx = builder.build_int_unsigned_div(idx, block_size, "")?;
    let block_addr = unsafe { builder.build_in_bounds_gep(mask_ptr, &[block_idx], "")? };
    let block = builder.build_load(block_addr, "")?.into_int_value();

    let all_ones = usize_t.const_all_ones();
    let one = usize_t.const_int(1, false);
    let idx_in_block = builder.build_int_unsigned_rem(idx, block_size, "")?;
    let new_block = if value {
        // Pad with ones.
        let (num_used, shifted) = match end {
            BitsToPad::Before => {
                let fst_block_used = builder.build_int_sub(block_size, idx_in_block, "")?;
                let rsh = builder.build_right_shift(all_ones, fst_block_used, false, "")?;
                (fst_block_used, rsh)
            }
            BitsToPad::After => {
                // 0<=`idx_in_block`<block_size from int_unsigned_rem, so add one to get
                // the number of used bits in the last block.
                let lst_block_used = builder.build_int_add(idx_in_block, one, "")?;
                let lsh = builder.build_left_shift(all_ones, lst_block_used, "")?;
                (lst_block_used, lsh)
            }
        };
        // If the shift amount is the block_size, LLVM defines the shift to be a no-op, but we want a zero.
        let all_used = builder.build_int_compare(IntPredicate::EQ, num_used, block_size, "")?;
        let pad = builder
            .build_select(all_used, usize_t.const_zero(), shifted, "")?
            .into_int_value();
        builder.build_or(block, pad, "")?
    } else {
        // Pad with zeroes.
        let pad = match end {
            BitsToPad::Before => builder.build_left_shift(all_ones, idx_in_block, "")?,
            BitsToPad::After => {
                // 0<=`idx_in_block`<block_size from int_unsigned_rem, so add one to get
                // the number of used bits in the last block.
                let lst_block_used = builder.build_int_add(idx_in_block, one, "")?;
                let lst_block_unused = builder.build_int_sub(block_size, lst_block_used, "")?;
                builder.build_right_shift(all_ones, lst_block_unused, false, "")?
            }
        };
        builder.build_and(block, pad, "")?
    };
    builder.build_store(block_addr, new_block)?;
    Ok(())
}

/// Emits a check that returns whether a specific array element is borrowed (true) or not (false).
pub fn build_is_borrowed_bit<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_ptr: PointerValue<'c>,
    idx: IntValue<'c>,
) -> Result<inkwell::values::IntValue<'c>> {
    // Wrap the check into a function instead of inlining
    const FUNC_NAME: &str = "__barray_is_borrowed";
    get_or_make_function(
        ctx,
        FUNC_NAME,
        [mask_ptr.into(), idx.into()],
        Some(ctx.iw_context().bool_type().into()),
        |ctx, [mask_ptr, idx]| {
            let (_, bit) = inspect_mask_idx_bit(ctx, mask_ptr, idx)?;
            Ok(Some(bit.into()))
        },
    )
    .map(|v| v.expect("i1 return value").into_int_value())
}

/// Emits a check that no array elements have been borrowed.
pub fn build_none_borrowed_check<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_ptr: PointerValue<'c>,
    offset: IntValue<'c>,
    size: u64,
) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    // Wrap the check into a function instead of inlining
    const FUNC_NAME: &str = "__barray_check_none_borrowed";
    let usize_t = usize_ty(&ctx.typing_session());
    let size = usize_t.const_int(size, false);
    get_or_make_function(
        ctx,
        FUNC_NAME,
        [mask_ptr.into(), offset.into(), size.into()],
        None,
        |ctx, [mask_ptr, offset, size]| {
            let mask_ptr = mask_ptr.into_pointer_value();
            let offset = offset.into_int_value();
            let size = size.into_int_value();
            // Pad unused bits to zero
            let info = MaskInfo {
                mask_ptr,
                offset,
                size,
            };
            check_all_mask_eq(ccg, ctx, &info, false, &ERR_SOME_BORROWED)?;
            Ok(None)
        },
    )?;
    Ok(())
}

/// Emits a check that all array elements have been borrowed.
pub fn build_all_borrowed_check<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    mask_ptr: PointerValue<'c>,
    offset: IntValue<'c>,
    size: u64,
) -> Result<()> {
    if size == 0 {
        return Ok(());
    }
    // Wrap the check into a function instead of inlining
    const FUNC_NAME: &str = "__barray_check_all_borrowed";
    let usize_t = usize_ty(&ctx.typing_session());
    let size = usize_t.const_int(size, false);
    get_or_make_function(
        ctx,
        FUNC_NAME,
        [mask_ptr.into(), offset.into(), size.into()],
        None,
        |ctx, [mask_ptr, offset, size]| {
            let mask_ptr = mask_ptr.into_pointer_value();
            let offset = offset.into_int_value();
            let size = size.into_int_value();
            // Pad unused bits to one
            let info = MaskInfo {
                mask_ptr,
                offset,
                size,
            };
            check_all_mask_eq(ccg, ctx, &info, true, &ERR_NOT_ALL_BORROWED)?;
            Ok(None)
        },
    )?;
    Ok(())
}

/// Emits a check that a specified (unsigned) index is less than the size of the array
pub fn build_bounds_check<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    size: u64,
    idx: IntValue<'c>,
) -> Result<()> {
    // Wrap the check into a function instead of inlining
    const FUNC_NAME: &str = "__barray_check_bounds";
    let size = usize_ty(&ctx.typing_session()).const_int(size, false);
    get_or_make_function(
        ctx,
        FUNC_NAME,
        [size.into(), idx.into()],
        None,
        |ctx, [size, idx]| {
            let size = size.into_int_value();
            let idx = idx.into_int_value();
            let in_bounds = ctx
                .builder()
                .build_int_compare(IntPredicate::ULT, idx, size, "")?;
            let ok_bb = ctx.build_positioned_new_block("ok", None, |_, bb| bb);
            let err_bb = ctx.build_positioned_new_block("out_of_bounds", None, |ctx, bb| {
                let err: &ConstError = &ERR_OUT_OF_BOUNDS;
                let err_val = ctx.emit_custom_const(err).unwrap();
                ccg.emit_panic(ctx, err_val)?;
                ctx.builder().build_unreachable()?;
                Ok(bb)
            })?;
            ctx.builder()
                .build_conditional_branch(in_bounds, ok_bb, err_bb)?;
            ctx.builder().position_at_end(ok_bb);
            Ok(None)
        },
    )?;
    Ok(())
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

    let (body_start_block, body_end_block, val) =
        ctx.build_positioned_new_block("", Some(exit_block), |ctx, body_start_bb| {
            let idx = ctx.builder().build_load(idx_ptr, "")?.into_int_value();
            let val = go(ctx, idx)?;
            let builder = ctx.builder();
            let body_end_bb = builder.get_insert_block().unwrap();
            let inc_idx = builder.build_int_add(idx, idx_ty.const_int(1, false), "")?;
            builder.build_store(idx_ptr, inc_idx)?;
            // Branch to the head is built later
            Ok((body_start_bb, body_end_bb, val))
        })?;

    let head_block = ctx.build_positioned_new_block("", Some(body_start_block), |ctx, bb| {
        let builder = ctx.builder();
        let idx = builder.build_load(idx_ptr, "")?.into_int_value();
        let cmp = builder.build_int_compare(IntPredicate::ULT, idx, iters, "")?;
        builder.build_conditional_branch(cmp, body_start_block, exit_block)?;
        Ok(bb)
    })?;

    let builder = ctx.builder();
    builder.build_unconditional_branch(head_block)?;
    builder.position_at_end(body_end_block);
    builder.build_unconditional_branch(head_block)?;
    ctx.builder().position_at_end(exit_block);
    Ok(val)
}

/// Emits an [`borrow_array::BArrayValue`].
pub fn emit_barray_value<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    value: &borrow_array::BArrayValue,
) -> Result<BasicValueEnum<'c>> {
    let ts = ctx.typing_session();
    let elem_ty = ts.llvm_type(value.get_element_type())?;
    let (elem_ptr, array_v) =
        build_barray_alloc(ctx, ccg, elem_ty, value.get_contents().len() as u64, false)?;
    for (i, v) in value.get_contents().iter().enumerate() {
        let llvm_v = emit_value(ctx, v)?;
        let idx = ts.iw_context().i32_type().const_int(i as u64, true);
        let elem_addr = unsafe { ctx.builder().build_in_bounds_gep(elem_ptr, &[idx], "")? };
        ctx.builder().build_store(elem_addr, llvm_v)?;
    }
    Ok(array_v.into())
}

/// Emits an [`BArrayOp`].
pub fn emit_barray_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayOp,
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
    let BArrayOp {
        def,
        elem_ty: ref hugr_elem_ty,
        size,
    } = op;
    let elem_ty = ts.llvm_type(hugr_elem_ty)?;
    match def {
        BArrayOpDef::new_array => {
            let (elem_ptr, array_v) = build_barray_alloc(ctx, ccg, elem_ty, size, false)?;
            let usize_t = usize_ty(&ctx.typing_session());
            for (i, v) in inputs.into_iter().enumerate() {
                let idx = usize_t.const_int(i as u64, true);
                let elem_addr = unsafe { ctx.builder().build_in_bounds_gep(elem_ptr, &[idx], "")? };
                ctx.builder().build_store(elem_addr, v)?;
            }
            outputs.finish(ctx.builder(), [array_v.into()])
        }
        BArrayOpDef::unpack => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::unpack expects one argument"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;

            let mut result = Vec::with_capacity(size as usize);
            let usize_t = usize_ty(&ctx.typing_session());

            for i in 0..size {
                let idx = ctx
                    .builder()
                    .build_int_add(offset, usize_t.const_int(i, false), "")?;
                MaskCheck::CheckNotBorrowed.emit(ccg, ctx, mask_ptr, idx)?;
                let elem_addr =
                    unsafe { ctx.builder().build_in_bounds_gep(elems_ptr, &[idx], "")? };
                let elem_v = ctx.builder().build_load(elem_addr, "")?;
                result.push(elem_v);
            }

            outputs.finish(ctx.builder(), result)
        }
        BArrayOpDef::get => {
            let [array_v, index_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::get expects two arguments"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();
            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("BArrayOp::get has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("BArrayOp::get output is not a sum type"))?
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
                    // inside `success_block` we know `index_v` to be in bounds
                    let index_v = ctx.builder().build_int_add(index_v, offset, "")?;
                    MaskCheck::CheckNotBorrowed.emit(ccg, ctx, mask_ptr, index_v)?;
                    let builder = ctx.builder();
                    let elem_addr =
                        unsafe { builder.build_in_bounds_gep(elems_ptr, &[index_v], "")? };
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
        BArrayOpDef::set => {
            let [array_v, index_v, value_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::set expects three arguments"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();

            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("BArrayOp::set has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("BArrayOp::set output is not a sum type"))?
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
                    // inside `success_block` we know `index_v` to be in bounds.
                    let index_v = ctx.builder().build_int_add(index_v, offset, "")?;
                    MaskCheck::CheckNotBorrowed.emit(ccg, ctx, mask_ptr, index_v)?;
                    let builder = ctx.builder();
                    let elem_addr =
                        unsafe { builder.build_in_bounds_gep(elems_ptr, &[index_v], "")? };
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
        BArrayOpDef::swap => {
            let [array_v, index1_v, index2_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::swap expects three arguments"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index1_v = index1_v.into_int_value();
            let index2_v = index2_v.into_int_value();

            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("BArrayOp::swap has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("BArrayOp::swap output is not a sum type"))?
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
                    let index1_v = builder.build_int_add(index1_v, offset, "")?;
                    let index2_v = builder.build_int_add(index2_v, offset, "")?;
                    MaskCheck::CheckNotBorrowed.emit(ccg, ctx, mask_ptr, index1_v)?;
                    MaskCheck::CheckNotBorrowed.emit(ccg, ctx, mask_ptr, index2_v)?;
                    let builder = ctx.builder();
                    let elem1_addr =
                        unsafe { builder.build_in_bounds_gep(elems_ptr, &[index1_v], "")? };
                    let elem1_v = builder.build_load(elem1_addr, "")?;
                    let elem2_addr =
                        unsafe { builder.build_in_bounds_gep(elems_ptr, &[index2_v], "")? };
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
        BArrayOpDef::pop_left => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::pop_left expects one argument"))?;
            let r = emit_pop_op(
                ccg,
                ctx,
                hugr_elem_ty.clone(),
                size,
                array_v.into_struct_value(),
                true,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        BArrayOpDef::pop_right => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::pop_right expects one argument"))?;
            let r = emit_pop_op(
                ccg,
                ctx,
                hugr_elem_ty.clone(),
                size,
                array_v.into_struct_value(),
                false,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        BArrayOpDef::discard_empty => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayOpDef::discard_empty expects one argument"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset: _,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            ccg.emit_free_array(ctx, elems_ptr)?;
            ccg.emit_free_array(ctx, mask_ptr)?;
            outputs.finish(ctx.builder(), [])
        }
        _ => todo!(),
    }
}

/// Emits an [`BArrayClone`] op.
pub fn emit_clone_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayClone,
    array_v: BasicValueEnum<'c>,
) -> Result<(BasicValueEnum<'c>, BasicValueEnum<'c>)> {
    let elem_ty = ctx.llvm_type(&op.elem_ty)?;
    let BArrayFatPtrComponents {
        elems_ptr,
        mask_ptr,
        offset,
    } = decompose_barray_fat_pointer(ctx.builder(), array_v)?;
    build_none_borrowed_check(ccg, ctx, mask_ptr, offset, op.size)?;
    let (other_ptr, other_array_v) = build_barray_alloc(ctx, ccg, elem_ty, op.size, false)?;
    let src_ptr = unsafe {
        ctx.builder()
            .build_in_bounds_gep(elems_ptr, &[offset], "")?
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

/// Emits an [`BArrayDiscard`] op.
pub fn emit_barray_discard<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    _op: BArrayDiscard,
    array_v: BasicValueEnum<'c>,
) -> Result<()> {
    let array_ptr =
        ctx.builder()
            .build_extract_value(array_v.into_struct_value(), 0, "array_ptr")?;
    ccg.emit_free_array(ctx, array_ptr.into_pointer_value())?;
    Ok(())
}

/// Emits the [`BArrayOpDef::pop_left`] and [`BArrayOpDef::pop_right`] operations.
fn emit_pop_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    elem_ty: HugrType,
    size: u64,
    array_v: StructValue<'c>,
    pop_left: bool,
) -> Result<BasicValueEnum<'c>> {
    let ts = ctx.typing_session();
    let builder = ctx.builder();
    let fp = decompose_barray_fat_pointer(builder, array_v.into())?;
    let ret_ty = ts.llvm_sum_type(option_type(vec![
        elem_ty.clone(),
        borrow_array_type(size.saturating_add_signed(-1), elem_ty),
    ]))?;
    if size == 0 {
        return Ok(ret_ty.build_tag(builder, 0, vec![])?.into());
    }
    let (elem_ptr, new_array_offset) = {
        if pop_left {
            let new_array_offset = builder.build_int_add(
                fp.offset,
                usize_ty(&ts).const_int(1, false),
                "new_offset",
            )?;
            MaskCheck::CheckNotBorrowed.emit(ccg, ctx, fp.mask_ptr, fp.offset)?;
            let elem_ptr = unsafe {
                ctx.builder()
                    .build_in_bounds_gep(fp.elems_ptr, &[fp.offset], "")
            }?;
            (elem_ptr, new_array_offset)
        } else {
            let idx =
                builder.build_int_add(fp.offset, usize_ty(&ts).const_int(size - 1, false), "")?;
            MaskCheck::CheckNotBorrowed.emit(ccg, ctx, fp.mask_ptr, idx)?;
            let elem_ptr = unsafe { ctx.builder().build_in_bounds_gep(fp.elems_ptr, &[idx], "") }?;
            (elem_ptr, fp.offset)
        }
    };
    let elem_v = ctx.builder().build_load(elem_ptr, "")?;
    let new_array_v = build_barray_fat_pointer(
        ctx,
        BArrayFatPtrComponents {
            offset: new_array_offset,
            ..fp
        },
    )?;

    Ok(ret_ty
        .build_tag(ctx.builder(), 1, vec![elem_v, new_array_v.into()])?
        .into())
}

/// Emits an [`BArrayRepeat`] op.
pub fn emit_repeat_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayRepeat,
    func: BasicValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let elem_ty = ctx.llvm_type(&op.elem_ty)?;
    let (ptr, array_v) = build_barray_alloc(ctx, ccg, elem_ty, op.size, false)?;
    let array_len = usize_ty(&ctx.typing_session()).const_int(op.size, false);
    build_loop(ctx, array_len, |ctx, idx| {
        let builder = ctx.builder();
        let func_ptr = CallableValue::try_from(func.into_pointer_value())
            .map_err(|()| anyhow!("BArrayOpDef::repeat expects a function pointer"))?;
        let v = builder
            .build_call(func_ptr, &[], "")?
            .try_as_basic_value()
            .left()
            .ok_or(anyhow!("BArrayOpDef::repeat function must return a value"))?;
        let elem_addr = unsafe { builder.build_in_bounds_gep(ptr, &[idx], "")? };
        builder.build_store(elem_addr, v)?;
        Ok(())
    })?;
    Ok(array_v.into())
}

/// Emits an [`BArrayScan`] op.
///
/// Returns the resulting array and the final values of the accumulators.
pub fn emit_scan_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayScan,
    src_array_v: StructValue<'c>,
    func: BasicValueEnum<'c>,
    initial_accs: &[BasicValueEnum<'c>],
) -> Result<(BasicValueEnum<'c>, Vec<BasicValueEnum<'c>>)> {
    let BArrayFatPtrComponents {
        elems_ptr: src_ptr,
        mask_ptr: src_mask_ptr,
        offset: src_offset,
    } = decompose_barray_fat_pointer(ctx.builder(), src_array_v.into())?;
    build_none_borrowed_check(ccg, ctx, src_mask_ptr, src_offset, op.size)?;
    let tgt_elem_ty = ctx.llvm_type(&op.tgt_ty)?;
    // TODO: If `sizeof(op.src_ty) >= sizeof(op.tgt_ty)`, we could reuse the memory
    // from `src` instead of allocating a fresh array
    let (tgt_ptr, tgt_array_v) = build_barray_alloc(ctx, ccg, tgt_elem_ty, op.size, false)?;
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
            .map_err(|()| anyhow!("BArrayOpDef::scan expects a function pointer"))?;
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
    ccg.emit_free_array(ctx, src_mask_ptr)?;
    let builder = ctx.builder();
    let final_accs = acc_ptrs
        .into_iter()
        .map(|ptr| builder.build_load(ptr, ""))
        .try_collect()?;
    Ok((tgt_array_v.into(), final_accs))
}

/// Emits an [`BArrayUnsafeOp`].
pub fn emit_barray_unsafe_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayUnsafeOp,
    inputs: Vec<BasicValueEnum<'c>>,
    outputs: RowPromise<'c>,
) -> Result<()> {
    let builder = ctx.builder();
    let BArrayUnsafeOp {
        def,
        elem_ty: ref hugr_elem_ty,
        size,
    } = op;

    match def {
        BArrayUnsafeOpDef::borrow => {
            let [array_v, index_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayUnsafeOpDef::borrow expects two arguments"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();
            build_bounds_check(ccg, ctx, size, index_v)?;
            let offset_index_v = ctx.builder().build_int_add(index_v, offset, "")?;
            MaskCheck::Borrow.emit(ccg, ctx, mask_ptr, offset_index_v)?;
            let builder = ctx.builder();
            let elem_addr =
                unsafe { builder.build_in_bounds_gep(elems_ptr, &[offset_index_v], "")? };
            let elem_v = builder.build_load(elem_addr, "")?;
            outputs.finish(builder, [array_v, elem_v])
        }
        BArrayUnsafeOpDef::r#return => {
            let [array_v, index_v, elem_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayUnsafeOpDef::return expects three arguments"))?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();
            build_bounds_check(ccg, ctx, size, index_v)?;
            let offset_index_v = ctx.builder().build_int_add(index_v, offset, "")?;
            MaskCheck::Return.emit(ccg, ctx, mask_ptr, offset_index_v)?;
            let builder = ctx.builder();
            let elem_addr =
                unsafe { builder.build_in_bounds_gep(elems_ptr, &[offset_index_v], "")? };
            builder.build_store(elem_addr, elem_v)?;
            outputs.finish(builder, [array_v])
        }
        BArrayUnsafeOpDef::discard_all_borrowed => {
            let [array_v] = inputs.try_into().map_err(|_| {
                anyhow!("BArrayUnsafeOpDef::discard_all_borrowed expects one argument")
            })?;
            let BArrayFatPtrComponents {
                elems_ptr,
                mask_ptr,
                offset,
            } = decompose_barray_fat_pointer(builder, array_v)?;
            build_all_borrowed_check(ccg, ctx, mask_ptr, offset, size)?;
            ccg.emit_free_array(ctx, elems_ptr)?;
            ccg.emit_free_array(ctx, mask_ptr)?;
            outputs.finish(ctx.builder(), [])
        }
        BArrayUnsafeOpDef::new_all_borrowed => {
            let elem_ty = ctx.llvm_type(hugr_elem_ty)?;
            let (_, array_v) = build_barray_alloc(ctx, ccg, elem_ty, size, true)?;
            outputs.finish(ctx.builder(), [array_v.into()])
        }
        BArrayUnsafeOpDef::is_borrowed => {
            let [array_v, index_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("BArrayUnsafeOpDef::is_borrowed expects two arguments"))?;
            let BArrayFatPtrComponents {
                mask_ptr, offset, ..
            } = decompose_barray_fat_pointer(builder, array_v)?;
            let index_v = index_v.into_int_value();
            build_bounds_check(ccg, ctx, size, index_v)?;
            let offset_index_v = ctx.builder().build_int_add(index_v, offset, "")?;
            // let bit = build_is_borrowed_check(ctx, mask_ptr, offset_index_v)?;
            let bit = build_is_borrowed_bit(ctx, mask_ptr, offset_index_v)?;
            outputs.finish(ctx.builder(), [array_v, bit.into()])
        }
        _ => todo!(),
    }
}

/// Emits an [`BArrayToArray`] op.
pub fn emit_to_array_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayToArray,
    barray_v: BasicValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let BArrayFatPtrComponents {
        elems_ptr,
        mask_ptr,
        offset,
    } = decompose_barray_fat_pointer(ctx.builder(), barray_v)?;
    build_none_borrowed_check(ccg, ctx, mask_ptr, offset, op.size)?;
    Ok(build_array_fat_pointer(ctx, elems_ptr, offset)?.into())
}

/// Emits an [`BArrayFromArray`] op.
pub fn emit_from_array_op<'c, H: HugrView<Node = Node>>(
    ccg: &impl BorrowArrayCodegen,
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: BArrayFromArray,
    array_v: BasicValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    // We reuse the allocation from the array but we have to allocate a fresh mask.
    // Note that the mask must have size at least `size + offset` so the offsets match up.
    let usize_t = usize_ty(&ctx.typing_session());
    let builder = ctx.builder();
    let (ptr, offset) = decompose_array_fat_pointer(builder, array_v)?;
    let size = usize_t.const_int(op.size, false);
    let mask_bits = builder.build_int_add(size, offset, "")?;
    let mask_blocks = builder.build_int_unsigned_div(mask_bits, usize_t.size_of(), "")?;
    // Increment by one to account for potential rounding down
    let mask_blocks = builder.build_int_add(mask_blocks, usize_t.const_int(1, false), "")?;
    let (mask_ptr, mask_size) = alloc_typed_array(ccg, ctx, usize_t.into(), mask_blocks)?;
    fill_mask(ctx, mask_ptr, mask_size, false)?;
    Ok(build_barray_fat_pointer(
        ctx,
        BArrayFatPtrComponents {
            elems_ptr: ptr,
            mask_ptr,
            offset,
        },
    )?
    .into())
}

#[cfg(test)]
mod test {
    use hugr_core::Wire;
    use hugr_core::builder::{DataflowHugr, HugrBuilder};
    use hugr_core::extension::prelude::either_type;
    use hugr_core::ops::Tag;
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
    use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
    use hugr_core::std_extensions::collections::array::ArrayOpBuilder;
    use hugr_core::std_extensions::collections::array::op_builder::build_all_borrow_array_ops;
    use hugr_core::std_extensions::collections::borrow_array::{
        self, BArrayOpBuilder, BArrayRepeat, BArrayScan, BArrayToArray, borrow_array_type,
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

    use crate::emit::test::PanicTestPreludeCodegen;
    use crate::extension::DefaultPreludeCodegen;
    use crate::types::HugrType;
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
                build_all_borrow_array_ops(builder.dfg_builder_endo([]).unwrap())
                    .finish_sub_container()
                    .unwrap();
                builder.finish_hugr().unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                let arr = builder.add_new_borrow_array(usize_t(), [us1, us2]).unwrap();
                let (_, arr) = builder
                    .add_borrow_array_get(usize_t(), 2, arr, us1)
                    .unwrap();
                builder.add_borrow_array_discard(usize_t(), 2, arr).unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                let arr = builder.add_new_borrow_array(usize_t(), [us1, us2]).unwrap();
                let (arr1, arr2) = builder.add_borrow_array_clone(usize_t(), 2, arr).unwrap();
                builder
                    .add_borrow_array_discard(usize_t(), 2, arr1)
                    .unwrap();
                builder
                    .add_borrow_array_discard(usize_t(), 2, arr2)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
        });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_array_value(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(STD_REG.to_owned())
            .with_outs(vec![borrow_array_type(2, usize_t())])
            .finish(|mut builder| {
                let vs = vec![ConstUsize::new(1).into(), ConstUsize::new(2).into()];
                let arr = builder.add_load_value(borrow_array::BArrayValue::new(usize_t(), vs));
                builder.finish_hugr_with_outputs([arr]).unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
    //             let arr = builder.add_load_value(borrow_array::BArrayValue::new(usize_t(), values));

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
    //             .add_default_borrow_array_extensions()
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
            borrow_array::EXTENSION.to_owned(),
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
                let arr = builder.add_new_borrow_array(usize_t(), [us1, us2]).unwrap();
                let i = builder.add_load_value(ConstUsize::new(index));
                let (get_r, arr) = builder.add_borrow_array_get(usize_t(), 2, arr, i).unwrap();
                builder.add_borrow_array_discard(usize_t(), 2, arr).unwrap();
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
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                let arr = builder
                    .add_new_borrow_array(int_ty.clone(), [i1, i2])
                    .unwrap();
                let index = builder.add_load_value(ConstUsize::new(index));
                let value = builder.add_load_value(ConstInt::new_u(3, value).unwrap());
                let get_r = builder
                    .add_borrow_array_set(int_ty.clone(), 2, arr, index, value)
                    .unwrap();
                let r = {
                    let res_sum_ty = {
                        let row = vec![int_ty.clone(), borrow_array_type(2, int_ty.clone())];
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
                        let (r, arr) = builder
                            .add_borrow_array_get(int_ty.clone(), 2, arr, us0)
                            .unwrap();
                        let [arr_0] = builder
                            .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                            .unwrap();
                        let (r, arr) = builder
                            .add_borrow_array_get(int_ty.clone(), 2, arr, us1)
                            .unwrap();
                        let [arr_1] = builder
                            .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                            .unwrap();
                        let elem_eq = builder.add_ieq(3, elem, expected_elem).unwrap();
                        let arr_0_eq = builder.add_ieq(3, arr_0, expected_arr_0).unwrap();
                        let arr_1_eq = builder.add_ieq(3, arr_1, expected_arr_1).unwrap();
                        let r = builder.add_and(elem_eq, arr_0_eq).unwrap();
                        let r = builder.add_and(r, arr_1_eq).unwrap();
                        builder
                            .add_borrow_array_discard(int_ty.clone(), 2, arr)
                            .unwrap();
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
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
        let arr_ty = borrow_array_type(2, int_ty.clone());
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let us0 = builder.add_load_value(ConstUsize::new(0));
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let i1 = builder.add_load_value(ConstInt::new_u(3, 1).unwrap());
                let i2 = builder.add_load_value(ConstInt::new_u(3, 2).unwrap());
                let arr = builder
                    .add_new_borrow_array(int_ty.clone(), [i1, i2])
                    .unwrap();

                let index1 = builder.add_load_value(ConstUsize::new(index1));
                let index2 = builder.add_load_value(ConstUsize::new(index2));
                let r = builder
                    .add_borrow_array_swap(int_ty.clone(), 2, arr, index1, index2)
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
                let (r, arr) = builder
                    .add_borrow_array_get(int_ty.clone(), 2, arr, us0)
                    .unwrap();
                let elem_0 = builder
                    .build_unwrap_sum::<1>(1, option_type(int_ty.clone()), r)
                    .unwrap()[0];
                let (r, arr) = builder
                    .add_borrow_array_get(int_ty.clone(), 2, arr, us1)
                    .unwrap();
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
                builder
                    .add_borrow_array_discard(int_ty.clone(), 2, arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
        let arr_ty = borrow_array_type(2, int_ty.clone());
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let idx = builder.add_load_value(ConstUsize::new(index));
                let i1 = builder.add_load_value(ConstInt::new_u(3, 1).unwrap());
                let i2 = builder.add_load_value(ConstInt::new_u(3, 2).unwrap());
                let inew = builder.add_load_value(ConstInt::new_u(3, new_v).unwrap());
                let arr = builder
                    .add_new_borrow_array(int_ty.clone(), [i1, i2])
                    .unwrap();

                let (arr, arr_clone) = builder
                    .add_borrow_array_clone(int_ty.clone(), 2, arr)
                    .unwrap();
                let r = builder
                    .add_borrow_array_set(int_ty.clone(), 2, arr, idx, inew)
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
                    .add_borrow_array_get(int_ty.clone(), 2, arr_clone, idx)
                    .unwrap();
                let [elem] = builder
                    .build_unwrap_sum(1, option_type(int_ty.clone()), r)
                    .unwrap();
                builder
                    .add_borrow_array_discard(int_ty.clone(), 2, arr)
                    .unwrap();
                builder
                    .add_borrow_array_discard(int_ty.clone(), 2, arr_clone)
                    .unwrap();
                builder.finish_hugr_with_outputs([elem]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                    .add_new_borrow_array(int_ty.clone(), new_array_args)
                    .unwrap();
                for (i, left) in from_left.iter().enumerate() {
                    let array_size = (array_contents.len() - i) as u64;
                    let pop_res = if *left {
                        builder
                            .add_borrow_array_pop_left(int_ty.clone(), array_size, arr)
                            .unwrap()
                    } else {
                        builder
                            .add_borrow_array_pop_right(int_ty.clone(), array_size, arr)
                            .unwrap()
                    };
                    let [elem, new_arr] = builder
                        .build_unwrap_sum(
                            1,
                            option_type(vec![
                                int_ty.clone(),
                                borrow_array_type(array_size - 1, int_ty.clone()),
                            ]),
                            pop_res,
                        )
                        .unwrap();
                    arr = new_arr;
                    r = builder.add_iadd(6, r, elem).unwrap();
                }
                builder
                    .add_borrow_array_discard(
                        int_ty.clone(),
                        (array_contents.len() - from_left.len()) as u64,
                        arr,
                    )
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                let array = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    array_contents
                        .iter()
                        .map(|&i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let array = builder.add_load_value(array);
                let unpacked = builder
                    .add_borrow_array_unpack(int_ty.clone(), array_contents.len() as u64, array)
                    .unwrap();
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for elem in unpacked {
                    r = builder.add_iadd(6, r, elem).unwrap();
                }

                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                let repeat = BArrayRepeat::new(int_ty.clone(), size);
                let arr = builder
                    .add_dataflow_op(repeat, vec![func_v])
                    .unwrap()
                    .out_wire(0);
                let idx_v = builder.add_load_value(ConstUsize::new(idx));
                let (get_res, arr) = builder
                    .add_borrow_array_get(int_ty.clone(), size, arr, idx_v)
                    .unwrap();
                let [elem] = builder
                    .build_unwrap_sum(1, option_type(vec![int_ty.clone()]), get_res)
                    .unwrap();
                builder
                    .add_borrow_array_discard(int_ty.clone(), size, arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([elem]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                    .add_new_borrow_array(int_ty.clone(), new_array_args)
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
                let scan = BArrayScan::new(int_ty.clone(), int_ty.clone(), vec![], size);
                let mut arr = builder
                    .add_dataflow_op(scan, [arr, func_v])
                    .unwrap()
                    .out_wire(0);

                for i in 0..size {
                    let array_size = size - i;
                    let pop_res = builder
                        .add_borrow_array_pop_left(int_ty.clone(), array_size, arr)
                        .unwrap();
                    let [elem, new_arr] = builder
                        .build_unwrap_sum(
                            1,
                            option_type(vec![
                                int_ty.clone(),
                                borrow_array_type(array_size - 1, int_ty.clone()),
                            ]),
                            pop_res,
                        )
                        .unwrap();
                    arr = new_arr;
                    r = builder.add_iadd(6, r, elem).unwrap();
                }
                builder
                    .add_borrow_array_discard_empty(int_ty.clone(), arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
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
                    .add_new_borrow_array(int_ty.clone(), new_array_args)
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
                let scan = BArrayScan::new(int_ty.clone(), Type::UNIT, vec![int_ty.clone()], size);
                let zero = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let [arr, sum] = builder
                    .add_dataflow_op(scan, [arr, func_v, zero])
                    .unwrap()
                    .outputs_arr();
                builder
                    .add_borrow_array_discard(Type::UNIT, size, arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([sum]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
                .add_default_int_extensions()
        });
        let expected: u64 = (0..size).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    fn build_pops(
        builder: &mut impl Dataflow,
        elem_ty: HugrType,
        size: u64,
        mut array: Wire,
        num_pops: u64,
    ) -> Wire {
        for i in 0..num_pops {
            let res = builder
                .add_borrow_array_pop_left(elem_ty.clone(), size - i, array)
                .unwrap();
            let [_, arr] = builder
                .build_unwrap_sum(
                    1,
                    option_type(vec![
                        elem_ty.clone(),
                        borrow_array_type(size - i - 1, elem_ty.clone()),
                    ]),
                    res,
                )
                .unwrap();
            array = arr;
        }
        array
    }

    #[rstest]
    #[case::single_block(48, 10, &[0, 11, 12, 13, 36, 37])]
    #[case::block_boundary1(65, 0, &[63, 64])]
    #[case::block_boundary2(65, 10, &[53, 54])]
    #[case::block_boundary3(129, 0, &[63, 64, 127, 128])]
    fn exec_borrow_return(
        mut exec_ctx: TestContext,
        #[case] mut size: u64,
        #[case] num_pops: u64,
        #[case] indices: &[u64],
    ) {
        // We build a HUGR that:
        // - Loads an array filled with 0..size
        // - Pops specified numbers from the left to introduce an offset
        // - Borrows the elements at the provided indices and sums them up
        // - Puts 0s in the borrowed places in reverse order
        // - Borrows the indices again

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let array = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (0..size)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let mut array = builder.add_load_value(array);
                array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                size -= num_pops;
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for &i in indices {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    let (arr, val) = builder
                        .add_borrow_array_borrow(int_ty.clone(), size, array, i)
                        .unwrap();
                    r = builder.add_iadd(6, r, val).unwrap();
                    array = arr;
                }
                let zero = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for &i in indices.iter().rev() {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    array = builder
                        .add_borrow_array_return(int_ty.clone(), size, array, i, zero)
                        .unwrap();
                }
                for &i in indices {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    let (arr, val) = builder
                        .add_borrow_array_borrow(int_ty.clone(), size, array, i)
                        .unwrap();
                    r = builder.add_iadd(6, r, val).unwrap();
                    array = arr;
                }
                builder
                    .add_borrow_array_discard(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
                .add_default_int_extensions()
        });
        let expected: u64 = indices.iter().map(|i| i + num_pops).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case::empty(0, 0)]
    #[case::basic(32, 0)]
    #[case::boundary(65, 0)]
    #[case::pop1(65, 10)]
    #[case::pop2(200, 32)]
    fn exec_discard_all_borrowed(
        mut exec_ctx: TestContext,
        #[case] mut size: u64,
        #[case] num_pops: u64,
    ) {
        // We build a HUGR that:
        // - Loads an array filled with 0..size
        // - Pops specified numbers from the left to introduce an offset
        // - Borrows the remaining elements and sums them up
        // - Discards the array using `discard_all_borrowed`

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let array = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (0..size)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let mut array = builder.add_load_value(array);
                array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                size -= num_pops;
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for i in 0..size {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    let (arr, val) = builder
                        .add_borrow_array_borrow(int_ty.clone(), size, array, i)
                        .unwrap();
                    r = builder.add_iadd(6, r, val).unwrap();
                    array = arr;
                }
                builder
                    .add_discard_all_borrowed(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
                .add_default_int_extensions()
        });
        let expected: u64 = (num_pops..size + num_pops).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case::small(10, 0)]
    #[case::small_pop(10, 2)]
    #[case::oneword(64, 0)]
    #[case::oneword_pop(64, 5)]
    #[case::big(97, 0)]
    #[case::big_pop(97, 5)]
    #[case::big_pop_until_small(97, 65)]
    fn exec_discard_all_borrowed2(
        mut exec_ctx: TestContext,
        #[case] size: u64,
        #[case] num_pops: u64,
    ) {
        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut array = builder.add_new_all_borrowed(int_ty.clone(), size).unwrap();
                let val = builder.add_load_value(ConstInt::new_u(6, 15).unwrap());
                for i in 0..num_pops {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    array = builder
                        .add_borrow_array_return(int_ty.clone(), size, array, i, val)
                        .unwrap();
                }
                let array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                builder
                    .add_discard_all_borrowed(int_ty.clone(), size - num_pops, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(PanicTestPreludeCodegen)
                .add_default_borrow_array_extensions(PanicTestPreludeCodegen)
                .add_default_int_extensions()
        });
        assert_eq!(&exec_ctx.exec_hugr_panicking(hugr, "main"), "");
    }

    #[rstest]
    #[case::basic(32, 0)]
    #[case::boundary(65, 0)]
    #[case::pop1(65, 10)]
    #[case::pop2(200, 32)]
    fn exec_conversion_roundtrip(
        mut exec_ctx: TestContext,
        #[case] mut size: u64,
        #[case] num_pops: u64,
    ) {
        // We build a HUGR that:
        // - Loads a borrow array filled with 0..size
        // - Pops specified numbers from the left to introduce an offset
        // - Converts it into a regular array
        // - Converts it back into a borrow array
        // - Borrows all elements, sums them up, and returns the sum

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                use hugr_core::std_extensions::collections::borrow_array::BArrayFromArray;

                let barray = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (0..size)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let barray = builder.add_load_value(barray);
                let barray = build_pops(&mut builder, int_ty.clone(), size, barray, num_pops);
                size -= num_pops;
                let array = builder
                    .add_dataflow_op(BArrayToArray::new(int_ty.clone(), size), [barray])
                    .unwrap()
                    .out_wire(0);
                let mut barray = builder
                    .add_dataflow_op(BArrayFromArray::new(int_ty.clone(), size), [array])
                    .unwrap()
                    .out_wire(0);

                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                for i in 0..size {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    let (arr, val) = builder
                        .add_borrow_array_borrow(int_ty.clone(), size, barray, i)
                        .unwrap();
                    r = builder.add_iadd(6, r, val).unwrap();
                    barray = arr;
                }
                builder
                    .add_discard_all_borrowed(int_ty.clone(), size, barray)
                    .unwrap();
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        let expected: u64 = (num_pops..size + num_pops).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case::oob(1, 0, [0, 1], "Index out of bounds")]
    #[case::double_borrow(32, 0, [0, 0], "Array element is already borrowed")]
    #[case::double_borrow_pop(32, 1, [1, 0], "Array element is already borrowed")]
    #[case::double_borrow_boundary(65, 1, [64, 63], "Array element is already borrowed")]
    #[case::pop_borrowed(32, 1, [0, 10], "Array element is already borrowed")]
    fn exec_borrow_panics(
        mut exec_ctx: TestContext,
        #[case] mut size: u64,
        #[case] num_pops: u64,
        #[case] indices: [u64; 2],
        #[case] msg: &str,
    ) {
        // We build a HUGR that:
        // - Loads an array filled with 0..size
        // - Borrows element `indices[0]`
        // - Pops specified numbers from the left to introduce an offset
        // - Borrows element `indices[1]`
        // - Checks that the program panics with the given message

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let array = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (0..size)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let mut array = builder.add_load_value(array);
                let i1 = builder.add_load_value(ConstUsize::new(indices[0]));
                let i2 = builder.add_load_value(ConstUsize::new(indices[1]));
                array = builder
                    .add_borrow_array_borrow(int_ty.clone(), size, array, i1)
                    .unwrap()
                    .0;
                array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                size -= num_pops;
                array = builder
                    .add_borrow_array_borrow(int_ty.clone(), size, array, i2)
                    .unwrap()
                    .0;
                builder
                    .add_borrow_array_discard(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(PanicTestPreludeCodegen)
                .add_default_borrow_array_extensions(PanicTestPreludeCodegen)
                .add_default_int_extensions()
        });
        assert_eq!(&exec_ctx.exec_hugr_panicking(hugr, "main"), msg);
    }

    #[rstest]
    #[case::oob(1, 0, [0, 1], "Index out of bounds")]
    #[case::pop_borrowed(32, 1, [1, 2], "Array element is already borrowed")]
    #[case::return_twice(32, 0, [0, 0], "Array already contains an element at this index")]
    #[case::return_twice_boundary(65, 0, [64, 64], "Array already contains an element at this index")]
    fn exec_return_panics(
        mut exec_ctx: TestContext,
        #[case] mut size: u64,
        #[case] num_pops: u64,
        #[case] indices: [u64; 2],
        #[case] msg: &str,
    ) {
        // We build a HUGR that:
        // - Creates an empty array with `new_all_borrowed`
        // - Returns an element to index `indices[0]`
        // - Pops specified numbers from the left to introduce an offset
        // - Returns an element to index `indices[1]`
        // - Checks that the program panics with the given message

        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut array = builder.add_new_all_borrowed(int_ty.clone(), size).unwrap();
                let i1 = builder.add_load_value(ConstUsize::new(indices[0]));
                let i2 = builder.add_load_value(ConstUsize::new(indices[1]));
                let zero = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                array = builder
                    .add_borrow_array_return(int_ty.clone(), size, array, i1, zero)
                    .unwrap();
                array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                size -= num_pops;
                array = builder
                    .add_borrow_array_return(int_ty.clone(), size, array, i2, zero)
                    .unwrap();
                builder
                    .add_borrow_array_discard(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(PanicTestPreludeCodegen)
                .add_default_borrow_array_extensions(PanicTestPreludeCodegen)
                .add_default_int_extensions()
        });
        assert_eq!(&exec_ctx.exec_hugr_panicking(hugr, "main"), msg);
    }

    #[rstest]
    fn exec_shift_by_64(mut exec_ctx: TestContext, #[values(false, true)] right: bool) {
        const SHIFTED_VAL: u64 = 35;
        // This test generates LLVM code to shift a 64-bit value right/left by 64 places.
        // This is a no-op in LLVM, rather than producing 0 as one might expect.
        use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
        let int_ty = int_type(6); // 64-bit integer
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let minus_one = builder.add_load_value(ConstInt::new_u(6, SHIFTED_VAL).unwrap());
                let shift = builder.add_load_value(ConstInt::new_u(6, 64).unwrap());
                let op = if right {
                    IntOpDef::ishr
                } else {
                    IntOpDef::ishl
                };
                let result = builder
                    .add_dataflow_op(op.with_log_width(6), [minus_one, shift])
                    .unwrap()
                    .out_wire(0);
                builder.finish_hugr_with_outputs([result]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), SHIFTED_VAL);
    }

    #[rstest]
    #[case::small(10, 0, 0)]
    #[case::small_right(10, 0, 9)]
    #[case::small_pop(10, 2, 0)]
    #[case::small_right_pop(10, 2, 7)]
    #[case::oneword(64, 0, 0)]
    #[case::oneword_pop(64, 2, 0)]
    #[case::big_right(97, 0, 96)]
    #[case::big_pop(97, 5, 0)]
    #[case::big_popmany(97, 65, 0)]
    fn exec_discard_all_borrowed_panic(
        mut exec_ctx: TestContext,
        #[case] size: u64,
        #[case] num_pops: u64,
        #[case] ret_index: u64,
    ) {
        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let mut array = builder.add_new_all_borrowed(int_ty.clone(), size).unwrap();
                let val = builder.add_load_value(ConstInt::new_u(6, 15).unwrap());
                for i in 0..num_pops {
                    let i = builder.add_load_value(ConstUsize::new(i));
                    array = builder
                        .add_borrow_array_return(int_ty.clone(), size, array, i, val)
                        .unwrap();
                }
                let array = build_pops(&mut builder, int_ty.clone(), size, array, num_pops);
                let size = size - num_pops;
                let ret_index = builder.add_load_value(ConstUsize::new(ret_index));
                let array = builder
                    .add_borrow_array_return(int_ty.clone(), size, array, ret_index, val)
                    .unwrap();
                builder
                    .add_discard_all_borrowed(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(PanicTestPreludeCodegen)
                .add_default_borrow_array_extensions(PanicTestPreludeCodegen)
                .add_default_int_extensions()
        });
        let msg = "Array contains non-borrowed elements and cannot be discarded";
        assert_eq!(&exec_ctx.exec_hugr_panicking(hugr, "main"), msg);
    }

    #[rstest]
    fn exec_to_array_panic(mut exec_ctx: TestContext) {
        let int_ty = int_type(6);
        let size = 10;
        let hugr = SimpleHugrConfig::new()
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let barray = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (0..size)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let barray = builder.add_load_value(barray);
                let idx = builder.add_load_value(ConstUsize::new(0));
                let (barray, _) = builder
                    .add_borrow_array_borrow(int_ty.clone(), size, barray, idx)
                    .unwrap();
                let array = builder
                    .add_dataflow_op(BArrayToArray::new(int_ty.clone(), size), [barray])
                    .unwrap()
                    .out_wire(0);
                builder
                    .add_array_discard(int_ty.clone(), size, array)
                    .unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(PanicTestPreludeCodegen)
                .add_default_borrow_array_extensions(PanicTestPreludeCodegen)
                .add_default_array_extensions()
                .add_default_int_extensions()
        });
        let msg = "Some array elements have been borrowed";
        assert_eq!(&exec_ctx.exec_hugr_panicking(hugr, "main"), msg);
    }

    #[rstest]
    fn exec_is_borrowed_basic(mut exec_ctx: TestContext) {
        // We build a HUGR that:
        // - Creates a borrow array [1,2,3]
        // - Borrows index 1
        // - Checks is_borrowed for indices 0, 1
        // - Returns 1 if [false, true], else 0
        let int_ty = int_type(6);
        let size = 3;
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish(|mut builder| {
                let barray = borrow_array::BArrayValue::new(
                    int_ty.clone(),
                    (1..=3)
                        .map(|i| ConstInt::new_u(6, i).unwrap().into())
                        .collect_vec(),
                );
                let barray = builder.add_load_value(barray);
                let idx1 = builder.add_load_value(ConstUsize::new(1));
                let (barray, _) = builder
                    .add_borrow_array_borrow(int_ty.clone(), size, barray, idx1)
                    .unwrap();

                let idx0 = builder.add_load_value(ConstUsize::new(0));
                let (arr, b0_bools) =
                    [idx0, idx1]
                        .iter()
                        .fold((barray, Vec::new()), |(arr, mut bools), idx| {
                            let (arr, b) = builder
                                .add_is_borrowed(int_ty.clone(), size, arr, *idx)
                                .unwrap();
                            bools.push(b);
                            (arr, bools)
                        });
                let [b0, b1] = b0_bools.try_into().unwrap();

                let b0 = builder.add_not(b0).unwrap(); // flip b0 to true
                let and01 = builder.add_and(b0, b1).unwrap();
                // convert bool to i1
                let i1 = builder
                    .add_dataflow_op(ConvertOpDef::ifrombool.without_log_width(), [and01])
                    .unwrap()
                    .out_wire(0);
                // widen i1 to i64
                let i_64 = builder
                    .add_dataflow_op(IntOpDef::iwiden_u.with_two_log_widths(0, 6), [i1])
                    .unwrap()
                    .out_wire(0);
                builder
                    .add_borrow_array_discard(int_ty.clone(), size, arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([i_64]).unwrap()
            });

        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_logic_extensions()
                .add_conversion_extensions()
                .add_default_borrow_array_extensions(DefaultPreludeCodegen)
                .add_default_int_extensions()
        });
        assert_eq!(1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }
}

use anyhow::Result;
use hugr_core::{HugrView, Node};
use inkwell::{
    AddressSpace,
    values::{BasicMetadataValueEnum, BasicValueEnum},
};

use crate::emit::func::EmitFuncContext;

/// Emits a call to the libc `void abort()` function.
pub fn emit_libc_abort<H: HugrView<Node = Node>>(context: &mut EmitFuncContext<H>) -> Result<()> {
    let iw_ctx = context.typing_session().iw_context();
    let abort_sig = iw_ctx.void_type().fn_type(&[], false);
    let abort = context.get_extern_func("abort", abort_sig)?;
    context.builder().build_call(abort, &[], "")?;
    Ok(())
}

/// Emits a call to the libc `int printf(char*, ...)` function.
pub fn emit_libc_printf<H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<H>,
    args: &[BasicMetadataValueEnum],
) -> Result<()> {
    let iw_ctx = context.typing_session().iw_context();
    let str_ty = iw_ctx.i8_type().ptr_type(AddressSpace::default());
    let printf_sig = iw_ctx.i32_type().fn_type(&[str_ty.into()], true);

    let printf = context.get_extern_func("printf", printf_sig)?;
    context.builder().build_call(printf, args, "")?;
    Ok(())
}

/// Emits a call to the libc `void* malloc(size_t size)` function.
pub fn emit_libc_malloc<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    size: BasicMetadataValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let iw_ctx = context.typing_session().iw_context();
    let malloc_sig = iw_ctx
        .i8_type()
        .ptr_type(AddressSpace::default())
        .fn_type(&[iw_ctx.i64_type().into()], false);
    let malloc = context.get_extern_func("malloc", malloc_sig)?;
    let res = context
        .builder()
        .build_call(malloc, &[size], "")?
        .try_as_basic_value()
        .unwrap_left();
    Ok(res)
}

/// Emits a call to the libc `void free(void* ptr)` function.
pub fn emit_libc_free<H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<H>,
    ptr: BasicMetadataValueEnum,
) -> Result<()> {
    let iw_ctx = context.typing_session().iw_context();
    let ptr_ty = iw_ctx.i8_type().ptr_type(AddressSpace::default());
    let ptr = context
        .builder()
        .build_bit_cast(ptr.into_pointer_value(), ptr_ty, "")?;

    let free_sig = iw_ctx.void_type().fn_type(&[ptr_ty.into()], false);
    let free = context.get_extern_func("free", free_sig)?;
    context.builder().build_call(free, &[ptr.into()], "")?;
    Ok(())
}

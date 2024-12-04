use anyhow::Result;
use hugr_core::HugrView;
use inkwell::{values::BasicMetadataValueEnum, AddressSpace};

use crate::emit::func::EmitFuncContext;

/// Emits a call to the libc `void abort()` function.
pub fn emit_libc_abort<H: HugrView>(context: &mut EmitFuncContext<H>) -> Result<()> {
    let iw_ctx = context.typing_session().iw_context();
    let abort_sig = iw_ctx.void_type().fn_type(&[], false);
    let abort = context.get_extern_func("abort", abort_sig)?;
    context.builder().build_call(abort, &[], "")?;
    Ok(())
}

/// Emits a call to the libc `int printf(char*, ...)` function.
pub fn emit_libc_printf<H: HugrView>(
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

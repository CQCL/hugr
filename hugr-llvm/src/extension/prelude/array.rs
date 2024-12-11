//! Codegen for prelude array operations.
use anyhow::{anyhow, Ok, Result};
use hugr_core::{
    extension::{
        prelude::{
            array::{ArrayRepeat, ArrayScan},
            array_type, option_type, ArrayOp, ArrayOpDef,
        },
        simple_op::MakeRegisteredOp,
    },
    ops::DataflowOpTrait as _,
    types::TypeEnum,
    HugrView,
};
use inkwell::{
    builder::{Builder, BuilderError},
    types::BasicType,
    values::{ArrayValue, BasicValue as _, BasicValueEnum, CallableValue, IntValue, PointerValue},
    IntPredicate,
};
use itertools::Itertools;

use crate::{
    emit::{deaggregate_call_result, EmitFuncContext, RowPromise},
    sum::LLVMSumType,
    types::{HugrType, TypingSession},
};

use super::PreludeCodegen;

/// Helper function to allocate an array on the stack.
///
/// Returns two pointers: The first one is a pointer to the first element of the
/// array (i.e. it is of type `array.get_element_type().ptr_type()`) whereas the
/// second one points to the whole array value, i.e. it is of type `array.ptr_type()`.
fn build_array_alloca<'c>(
    builder: &Builder<'c>,
    array: ArrayValue<'c>,
) -> Result<(PointerValue<'c>, PointerValue<'c>), BuilderError> {
    let array_ty = array.get_type();
    let array_len: IntValue<'c> = {
        let ctx = builder.get_insert_block().unwrap().get_context();
        ctx.i32_type().const_int(array_ty.len() as u64, false)
    };
    let ptr = builder.build_array_alloca(array_ty.get_element_type(), array_len, "")?;
    let array_ptr = builder
        .build_bit_cast(ptr, array_ty.ptr_type(Default::default()), "")?
        .into_pointer_value();
    builder.build_store(array_ptr, array)?;
    Result::Ok((ptr, array_ptr))
}

/// Helper function to allocate an array on the stack and pass a pointer to it
/// to a closure.
///
/// The pointer forwarded to the closure is a pointer to the first element of
/// the array. I.e. it is of type `array.get_element_type().ptr_type()` not
/// `array.ptr_type()`
fn with_array_alloca<'c, T, E: From<BuilderError>>(
    builder: &Builder<'c>,
    array: ArrayValue<'c>,
    go: impl FnOnce(PointerValue<'c>) -> Result<T, E>,
) -> Result<T, E> {
    let (ptr, _) = build_array_alloca(builder, array)?;
    go(ptr)
}

/// Helper function to build a loop that repeats for a given number of iterations.
///
/// The provided closure is called to build the loop body. Afterwards, the builder is positioned at
/// the end of the loop exit block.
fn build_loop<'c, T, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    iters: IntValue<'c>,
    go: impl FnOnce(&mut EmitFuncContext<'c, '_, H>, IntValue<'c>) -> Result<T>,
) -> Result<T> {
    let builder = ctx.builder();
    let idx_ty = ctx.iw_context().i32_type();
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

pub fn emit_array_op<'c, H: HugrView>(
    pcg: &impl PreludeCodegen,
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
        ref elem_ty,
        size,
    } = op;
    let llvm_array_ty = pcg
        .array_type(&ts, ts.llvm_type(elem_ty)?, size)
        .as_basic_type_enum()
        .into_array_type();
    match def {
        ArrayOpDef::new_array => {
            let mut array_v = llvm_array_ty.get_undef();
            for (i, v) in inputs.into_iter().enumerate() {
                array_v = builder
                    .build_insert_value(array_v, v, i as u32, "")?
                    .into_array_value();
            }
            outputs.finish(builder, [array_v.as_basic_value_enum()])
        }
        ArrayOpDef::get => {
            let [array_v, index_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::get expects two arguments"))?;
            let array_v = array_v.into_array_value();
            let index_v = index_v.into_int_value();
            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("ArrayOp::get has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("ArrayOp::get output is not a sum type"))?
                };
                LLVMSumType::try_new(&ts, st.clone())?
            };

            let exit_rmb = ctx.new_row_mail_box([res_hugr_ty], "")?;

            let exit_block = ctx.build_positioned_new_block("", None, |ctx, bb| {
                outputs.finish(ctx.builder(), exit_rmb.read_vec(ctx.builder(), [])?)?;
                Ok(bb)
            })?;

            let success_block =
                ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let elem_v = with_array_alloca(builder, array_v, |ptr| {
                        // inside `success_block` we know `index_v` to be in
                        // bounds.
                        let elem_addr =
                            unsafe { builder.build_in_bounds_gep(ptr, &[index_v], "")? };
                        builder.build_load(elem_addr, "")
                    })?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![elem_v])?;
                    exit_rmb.write(ctx.builder(), [success_v])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v = res_sum_ty.build_tag(builder, 0, vec![])?;
                    exit_rmb.write(ctx.builder(), [failure_v])?;
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
            let [array_v0, index_v, value_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::set expects three arguments"))?;
            let array_v = array_v0.into_array_value();
            let index_v = index_v.into_int_value();

            let res_hugr_ty = sig
                .output()
                .get(0)
                .ok_or(anyhow!("ArrayOp::set has no outputs"))?;

            let res_sum_ty = {
                let TypeEnum::Sum(st) = res_hugr_ty.as_type_enum() else {
                    Err(anyhow!("ArrayOp::set output is not a sum type"))?
                };
                LLVMSumType::try_new(&ts, st.clone())?
            };

            let exit_rmb = ctx.new_row_mail_box([res_hugr_ty], "")?;

            let exit_block = ctx.build_positioned_new_block("", None, |ctx, bb| {
                outputs.finish(ctx.builder(), exit_rmb.read_vec(ctx.builder(), [])?)?;
                Ok(bb)
            })?;

            let success_block =
                ctx.build_positioned_new_block("", Some(exit_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let (elem_v, array_v) = with_array_alloca(builder, array_v, |ptr| {
                        // inside `success_block` we know `index_v` to be in
                        // bounds.
                        let elem_addr =
                            unsafe { builder.build_in_bounds_gep(ptr, &[index_v], "")? };
                        let elem_v = builder.build_load(elem_addr, "")?;
                        builder.build_store(elem_addr, value_v)?;
                        let ptr = builder
                            .build_bit_cast(
                                ptr,
                                array_v.get_type().ptr_type(Default::default()),
                                "",
                            )?
                            .into_pointer_value();
                        let array_v = builder.build_load(ptr, "")?;
                        Ok((elem_v, array_v))
                    })?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![elem_v, array_v])?;
                    exit_rmb.write(ctx.builder(), [success_v])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v =
                        res_sum_ty.build_tag(builder, 0, vec![value_v, array_v.into()])?;
                    exit_rmb.write(ctx.builder(), [failure_v])?;
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
            let [array_v0, index1_v, index2_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::swap expects three arguments"))?;
            let array_v = array_v0.into_array_value();
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
                LLVMSumType::try_new(&ts, st.clone())?
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
                    let array_v = with_array_alloca(builder, array_v, |ptr| {
                        // inside `success_block` we know `index1_v` and `index2_v`
                        // to be in bounds.
                        let elem1_addr =
                            unsafe { builder.build_in_bounds_gep(ptr, &[index1_v], "")? };
                        let elem1_v = builder.build_load(elem1_addr, "")?;
                        let elem2_addr =
                            unsafe { builder.build_in_bounds_gep(ptr, &[index2_v], "")? };
                        let elem2_v = builder.build_load(elem2_addr, "")?;
                        builder.build_store(elem1_addr, elem2_v)?;
                        builder.build_store(elem2_addr, elem1_v)?;
                        let ptr = builder
                            .build_bit_cast(
                                ptr,
                                array_v.get_type().ptr_type(Default::default()),
                                "",
                            )?
                            .into_pointer_value();
                        builder.build_load(ptr, "")
                    })?;
                    let success_v = res_sum_ty.build_tag(builder, 1, vec![array_v])?;
                    exit_rmb.write(ctx.builder(), [success_v])?;
                    builder.build_unconditional_branch(exit_block)?;
                    Ok(bb)
                })?;

            let failure_block =
                ctx.build_positioned_new_block("", Some(success_block), |ctx, bb| {
                    let builder = ctx.builder();
                    let failure_v = res_sum_ty.build_tag(builder, 0, vec![array_v.into()])?;
                    exit_rmb.write(ctx.builder(), [failure_v])?;
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
                builder,
                &ts,
                elem_ty.clone(),
                size,
                array_v.into_array_value(),
                true,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        ArrayOpDef::pop_right => {
            let [array_v] = inputs
                .try_into()
                .map_err(|_| anyhow!("ArrayOpDef::pop_right expects one argument"))?;
            let r = emit_pop_op(
                builder,
                &ts,
                elem_ty.clone(),
                size,
                array_v.into_array_value(),
                false,
            )?;
            outputs.finish(ctx.builder(), [r])
        }
        ArrayOpDef::discard_empty => Ok(()),
        _ => todo!(),
    }
}

/// Helper function to emit the pop operations.
fn emit_pop_op<'c>(
    builder: &Builder<'c>,
    ts: &TypingSession<'c, '_>,
    elem_ty: HugrType,
    size: u64,
    array_v: ArrayValue<'c>,
    pop_left: bool,
) -> Result<BasicValueEnum<'c>> {
    let ret_ty = LLVMSumType::try_new(
        ts,
        option_type(vec![
            elem_ty.clone(),
            array_type(size.saturating_add_signed(-1), elem_ty),
        ]),
    )?;
    if size == 0 {
        return ret_ty.build_tag(builder, 0, vec![]);
    }
    let ctx = builder.get_insert_block().unwrap().get_context();
    let (elem_v, array_v) = with_array_alloca(builder, array_v, |ptr| {
        let (elem_ptr, ptr) = {
            if pop_left {
                let rest_ptr =
                    unsafe { builder.build_gep(ptr, &[ctx.i32_type().const_int(1, false)], "") }?;
                (ptr, rest_ptr)
            } else {
                let elem_ptr = unsafe {
                    builder.build_gep(ptr, &[ctx.i32_type().const_int(size - 1, false)], "")
                }?;
                (elem_ptr, ptr)
            }
        };
        let elem_v = builder.build_load(elem_ptr, "")?;
        let new_array_ty = array_v
            .get_type()
            .get_element_type()
            .array_type(size as u32 - 1);
        let ptr = builder
            .build_bit_cast(ptr, new_array_ty.ptr_type(Default::default()), "")?
            .into_pointer_value();
        let array_v = builder.build_load(ptr, "")?;
        Ok((elem_v, array_v))
    })?;
    ret_ty.build_tag(builder, 1, vec![elem_v, array_v])
}

/// Emits an [ArrayRepeat] op.
pub fn emit_repeat_op<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayRepeat,
    func: BasicValueEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let builder = ctx.builder();
    let array_len = ctx.iw_context().i32_type().const_int(op.size, false);
    let array_ty = ctx.llvm_type(&op.elem_ty)?.array_type(op.size as u32);
    let (ptr, array_ptr) = build_array_alloca(builder, array_ty.get_undef())?;
    build_loop(ctx, array_len, |ctx, idx| {
        let builder = ctx.builder();
        let func_ptr = CallableValue::try_from(func.into_pointer_value())
            .map_err(|_| anyhow!("ArrayOpDef::repeat expects a function pointer"))?;
        let v = builder
            .build_call(func_ptr, &[], "")?
            .try_as_basic_value()
            .left()
            .ok_or(anyhow!("ArrayOpDef::repeat function must return a value"))?;
        let elem_addr = unsafe { builder.build_in_bounds_gep(ptr, &[idx], "")? };
        builder.build_store(elem_addr, v)?;
        Ok(())
    })?;

    let builder = ctx.builder();
    let array_v = builder.build_load(array_ptr, "")?;
    Ok(array_v)
}

/// Emits an [ArrayScan] op.
///
/// Returns the resulting array and the final values of the accumulators.
pub fn emit_scan_op<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    op: ArrayScan,
    src_array: BasicValueEnum<'c>,
    func: BasicValueEnum<'c>,
    initial_accs: &[BasicValueEnum<'c>],
) -> Result<(BasicValueEnum<'c>, Vec<BasicValueEnum<'c>>)> {
    let builder = ctx.builder();
    let ts = ctx.typing_session();
    let array_len = ctx.iw_context().i32_type().const_int(op.size, false);
    let tgt_array_ty = ts.llvm_type(&op.tgt_ty)?.array_type(op.size as u32);
    let (src_ptr, _) = build_array_alloca(builder, src_array.into_array_value())?;
    let (tgt_ptr, tgt_array_ptr) = build_array_alloca(builder, tgt_array_ty.get_undef())?;

    let acc_tys: Vec<_> = op.acc_tys.iter().map(|ty| ts.llvm_type(ty)).try_collect()?;
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
            .map_err(|_| anyhow!("ArrayOpDef::scan expects a function pointer"))?;
        let src_elem_addr = unsafe { builder.build_in_bounds_gep(src_ptr, &[idx], "")? };
        let src_elem = builder.build_load(src_elem_addr, "")?;
        let mut args = vec![src_elem.into()];
        for ptr in acc_ptrs.iter() {
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

    let builder = ctx.builder();
    let tgt_array_v = builder.build_load(tgt_array_ptr, "")?;
    let final_accs = acc_ptrs
        .into_iter()
        .map(|ptr| builder.build_load(ptr, ""))
        .try_collect()?;
    Ok((tgt_array_v, final_accs))
}

#[cfg(test)]
mod test {
    use hugr_core::builder::Container as _;
    use hugr_core::extension::prelude::array::ArrayRepeat;
    use hugr_core::extension::ExtensionSet;
    use hugr_core::ops::Tag;
    use hugr_core::types::Type;
    use hugr_core::{
        builder::{Dataflow, DataflowSubContainer, SubContainer},
        extension::{
            prelude::{
                self, array::ArrayScan, array_type, bool_t, option_type, usize_t, ConstUsize,
                UnwrapBuilder as _,
            },
            ExtensionRegistry,
        },
        ops::Value,
        std_extensions::{
            arithmetic::{
                int_ops::{self},
                int_types::{self, int_type, ConstInt},
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
        custom::CodegenExtsBuilder,
        emit::test::SimpleHugrConfig,
        test::{exec_ctx, llvm_ctx, TestContext},
        utils::{array_op_builder, ArrayOpBuilder, IntOpBuilder, LogicOpBuilder},
    };

    #[rstest]
    fn emit_all_ops(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                array_op_builder::test::all_array_ops(builder.dfg_builder_endo([]).unwrap())
                    .finish_sub_container()
                    .unwrap();
                builder.finish_sub_container().unwrap()
            });
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_get(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let us1 = builder.add_load_value(ConstUsize::new(1));
                let us2 = builder.add_load_value(ConstUsize::new(2));
                let arr = builder.add_new_array(usize_t(), [us1, us2]).unwrap();
                builder.add_array_get(usize_t(), 2, arr, us1).unwrap();
                builder.finish_with_outputs([]).unwrap()
            });
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    fn exec_registry() -> ExtensionRegistry {
        ExtensionRegistry::new([
            int_types::EXTENSION.to_owned(),
            int_ops::EXTENSION.to_owned(),
            logic::EXTENSION.to_owned(),
            prelude::PRELUDE.to_owned(),
        ])
    }

    fn exec_extension_set() -> ExtensionSet {
        ExtensionSet::from_iter([
            int_types::EXTENSION_ID,
            int_ops::EXTENSION_ID,
            logic::EXTENSION_ID,
            prelude::PRELUDE_ID,
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
                let get_r = builder.add_array_get(usize_t(), 2, arr, i).unwrap();
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
                builder.finish_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
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
            .finish_with_exts(|mut builder, reg| {
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
                        let [arr_0] = {
                            let r = builder.add_array_get(int_ty.clone(), 2, arr, us0).unwrap();
                            builder
                                .build_unwrap_sum(reg, 1, option_type(int_ty.clone()), r)
                                .unwrap()
                        };
                        let [arr_1] = {
                            let r = builder.add_array_get(int_ty.clone(), 2, arr, us1).unwrap();
                            builder
                                .build_unwrap_sum(reg, 1, option_type(int_ty.clone()), r)
                                .unwrap()
                        };
                        let elem_eq = builder.add_ieq(3, elem, expected_elem).unwrap();
                        let arr_0_eq = builder.add_ieq(3, arr_0, expected_arr_0).unwrap();
                        let arr_1_eq = builder.add_ieq(3, arr_1, expected_arr_1).unwrap();
                        let r = builder.add_and(elem_eq, arr_0_eq).unwrap();
                        let r = builder.add_and(r, arr_1_eq).unwrap();
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
                builder.finish_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_int_extensions()
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
            .finish_with_exts(|mut builder, reg| {
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
                let elem_0 = {
                    let r = builder.add_array_get(int_ty.clone(), 2, arr, us0).unwrap();
                    builder
                        .build_unwrap_sum::<1>(reg, 1, option_type(int_ty.clone()), r)
                        .unwrap()[0]
                };
                let elem_1 = {
                    let r = builder.add_array_get(int_ty.clone(), 2, arr, us1).unwrap();
                    builder
                        .build_unwrap_sum::<1>(reg, 1, option_type(int_ty), r)
                        .unwrap()[0]
                };
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
                builder.finish_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_int_extensions()
                .add_logic_extensions()
        });
        assert_eq!(1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(true, 0, 0)]
    #[case(true, 1, 1)]
    #[case(true, 2, 3)]
    #[case(true, 3, 7)]
    #[case(false, 0, 0)]
    #[case(false, 1, 4)]
    #[case(false, 2, 6)]
    #[case(false, 3, 7)]
    fn exec_pop(
        mut exec_ctx: TestContext,
        #[case] from_left: bool,
        #[case] num: usize,
        #[case] expected: u64,
    ) {
        // We build a HUGR that:
        // - Creates an array: [1,2,4]
        // - Pops `num` elements from the left or right
        // - Returns the sum of the popped elements
        let array_contents = [1, 2, 4];
        let int_ty = int_type(6);
        let hugr = SimpleHugrConfig::new()
            .with_outs(int_ty.clone())
            .with_extensions(exec_registry())
            .finish_with_exts(|mut builder, reg| {
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let new_array_args = array_contents
                    .iter()
                    .map(|&i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let mut arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();
                for i in 0..num {
                    let array_size = (array_contents.len() - i) as u64;
                    let pop_res = if from_left {
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
                            reg,
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
                builder.finish_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| cge.add_default_prelude_extensions().add_int_extensions());
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
            .finish_with_exts(|mut builder, reg| {
                let mut func = builder
                    .define_function(
                        "foo",
                        Signature::new(vec![], vec![int_ty.clone()])
                            .with_extension_delta(exec_extension_set()),
                    )
                    .unwrap();
                let v = func.add_load_value(ConstInt::new_u(6, value).unwrap());
                let func_id = func.finish_with_outputs(vec![v]).unwrap();
                let func_v = builder.load_func(func_id.handle(), &[]).unwrap();
                let repeat = ArrayRepeat::new(int_ty.clone(), size, exec_extension_set());
                let arr = builder
                    .add_dataflow_op(repeat, vec![func_v])
                    .unwrap()
                    .out_wire(0);
                let idx_v = builder.add_load_value(ConstUsize::new(idx));
                let get_res = builder
                    .add_array_get(int_ty.clone(), size, arr, idx_v)
                    .unwrap();
                let [elem] = builder
                    .build_unwrap_sum(reg, 1, option_type(vec![int_ty.clone()]), get_res)
                    .unwrap();
                builder.finish_with_outputs([elem]).unwrap()
            });
        exec_ctx.add_extensions(|cge| cge.add_default_prelude_extensions().add_int_extensions());
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
            .finish_with_exts(|mut builder, reg| {
                let mut r = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let new_array_args = (0..size)
                    .map(|i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();

                let mut func = builder
                    .define_function(
                        "foo",
                        Signature::new(vec![int_ty.clone()], vec![int_ty.clone()])
                            .with_extension_delta(exec_extension_set()),
                    )
                    .unwrap();
                let [elem] = func.input_wires_arr();
                let delta = func.add_load_value(ConstInt::new_u(6, inc).unwrap());
                let out = func.add_iadd(6, elem, delta).unwrap();
                let func_id = func.finish_with_outputs(vec![out]).unwrap();
                let func_v = builder.load_func(func_id.handle(), &[]).unwrap();
                let scan = ArrayScan::new(
                    int_ty.clone(),
                    int_ty.clone(),
                    vec![],
                    size,
                    exec_extension_set(),
                );
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
                            reg,
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
                builder.finish_with_outputs([r]).unwrap()
            });
        exec_ctx.add_extensions(|cge| cge.add_default_prelude_extensions().add_int_extensions());
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
            .finish_with_exts(|mut builder, _reg| {
                let new_array_args = (0..size)
                    .map(|i| builder.add_load_value(ConstInt::new_u(6, i).unwrap()))
                    .collect_vec();
                let arr = builder
                    .add_new_array(int_ty.clone(), new_array_args)
                    .unwrap();

                let mut func = builder
                    .define_function(
                        "foo",
                        Signature::new(
                            vec![int_ty.clone(), int_ty.clone()],
                            vec![Type::UNIT, int_ty.clone()],
                        )
                        .with_extension_delta(exec_extension_set()),
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
                let scan = ArrayScan::new(
                    int_ty.clone(),
                    Type::UNIT,
                    vec![int_ty.clone()],
                    size,
                    exec_extension_set(),
                );
                let zero = builder.add_load_value(ConstInt::new_u(6, 0).unwrap());
                let sum = builder
                    .add_dataflow_op(scan, [arr, func_v, zero])
                    .unwrap()
                    .out_wire(1);
                builder.finish_with_outputs([sum]).unwrap()
            });
        exec_ctx.add_extensions(|cge| cge.add_default_prelude_extensions().add_int_extensions());
        let expected: u64 = (0..size).sum();
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }
}

use anyhow::{Result, anyhow, bail, ensure};

use hugr_core::{
    HugrView, Node,
    extension::{
        prelude::{ConstError, bool_t, sum_with_error},
        simple_op::MakeExtensionOp,
    },
    ops::{DataflowOpTrait as _, constant::Value, custom::ExtensionOp},
    std_extensions::arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES},
    types::{TypeEnum, TypeRow},
};

use inkwell::{FloatPredicate, IntPredicate, types::IntType, values::BasicValue};

use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{
        EmitOpArgs,
        func::EmitFuncContext,
        ops::{emit_custom_unary_op, emit_value},
    },
    extension::int::get_width_arg,
    sum::LLVMSumValue,
    types::HugrType,
};

/// Returns the largest and smallest values that can be represented by an
/// integer of the given `width`.
///
/// The elements of the tuple are:
///  - The most negative signed integer
///  - The most positive signed integer
///  - The largest unsigned integer
#[must_use]
pub fn int_type_bounds(width: u32) -> (i64, i64, u64) {
    assert!(width <= 64);
    (
        i64::MIN >> (64 - width),
        i64::MAX >> (64 - width),
        u64::MAX >> (64 - width),
    )
}

fn build_trunc_op<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    signed: bool,
    log_width: u64,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
) -> Result<()> {
    let hugr_int_ty = INT_TYPES[log_width as usize].clone();
    let hugr_sum_ty = sum_with_error(vec![hugr_int_ty.clone()]);
    // TODO: it would be nice to get this info out of `ops.node()`, this would
    // require adding appropriate methods to `ConvertOpDef`. In the meantime, we
    // assert that the output types are as we expect.
    debug_assert_eq!(
        TypeRow::from(vec![HugrType::from(hugr_sum_ty.clone())]),
        args.node().signature().output
    );

    let Some(int_ty) = IntType::try_from(context.llvm_type(&hugr_int_ty)?).ok() else {
        bail!("Expected `arithmetic.int` to lower to an llvm integer")
    };

    let sum_ty = context.llvm_sum_type(hugr_sum_ty)?;

    let (width, (int_min_value_s, int_max_value_s, int_max_value_u)) = {
        ensure!(
            log_width <= 6,
            "Expected log_width of output to be <= 6, found: {log_width}"
        );
        let width = 1 << log_width;
        (width, int_type_bounds(width))
    };

    emit_custom_unary_op(context, args, |ctx, arg, _| {
        // We have to check if the conversion will work, so we
        // make the maximum int and convert to a float, then compare
        // with the function input.
        let flt_max = ctx.iw_context().f64_type().const_float(if signed {
            int_max_value_s as f64
        } else {
            int_max_value_u as f64
        });

        let within_upper_bound = ctx.builder().build_float_compare(
            FloatPredicate::OLT,
            arg.into_float_value(),
            flt_max,
            "within_upper_bound",
        )?;

        let flt_min = ctx.iw_context().f64_type().const_float(if signed {
            int_min_value_s as f64
        } else {
            0.0
        });

        let within_lower_bound = ctx.builder().build_float_compare(
            FloatPredicate::OLE,
            flt_min,
            arg.into_float_value(),
            "within_lower_bound",
        )?;

        // N.B. If the float value is NaN, we will never succeed.
        let success = ctx
            .builder()
            .build_and(within_upper_bound, within_lower_bound, "success")
            .unwrap();

        // Perform the conversion unconditionally, which will result
        // in a poison value if the input was too large. We will
        // decide whether we return it based on the result of our
        // earlier check.
        let trunc_result = if signed {
            ctx.builder()
                .build_float_to_signed_int(arg.into_float_value(), int_ty, "trunc_result")
        } else {
            ctx.builder().build_float_to_unsigned_int(
                arg.into_float_value(),
                int_ty,
                "trunc_result",
            )
        }?
        .as_basic_value_enum();

        let err_msg = Value::extension(ConstError::new(
            2,
            format!("Float value too big to convert to int of given width ({width})"),
        ));

        let err_val = emit_value(ctx, &err_msg)?;
        let failure = sum_ty.build_tag(ctx.builder(), 0, vec![err_val]).unwrap();
        let trunc_result = sum_ty
            .build_tag(ctx.builder(), 1, vec![trunc_result])
            .unwrap();

        let final_result = ctx
            .builder()
            .build_select(success, trunc_result, failure, "")
            .unwrap();
        Ok(vec![final_result])
    })
}

fn emit_conversion_op<'c, H: HugrView<Node = Node>>(
    context: &mut EmitFuncContext<'c, '_, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    conversion_op: ConvertOpDef,
) -> Result<()> {
    match conversion_op {
        ConvertOpDef::trunc_u | ConvertOpDef::trunc_s => {
            let signed = conversion_op == ConvertOpDef::trunc_s;
            let log_width = get_width_arg(&args, &conversion_op)?;
            build_trunc_op(context, signed, log_width, args)
        }

        ConvertOpDef::convert_u => emit_custom_unary_op(context, args, |ctx, arg, out_tys| {
            let out_ty = out_tys.last().unwrap();
            Ok(vec![
                ctx.builder()
                    .build_unsigned_int_to_float(
                        arg.into_int_value(),
                        out_ty.into_float_type(),
                        "",
                    )?
                    .as_basic_value_enum(),
            ])
        }),

        ConvertOpDef::convert_s => emit_custom_unary_op(context, args, |ctx, arg, out_tys| {
            let out_ty = out_tys.last().unwrap();
            Ok(vec![
                ctx.builder()
                    .build_signed_int_to_float(arg.into_int_value(), out_ty.into_float_type(), "")?
                    .as_basic_value_enum(),
            ])
        }),
        // These ops convert between hugr's `USIZE` and u64. The former is
        // implementation-dependent and we define them to be the same.
        // Hence our implementation is a noop.
        ConvertOpDef::itousize | ConvertOpDef::ifromusize => {
            emit_custom_unary_op(context, args, |_, arg, _| Ok(vec![arg]))
        }
        ConvertOpDef::itobool | ConvertOpDef::ifrombool => {
            assert!(conversion_op.type_args().is_empty()); // Always 1-bit int <-> bool
            let i0_ty = context
                .typing_session()
                .llvm_type(&INT_TYPES[0])?
                .into_int_type();
            let sum_ty = context
                .typing_session()
                .llvm_sum_type(match bool_t().as_type_enum() {
                    TypeEnum::Sum(st) => st.clone(),
                    _ => panic!("Hugr prelude bool_t() not a Sum"),
                })?;

            emit_custom_unary_op(context, args, |ctx, arg, _| {
                let res = if conversion_op == ConvertOpDef::itobool {
                    let is1 = ctx.builder().build_int_compare(
                        IntPredicate::EQ,
                        arg.into_int_value(),
                        i0_ty.const_int(1, false),
                        "eq1",
                    )?;
                    let sum_f = sum_ty.build_tag(ctx.builder(), 0, vec![])?;
                    let sum_t = sum_ty.build_tag(ctx.builder(), 1, vec![])?;
                    ctx.builder().build_select(is1, sum_t, sum_f, "")?
                } else {
                    let tag_ty = sum_ty.tag_type();
                    let tag = LLVMSumValue::try_new(arg, sum_ty)?.build_get_tag(ctx.builder())?;
                    let is_true = ctx.builder().build_int_compare(
                        IntPredicate::EQ,
                        tag,
                        tag_ty.const_int(1, false),
                        "",
                    )?;
                    ctx.builder().build_select(
                        is_true,
                        i0_ty.const_int(1, false),
                        i0_ty.const_int(0, false),
                        "",
                    )?
                };
                Ok(vec![res])
            })
        }
        ConvertOpDef::bytecast_int64_to_float64 => {
            emit_custom_unary_op(context, args, |ctx, arg, outs| {
                let [out] = outs.try_into()?;
                Ok(vec![ctx.builder().build_bit_cast(arg, out, "")?])
            })
        }
        ConvertOpDef::bytecast_float64_to_int64 => {
            emit_custom_unary_op(context, args, |ctx, arg, outs| {
                let [out] = outs.try_into()?;
                Ok(vec![ctx.builder().build_bit_cast(arg, out, "")?])
            })
        }
        _ => Err(anyhow!(
            "Conversion op not implemented: {:?}",
            args.node().as_ref()
        )),
    }
}

#[derive(Clone, Debug)]
pub struct ConversionExtension;

impl CodegenExtension for ConversionExtension {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder.simple_extension_op(emit_conversion_op)
    }
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    #[must_use]
    pub fn add_conversion_extensions(self) -> Self {
        self.add_extension(ConversionExtension)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::check_emission;
    use crate::emit::test::{DFGW, SimpleHugrConfig};
    use crate::test::{TestContext, exec_ctx, llvm_ctx};
    use hugr_core::builder::{DataflowHugr, SubContainer};
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::std_extensions::arithmetic::float_types::ConstF64;
    use hugr_core::std_extensions::arithmetic::int_types::ConstInt;
    use hugr_core::{
        Hugr,
        builder::{Dataflow, DataflowSubContainer},
        extension::prelude::{ConstUsize, PRELUDE_REGISTRY, usize_t},
        std_extensions::arithmetic::{
            conversions::{ConvertOpDef, EXTENSION},
            float_types::float64_type,
            int_types::INT_TYPES,
        },
        types::Type,
    };
    use rstest::rstest;

    fn test_conversion_op(
        name: impl AsRef<str>,
        in_type: Type,
        out_type: Type,
        int_width: u8,
    ) -> Hugr {
        SimpleHugrConfig::new()
            .with_ins(vec![in_type.clone()])
            .with_outs(vec![out_type.clone()])
            .with_extensions(STD_REG.clone())
            .finish(|mut hugr_builder| {
                let [in1] = hugr_builder.input_wires_arr();
                let ext_op = EXTENSION
                    .instantiate_extension_op(name.as_ref(), [u64::from(int_width).into()])
                    .unwrap();
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, [in1])
                    .unwrap()
                    .outputs();
                hugr_builder.finish_hugr_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    #[case("convert_u", 4)]
    #[case("convert_s", 5)]
    fn test_convert(mut llvm_ctx: TestContext, #[case] op_name: &str, #[case] log_width: u8) -> () {
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_default_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
        });
        let in_ty = INT_TYPES[log_width as usize].clone();
        let out_ty = float64_type();
        let hugr = test_conversion_op(op_name, in_ty, out_ty, log_width);
        check_emission!(op_name, hugr, llvm_ctx);
    }

    #[rstest]
    #[case("trunc_u", 6)]
    #[case("trunc_s", 5)]
    fn test_truncation(
        mut llvm_ctx: TestContext,
        #[case] op_name: &str,
        #[case] log_width: u8,
    ) -> () {
        llvm_ctx.add_extensions(|builder| {
            builder
                .add_default_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
                .add_default_prelude_extensions()
        });
        let in_ty = float64_type();
        let out_ty = sum_with_error(INT_TYPES[log_width as usize].clone());
        let hugr = test_conversion_op(op_name, in_ty, out_ty.into(), log_width);
        check_emission!(op_name, hugr, llvm_ctx);
    }

    #[rstest]
    #[case("itobool", true)]
    #[case("ifrombool", false)]
    fn test_intbool_emit(
        mut llvm_ctx: TestContext,
        #[case] op_name: &str,
        #[case] input_int: bool,
    ) {
        let mut tys = [INT_TYPES[0].clone(), bool_t()];
        if !input_int {
            tys.reverse();
        }
        let [in_t, out_t] = tys;
        llvm_ctx.add_extensions(|builder| {
            builder
                .add_default_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
        });
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![in_t])
            .with_outs(vec![out_t])
            .with_extensions(STD_REG.to_owned())
            .finish(|mut hugr_builder| {
                let [in1] = hugr_builder.input_wires_arr();
                let ext_op = EXTENSION.instantiate_extension_op(op_name, []).unwrap();
                let [out1] = hugr_builder
                    .add_dataflow_op(ext_op, [in1])
                    .unwrap()
                    .outputs_arr();
                hugr_builder.finish_hugr_with_outputs([out1]).unwrap()
            });
        check_emission!(op_name, hugr, llvm_ctx);
    }

    #[rstest]
    fn my_test_exec(mut exec_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder: DFGW| {
                let konst = builder.add_load_value(ConstUsize::new(42));
                builder.finish_hugr_with_outputs([konst]).unwrap()
            });
        exec_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        assert_eq!(42, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(0)]
    #[case(42)]
    #[case(18_446_744_073_709_551_615)]
    fn usize_roundtrip(mut exec_ctx: TestContext, #[case] val: u64) -> () {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(STD_REG.clone())
            .finish(|mut builder: DFGW| {
                let k = builder.add_load_value(ConstUsize::new(val));
                let [int] = builder
                    .add_dataflow_op(ConvertOpDef::ifromusize.without_log_width(), [k])
                    .unwrap()
                    .outputs_arr();
                let [usize_] = builder
                    .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [int])
                    .unwrap()
                    .outputs_arr();
                builder.finish_hugr_with_outputs([usize_]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_default_int_extensions()
                .add_conversion_extensions()
                .add_default_prelude_extensions()
        });
        assert_eq!(val, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    fn roundtrip_hugr(val: u64, signed: bool) -> Hugr {
        let int64 = INT_TYPES[6].clone();
        SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(STD_REG.clone())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(val));
                let [int] = builder
                    .add_dataflow_op(ConvertOpDef::ifromusize.without_log_width(), [k])
                    .unwrap()
                    .outputs_arr();
                let [flt] = {
                    let op = if signed {
                        ConvertOpDef::convert_s.with_log_width(6)
                    } else {
                        ConvertOpDef::convert_u.with_log_width(6)
                    };
                    builder.add_dataflow_op(op, [int]).unwrap().outputs_arr()
                };

                let [int_or_err] = {
                    let op = if signed {
                        ConvertOpDef::trunc_s.with_log_width(6)
                    } else {
                        ConvertOpDef::trunc_u.with_log_width(6)
                    };
                    builder.add_dataflow_op(op, [flt]).unwrap().outputs_arr()
                };
                let sum_ty = sum_with_error(int64.clone());
                let variants = (0..sum_ty.num_variants())
                    .map(|i| sum_ty.get_variant(i).unwrap().clone().try_into().unwrap());
                let mut cond_b = builder
                    .conditional_builder((variants, int_or_err), [], vec![int64].into())
                    .unwrap();
                let win_case = cond_b.case_builder(1).unwrap();
                let [win_in] = win_case.input_wires_arr();
                win_case.finish_with_outputs([win_in]).unwrap();
                let mut lose_case = cond_b.case_builder(0).unwrap();
                let const_999 = lose_case.add_load_value(ConstUsize::new(999));
                let [const_999] = lose_case
                    .add_dataflow_op(ConvertOpDef::ifromusize.without_log_width(), [const_999])
                    .unwrap()
                    .outputs_arr();
                lose_case.finish_with_outputs([const_999]).unwrap();

                let cond = cond_b.finish_sub_container().unwrap();

                let [cond_result] = cond.outputs_arr();

                let [usize_] = builder
                    .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [cond_result])
                    .unwrap()
                    .outputs_arr();
                builder.finish_hugr_with_outputs([usize_]).unwrap()
            })
    }

    fn add_extensions(ctx: &mut TestContext) {
        ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
                .add_default_int_extensions()
        });
    }

    #[rstest]
    // Exact roundtrip conversion is defined on values up to 2**53 for f64.
    #[case(0)]
    #[case(3)]
    #[case(255)]
    #[case(4294967295)]
    #[case(42)]
    #[case(18_000_000_000_000_000_000)]
    fn roundtrip_unsigned(mut exec_ctx: TestContext, #[case] val: u64) {
        add_extensions(&mut exec_ctx);
        let hugr = roundtrip_hugr(val, false);
        assert_eq!(val, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    // Exact roundtrip conversion is defined on values up to 2**53 for f64.
    #[case(0)]
    #[case(3)]
    #[case(255)]
    #[case(4294967295)]
    #[case(42)]
    #[case(-9_000_000_000_000_000_000)]
    fn roundtrip_signed(mut exec_ctx: TestContext, #[case] val: i64) {
        add_extensions(&mut exec_ctx);
        let hugr = roundtrip_hugr(val as u64, true);
        assert_eq!(val, exec_ctx.exec_hugr_u64(hugr, "main") as i64);
    }

    // For unisgined ints larger than (1 << 54) - 1, f64s do not have enough
    // precision to exactly roundtrip the int.
    // The exact behaviour of the round-trip is is platform-dependent.
    #[rstest]
    #[case(u64::MAX)]
    #[case(u64::MAX - 1)]
    #[case(u64::MAX - (1 << 1))]
    #[case(u64::MAX - (1 << 2))]
    #[case(u64::MAX - (1 << 3))]
    #[case(u64::MAX - (1 << 4))]
    #[case(u64::MAX - (1 << 5))]
    #[case(u64::MAX - (1 << 6))]
    #[case(u64::MAX - (1 << 7))]
    #[case(u64::MAX - (1 << 8))]
    #[case(u64::MAX - (1 << 9))]
    #[case(u64::MAX - (1 << 10))]
    #[case(u64::MAX - (1 << 11))]
    fn approx_roundtrip_unsigned(mut exec_ctx: TestContext, #[case] val: u64) {
        add_extensions(&mut exec_ctx);

        let hugr = roundtrip_hugr(val, false);
        let result = exec_ctx.exec_hugr_u64(hugr, "main");
        let (v_r_max, v_r_min) = (val.max(result), val.min(result));
        // If val is too large the `trunc_u` op in `hugr` will return None.
        // In this case the hugr returns the magic number `999`.
        assert!(result == 999 || (v_r_max - v_r_min) < 1 << 10);
    }

    #[rstest]
    #[case(i64::MAX)]
    #[case(i64::MAX - 1)]
    #[case(i64::MAX - (1 << 1))]
    #[case(i64::MAX - (1 << 2))]
    #[case(i64::MAX - (1 << 3))]
    #[case(i64::MAX - (1 << 4))]
    #[case(i64::MAX - (1 << 5))]
    #[case(i64::MAX - (1 << 6))]
    #[case(i64::MAX - (1 << 7))]
    #[case(i64::MAX - (1 << 8))]
    #[case(i64::MAX - (1 << 9))]
    #[case(i64::MAX - (1 << 10))]
    #[case(i64::MAX - (1 << 11))]
    #[case(i64::MIN)]
    #[case(i64::MIN + 1)]
    #[case(i64::MIN + (1 << 1))]
    #[case(i64::MIN + (1 << 2))]
    #[case(i64::MIN + (1 << 3))]
    #[case(i64::MIN + (1 << 4))]
    #[case(i64::MIN + (1 << 5))]
    #[case(i64::MIN + (1 << 6))]
    #[case(i64::MIN + (1 << 7))]
    #[case(i64::MIN + (1 << 8))]
    #[case(i64::MIN + (1 << 9))]
    #[case(i64::MIN + (1 << 10))]
    #[case(i64::MIN + (1 << 11))]
    fn approx_roundtrip_signed(mut exec_ctx: TestContext, #[case] val: i64) {
        add_extensions(&mut exec_ctx);

        let hugr = roundtrip_hugr(val as u64, true);
        let result = exec_ctx.exec_hugr_u64(hugr, "main") as i64;
        // If val.abs() is too large the `trunc_s` op in `hugr` will return None.
        // In this case the hugr returns the magic number `999`.
        assert!(result == 999 || (val - result).abs() < 1 << 10);
    }

    #[rstest]
    fn itobool_cond(mut exec_ctx: TestContext, #[values(0, 1)] i: u64) {
        use hugr_core::type_row;

        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![usize_t()])
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstInt::new_u(0, i).unwrap());
                let ext_op = EXTENSION.instantiate_extension_op("itobool", []).unwrap();
                let [b] = builder.add_dataflow_op(ext_op, [i]).unwrap().outputs_arr();
                let mut cond = builder
                    .conditional_builder(
                        ([type_row![], type_row![]], b),
                        [],
                        vec![usize_t()].into(),
                    )
                    .unwrap();
                let mut case_false = cond.case_builder(0).unwrap();
                let false_result = case_false.add_load_value(ConstUsize::new(1));
                case_false.finish_with_outputs([false_result]).unwrap();
                let mut case_true = cond.case_builder(1).unwrap();
                let true_result = case_true.add_load_value(ConstUsize::new(6));
                case_true.finish_with_outputs([true_result]).unwrap();
                let res = cond.finish_sub_container().unwrap();
                builder.finish_hugr_with_outputs(res.outputs()).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(i * 5 + 1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    fn itobool_roundtrip(mut exec_ctx: TestContext, #[values(0, 1)] i: u64) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![INT_TYPES[0].clone()])
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstInt::new_u(0, i).unwrap());
                let i2b = EXTENSION.instantiate_extension_op("itobool", []).unwrap();
                let [b] = builder.add_dataflow_op(i2b, [i]).unwrap().outputs_arr();
                let b2i = EXTENSION.instantiate_extension_op("ifrombool", []).unwrap();
                let [i] = builder.add_dataflow_op(b2i, [b]).unwrap().outputs_arr();
                builder.finish_hugr_with_outputs([i]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_default_int_extensions()
        });
        assert_eq!(i, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case(42.0)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    #[case(f64::NAN)]
    #[case(-0.0f64)]
    #[case(0.0f64)]
    fn bytecast_int64_to_float64(mut exec_ctx: TestContext, #[case] f: f64) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(float64_type())
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstInt::new_u(6, f.to_bits()).unwrap());
                let i2f = EXTENSION
                    .instantiate_extension_op("bytecast_int64_to_float64", [])
                    .unwrap();
                let [f] = builder.add_dataflow_op(i2f, [i]).unwrap().outputs_arr();
                builder.finish_hugr_with_outputs([f]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_default_int_extensions()
                .add_float_extensions()
        });
        let hugr_f = exec_ctx.exec_hugr_f64(hugr, "main");
        assert!((f.is_nan() && hugr_f.is_nan()) || f == hugr_f);
    }

    #[rstest]
    #[case(42.0)]
    #[case(-0.0f64)]
    #[case(0.0f64)]
    fn bytecast_float64_to_int64(mut exec_ctx: TestContext, #[case] f: f64) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(INT_TYPES[6].clone())
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder| {
                let f = builder.add_load_value(ConstF64::new(f));
                let f2i = EXTENSION
                    .instantiate_extension_op("bytecast_float64_to_int64", [])
                    .unwrap();
                let [i] = builder.add_dataflow_op(f2i, [f]).unwrap().outputs_arr();
                builder.finish_hugr_with_outputs([i]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_default_int_extensions()
                .add_float_extensions()
        });
        assert_eq!(f.to_bits(), exec_ctx.exec_hugr_u64(hugr, "main"));
    }
}

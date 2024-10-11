use super::{CodegenExtension, CodegenExtsBuilder};

use anyhow::{anyhow, Result};

use hugr::{
    extension::{
        prelude::{sum_with_error, ConstError, BOOL_T},
        simple_op::MakeExtensionOp,
    },
    ops::{constant::Value, custom::ExtensionOp},
    std_extensions::arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES},
    types::{TypeArg, TypeEnum},
    HugrView,
};

use inkwell::{values::BasicValue, FloatPredicate, IntPredicate};

use crate::{
    emit::{
        func::EmitFuncContext,
        ops::{emit_custom_unary_op, emit_value},
        EmitOpArgs,
    },
    sum::LLVMSumValue,
};

fn build_trunc_op<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    signed: bool,
    log_width: u64,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
) -> Result<()> {
    // Note: This logic is copied from `llvm_type` in the IntTypes
    // extension. We need to have a common source of truth for this.
    let (width, (int_min_value_s, int_max_value_s), int_max_value_u) = match log_width {
        0..=3 => (8, (i8::MIN as i64, i8::MAX as i64), u8::MAX as u64),
        4 => (16, (i16::MIN as i64, i16::MAX as i64), u16::MAX as u64),
        5 => (32, (i32::MIN as i64, i32::MAX as i64), u32::MAX as u64),
        6 => (64, (i64::MIN, i64::MAX), u64::MAX),
        m => return Err(anyhow!("ConversionEmitter: unsupported log_width: {}", m)),
    };

    let hugr_int_ty = INT_TYPES[log_width as usize].clone();
    let int_ty = context
        .typing_session()
        .llvm_type(&hugr_int_ty)?
        .into_int_type();

    let hugr_sum_ty = sum_with_error(vec![hugr_int_ty]);
    let sum_ty = context.typing_session().llvm_sum_type(hugr_sum_ty)?;

    emit_custom_unary_op(context, args, |ctx, arg, _| {
        // We have to check if the conversion will work, so we
        // make the maximum int and convert to a float, then compare
        // with the function input.
        let flt_max = if signed {
            ctx.iw_context()
                .f64_type()
                .const_float(int_max_value_s as f64)
        } else {
            ctx.iw_context()
                .f64_type()
                .const_float(int_max_value_u as f64)
        };

        let within_upper_bound = ctx.builder().build_float_compare(
            FloatPredicate::OLE,
            arg.into_float_value(),
            flt_max,
            "within_upper_bound",
        )?;

        let flt_min = if signed {
            ctx.iw_context()
                .f64_type()
                .const_float(int_min_value_s as f64)
        } else {
            ctx.iw_context().f64_type().const_float(0.0)
        };

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
            format!(
                "Float value too big to convert to int of given width ({})",
                width
            ),
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

fn emit_conversion_op<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, H>,
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    conversion_op: ConvertOpDef,
) -> Result<()> {
    match conversion_op {
        ConvertOpDef::trunc_u | ConvertOpDef::trunc_s => {
            let signed = conversion_op == ConvertOpDef::trunc_s;
            let Some(TypeArg::BoundedNat { n: log_width }) = args.node().args().last().cloned()
            else {
                panic!("This op should have one type arg only: the log-width of the int we're truncating to.: {:?}", conversion_op.type_args())
            };

            build_trunc_op(context, signed, log_width, args)
        }

        ConvertOpDef::convert_u => emit_custom_unary_op(context, args, |ctx, arg, out_tys| {
            let out_ty = out_tys.last().unwrap();
            Ok(vec![ctx
                .builder()
                .build_unsigned_int_to_float(arg.into_int_value(), out_ty.into_float_type(), "")?
                .as_basic_value_enum()])
        }),

        ConvertOpDef::convert_s => emit_custom_unary_op(context, args, |ctx, arg, out_tys| {
            let out_ty = out_tys.last().unwrap();
            Ok(vec![ctx
                .builder()
                .build_signed_int_to_float(arg.into_int_value(), out_ty.into_float_type(), "")?
                .as_basic_value_enum()])
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
                .llvm_sum_type(match BOOL_T.as_type_enum() {
                    TypeEnum::Sum(st) => st.clone(),
                    _ => panic!("Hugr prelude BOOL_T not a Sum"),
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
                    let tag_ty = sum_ty.get_tag_type();
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
        _ => Err(anyhow!(
            "Conversion op not implemented: {:?}",
            args.node().as_ref()
        )),
    }
}

#[derive(Clone, Debug)]
pub struct ConversionExtension;

impl CodegenExtension for ConversionExtension {
    fn add_extension<'a, H: HugrView + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder.simple_extension_op(emit_conversion_op)
    }
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    pub fn add_conversion_extensions(self) -> Self {
        self.add_extension(ConversionExtension)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::check_emission;
    use crate::emit::test::{SimpleHugrConfig, DFGW};
    use crate::test::{exec_ctx, llvm_ctx, TestContext};
    use hugr::builder::SubContainer;
    use hugr::std_extensions::arithmetic::int_types::ConstInt;
    use hugr::{
        builder::{Dataflow, DataflowSubContainer},
        extension::prelude::{ConstUsize, PRELUDE_REGISTRY, USIZE_T},
        std_extensions::arithmetic::{
            conversions::{ConvertOpDef, CONVERT_OPS_REGISTRY, EXTENSION},
            float_types::FLOAT64_TYPE,
            int_types::INT_TYPES,
        },
        types::Type,
        Hugr,
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
            .with_extensions(CONVERT_OPS_REGISTRY.clone())
            .finish(|mut hugr_builder| {
                let [in1] = hugr_builder.input_wires_arr();
                let ext_op = EXTENSION
                    .instantiate_extension_op(
                        name.as_ref(),
                        [(int_width as u64).into()],
                        &CONVERT_OPS_REGISTRY,
                    )
                    .unwrap();
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, [in1])
                    .unwrap()
                    .outputs();
                hugr_builder.finish_with_outputs(outputs).unwrap()
            })
    }

    #[rstest]
    #[case("convert_u", 4)]
    #[case("convert_s", 5)]
    fn test_convert(mut llvm_ctx: TestContext, #[case] op_name: &str, #[case] log_width: u8) -> () {
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
        });
        let in_ty = INT_TYPES[log_width as usize].clone();
        let out_ty = FLOAT64_TYPE;
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
                .add_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
                .add_default_prelude_extensions()
        });
        let in_ty = FLOAT64_TYPE;
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
        let mut tys = [INT_TYPES[0].clone(), BOOL_T];
        if !input_int {
            tys.reverse()
        };
        let [in_t, out_t] = tys;
        llvm_ctx.add_extensions(|builder| {
            builder
                .add_int_extensions()
                .add_float_extensions()
                .add_conversion_extensions()
        });
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![in_t])
            .with_outs(vec![out_t])
            .with_extensions(CONVERT_OPS_REGISTRY.to_owned())
            .finish(|mut hugr_builder| {
                let [in1] = hugr_builder.input_wires_arr();
                let ext_op = EXTENSION
                    .instantiate_extension_op(op_name, [], &CONVERT_OPS_REGISTRY)
                    .unwrap();
                let [out1] = hugr_builder
                    .add_dataflow_op(ext_op, [in1])
                    .unwrap()
                    .outputs_arr();
                hugr_builder.finish_with_outputs([out1]).unwrap()
            });
        check_emission!(op_name, hugr, llvm_ctx);
    }

    #[rstest]
    fn my_test_exec(mut exec_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(USIZE_T)
            .with_extensions(PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder: DFGW| {
                let konst = builder.add_load_value(ConstUsize::new(42));
                builder.finish_with_outputs([konst]).unwrap()
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
            .with_outs(USIZE_T)
            .with_extensions(CONVERT_OPS_REGISTRY.clone())
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
                builder.finish_with_outputs([usize_]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_int_extensions()
                .add_conversion_extensions()
                .add_default_prelude_extensions()
        });
        assert_eq!(val, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    fn roundtrip_hugr(val: u64) -> Hugr {
        let int64 = INT_TYPES[6].clone();
        SimpleHugrConfig::new()
            .with_outs(USIZE_T)
            .with_extensions(CONVERT_OPS_REGISTRY.clone())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(val));
                let [int] = builder
                    .add_dataflow_op(ConvertOpDef::ifromusize.without_log_width(), [k])
                    .unwrap()
                    .outputs_arr();
                let [flt] = builder
                    .add_dataflow_op(ConvertOpDef::convert_u.with_log_width(6), [int])
                    .unwrap()
                    .outputs_arr();
                let [int_or_err] = builder
                    .add_dataflow_op(ConvertOpDef::trunc_u.with_log_width(6), [flt])
                    .unwrap()
                    .outputs_arr();
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
                builder.finish_with_outputs([usize_]).unwrap()
            })
    }

    fn add_extensions(ctx: &mut TestContext) {
        ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
                .add_int_extensions()
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
    fn roundtrip(mut exec_ctx: TestContext, #[case] val: u64) {
        add_extensions(&mut exec_ctx);
        let hugr = roundtrip_hugr(val);
        assert_eq!(val, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    // N.B.: There's some strange behaviour at the upper end of the ints - the
    // first case gets converted to something that's off by 1,000, but the second
    // (which is (2 ^ 64) - 1) gets converted to (2 ^ 32) - off by 9 million!
    // The fact that the first case works as expected  tells me this isn't to do
    // with int widths - maybe a floating point expert could explain that this
    // is standard behaviour...
    #[rstest]
    #[case(18_446_744_073_709_550_000, 18_446_744_073_709_549_568)]
    #[case(18_446_744_073_709_551_615, 9_223_372_036_854_775_808)] // 2 ^ 63
    fn approx_roundtrip(mut exec_ctx: TestContext, #[case] val: u64, #[case] expected: u64) {
        add_extensions(&mut exec_ctx);
        let hugr = roundtrip_hugr(val);
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    fn itobool_cond(mut exec_ctx: TestContext, #[values(0, 1)] i: u64) {
        use hugr::type_row;

        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![USIZE_T])
            .with_extensions(CONVERT_OPS_REGISTRY.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstInt::new_u(0, i).unwrap());
                let ext_op = EXTENSION
                    .instantiate_extension_op("itobool", [], &CONVERT_OPS_REGISTRY)
                    .unwrap();
                let [b] = builder.add_dataflow_op(ext_op, [i]).unwrap().outputs_arr();
                let mut cond = builder
                    .conditional_builder(([type_row![], type_row![]], b), [], type_row![USIZE_T])
                    .unwrap();
                let mut case_false = cond.case_builder(0).unwrap();
                let false_result = case_false.add_load_value(ConstUsize::new(1));
                case_false.finish_with_outputs([false_result]).unwrap();
                let mut case_true = cond.case_builder(1).unwrap();
                let true_result = case_true.add_load_value(ConstUsize::new(6));
                case_true.finish_with_outputs([true_result]).unwrap();
                let res = cond.finish_sub_container().unwrap();
                builder.finish_with_outputs(res.outputs()).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_int_extensions()
        });
        assert_eq!(i * 5 + 1, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    fn itobool_roundtrip(mut exec_ctx: TestContext, #[values(0, 1)] i: u64) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![INT_TYPES[0].clone()])
            .with_extensions(CONVERT_OPS_REGISTRY.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstInt::new_u(0, i).unwrap());
                let i2b = EXTENSION
                    .instantiate_extension_op("itobool", [], &CONVERT_OPS_REGISTRY)
                    .unwrap();
                let [b] = builder.add_dataflow_op(i2b, [i]).unwrap().outputs_arr();
                let b2i = EXTENSION
                    .instantiate_extension_op("ifrombool", [], &CONVERT_OPS_REGISTRY)
                    .unwrap();
                let [i] = builder.add_dataflow_op(b2i, [b]).unwrap().outputs_arr();
                builder.finish_with_outputs([i]).unwrap()
            });
        exec_ctx.add_extensions(|builder| {
            builder
                .add_conversion_extensions()
                .add_default_prelude_extensions()
                .add_int_extensions()
        });
        assert_eq!(i, exec_ctx.exec_hugr_u64(hugr, "main"));
    }
}

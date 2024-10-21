use anyhow::{anyhow, bail, Result};

use hugr::{
    extension::prelude::{option_type, ConstError},
    ops::ExtensionOp,
    HugrView,
};
use inkwell::{
    types::FloatType,
    values::{FloatValue, IntValue},
    FloatPredicate,
};
use lazy_static::lazy_static;

use crate::{
    custom::CodegenExtsBuilder,
    emit::{emit_value, get_intrinsic, EmitFuncContext, EmitOpArgs},
    types::TypingSession,
    CodegenExtension,
};

use tket2::extension::rotation::{
    ConstRotation, RotationOp, ROTATION_CUSTOM_TYPE, ROTATION_EXTENSION_ID, ROTATION_TYPE,
};

use super::{DefaultPreludeCodegen, PreludeCodegen};

/// A codegen extension for the `tket2.rotation` extension.
///
/// We lower [ROTATION_CUSTOM_TYPE] to an `f64`, representing a number of half-turns.
///
/// A `RotationCodegenExtension` carries a `PCG`, which should impl
/// [PreludeCodegen]. This is used to [PreludeCodegen::emit_panic] when lowering panicking ops.
#[derive(Clone)]
pub struct RotationCodegenExtension<PCG> {
    prelude_codegen: PCG,
    from_halfturns_err: ConstError,
}

lazy_static! {
    /// The error emitted when panicking in the lowering of
    /// `tket2.rotation.from_halfturns_unchecked` by
    /// [DEFAULT_ROTATION_EXTENSION].
    pub static ref DEFAULT_FROM_HALFTURNS_ERROR: ConstError =
        ConstError::new(1, "tket2.rotation.from_halfturns_unchecked failed");

    /// The codegen extension that is registered by
    /// [CodegenExtsBuilder::add_default_rotation_extensions].
    pub static ref DEFAULT_ROTATION_EXTENSION: RotationCodegenExtension<DefaultPreludeCodegen> = RotationCodegenExtension::new(DefaultPreludeCodegen);
}

fn llvm_angle_type<'c>(ts: &TypingSession<'c, '_>) -> FloatType<'c> {
    ts.iw_context().f64_type()
}

impl<PCG: PreludeCodegen> RotationCodegenExtension<PCG> {
    /// Returns a new RotationCodegenExtension with the given [PreludeCodegen].
    pub fn new(prelude_codegen: PCG) -> Self {
        Self {
            prelude_codegen,
            from_halfturns_err: DEFAULT_FROM_HALFTURNS_ERROR.to_owned(),
        }
    }

    /// Returns a new RotationCodegenExtension the given `from_halfturns_err`.
    ///
    /// While lowering a `tket2.rotation.from_halfturns_unchecked` op we must
    /// panic in some codepaths. This function allows customising the panic
    /// message. The default panic message is [static@DEFAULT_FROM_HALFTURNS_ERROR].
    pub fn with_from_halfturns_err(mut self, from_halfturns_err: ConstError) -> Self {
        self.from_halfturns_err = from_halfturns_err;
        self
    }

    /// returns (float, bool) where the float is the number of halfturns, or
    /// poison. If the bool is true then the float is not poison.
    fn emit_from_halfturns<'c, H: HugrView>(
        &self,
        context: &mut EmitFuncContext<'c, '_, H>,
        half_turns: FloatValue<'c>,
    ) -> Result<(FloatValue<'c>, IntValue<'c>)> {
        let angle_ty = llvm_angle_type(&context.typing_session());
        let builder = context.builder();

        // We must distinguish {NaNs, infinities} from finite
        // values. The `llvm.is.fpclass` intrinsic was introduced in llvm 15
        // and is the best way to do so. For now we are using llvm
        // 14, and so we use 3 `feq`s.
        // Below is commented code that we can use once we support llvm 15.
        #[cfg(feature = "llvm14-0")]
        let half_turns_ok = {
            let is_pos_inf = builder.build_float_compare(
                FloatPredicate::OEQ,
                half_turns,
                angle_ty.const_float(f64::INFINITY),
                "",
            )?;
            let is_neg_inf = builder.build_float_compare(
                FloatPredicate::OEQ,
                half_turns,
                angle_ty.const_float(f64::NEG_INFINITY),
                "",
            )?;
            let is_nan = builder.build_float_compare(
                FloatPredicate::UNO,
                half_turns,
                angle_ty.const_zero(),
                "",
            )?;
            builder.build_not(
                builder.build_or(builder.build_or(is_pos_inf, is_neg_inf, "")?, is_nan, "")?,
                "",
            )?
        };
        // let half_turns_ok = {
        //     let i32_ty = self.0.iw_context().i32_type();
        //     let builder = self.0.builder();
        //     let is_fpclass = get_intrinsic(module, "llvm.is.fpclass", [float_ty.as_basic_type_enum(), i32_ty.as_basic_type_enum()])?;
        //     // Here we pick out the following floats:
        //     //  - bit 0: Signalling Nan
        //     //  - bit 3: Negative normal
        //     //  - bit 8: Positive normal
        //     let test = i32_ty.const_int((1 << 0) | (1 << 3) | (1 << 8), false);
        //     builder
        //         .build_call(is_fpclass, &[rads.into(), test.into()], "")?
        //         .try_as_basic_value()
        //         .left()
        //         .ok_or(anyhow!("llvm.is.fpclass has no return value"))?
        //         .into_int_value()
        // };
        Ok((half_turns, half_turns_ok))
    }

    fn emit_rotation_op<'c, H: HugrView>(
        &self,
        context: &mut EmitFuncContext<'c, '_, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
        op: RotationOp,
    ) -> Result<()> {
        let ts = context.typing_session();
        let module = context.get_current_module();
        let builder = context.builder();
        let angle_ty = llvm_angle_type(&ts);

        match op {
            RotationOp::radd => {
                let [lhs, rhs] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::radd expects two arguments"))?;
                let (lhs, rhs) = (lhs.into_float_value(), rhs.into_float_value());
                let r = builder.build_float_add(lhs, rhs, "")?;
                args.outputs.finish(builder, [r.into()])
            }
            RotationOp::from_halfturns_unchecked => {
                let [half_turns] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::from_halfturns expects one arguments"))?;
                let (half_turns, half_turns_ok) =
                    self.emit_from_halfturns(context, half_turns.into_float_value())?;

                let fail_block = context.build_positioned_new_block("", None, |context, bb| {
                    let err = emit_value(context, &self.from_halfturns_err.clone().into())?;
                    self.prelude_codegen.emit_panic(context, err)?;
                    context.builder().build_unreachable()?;
                    anyhow::Ok(bb)
                })?;

                let success_block =
                    context.build_positioned_new_block("", None, |context, bb| {
                        args.outputs
                            .finish(context.builder(), [half_turns.into()])?;
                        anyhow::Ok(bb)
                    })?;

                context.builder().build_conditional_branch(
                    half_turns_ok,
                    success_block,
                    fail_block,
                )?;
                context.builder().position_at_end(success_block);
                Ok(())
            }
            RotationOp::from_halfturns => {
                let [half_turns] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::from_halfturns expects one arguments"))?;
                let (half_turns, half_turns_ok) =
                    self.emit_from_halfturns(context, half_turns.into_float_value())?;

                let builder = context.builder();
                let result_sum_type = ts.llvm_sum_type(option_type(ROTATION_TYPE))?;
                let success = result_sum_type.build_tag(builder, 1, vec![half_turns.into()])?;
                let failure = result_sum_type.build_tag(builder, 0, vec![])?;
                let result = builder.build_select(half_turns_ok, success, failure, "")?;
                args.outputs.finish(builder, [result])
            }
            RotationOp::to_halfturns => {
                let [half_turns] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::tohalfturns expects one argument"))?;
                let half_turns = half_turns.into_float_value();

                // normalised_half_turns is in the interval 0..2
                let normalised_half_turns = {
                    // normalised_rads = (half_turns/2 - floor(half_turns/2)) * 2
                    // note that floor(x) gives the largest integral value less
                    // than or equal to x so this deals with both positive and
                    // negative rads.
                    let turns =
                        builder.build_float_div(half_turns, angle_ty.const_float(2.0), "")?;
                    let floor_turns = {
                        let floor = get_intrinsic(module, "llvm.floor", [angle_ty.into()])?;
                        builder
                            .build_call(floor, &[turns.into()], "")?
                            .try_as_basic_value()
                            .left()
                            .ok_or(anyhow!("llvm.floor has no return value"))?
                            .into_float_value()
                    };
                    let normalised_turns = builder.build_float_sub(turns, floor_turns, "")?;
                    builder.build_float_mul(normalised_turns, angle_ty.const_float(2.0), "")?
                };
                args.outputs.finish(builder, [normalised_half_turns.into()])
            }
            op => bail!("Unsupported op: {op:?}"),
        }
    }
}

impl<PCG: PreludeCodegen> CodegenExtension for RotationCodegenExtension<PCG> {
    fn add_extension<'a, H: HugrView + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type(
                (ROTATION_EXTENSION_ID, ROTATION_CUSTOM_TYPE.name().clone()),
                |ts, _| Ok(llvm_angle_type(&ts).into()),
            )
            .custom_const::<ConstRotation>(|context, rotation| {
                let angle_ty = llvm_angle_type(&context.typing_session());
                Ok(angle_ty.const_float(rotation.half_turns()).into())
            })
            .simple_extension_op(move |context, args, op| self.emit_rotation_op(context, args, op))
    }
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    pub fn add_default_rotation_extensions(self) -> Self {
        self.add_extension(DEFAULT_ROTATION_EXTENSION.to_owned())
    }
}

#[cfg(test)]
mod test {

    use hugr::{
        builder::{Dataflow, DataflowSubContainer as _, SubContainer},
        extension::ExtensionSet,
        ops::{
            constant::{CustomConst, TryHash},
            OpName,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
    };
    use inkwell::values::BasicValueEnum;
    use rstest::rstest;
    use tket2::extension::rotation::{RotationOpBuilder as _, ROTATION_TYPE};

    use crate::utils::UnwrapBuilder;
    use crate::{
        check_emission,
        emit::test::SimpleHugrConfig,
        test::{exec_ctx, llvm_ctx, TestContext},
        types::HugrType,
    };

    use super::*;

    #[rstest]
    fn emit_all_ops(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![FLOAT64_TYPE])
            .with_extensions(tket2::extension::REGISTRY.to_owned())
            .finish_with_exts(|mut builder, reg| {
                let [a1] = builder.input_wires_arr();
                let rot1 = builder.add_from_halfturns_unchecked(a1).unwrap();
                let half_turns = builder.add_to_halfturns(rot1).unwrap();
                let [rot2] = {
                    let mb_rot = builder.add_from_halfturns(half_turns).unwrap();
                    builder
                        .build_unwrap_sum(reg, 1, option_type(ROTATION_TYPE), mb_rot)
                        .unwrap()
                };
                let _ = builder
                    .add_dataflow_op(RotationOp::radd, [rot1, rot2])
                    .unwrap();
                builder.finish_sub_container().unwrap()
            });
        llvm_ctx.add_extensions(|cge| {
            cge.add_default_rotation_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
        });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    #[case(ConstRotation::new(1.0).unwrap(), ConstRotation::new(0.5).unwrap(), 1.5)]
    #[case(ConstRotation::PI, ConstRotation::new(1.5).unwrap(), 0.5)]
    fn exec_aadd(
        mut exec_ctx: TestContext,
        #[case] angle1: ConstRotation,
        #[case] angle2: ConstRotation,
        #[case] expected_half_turns: f64,
    ) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(tket2::extension::REGISTRY.to_owned())
            .with_outs(FLOAT64_TYPE)
            .finish(|mut builder| {
                let rot2 = builder.add_load_value(angle1);
                let rot1 = builder.add_load_value(angle2);
                let rot = builder
                    .add_dataflow_op(RotationOp::radd, [rot1, rot2])
                    .unwrap()
                    .out_wire(0);
                let value = builder.add_to_halfturns(rot).unwrap();

                builder.finish_with_outputs([value]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_rotation_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
        });
        let half_turns = exec_ctx.exec_hugr_f64(hugr, "main");
        let epsilon = 0.0000000000001; // chosen without too much thought
        assert!(
            f64::abs(expected_half_turns - half_turns) < epsilon,
            "abs({expected_half_turns} - {half_turns}) >= {epsilon}"
        );
    }

    #[rstest]
    #[case(ConstRotation::PI, 1.0)]
    #[case(ConstRotation::TAU, 0.0)]
    #[case(ConstRotation::PI_2, 0.5)]
    #[case(ConstRotation::PI_4, 0.25)]
    fn exec_to_halfturns(
        mut exec_ctx: TestContext,
        #[case] angle: ConstRotation,
        #[case] expected_halfturns: f64,
    ) {
        let hugr = SimpleHugrConfig::new()
            .with_extensions(tket2::extension::REGISTRY.to_owned())
            .with_outs(FLOAT64_TYPE)
            .finish(|mut builder| {
                let rot = builder.add_load_value(angle);
                let halfturns = builder.add_to_halfturns(rot).unwrap();
                builder.finish_with_outputs([halfturns]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_rotation_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
        });

        let halfturns = exec_ctx.exec_hugr_f64(hugr, "main");
        let epsilon = 0.000000000001; // chosen without too much thought
        assert!(
            f64::abs(expected_halfturns - halfturns) < epsilon,
            "abs({expected_halfturns} - {halfturns}) >= {epsilon}"
        );
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct NonFiniteConst64(f64);

    #[typetag::serde]
    impl CustomConst for NonFiniteConst64 {
        fn name(&self) -> OpName {
            "NonFiniteConst64".into()
        }

        fn extension_reqs(&self) -> ExtensionSet {
            float_types::EXTENSION_ID.into()
        }

        fn get_type(&self) -> HugrType {
            FLOAT64_TYPE
        }
    }

    impl TryHash for NonFiniteConst64 {}

    fn add_nonfinite_const_extensions<'a, H: HugrView + 'a>(
        cem: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H> {
        fn emit_nonfinite_const<'c, H: HugrView>(
            context: &mut EmitFuncContext<'c, '_, H>,
            konst: &NonFiniteConst64,
        ) -> Result<BasicValueEnum<'c>> {
            Ok(context.iw_context().f64_type().const_float(konst.0).into())
        }
        cem.custom_const(emit_nonfinite_const)
    }

    impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
        fn add_nonfinite_const_extensions(self) -> Self {
            add_nonfinite_const_extensions(self)
        }
    }

    #[rstest]
    #[case(1.0, Some(1.0))]
    #[case(-1.0, Some(1.0))]
    #[case(0.5, Some(0.5))]
    #[case(-0.5, Some(1.5))]
    #[case(0.25, Some(0.25))]
    #[case(-0.25, Some(1.75))]
    #[case(13.5, Some(1.5))]
    #[case(-13.5, Some(0.5))]
    #[case(f64::NAN, None)]
    #[case(f64::INFINITY, None)]
    #[case(f64::NEG_INFINITY, None)]
    fn exec_from_halfturns(
        mut exec_ctx: TestContext,
        #[case] halfturns: f64,
        #[case] expected_halfturns: Option<f64>,
    ) {
        use hugr::{ops::Value, type_row};

        let hugr = SimpleHugrConfig::new()
            .with_extensions(tket2::extension::REGISTRY.to_owned())
            .with_outs(FLOAT64_TYPE)
            .finish(|mut builder| {
                let konst: Value = if halfturns.is_finite() {
                    ConstF64::new(halfturns).into()
                } else {
                    NonFiniteConst64(halfturns).into()
                };
                let halfturns = {
                    let halfturns = builder.add_load_value(konst);
                    let mb_rot = builder.add_from_halfturns(halfturns).unwrap();
                    let mut conditional = builder
                        .conditional_builder(
                            ([type_row![], type_row![ROTATION_TYPE]], mb_rot),
                            [],
                            type_row![FLOAT64_TYPE],
                        )
                        .unwrap();
                    {
                        let mut failure_case = conditional.case_builder(0).unwrap();
                        let neg_one = failure_case.add_load_value(ConstF64::new(-1.0));
                        failure_case.finish_with_outputs([neg_one]).unwrap();
                    }
                    {
                        let mut success_case = conditional.case_builder(1).unwrap();
                        let [rotation] = success_case.input_wires_arr();
                        let halfturns = success_case.add_to_halfturns(rotation).unwrap();
                        success_case.finish_with_outputs([halfturns]).unwrap();
                    }
                    conditional.finish_sub_container().unwrap().out_wire(0)
                };
                builder.finish_with_outputs([halfturns]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_rotation_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
                .add_nonfinite_const_extensions()
        });

        let r = exec_ctx.exec_hugr_f64(hugr, "main");
        // chosen without too much thought, except that a f64 has 53 bits of
        // precision so 1 << 11 is the lowest reasonable value.
        let epsilon = 0.0000000000001; // chosen without too much thought

        let expected_halfturns = expected_halfturns.unwrap_or(-1.0);
        assert!((expected_halfturns - r).abs() < epsilon);
    }
}

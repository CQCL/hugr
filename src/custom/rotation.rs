use anyhow::{anyhow, bail, Result};
use std::any::TypeId;

use hugr::{
    extension::{prelude::option_type, simple_op::MakeOpDef, ExtensionId},
    ops::{constant::CustomConst, ExtensionOp},
    types::CustomType,
    HugrView,
};
use inkwell::{
    types::{BasicTypeEnum, FloatType},
    values::BasicValueEnum,
    FloatPredicate,
};

use crate::{
    emit::{get_intrinsic, EmitFuncContext, EmitOpArgs},
    types::TypingSession,
};

use super::{CodegenExtension, CodegenExtsMap};

use tket2::extension::rotation::{
    ConstRotation, RotationOp, ROTATION_CUSTOM_TYPE, ROTATION_EXTENSION_ID, ROTATION_TYPE,
};

/// A codegen extension for the `tket2.rotation` extension.
///
/// We lower [ROTATION_CUSTOM_TYPE] to an `f64`, representing a number of half-turns.
pub struct RotationCodegenExtension;

fn llvm_angle_type<'c, H: HugrView>(ts: &TypingSession<'c, H>) -> FloatType<'c> {
    ts.iw_context().f64_type()
}

impl<H: HugrView> CodegenExtension<H> for RotationCodegenExtension {
    fn extension(&self) -> ExtensionId {
        ROTATION_EXTENSION_ID
    }

    fn llvm_type<'c>(
        &self,
        context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        if hugr_type == &ROTATION_CUSTOM_TYPE {
            Ok(llvm_angle_type(context).into())
        } else {
            bail!("Unsupported type: {hugr_type}")
        }
    }
    fn emit_extension_op<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let ts = context.typing_session();
        let module = context.get_current_module();
        let builder = context.builder();
        let angle_ty = llvm_angle_type(&ts);

        match RotationOp::from_op(&args.node())? {
            RotationOp::radd => {
                let [lhs, rhs] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::radd expects two arguments"))?;
                let (lhs, rhs) = (lhs.into_float_value(), rhs.into_float_value());
                let r = builder.build_float_add(lhs, rhs, "")?;
                args.outputs.finish(builder, [r.into()])
            }
            RotationOp::from_halfturns => {
                let [half_turns] = args
                    .inputs
                    .try_into()
                    .map_err(|_| anyhow!("RotationOp::from_halfturns expects one arguments"))?;
                let half_turns = half_turns.into_float_value();

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
                        builder.build_or(
                            builder.build_or(is_pos_inf, is_neg_inf, "")?,
                            is_nan,
                            "",
                        )?,
                        "",
                    )?
                };
                // let rads_ok = {
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

                let result_sum_type = ts.llvm_sum_type(option_type(ROTATION_TYPE))?;
                let rads_success =
                    result_sum_type.build_tag(builder, 1, vec![half_turns.into()])?;
                let rads_failure = result_sum_type.build_tag(builder, 0, vec![])?;
                let result = builder.build_select(half_turns_ok, rads_success, rads_failure, "")?;
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

    fn supported_consts(&self) -> std::collections::HashSet<std::any::TypeId> {
        let of = TypeId::of::<ConstRotation>();
        [of].into_iter().collect()
    }

    fn load_constant<'c>(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        let Some(rotation) = konst.downcast_ref::<ConstRotation>() else {
            return Ok(None);
        };
        let angle_type = llvm_angle_type(&context.typing_session());
        Ok(Some(angle_type.const_float(rotation.half_turns()).into()))
    }
}

pub fn add_rotation_extensions<H: HugrView>(cge: CodegenExtsMap<'_, H>) -> CodegenExtsMap<'_, H> {
    cge.add_cge(RotationCodegenExtension)
}

impl<'c, H: HugrView> CodegenExtsMap<'c, H> {
    pub fn add_rotation_extensions(self) -> Self {
        add_rotation_extensions(self)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use hugr::{
        builder::{Dataflow, DataflowSubContainer as _, SubContainer},
        extension::ExtensionSet,
        ops::OpName,
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
    };
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
            .with_ins(vec![ROTATION_TYPE])
            .with_extensions(tket2::extension::REGISTRY.to_owned())
            .finish_with_exts(|mut builder, reg| {
                let [rot1] = builder.input_wires_arr();
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
            cge.add_rotation_extensions()
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
            cge.add_rotation_extensions()
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
            cge.add_rotation_extensions()
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

    struct NonFiniteConst64CodegenExtension;

    impl<H: HugrView> CodegenExtension<H> for NonFiniteConst64CodegenExtension {
        fn extension(&self) -> ExtensionId {
            ExtensionId::new_unchecked("NonFiniteConst64")
        }

        fn llvm_type<'c>(
            &self,
            _: &TypingSession<'c, H>,
            _: &CustomType,
        ) -> Result<BasicTypeEnum<'c>> {
            panic!("no types")
        }

        fn emit_extension_op<'c>(
            &self,
            _: &mut EmitFuncContext<'c, H>,
            _: EmitOpArgs<'c, '_, ExtensionOp, H>,
        ) -> Result<()> {
            panic!("no ops")
        }

        fn supported_consts(&self) -> HashSet<TypeId> {
            let of = TypeId::of::<NonFiniteConst64>();
            [of].into_iter().collect()
        }

        fn load_constant<'c>(
            &self,
            context: &mut EmitFuncContext<'c, H>,
            konst: &dyn CustomConst,
        ) -> Result<Option<BasicValueEnum<'c>>> {
            let Some(NonFiniteConst64(f)) = konst.downcast_ref::<NonFiniteConst64>() else {
                panic!("load_constant")
            };
            Ok(Some(context.iw_context().f64_type().const_float(*f).into()))
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
            cge.add_rotation_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
                .add_cge(NonFiniteConst64CodegenExtension)
        });

        let r = exec_ctx.exec_hugr_f64(hugr, "main");
        // chosen without too much thought, except that a f64 has 53 bits of
        // precision so 1 << 11 is the lowest reasonable value.
        let epsilon = 0.0000000000001; // chosen without too much thought

        let expected_halfturns = expected_halfturns.unwrap_or(-1.0);
        assert!((expected_halfturns - r).abs() < epsilon);
    }
}

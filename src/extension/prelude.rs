use anyhow::{anyhow, Ok, Result};
use hugr::{
    extension::{
        prelude::{
            self, ArrayOp, ArrayOpDef, ConstError, ConstExternalSymbol, ConstString, ConstUsize,
            MakeTuple, TupleOpDef, UnpackTuple, ARRAY_TYPE_NAME, ERROR_CUSTOM_TYPE, ERROR_TYPE,
            STRING_CUSTOM_TYPE,
        },
        simple_op::MakeExtensionOp as _,
    },
    ops::constant::CustomConst,
    types::{SumType, TypeArg},
    HugrView,
};
use inkwell::{
    types::{BasicType, BasicTypeEnum, IntType},
    values::{BasicValue as _, BasicValueEnum, IntValue},
    AddressSpace,
};
use itertools::Itertools;

use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{
        func::EmitFuncContext,
        libc::{emit_libc_abort, emit_libc_printf},
        RowPromise,
    },
    sum::LLVMSumValue,
    types::TypingSession,
};

pub mod array;

/// A helper trait for implementing [CodegenExtension]s for
/// [hugr::extension::prelude].
///
/// All methods have sensible defaults provided, and [DefaultPreludeCodegen] is
/// trivial implementation of this trait, which delegates everything to those
/// default implementations.
///
/// TODO several types and ops are unimplemented. We expect to add methods to
/// this trait as necessary, allowing downstream users to customise the lowering
/// of `prelude`.
pub trait PreludeCodegen: Clone {
    /// Return the llvm type of [hugr::extension::prelude::USIZE_T]. That type
    /// must be an [IntType].
    fn usize_type<'c>(&self, session: &TypingSession<'c>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [hugr::extension::prelude::QB_T].
    fn qubit_type<'c>(&self, session: &TypingSession<'c>) -> impl BasicType<'c> {
        session.iw_context().i16_type()
    }

    /// Return the llvm type of [hugr::extension::prelude::array_type].
    fn array_type<'c>(
        &self,
        _session: &TypingSession<'c>,
        elem_ty: BasicTypeEnum<'c>,
        size: u64,
    ) -> impl BasicType<'c> {
        elem_ty.array_type(size as u32)
    }

    /// Emit a [hugr::extension::prelude::ArrayOp].
    fn emit_array_op<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, H>,
        op: ArrayOp,
        inputs: Vec<BasicValueEnum<'c>>,
        outputs: RowPromise<'c>,
    ) -> Result<()> {
        array::emit_array_op(self, ctx, op, inputs, outputs)
    }

    /// Emit a [hugr::extension::prelude::PRINT_OP_ID] node.
    fn emit_print<H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        text: BasicValueEnum,
    ) -> Result<()> {
        let format_str = ctx
            .builder()
            .build_global_string_ptr("%s\n", "prelude.print_template")?
            .as_basic_value_enum();
        emit_libc_printf(ctx, &[format_str.into(), text.into()])
    }

    /// Emit a [hugr::extension::prelude::PANIC_OP_ID] node.
    fn emit_panic<H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        signal: IntValue,
        msg: BasicValueEnum,
    ) -> Result<()> {
        let format_str = ctx
            .builder()
            .build_global_string_ptr(
                "Program panicked (signal %i): %s\n",
                "prelude.panic_template",
            )?
            .as_basic_value_enum();
        emit_libc_printf(ctx, &[format_str.into(), signal.into(), msg.into()])?;
        emit_libc_abort(ctx)
    }
}

/// A trivial implementation of [PreludeCodegen] which passes all methods
/// through to their default implementations.
#[derive(Default, Clone)]
pub struct DefaultPreludeCodegen;

impl PreludeCodegen for DefaultPreludeCodegen {}

#[derive(Clone, Debug, Default)]
pub struct PreludeCodegenExtension<PCG>(PCG);

impl<PCG: PreludeCodegen> PreludeCodegenExtension<PCG> {
    pub fn new(pcg: PCG) -> Self {
        Self(pcg)
    }
}

impl<PCG: PreludeCodegen> From<PCG> for PreludeCodegenExtension<PCG> {
    fn from(pcg: PCG) -> Self {
        Self::new(pcg)
    }
}

impl<PCG: PreludeCodegen> CodegenExtension for PreludeCodegenExtension<PCG> {
    fn add_extension<'a, H: HugrView + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        add_prelude_extensions(builder, self.0)
    }
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsBuilder] using `pcg`
    /// as the implementation.
    pub fn add_default_prelude_extensions(self) -> Self {
        self.add_prelude_extensions(DefaultPreludeCodegen)
    }

    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsBuilder] using
    /// [DefaultPreludeCodegen] as the implementation.
    pub fn add_prelude_extensions(self, pcg: impl PreludeCodegen + 'a) -> Self {
        self.add_extension(PreludeCodegenExtension::from(pcg))
    }
}

/// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using `pcg`
/// as the implementation.
fn add_prelude_extensions<'a, H: HugrView + 'a>(
    cem: CodegenExtsBuilder<'a, H>,
    pcg: impl PreludeCodegen + 'a,
) -> CodegenExtsBuilder<'a, H> {
    cem.custom_type((prelude::PRELUDE_ID, "qubit".into()), {
        let pcg = pcg.clone();
        move |ts, _| Ok(pcg.qubit_type(&ts).as_basic_type_enum())
    })
    .custom_type((prelude::PRELUDE_ID, "usize".into()), {
        let pcg = pcg.clone();
        move |ts, _| Ok(pcg.usize_type(&ts).as_basic_type_enum())
    })
    .custom_type((prelude::PRELUDE_ID, STRING_CUSTOM_TYPE.name().clone()), {
        move |ts, _| {
            // TODO allow customising string type
            Ok(ts
                .iw_context()
                .i8_type()
                .ptr_type(AddressSpace::default())
                .into())
        }
    })
    .custom_type((prelude::PRELUDE_ID, ERROR_CUSTOM_TYPE.name().clone()), {
        move |ts, _| {
            let ctx: &inkwell::context::Context = ts.iw_context();
            let signal_ty = ctx.i32_type().into();
            let message_ty = ctx.i8_type().ptr_type(AddressSpace::default()).into();
            Ok(ctx.struct_type(&[signal_ty, message_ty], false).into())
        }
    })
    .custom_type((prelude::PRELUDE_ID, ARRAY_TYPE_NAME.into()), {
        let pcg = pcg.clone();
        move |ts, hugr_type| {
            let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = hugr_type.args() else {
                return Err(anyhow!("Invalid type args for array type"));
            };
            let elem_ty = ts.llvm_type(ty)?;
            Ok(pcg.array_type(&ts, elem_ty, *n).as_basic_type_enum())
        }
    })
    .custom_const::<ConstUsize>(|context, k| {
        let ty: IntType = context
            .llvm_type(&k.get_type())?
            .try_into()
            .map_err(|_| anyhow!("Failed to get ConstUsize as IntType"))?;
        Ok(ty.const_int(k.value(), false).into())
    })
    .custom_const::<ConstExternalSymbol>(|context, k| {
        // TODO we should namespace these symbols
        // https://github.com/CQCL/hugr-llvm/issues/120
        let llvm_type = context.llvm_type(&k.get_type())?;
        let global = context.get_global(&k.symbol, llvm_type, k.constant)?;
        Ok(context
            .builder()
            .build_load(global.as_pointer_value(), &k.symbol)?)
    })
    .custom_const::<ConstString>(|context, k| {
        // TODO we should allow overriding the representation of strings
        let s = context.builder().build_global_string_ptr(k.value(), "")?;
        Ok(s.as_basic_value_enum())
    })
    .custom_const::<ConstError>(|context, k| {
        let builder = context.builder();
        let err_ty = context.llvm_type(&ERROR_TYPE)?.into_struct_type();
        let signal = err_ty
            .get_field_type_at_index(0)
            .unwrap()
            .into_int_type()
            .const_int(k.signal as u64, false);
        let message = builder
            .build_global_string_ptr(&k.message, "")?
            .as_basic_value_enum();
        let err = err_ty.const_named_struct(&[signal.into(), message]);
        Ok(err.into())
    })
    .simple_extension_op::<TupleOpDef>(|context, args, op| match op {
        TupleOpDef::UnpackTuple => {
            let unpack_tuple = UnpackTuple::from_extension_op(args.node().as_ref())?;
            let llvm_sum_type = context.llvm_sum_type(SumType::new([unpack_tuple.0]))?;
            let llvm_sum_value = args
                .inputs
                .into_iter()
                .exactly_one()
                .map_err(|_| anyhow!("UnpackTuple does not have exactly one input"))
                .and_then(|v| LLVMSumValue::try_new(v, llvm_sum_type))?;
            let rs = llvm_sum_value.build_untag(context.builder(), 0)?;
            args.outputs.finish(context.builder(), rs)
        }
        TupleOpDef::MakeTuple => {
            let make_tuple = MakeTuple::from_extension_op(args.node().as_ref())?;
            let llvm_sum_type = context.llvm_sum_type(SumType::new([make_tuple.0]))?;
            let r = llvm_sum_type.build_tag(context.builder(), 0, args.inputs)?;
            args.outputs.finish(context.builder(), [r])
        }
        _ => Err(anyhow!("Unsupported TupleOpDef")),
    })
    .simple_extension_op::<ArrayOpDef>({
        let pcg = pcg.clone();
        move |context, args, _| {
            pcg.emit_array_op(
                context,
                ArrayOp::from_extension_op(args.node().as_ref())?,
                args.inputs,
                args.outputs,
            )
        }
    })
    .extension_op(prelude::PRELUDE_ID, prelude::PRINT_OP_ID, {
        let pcg = pcg.clone();
        move |context, args| {
            let text = args.inputs[0];
            pcg.emit_print(context, text)?;
            args.outputs.finish(context.builder(), [])
        }
    })
    .extension_op(prelude::PRELUDE_ID, prelude::PANIC_OP_ID, {
        let pcg = pcg.clone();
        move |context, args| {
            let builder = context.builder();
            let err = args.inputs[0].into_struct_value();
            let signal = builder.build_extract_value(err, 0, "")?.into_int_value();
            let msg = builder.build_extract_value(err, 1, "")?;
            pcg.emit_panic(context, signal, msg)?;
            let returns = args
                .outputs
                .get_types()
                .map(|ty| ty.const_zero())
                .collect_vec();
            args.outputs.finish(context.builder(), returns)
        }
    })
}

#[cfg(test)]
mod test {
    use hugr::builder::{Dataflow, DataflowSubContainer};
    use hugr::extension::{PRELUDE, PRELUDE_REGISTRY};
    use hugr::types::{Type, TypeArg};
    use hugr::{type_row, Hugr};
    use prelude::{BOOL_T, PANIC_OP_ID, PRINT_OP_ID, QB_T, USIZE_T};
    use rstest::rstest;

    use crate::check_emission;
    use crate::custom::CodegenExtsBuilder;
    use crate::emit::test::SimpleHugrConfig;
    use crate::test::{llvm_ctx, TestContext};
    use crate::types::HugrType;

    use super::*;

    #[derive(Clone)]
    struct TestPreludeCodegen;
    impl PreludeCodegen for TestPreludeCodegen {
        fn usize_type<'c>(&self, session: &TypingSession<'c>) -> IntType<'c> {
            session.iw_context().i32_type()
        }

        fn qubit_type<'c>(&self, session: &TypingSession<'c>) -> impl BasicType<'c> {
            session.iw_context().f64_type()
        }
    }

    #[rstest]
    fn prelude_extension_types(llvm_ctx: TestContext) {
        let iw_context = llvm_ctx.iw_context();
        let type_converter = CodegenExtsBuilder::<Hugr>::default()
            .add_prelude_extensions(TestPreludeCodegen)
            .finish()
            .type_converter;
        let session = type_converter.session(iw_context);

        assert_eq!(
            iw_context.i32_type().as_basic_type_enum(),
            session.llvm_type(&USIZE_T).unwrap()
        );
        assert_eq!(
            iw_context.f64_type().as_basic_type_enum(),
            session.llvm_type(&QB_T).unwrap()
        );
    }

    #[rstest]
    fn prelude_extension_types_in_test_context(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(|x| x.add_prelude_extensions(TestPreludeCodegen));
        let tc = llvm_ctx.get_typing_session();
        assert_eq!(
            llvm_ctx.iw_context().i32_type().as_basic_type_enum(),
            tc.llvm_type(&USIZE_T).unwrap()
        );
        assert_eq!(
            llvm_ctx.iw_context().f64_type().as_basic_type_enum(),
            tc.llvm_type(&QB_T).unwrap()
        );
    }

    #[rstest::fixture]
    fn prelude_llvm_ctx(mut llvm_ctx: TestContext) -> TestContext {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        llvm_ctx
    }

    #[rstest]
    fn prelude_const_usize(prelude_llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(USIZE_T)
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(17));
                builder.finish_with_outputs([k]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_const_external_symbol(prelude_llvm_ctx: TestContext) {
        let konst1 = ConstExternalSymbol::new("sym1", USIZE_T, true);
        let konst2 = ConstExternalSymbol::new(
            "sym2",
            HugrType::new_sum([type_row![USIZE_T, HugrType::new_unit_sum(3)], type_row![]]),
            false,
        );

        let hugr = SimpleHugrConfig::new()
            .with_outs(vec![konst1.get_type(), konst2.get_type()])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k1 = builder.add_load_value(konst1);
                let k2 = builder.add_load_value(konst2);
                builder.finish_with_outputs([k1, k2]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_make_tuple(prelude_llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![BOOL_T, BOOL_T])
            .with_outs(Type::new_tuple(vec![BOOL_T, BOOL_T]))
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let in_wires = builder.input_wires();
                let r = builder.make_tuple(in_wires).unwrap();
                builder.finish_with_outputs([r]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_unpack_tuple(prelude_llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(Type::new_tuple(vec![BOOL_T, BOOL_T]))
            .with_outs(vec![BOOL_T, BOOL_T])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let unpack = builder
                    .add_dataflow_op(
                        UnpackTuple::new(vec![BOOL_T, BOOL_T].into()),
                        builder.input_wires(),
                    )
                    .unwrap();
                builder.finish_with_outputs(unpack.outputs()).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_panic(prelude_llvm_ctx: TestContext) {
        let error_val = ConstError::new(42, "PANIC");
        const TYPE_ARG_Q: TypeArg = TypeArg::Type { ty: QB_T };
        let type_arg_2q: TypeArg = TypeArg::Sequence {
            elems: vec![TYPE_ARG_Q, TYPE_ARG_Q],
        };
        let panic_op = PRELUDE
            .instantiate_extension_op(
                &PANIC_OP_ID,
                [type_arg_2q.clone(), type_arg_2q.clone()],
                &PRELUDE_REGISTRY,
            )
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![QB_T, QB_T])
            .with_outs(vec![QB_T, QB_T])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let [q0, q1] = builder.input_wires_arr();
                let err = builder.add_load_value(error_val);
                let [q0, q1] = builder
                    .add_dataflow_op(panic_op, [err, q0, q1])
                    .unwrap()
                    .outputs_arr();
                builder.finish_with_outputs([q0, q1]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_print(prelude_llvm_ctx: TestContext) {
        let greeting: ConstString = ConstString::new("Hello, world!".into());
        let print_op = PRELUDE
            .instantiate_extension_op(&PRINT_OP_ID, [], &PRELUDE_REGISTRY)
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let greeting_out = builder.add_load_value(greeting);
                builder.add_dataflow_op(print_op, [greeting_out]).unwrap();
                builder.finish_with_outputs([]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }
}

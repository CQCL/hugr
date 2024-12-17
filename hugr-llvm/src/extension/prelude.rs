use anyhow::{anyhow, bail, ensure, Ok, Result};
use hugr_core::extension::prelude::generic::LoadNat;
use hugr_core::extension::prelude::{
    self, error_type, generic, ConstError, ConstExternalSymbol, ConstString, ConstUsize, MakeTuple,
    TupleOpDef, UnpackTuple,
};
use hugr_core::extension::prelude::{ERROR_TYPE_NAME, STRING_TYPE_NAME};
use hugr_core::types::TypeArg;
use hugr_core::{
    extension::simple_op::MakeExtensionOp as _, ops::constant::CustomConst, types::SumType,
    HugrView,
};
use inkwell::{
    types::{BasicType, IntType, PointerType},
    values::{BasicValue as _, BasicValueEnum, StructValue},
    AddressSpace,
};
use itertools::Itertools;

use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{
        func::EmitFuncContext,
        libc::{emit_libc_abort, emit_libc_printf},
    },
    sum::LLVMSumValue,
    types::TypingSession,
};

/// A helper trait for customising the lowering [hugr_core::extension::prelude]
/// types, [CustomConst]s, and ops.
///
/// All methods have sensible defaults provided, and [DefaultPreludeCodegen] is
/// a trivial implementation of this trait which delegates everything to those
/// default implementations.
pub trait PreludeCodegen: Clone {
    /// Return the llvm type of [hugr_core::extension::prelude::usize_t]. That type
    /// must be an [IntType].
    fn usize_type<'c>(&self, session: &TypingSession<'c, '_>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [hugr_core::extension::prelude::qb_t].
    fn qubit_type<'c>(&self, session: &TypingSession<'c, '_>) -> impl BasicType<'c> {
        session.iw_context().i16_type()
    }

    /// Return the llvm type of [hugr_core::extension::prelude::error_type()].
    ///
    /// The returned type must always match the type of the returned value of
    /// [Self::emit_const_error], and the `err` argument of [Self::emit_panic].
    ///
    /// The default implementation is a struct type with an i32 field and an i8*
    /// field for the code and message.
    fn error_type<'c>(&self, session: &TypingSession<'c, '_>) -> Result<impl BasicType<'c>> {
        let ctx = session.iw_context();
        Ok(session.iw_context().struct_type(
            &[
                ctx.i32_type().into(),
                ctx.i8_type().ptr_type(AddressSpace::default()).into(),
            ],
            false,
        ))
    }

    /// Emit a [hugr_core::extension::prelude::PRINT_OP_ID] node.
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

    /// Emit instructions to materialise an LLVM value representing `err`.
    ///
    /// The type of the returned value must match [Self::error_type].
    ///
    /// The default implementation materialises an LLVM struct with the
    /// [ConstError::signal] and [ConstError::message] of `err`.
    fn emit_const_error<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        err: &ConstError,
    ) -> Result<BasicValueEnum<'c>> {
        let builder = ctx.builder();
        let err_ty = ctx.llvm_type(&error_type())?.into_struct_type();
        let signal = err_ty
            .get_field_type_at_index(0)
            .unwrap()
            .into_int_type()
            .const_int(err.signal as u64, false);
        let message = builder
            .build_global_string_ptr(&err.message, "")?
            .as_basic_value_enum();
        let err = err_ty.const_named_struct(&[signal.into(), message]);
        Ok(err.into())
    }

    /// Emit instructions to halt execution with the error `err`.
    ///
    /// The type of `err` must match that returned from [Self::error_type].
    ///
    /// The default implementation emits calls to libc's `printf` and `abort`.
    ///
    /// Note that implementations of `emit_panic` must not emit `unreachable`
    /// terminators, that, if appropriate, is the responsibility of the caller.
    fn emit_panic<H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        let format_str = ctx
            .builder()
            .build_global_string_ptr(
                "Program panicked (signal %i): %s\n",
                "prelude.panic_template",
            )?
            .as_basic_value_enum();
        let Some(err) = StructValue::try_from(err).ok() else {
            bail!("emit_panic: Expected err value to be a struct type")
        };
        ensure!(err.get_type().count_fields() == 2);
        let signal = ctx.builder().build_extract_value(err, 0, "")?;
        ensure!(signal.get_type() == ctx.iw_context().i32_type().as_basic_type_enum());
        let msg = ctx.builder().build_extract_value(err, 1, "")?;
        ensure!(PointerType::try_from(msg.get_type()).is_ok());
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
    .custom_type((prelude::PRELUDE_ID, STRING_TYPE_NAME.clone()), {
        move |ts, _| {
            // TODO allow customising string type
            Ok(ts
                .iw_context()
                .i8_type()
                .ptr_type(AddressSpace::default())
                .into())
        }
    })
    .custom_type((prelude::PRELUDE_ID, ERROR_TYPE_NAME.clone()), {
        let pcg = pcg.clone();
        move |ts, _| Ok(pcg.error_type(&ts)?.as_basic_type_enum())
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
    .custom_const::<ConstError>({
        let pcg = pcg.clone();
        move |context, k| {
            let err = pcg.emit_const_error(context, k)?;
            ensure!(
                err.get_type()
                    == pcg
                        .error_type(&context.typing_session())?
                        .as_basic_type_enum()
            );
            Ok(err)
        }
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
            let err = args.inputs[0];
            ensure!(
                err.get_type()
                    == pcg
                        .error_type(&context.typing_session())?
                        .as_basic_type_enum()
            );
            pcg.emit_panic(context, err)?;
            let returns = args
                .outputs
                .get_types()
                .map(|ty| ty.const_zero())
                .collect_vec();
            args.outputs.finish(context.builder(), returns)
        }
    })
    .extension_op(prelude::PRELUDE_ID, generic::LOAD_NAT_OP_ID, {
        let pcg = pcg.clone();
        move |context, args| {
            let load_nat = LoadNat::from_extension_op(args.node().as_ref())?;
            let v = match load_nat.get_nat() {
                TypeArg::BoundedNat { n } => pcg
                    .usize_type(&context.typing_session())
                    .const_int(n, false),
                arg => bail!("Unexpected type arg for LoadNat: {}", arg),
            };
            args.outputs.finish(context.builder(), vec![v.into()])
        }
    })
}

#[cfg(test)]
mod test {
    use hugr_core::builder::{Dataflow, DataflowSubContainer};
    use hugr_core::extension::PRELUDE;
    use hugr_core::types::{Type, TypeArg};
    use hugr_core::{type_row, Hugr};
    use prelude::{bool_t, qb_t, usize_t, PANIC_OP_ID, PRINT_OP_ID};
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
        fn usize_type<'c>(&self, session: &TypingSession<'c, '_>) -> IntType<'c> {
            session.iw_context().i32_type()
        }

        fn qubit_type<'c>(&self, session: &TypingSession<'c, '_>) -> impl BasicType<'c> {
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
            session.llvm_type(&usize_t()).unwrap()
        );
        assert_eq!(
            iw_context.f64_type().as_basic_type_enum(),
            session.llvm_type(&qb_t()).unwrap()
        );
    }

    #[rstest]
    fn prelude_extension_types_in_test_context(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(|x| x.add_prelude_extensions(TestPreludeCodegen));
        let tc = llvm_ctx.get_typing_session();
        assert_eq!(
            llvm_ctx.iw_context().i32_type().as_basic_type_enum(),
            tc.llvm_type(&usize_t()).unwrap()
        );
        assert_eq!(
            llvm_ctx.iw_context().f64_type().as_basic_type_enum(),
            tc.llvm_type(&qb_t()).unwrap()
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
            .with_outs(usize_t())
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(17));
                builder.finish_with_outputs([k]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_const_external_symbol(prelude_llvm_ctx: TestContext) {
        let konst1 = ConstExternalSymbol::new("sym1", usize_t(), true);
        let konst2 = ConstExternalSymbol::new(
            "sym2",
            HugrType::new_sum([
                vec![usize_t(), HugrType::new_unit_sum(3)].into(),
                type_row![],
            ]),
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
            .with_ins(vec![bool_t(), bool_t()])
            .with_outs(Type::new_tuple(vec![bool_t(), bool_t()]))
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
            .with_ins(Type::new_tuple(vec![bool_t(), bool_t()]))
            .with_outs(vec![bool_t(), bool_t()])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let unpack = builder
                    .add_dataflow_op(
                        UnpackTuple::new(vec![bool_t(), bool_t()].into()),
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
        let type_arg_q: TypeArg = TypeArg::Type { ty: qb_t() };
        let type_arg_2q: TypeArg = TypeArg::Sequence {
            elems: vec![type_arg_q.clone(), type_arg_q],
        };
        let panic_op = PRELUDE
            .instantiate_extension_op(&PANIC_OP_ID, [type_arg_2q.clone(), type_arg_2q.clone()])
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![qb_t(), qb_t()])
            .with_outs(vec![qb_t(), qb_t()])
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
        let print_op = PRELUDE.instantiate_extension_op(&PRINT_OP_ID, []).unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let greeting_out = builder.add_load_value(greeting);
                builder.add_dataflow_op(print_op, [greeting_out]).unwrap();
                builder.finish_with_outputs([]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_load_nat(prelude_llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let v = builder
                    .add_dataflow_op(LoadNat::new(TypeArg::BoundedNat { n: 42 }), vec![])
                    .unwrap()
                    .out_wire(0);
                builder.finish_with_outputs([v]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }
}

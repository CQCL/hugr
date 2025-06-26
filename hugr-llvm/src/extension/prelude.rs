use anyhow::{Ok, Result, anyhow, bail, ensure};
use hugr_core::Node;
use hugr_core::extension::prelude::generic::LoadNat;
use hugr_core::extension::prelude::{
    self, ConstError, ConstExternalSymbol, ConstString, ConstUsize, MakeTuple, TupleOpDef,
    UnpackTuple, error_type, generic,
};
use hugr_core::extension::prelude::{ERROR_TYPE_NAME, STRING_TYPE_NAME};
use hugr_core::ops::ExtensionOp;
use hugr_core::types::TypeArg;
use hugr_core::{
    HugrView, extension::simple_op::MakeExtensionOp as _, ops::constant::CustomConst,
    types::SumType,
};
use inkwell::{
    AddressSpace,
    types::{BasicType, IntType, PointerType},
    values::{BasicValue as _, BasicValueEnum, StructValue},
};
use itertools::Itertools;

use crate::emit::EmitOpArgs;
use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{
        func::EmitFuncContext,
        libc::{emit_libc_abort, emit_libc_printf},
    },
    sum::LLVMSumValue,
    types::TypingSession,
};

/// A helper trait for customising the lowering [`hugr_core::extension::prelude`]
/// types, [`CustomConst`]s, and ops.
///
/// All methods have sensible defaults provided, and [`DefaultPreludeCodegen`] is
/// a trivial implementation of this trait which delegates everything to those
/// default implementations.
pub trait PreludeCodegen: Clone {
    /// Return the llvm type of [`hugr_core::extension::prelude::usize_t`]. That type
    /// must be an [`IntType`].
    fn usize_type<'c>(&self, session: &TypingSession<'c, '_>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [`hugr_core::extension::prelude::qb_t`].
    fn qubit_type<'c>(&self, session: &TypingSession<'c, '_>) -> impl BasicType<'c> {
        session.iw_context().i16_type()
    }

    /// Return the llvm type of [`hugr_core::extension::prelude::error_type()`].
    ///
    /// The returned type must always match the type of the returned value of
    /// [`Self::emit_const_error`], and the `err` argument of [`Self::emit_panic`].
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

    /// Return the llvm type of [`hugr_core::extension::prelude::string_type()`].
    ///
    /// The returned type must always match the type of the returned value of
    /// [`Self::emit_const_string`], and the `text` argument of [`Self::emit_print`].
    ///
    /// The default implementation is i8*.
    fn string_type<'c>(&self, session: &TypingSession<'c, '_>) -> Result<impl BasicType<'c>> {
        Ok(session
            .iw_context()
            .i8_type()
            .ptr_type(AddressSpace::default()))
    }

    /// Emit a [`hugr_core::extension::prelude::PRINT_OP_ID`] node.
    fn emit_print<H: HugrView<Node = Node>>(
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
    /// The type of the returned value must match [`Self::error_type`].
    ///
    /// The default implementation materialises an LLVM struct with the
    /// [`ConstError::signal`] and [`ConstError::message`] of `err`.
    fn emit_const_error<'c, H: HugrView<Node = Node>>(
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
            .const_int(u64::from(err.signal), false);
        let message = builder
            .build_global_string_ptr(&err.message, "")?
            .as_basic_value_enum();
        let err = err_ty.const_named_struct(&[signal.into(), message]);
        Ok(err.into())
    }

    /// Emit instructions to construct an error value from a signal and message.
    ///
    /// The type of the returned value must match [`Self::error_type`].
    ///
    /// The default implementation constructs a struct with the given signal and message.
    fn emit_make_error<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        signal: BasicValueEnum<'c>,
        message: BasicValueEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        let builder = ctx.builder();

        // The usize signal is an i64 but error struct stores an i32.
        let i32_type = ctx.typing_session().iw_context().i32_type();
        let signal_int = signal.into_int_value();
        let signal_truncated = builder.build_int_truncate(signal_int, i32_type, "")?;

        // Construct the error struct as runtime value.
        let err_ty = ctx.llvm_type(&error_type())?.into_struct_type();
        let undef = err_ty.get_undef();
        let err_with_sig = builder
            .build_insert_value(undef, signal_truncated, 0, "")?
            .into_struct_value();
        let err_complete = builder
            .build_insert_value(err_with_sig, message, 1, "")?
            .into_struct_value();

        Ok(err_complete.into())
    }

    /// Emit instructions to halt execution with the error `err`.
    ///
    /// The type of `err` must match that returned from [`Self::error_type`].
    ///
    /// The default implementation emits calls to libc's `printf` and `abort`.
    ///
    /// Note that implementations of `emit_panic` must not emit `unreachable`
    /// terminators, that, if appropriate, is the responsibility of the caller.
    fn emit_panic<H: HugrView<Node = Node>>(
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

    /// Emit instructions to halt execution with the error `err`.
    ///
    /// The type of `err` must match that returned from [`Self::error_type`].
    ///
    /// The default implementation emits calls to libc's `printf` and `abort`,
    /// matching the default implementation of [`Self::emit_panic`].
    ///
    /// Note that implementations of `emit_panic` must not emit `unreachable`
    /// terminators, that, if appropriate, is the responsibility of the caller.
    fn emit_exit<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        self.emit_panic(ctx, err)
    }

    /// Emit instructions to materialise an LLVM value representing `str`.
    ///
    /// The type of the returned value must match [`Self::string_type`].
    ///
    /// The default implementation creates a global C string.
    fn emit_const_string<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        str: &ConstString,
    ) -> Result<BasicValueEnum<'c>> {
        let default_str_type = ctx
            .iw_context()
            .i8_type()
            .ptr_type(AddressSpace::default())
            .as_basic_type_enum();
        let str_type = ctx.llvm_type(&str.get_type())?.as_basic_type_enum();
        ensure!(
            str_type == default_str_type,
            "The default implementation of PreludeCodegen::string_type was overridden, but the default implementation of emit_const_string was not. String type is: {str_type}"
        );
        let s = ctx.builder().build_global_string_ptr(str.value(), "")?;
        Ok(s.as_basic_value_enum())
    }

    fn emit_barrier<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        // By default, treat barriers as no-ops.
        args.outputs.finish(ctx.builder(), args.inputs)
    }
}

/// A trivial implementation of [`PreludeCodegen`] which passes all methods
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
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        add_prelude_extensions(builder, self.0)
    }
}

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [`PreludeCodegenExtension`] to the given [`CodegenExtsBuilder`] using `pcg`
    /// as the implementation.
    #[must_use]
    pub fn add_default_prelude_extensions(self) -> Self {
        self.add_prelude_extensions(DefaultPreludeCodegen)
    }

    /// Add a [`PreludeCodegenExtension`] to the given [`CodegenExtsBuilder`] using
    /// [`DefaultPreludeCodegen`] as the implementation.
    pub fn add_prelude_extensions(self, pcg: impl PreludeCodegen + 'a) -> Self {
        self.add_extension(PreludeCodegenExtension::from(pcg))
    }
}

/// Add a [`PreludeCodegenExtension`] to the given [`CodegenExtsBuilder`] using `pcg`
/// as the implementation.
pub fn add_prelude_extensions<'a, H: HugrView<Node = Node> + 'a>(
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
    .custom_type((prelude::PRELUDE_ID, ERROR_TYPE_NAME.clone()), {
        let pcg = pcg.clone();
        move |ts, _| Ok(pcg.error_type(&ts)?.as_basic_type_enum())
    })
    .custom_type((prelude::PRELUDE_ID, STRING_TYPE_NAME.clone()), {
        let pcg = pcg.clone();
        move |ts, _| Ok(pcg.string_type(&ts)?.as_basic_type_enum())
    })
    .custom_const::<ConstUsize>(|context, k| {
        let ty: IntType = context
            .llvm_type(&k.get_type())?
            .try_into()
            .map_err(|()| anyhow!("Failed to get ConstUsize as IntType"))?;
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
    .custom_const::<ConstString>({
        let pcg = pcg.clone();
        move |context, k| {
            let err = pcg.emit_const_string(context, k)?;
            ensure!(
                err.get_type()
                    == pcg
                        .string_type(&context.typing_session())?
                        .as_basic_type_enum()
            );
            Ok(err)
        }
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
    .extension_op(prelude::PRELUDE_ID, prelude::NOOP_OP_ID, |context, args| {
        args.outputs.finish(context.builder(), args.inputs)
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
            args.outputs.finish(context.builder(), [r.into()])
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
    .extension_op(prelude::PRELUDE_ID, prelude::MAKE_ERROR_OP_ID, {
        let pcg = pcg.clone();
        move |context, args| {
            let signal = args.inputs[0];
            let message = args.inputs[1];
            ensure!(
                message.get_type()
                    == pcg
                        .string_type(&context.typing_session())?
                        .as_basic_type_enum(),
                signal.get_type() == pcg.usize_type(&context.typing_session()).into()
            );
            let err = pcg.emit_make_error(context, signal, message)?;
            args.outputs.finish(context.builder(), [err])
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
                .map(inkwell::types::BasicTypeEnum::const_zero)
                .collect_vec();
            args.outputs.finish(context.builder(), returns)
        }
    })
    .extension_op(prelude::PRELUDE_ID, prelude::EXIT_OP_ID, {
        // by default treat an exit like a panic
        let pcg = pcg.clone();
        move |context, args| {
            let err = args.inputs[0];
            ensure!(
                err.get_type()
                    == pcg
                        .error_type(&context.typing_session())?
                        .as_basic_type_enum()
            );
            pcg.emit_exit(context, err)?;
            let returns = args
                .outputs
                .get_types()
                .map(inkwell::types::BasicTypeEnum::const_zero)
                .collect_vec();
            args.outputs.finish(context.builder(), returns)
        }
    })
    .extension_op(prelude::PRELUDE_ID, generic::LOAD_NAT_OP_ID.clone(), {
        let pcg = pcg.clone();
        move |context, args| {
            let load_nat = LoadNat::from_extension_op(args.node().as_ref())?;
            let v = match load_nat.get_nat() {
                TypeArg::BoundedNat(n) => pcg
                    .usize_type(&context.typing_session())
                    .const_int(n, false),
                arg => bail!("Unexpected type arg for LoadNat: {}", arg),
            };
            args.outputs.finish(context.builder(), vec![v.into()])
        }
    })
    .extension_op(prelude::PRELUDE_ID, prelude::BARRIER_OP_ID, {
        let pcg = pcg.clone();
        move |context, args| pcg.emit_barrier(context, args)
    })
}

#[cfg(test)]
mod test {
    use hugr_core::builder::{Dataflow, DataflowHugr};
    use hugr_core::extension::PRELUDE;
    use hugr_core::extension::prelude::{EXIT_OP_ID, MAKE_ERROR_OP_ID, Noop};
    use hugr_core::types::{Term, Type};
    use hugr_core::{Hugr, type_row};
    use prelude::{PANIC_OP_ID, PRINT_OP_ID, bool_t, qb_t, usize_t};
    use rstest::{fixture, rstest};

    use crate::check_emission;
    use crate::custom::CodegenExtsBuilder;
    use crate::emit::test::SimpleHugrConfig;
    use crate::test::{TestContext, exec_ctx, llvm_ctx};
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
                builder.finish_hugr_with_outputs([k]).unwrap()
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
                builder.finish_hugr_with_outputs([k1, k2]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_noop(prelude_llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(usize_t())
            .with_outs(usize_t())
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let in_wires = builder.input_wires();
                let r = builder
                    .add_dataflow_op(Noop::new(usize_t()), in_wires)
                    .unwrap()
                    .outputs();
                builder.finish_hugr_with_outputs(r).unwrap()
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
                builder.finish_hugr_with_outputs([r]).unwrap()
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
                builder.finish_hugr_with_outputs(unpack.outputs()).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_panic(prelude_llvm_ctx: TestContext) {
        let error_val = ConstError::new(42, "PANIC");
        let type_arg_q: Term = qb_t().into();
        let type_arg_2q = Term::new_list([type_arg_q.clone(), type_arg_q]);
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
                builder.finish_hugr_with_outputs([q0, q1]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_exit(prelude_llvm_ctx: TestContext) {
        let error_val = ConstError::new(42, "EXIT");
        let type_arg_q: Term = qb_t().into();
        let type_arg_2q = Term::new_list([type_arg_q.clone(), type_arg_q]);
        let exit_op = PRELUDE
            .instantiate_extension_op(&EXIT_OP_ID, [type_arg_2q.clone(), type_arg_2q.clone()])
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![qb_t(), qb_t()])
            .with_outs(vec![qb_t(), qb_t()])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let [q0, q1] = builder.input_wires_arr();
                let err = builder.add_load_value(error_val);
                let [q0, q1] = builder
                    .add_dataflow_op(exit_op, [err, q0, q1])
                    .unwrap()
                    .outputs_arr();
                builder.finish_hugr_with_outputs([q0, q1]).unwrap()
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
                builder.finish_hugr_with_outputs([]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_make_error(prelude_llvm_ctx: TestContext) {
        let sig: ConstUsize = ConstUsize::new(100);
        let msg: ConstString = ConstString::new("Error!".into());

        let make_error_op = PRELUDE
            .instantiate_extension_op(&MAKE_ERROR_OP_ID, [])
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .with_outs(error_type())
            .finish(|mut builder| {
                let sig_out = builder.add_load_value(sig);
                let msg_out = builder.add_load_value(msg);
                let [err] = builder
                    .add_dataflow_op(make_error_op, [sig_out, msg_out])
                    .unwrap()
                    .outputs_arr();
                builder.finish_hugr_with_outputs([err]).unwrap()
            });

        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[rstest]
    fn prelude_make_error_and_panic(prelude_llvm_ctx: TestContext) {
        let sig: ConstUsize = ConstUsize::new(100);
        let msg: ConstString = ConstString::new("Error!".into());

        let make_error_op = PRELUDE
            .instantiate_extension_op(&MAKE_ERROR_OP_ID, [])
            .unwrap();

        let panic_op = PRELUDE
            .instantiate_extension_op(&PANIC_OP_ID, [Term::new_list([]), Term::new_list([])])
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let sig_out = builder.add_load_value(sig);
                let msg_out = builder.add_load_value(msg);
                let [err] = builder
                    .add_dataflow_op(make_error_op, [sig_out, msg_out])
                    .unwrap()
                    .outputs_arr();
                builder.add_dataflow_op(panic_op, [err]).unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
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
                    .add_dataflow_op(LoadNat::new(42u64.into()), vec![])
                    .unwrap()
                    .out_wire(0);
                builder.finish_hugr_with_outputs([v]).unwrap()
            });
        check_emission!(hugr, prelude_llvm_ctx);
    }

    #[fixture]
    fn barrier_hugr() -> Hugr {
        SimpleHugrConfig::new()
            .with_outs(vec![usize_t()])
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let i = builder.add_load_value(ConstUsize::new(42));
                let [w1, _w2] = builder.add_barrier([i, i]).unwrap().outputs_arr();
                builder.finish_hugr_with_outputs([w1]).unwrap()
            })
    }

    #[rstest]
    fn prelude_barrier(prelude_llvm_ctx: TestContext, barrier_hugr: Hugr) {
        check_emission!(barrier_hugr, prelude_llvm_ctx);
    }
    #[rstest]
    fn prelude_barrier_exec(mut exec_ctx: TestContext, barrier_hugr: Hugr) {
        exec_ctx.add_extensions(|cem| add_prelude_extensions(cem, TestPreludeCodegen));
        assert_eq!(exec_ctx.exec_hugr_u64(barrier_hugr, "main"), 42);
    }
}

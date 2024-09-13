use std::{any::TypeId, collections::HashSet};

use anyhow::{anyhow, Ok, Result};
use hugr::{
    extension::{
        prelude::{
            self, ArrayOp, ConstError, ConstExternalSymbol, ConstString, ConstUsize, MakeTuple,
            UnpackTuple, ERROR_CUSTOM_TYPE, ERROR_TYPE, PANIC_OP_ID, PRINT_OP_ID, QB_T,
            STRING_CUSTOM_TYPE, USIZE_T,
        },
        simple_op::MakeExtensionOp as _,
    },
    ops::{constant::CustomConst, ExtensionOp},
    types::{SumType, TypeArg, TypeEnum},
    HugrView,
};
use inkwell::{
    types::{BasicType, BasicTypeEnum, IntType},
    values::{BasicValue as _, BasicValueEnum, IntValue},
    AddressSpace,
};
use itertools::Itertools;

use crate::{
    emit::{
        func::EmitFuncContext,
        libc::{emit_libc_abort, emit_libc_printf},
        EmitOp, EmitOpArgs, RowPromise,
    },
    sum::LLVMSumValue,
    types::TypingSession,
};

use super::{CodegenExtension, CodegenExtsMap};

pub mod array;

/// A helper trait for implementing [CodegenExtension]s for
/// [hugr::extension::prelude].
///
/// All methods have sensible defaults provided, and [DefaultPreludeCodegen] is
/// trivial implementation of this trait, which delegates everything to those
/// default implementations.
///
/// One should use either [PreludeCodegenExtension::new], or
/// [CodegenExtsMap::add_prelude_extensions] to work with the
/// [CodegenExtension].
///
/// TODO several types and ops are unimplemented. We expect to add methods to
/// this trait as necessary, allowing downstream users to customise the lowering
/// of `prelude`.
pub trait PreludeCodegen: Clone {
    /// Return the llvm type of [hugr::extension::prelude::USIZE_T]. That type
    /// must be an [IntType].
    fn usize_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> IntType<'c> {
        session.iw_context().i64_type()
    }

    /// Return the llvm type of [hugr::extension::prelude::QB_T].
    fn qubit_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> impl BasicType<'c> {
        session.iw_context().i16_type()
    }

    /// Return the llvm type of [hugr::extension::prelude::array_type].
    fn array_type<'c, H: HugrView>(
        &self,
        _session: &TypingSession<'c, H>,
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

struct PreludeEmitter<'c, 'd, H: HugrView, PCG: PreludeCodegen>(
    &'d mut EmitFuncContext<'c, H>,
    PCG,
);

impl<'c, 'a, H: HugrView, PCG: PreludeCodegen> EmitOp<'c, ExtensionOp, H>
    for PreludeEmitter<'c, 'a, H, PCG>
{
    fn emit(&mut self, args: EmitOpArgs<'c, ExtensionOp, H>) -> Result<()> {
        let node = args.node();
        let name = node.def().name();
        if let Result::Ok(make_tuple) = MakeTuple::from_extension_op(&node) {
            let llvm_sum_type = self.0.llvm_sum_type(SumType::new([make_tuple.0]))?;
            let r = llvm_sum_type.build_tag(self.0.builder(), 0, args.inputs)?;
            args.outputs.finish(self.0.builder(), [r])
        } else if let Result::Ok(unpack_tuple) = UnpackTuple::from_extension_op(&node) {
            let llvm_sum_type = self.0.llvm_sum_type(SumType::new([unpack_tuple.0]))?;
            let llvm_sum_value = args
                .inputs
                .into_iter()
                .exactly_one()
                .map_err(|_| anyhow!("UnpackTuple does not have exactly one input"))
                .and_then(|v| LLVMSumValue::try_new(v, llvm_sum_type))?;
            let rs = llvm_sum_value.build_untag(self.0.builder(), 0)?;
            args.outputs.finish(self.0.builder(), rs)
        } else if let Result::Ok(op) = ArrayOp::from_extension_op(&node) {
            self.1.emit_array_op(self.0, op, args.inputs, args.outputs)
        } else if *name == PRINT_OP_ID {
            let text = args.inputs[0];
            self.1.emit_print(self.0, text)?;
            args.outputs.finish(self.0.builder(), [])
        } else if *name == PANIC_OP_ID {
            let builder = self.0.builder();
            let err = args.inputs[0].into_struct_value();
            let signal = builder.build_extract_value(err, 0, "")?.into_int_value();
            let msg = builder.build_extract_value(err, 1, "")?;
            self.1.emit_panic(self.0, signal, msg)?;
            let returns = args
                .outputs
                .get_types()
                .map(|ty| ty.const_zero())
                .collect_vec();
            args.outputs.finish(self.0.builder(), returns)
        } else {
            Err(anyhow!("PreludeEmitter: Unknown op: {}", name))
        }
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

impl<'c, H: HugrView, PCG: PreludeCodegen> CodegenExtension<'c, H>
    for PreludeCodegenExtension<PCG>
{
    fn extension(&self) -> hugr::extension::ExtensionId {
        prelude::PRELUDE_ID
    }

    fn llvm_type(
        &self,
        ts: &crate::types::TypingSession<'c, H>,
        hugr_type: &hugr::types::CustomType,
    ) -> anyhow::Result<BasicTypeEnum<'c>> {
        let TypeEnum::Extension(qubit_custom_type) = QB_T.as_type_enum().clone() else {
            panic!("Qubit is not a custom type: {QB_T:?}");
        };
        let TypeEnum::Extension(usize_custom_type) = USIZE_T.as_type_enum().clone() else {
            panic!("usize is not a custom type: {USIZE_T:?}");
        };
        if &qubit_custom_type == hugr_type {
            Ok(self.0.qubit_type(ts).as_basic_type_enum())
        } else if &usize_custom_type == hugr_type {
            Ok(self.0.usize_type(ts).as_basic_type_enum())
        } else if &STRING_CUSTOM_TYPE == hugr_type {
            Ok(ts
                .iw_context()
                .i8_type()
                .ptr_type(AddressSpace::default())
                .into())
        } else if &ERROR_CUSTOM_TYPE == hugr_type {
            let ctx: &inkwell::context::Context = ts.iw_context();
            let signal_ty = ctx.i32_type().into();
            let message_ty = ctx.i8_type().ptr_type(AddressSpace::default()).into();
            Ok(ctx.struct_type(&[signal_ty, message_ty], false).into())
        } else if hugr_type.name() == "array" {
            let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = hugr_type.args() else {
                return Err(anyhow!("Invalid type args for array type"));
            };
            let elem_ty = ts.llvm_type(ty)?;
            Ok(self.0.array_type(ts, elem_ty, *n).as_basic_type_enum())
        } else {
            Err(anyhow::anyhow!(
                "Type not supported by prelude extension: {hugr_type:?}"
            ))
        }
    }

    fn emitter<'a>(
        &'a self,
        ctx: &'a mut EmitFuncContext<'c, H>,
    ) -> Box<dyn EmitOp<'c, ExtensionOp, H> + 'a> {
        Box::new(PreludeEmitter(ctx, self.0.clone()))
    }

    fn supported_consts(&self) -> HashSet<TypeId> {
        [
            TypeId::of::<ConstUsize>(),
            TypeId::of::<ConstExternalSymbol>(),
            TypeId::of::<ConstError>(),
            TypeId::of::<ConstString>(),
        ]
        .into_iter()
        .collect()
    }

    fn load_constant(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        if let Some(k) = konst.downcast_ref::<ConstUsize>() {
            let ty: IntType<'c> = context
                .llvm_type(&k.get_type())?
                .try_into()
                .map_err(|_| anyhow!("Failed to get ConstUsize as IntType"))?;
            Ok(Some(ty.const_int(k.value(), false).into()))
        } else if let Some(k) = konst.downcast_ref::<ConstExternalSymbol>() {
            let llvm_type = context.llvm_type(&k.get_type())?;
            let global = context.get_global(&k.symbol, llvm_type, k.constant)?;
            Ok(Some(
                context
                    .builder()
                    .build_load(global.as_pointer_value(), &k.symbol)?,
            ))
        } else if let Some(k) = konst.downcast_ref::<ConstString>() {
            let s = context.builder().build_global_string_ptr(k.value(), "")?;
            Ok(Some(s.as_basic_value_enum()))
        } else if let Some(k) = konst.downcast_ref::<ConstError>() {
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
            Ok(Some(err.into()))
        } else {
            Ok(None)
        }
    }
}

/// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using `pcg`
/// as the implementation.
pub fn add_prelude_extensions<'c, H: HugrView>(
    cem: CodegenExtsMap<'c, H>,
    pcg: impl PreludeCodegen + 'c,
) -> CodegenExtsMap<'c, H> {
    cem.add_cge(PreludeCodegenExtension(pcg))
}

/// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using
/// [DefaultPreludeCodegen] as the implementation.
pub fn add_default_prelude_extensions<H: HugrView>(cem: CodegenExtsMap<H>) -> CodegenExtsMap<H> {
    cem.add_cge(PreludeCodegenExtension::from(DefaultPreludeCodegen))
}

impl<'c, H: HugrView> CodegenExtsMap<'c, H> {
    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using `pcg`
    /// as the implementation.
    pub fn add_default_prelude_extensions(self) -> Self {
        add_default_prelude_extensions(self)
    }

    /// Add a [PreludeCodegenExtension] to the given [CodegenExtsMap] using
    /// [DefaultPreludeCodegen] as the implementation.
    pub fn add_prelude_extensions(self, builder: impl PreludeCodegen + 'c) -> Self {
        add_prelude_extensions(self, builder)
    }
}

#[cfg(test)]
mod test {
    use hugr::builder::{Dataflow, DataflowSubContainer};
    use hugr::extension::{PRELUDE, PRELUDE_REGISTRY};
    use hugr::type_row;
    use hugr::types::{Type, TypeArg};
    use prelude::BOOL_T;
    use rstest::rstest;

    use crate::check_emission;
    use crate::emit::test::SimpleHugrConfig;
    use crate::test::{llvm_ctx, TestContext};
    use crate::types::HugrType;

    use super::*;

    #[derive(Clone)]
    struct TestPreludeCodegen;
    impl PreludeCodegen for TestPreludeCodegen {
        fn usize_type<'c, H: HugrView>(&self, session: &TypingSession<'c, H>) -> IntType<'c> {
            session.iw_context().i32_type()
        }

        fn qubit_type<'c, H: HugrView>(
            &self,
            session: &TypingSession<'c, H>,
        ) -> impl BasicType<'c> {
            session.iw_context().f64_type()
        }
    }

    #[rstest]
    fn prelude_extension_types(llvm_ctx: TestContext) {
        let ctx = llvm_ctx.iw_context();
        let ext: PreludeCodegenExtension<TestPreludeCodegen> = TestPreludeCodegen.into();
        let tc = llvm_ctx.get_typing_session();

        let TypeEnum::Extension(qb_ct) = QB_T.as_type_enum().clone() else {
            unreachable!()
        };
        let TypeEnum::Extension(usize_ct) = USIZE_T.as_type_enum().clone() else {
            unreachable!()
        };

        assert_eq!(
            ctx.i32_type().as_basic_type_enum(),
            ext.llvm_type(&tc, &usize_ct).unwrap()
        );
        assert_eq!(
            ctx.f64_type().as_basic_type_enum(),
            ext.llvm_type(&tc, &qb_ct).unwrap()
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

    #[rstest]
    fn prelude_const_usize(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_default_prelude_extensions);

        let hugr = SimpleHugrConfig::new()
            .with_outs(USIZE_T)
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let k = builder.add_load_value(ConstUsize::new(17));
                builder.finish_with_outputs([k]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_const_external_symbol(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(add_default_prelude_extensions);
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
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_make_tuple(mut llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![BOOL_T, BOOL_T])
            .with_outs(Type::new_tuple(vec![BOOL_T, BOOL_T]))
            .with_extensions(prelude::PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder| {
                let in_wires = builder.input_wires();
                let r = builder.make_tuple(in_wires).unwrap();
                builder.finish_with_outputs([r]).unwrap()
            });
        llvm_ctx.add_extensions(add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_unpack_tuple(mut llvm_ctx: TestContext) {
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
        llvm_ctx.add_extensions(add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_panic(mut llvm_ctx: TestContext) {
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

        llvm_ctx.add_extensions(add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn prelude_print(mut llvm_ctx: TestContext) {
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

        llvm_ctx.add_extensions(add_default_prelude_extensions);
        check_emission!(hugr, llvm_ctx);
    }
}

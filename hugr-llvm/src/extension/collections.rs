use anyhow::{anyhow, Ok, Result};
use hugr_core::core::Either;
use hugr_core::core::Either::{Left, Right};
use hugr_core::{
    extension::prelude::{either_type, option_type},
    ops::{constant::CustomConst, ExtensionOp, NamedOp},
    std_extensions::collections::{self, ListOp, ListValue},
    types::{SumType, Type, TypeArg},
    HugrView,
};
use inkwell::{
    types::{BasicType, BasicTypeEnum, FunctionType},
    values::{BasicValueEnum, IntValue, PointerValue},
    AddressSpace,
};
use itertools::Itertools;

use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{emit_value, func::EmitFuncContext, EmitOpArgs},
    types::{HugrType, TypingSession},
};

/// A helper trait for customising the lowering [hugr_core::std_extensions::collections]
/// types, [CustomConst]s, and ops.
///
/// All methods have defaults provided that call out to runtime functions, and
/// [DefaultCollectionsCodegen] is a trivial implementation of this trait which
/// delegates everything to those default implementations.
pub trait CollectionsCodegen: Clone {
    /// Return the llvm type of [hugr_core::std_extensions::collections::LIST_TYPENAME].
    fn list_type<'c>(&self, session: TypingSession<'c, '_>) -> BasicTypeEnum<'c> {
        session
            .iw_context()
            .i8_type()
            .ptr_type(AddressSpace::default())
            .into()
    }

    /// Helper function to compute the signature of runtime functions.
    fn rt_func_sig<'c>(
        &self,
        ts: TypingSession<'c, '_>,
        fallible: bool,
        indexed: bool,
        inout: bool,
    ) -> FunctionType<'c> {
        let iwc = ts.iw_context();
        let mut args = vec![self.list_type(ts).into()];
        if indexed {
            args.push(iwc.i64_type().into());
        }
        if inout {
            args.push(iwc.i8_type().ptr_type(AddressSpace::default()).into());
        }
        if fallible {
            iwc.bool_type().fn_type(&args, false)
        } else {
            iwc.void_type().fn_type(&args, false)
        }
    }

    /// Emits a call to the runtime function that allocates a new list.
    fn rt_list_new<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        capacity: IntValue<'c>,
        elem_size: IntValue<'c>,
        alignment: IntValue<'c>,
        destructor: PointerValue<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        let session = ctx.typing_session();
        let iwc = session.iw_context();
        let func_ty = self.list_type(session).fn_type(
            &[
                iwc.i64_type().into(), // Capacity
                iwc.i64_type().into(), // Single element size in bytes
                iwc.i64_type().into(), // Element alignment
                // Pointer to element destructor
                iwc.i8_type().ptr_type(AddressSpace::default()).into(),
            ],
            false,
        );
        let func = ctx.get_extern_func("__rt__list__new", func_ty)?;
        let list = ctx
            .builder()
            .build_call(
                func,
                &[
                    capacity.into(),
                    elem_size.into(),
                    alignment.into(),
                    destructor.into(),
                ],
                "",
            )?
            .try_as_basic_value()
            .unwrap_left();
        Ok(list)
    }

    /// Emits a call to the runtime function that pushes to a list.
    fn rt_list_push<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        elem_ptr: PointerValue<'c>,
    ) -> Result<()> {
        let sig = self.rt_func_sig(ctx.typing_session(), false, false, true);
        let func = ctx.get_extern_func("__rt__list__push", sig)?;
        ctx.builder()
            .build_call(func, &[list.into(), elem_ptr.into()], "")?;
        Ok(())
    }

    /// Emits a call to the runtime function that pops from a list.
    fn rt_list_pop<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        out_ptr: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        let sig = self.rt_func_sig(ctx.typing_session(), true, false, true);
        let func = ctx.get_extern_func("__rt__list__pop", sig)?;
        let res = ctx
            .builder()
            .build_call(func, &[list.into(), out_ptr.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        Ok(res)
    }

    /// Emits a call to the runtime function that retrives an element from a list.
    fn rt_list_get<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        idx: IntValue<'c>,
        out_ptr: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        let sig = self.rt_func_sig(ctx.typing_session(), true, true, true);
        let func = ctx.get_extern_func("__rt__list__get", sig)?;
        let res = ctx
            .builder()
            .build_call(func, &[list.into(), idx.into(), out_ptr.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        Ok(res)
    }

    /// Emits a call to the runtime function that updates an element in a list.
    fn rt_list_set<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        idx: IntValue<'c>,
        elem_ptr: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        let sig = self.rt_func_sig(ctx.typing_session(), true, true, true);
        let func = ctx.get_extern_func("__rt__list__set", sig)?;
        let res = ctx
            .builder()
            .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        Ok(res)
    }

    /// Emits a call to the runtime function that inserts an element into a list.
    fn rt_list_insert<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        idx: IntValue<'c>,
        elem_ptr: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        let sig = self.rt_func_sig(ctx.typing_session(), true, true, true);
        let func = ctx.get_extern_func("__rt__list__insert", sig)?;
        let res = ctx
            .builder()
            .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        Ok(res)
    }

    /// Emits a call to the runtime function that computes the length of a list.
    fn rt_list_length<'c, H: HugrView>(
        &self,
        ctx: &EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
    ) -> Result<IntValue<'c>> {
        let session = ctx.typing_session();
        let func_ty = session
            .iw_context()
            .i64_type()
            .fn_type(&[self.list_type(session).into()], false);
        let func = ctx.get_extern_func("__rt__list__length", func_ty)?;
        let res = ctx
            .builder()
            .build_call(func, &[list.into()], "")?
            .try_as_basic_value()
            .unwrap_left()
            .into_int_value();
        Ok(res)
    }

    /// Emit instructions to materialise an LLVM value representing
    /// a [ListValue].
    fn emit_const_list<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        elems: Vec<BasicValueEnum<'c>>,
        elem_ty: BasicTypeEnum<'c>,
    ) -> Result<BasicValueEnum<'c>> {
        let iwc = ctx.typing_session().iw_context();
        let capacity = iwc.i64_type().const_int(elems.len() as u64, false);
        let elem_size = elem_ty.size_of().unwrap();
        let alignment = iwc.i64_type().const_int(8, false);
        // TODO: Lookup destructor for elem_ty
        let destructor = iwc.i8_type().ptr_type(AddressSpace::default()).const_null();
        let list = self.rt_list_new(ctx, capacity, elem_size, alignment, destructor)?;
        for elem in elems {
            self.emit_push(ctx, list, elem)?;
        }
        Ok(list)
    }

    /// Emit a [ListOp::push] node.
    fn emit_push<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        elem: BasicValueEnum<'c>,
    ) -> Result<()> {
        let elem_ptr = build_alloca_i8_ptr(ctx, Right(elem))?;
        self.rt_list_push(ctx, list, elem_ptr)
    }

    /// Emit a [ListOp::pop] node.
    fn emit_pop<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        hugr_elem_ty: HugrType,
    ) -> Result<BasicValueEnum<'c>> {
        let elem_ty = ctx.llvm_type(&hugr_elem_ty)?;
        let out_ptr = build_alloca_i8_ptr(ctx, Left(elem_ty))?;
        let pop_ok = self.rt_list_pop(ctx, list, out_ptr)?;
        let elem = build_load_i8_ptr(ctx, out_ptr, elem_ty)?;
        build_option(ctx, pop_ok, elem, hugr_elem_ty)
    }

    /// Emit a [ListOp::get] node.
    fn emit_get<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        index: IntValue<'c>,
        hugr_elem_ty: HugrType,
    ) -> Result<BasicValueEnum<'c>> {
        let elem_ty = ctx.llvm_type(&hugr_elem_ty)?;
        let out_ptr = build_alloca_i8_ptr(ctx, Left(elem_ty))?;
        let get_ok = self.rt_list_get(ctx, list, index, out_ptr)?;
        let elem = build_load_i8_ptr(ctx, out_ptr, elem_ty)?;
        build_option(ctx, get_ok, elem, hugr_elem_ty)
    }

    /// Emit a [ListOp::set] node.
    fn emit_set<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        index: IntValue<'c>,
        elem: BasicValueEnum<'c>,
        hugr_elem_ty: HugrType,
    ) -> Result<BasicValueEnum<'c>> {
        let elem_ptr = build_alloca_i8_ptr(ctx, Right(elem))?;
        let set_ok = self.rt_list_set(ctx, list, index, elem_ptr)?;
        let old_elem = build_load_i8_ptr(ctx, elem_ptr, elem.get_type())?;
        build_ok_or_else(
            ctx,
            set_ok,
            elem,
            hugr_elem_ty.clone(),
            old_elem,
            hugr_elem_ty,
        )
    }

    /// Emit a [ListOp::insert] node.
    fn emit_insert<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
        index: IntValue<'c>,
        elem: BasicValueEnum<'c>,
        hugr_elem_ty: HugrType,
    ) -> Result<BasicValueEnum<'c>> {
        let elem_ptr = build_alloca_i8_ptr(ctx, Right(elem))?;
        let insert_ok = self.rt_list_insert(ctx, list, index, elem_ptr)?;
        let unit = ctx
            .llvm_sum_type(SumType::new_unary(1))?
            .build_tag(ctx.builder(), 0, vec![])?;
        build_ok_or_else(ctx, insert_ok, unit, Type::UNIT, elem, hugr_elem_ty)
    }

    /// Emit a [ListOp::length] node.
    fn emit_length<'c, H: HugrView>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        list: BasicValueEnum<'c>,
    ) -> Result<IntValue<'c>> {
        Ok(self.rt_list_length(ctx, list)?)
    }
}

fn build_option<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    is_some: IntValue<'c>,
    some_value: BasicValueEnum<'c>,
    hugr_ty: HugrType,
) -> Result<BasicValueEnum<'c>> {
    let option_ty = ctx.llvm_sum_type(option_type(hugr_ty))?;
    let builder = ctx.builder();
    let some = option_ty.build_tag(builder, 1, vec![some_value])?;
    let none = option_ty.build_tag(builder, 0, vec![])?;
    let option = builder.build_select(is_some, some, none, "")?;
    Ok(option)
}

fn build_ok_or_else<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    is_ok: IntValue<'c>,
    ok_value: BasicValueEnum<'c>,
    ok_hugr_ty: HugrType,
    else_value: BasicValueEnum<'c>,
    else_hugr_ty: HugrType,
) -> Result<BasicValueEnum<'c>> {
    let either_ty = ctx.llvm_sum_type(either_type(else_hugr_ty, ok_hugr_ty))?;
    let builder = ctx.builder();
    let left = either_ty.build_tag(builder, 0, vec![else_value])?;
    let right = either_ty.build_tag(builder, 1, vec![ok_value])?;
    let either = builder.build_select(is_ok, right, left, "")?;
    Ok(either)
}

/// Helper function to allocate space on the stack for a given type.
///
/// Returns an i8 pointer to the allocated memory.
fn build_alloca_i8_ptr<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ty_or_val: Either<BasicTypeEnum<'c>, BasicValueEnum<'c>>,
) -> Result<PointerValue<'c>> {
    let builder = ctx.builder();
    let ty = match ty_or_val {
        Left(ty) => ty,
        Right(val) => val.get_type(),
    };
    let ptr = builder.build_alloca(ty, "")?;

    if let Right(val) = ty_or_val {
        builder.build_store(ptr, val)?;
    }
    let i8_ptr = builder.build_pointer_cast(
        ptr,
        ctx.iw_context().i8_type().ptr_type(AddressSpace::default()),
        "",
    )?;
    Ok(i8_ptr)
}

/// Helper function to load a value from an i8 pointer.
fn build_load_i8_ptr<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    i8_ptr: PointerValue<'c>,
    ty: BasicTypeEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let builder = ctx.builder();
    let ptr = builder.build_pointer_cast(i8_ptr, ty.ptr_type(AddressSpace::default()), "")?;
    let val = builder.build_load(ptr, "")?;
    Ok(val)
}

/// A trivial implementation of [CollectionsCodegen] which passes all methods
/// through to their default implementations.
#[derive(Default, Clone)]
pub struct DefaultCollectionsCodegen;

impl CollectionsCodegen for DefaultCollectionsCodegen {}

#[derive(Clone, Debug, Default)]
pub struct CollectionsCodegenExtension<CCG>(CCG);

impl<CCG: CollectionsCodegen> CollectionsCodegenExtension<CCG> {
    pub fn new(ccg: CCG) -> Self {
        Self(ccg)
    }
}

impl<CCG: CollectionsCodegen> From<CCG> for CollectionsCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}

impl<CCG: CollectionsCodegen> CodegenExtension for CollectionsCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        add_collections_extensions(builder, self.0)
    }
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [CollectionsCodegenExtension] to the given [CodegenExtsBuilder] using `ccg`
    /// as the implementation.
    pub fn add_default_collections_extensions(self) -> Self {
        self.add_collections_extensions(DefaultCollectionsCodegen)
    }

    /// Add a [CollectionsCodegenExtension] to the given [CodegenExtsBuilder] using
    /// [DefaultCollectionsCodegen] as the implementation.
    pub fn add_collections_extensions(self, ccg: impl CollectionsCodegen + 'a) -> Self {
        self.add_extension(CollectionsCodegenExtension::from(ccg))
    }
}

fn emit_list_op<'c, H: HugrView>(
    context: &mut EmitFuncContext<'c, '_, H>,
    ccg: &(impl CollectionsCodegen + 'c),
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: ListOp,
) -> Result<()> {
    let hugr_elem_ty = match args.node().args() {
        [TypeArg::Type { ty }] => ty.clone(),
        _ => {
            return Err(anyhow!("Collections: invalid type args for list op"));
        }
    };
    match op {
        ListOp::push => {
            let [list, elem] = args.inputs.try_into().unwrap();
            ccg.emit_push(context, list, elem)?;
            args.outputs.finish(context.builder(), vec![list])?;
        }
        ListOp::pop => {
            let [list] = args.inputs.try_into().unwrap();
            let elem_opt = ccg.emit_pop(context, list, hugr_elem_ty)?;
            args.outputs
                .finish(context.builder(), vec![list, elem_opt])?;
        }
        ListOp::get => {
            let [list, idx] = args.inputs.try_into().unwrap();
            let elem_opt = ccg.emit_get(context, list, idx.into_int_value(), hugr_elem_ty)?;
            args.outputs.finish(context.builder(), vec![elem_opt])?;
        }
        ListOp::set => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let ok = ccg.emit_set(context, list, idx.into_int_value(), elem, hugr_elem_ty)?;
            args.outputs.finish(context.builder(), vec![list, ok])?;
        }
        ListOp::insert => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let ok = ccg.emit_insert(context, list, idx.into_int_value(), elem, hugr_elem_ty)?;
            args.outputs.finish(context.builder(), vec![list, ok])?;
        }
        ListOp::length => {
            let [list] = args.inputs.try_into().unwrap();
            let length = ccg.emit_length(context, list)?;
            args.outputs
                .finish(context.builder(), vec![list, length.into()])?;
        }
        _ => return Err(anyhow!("Collections: unimplemented op: {}", op.name())),
    }
    Ok(())
}

/// Add a [CollectionsCodegenExtension] to the given [CodegenExtsBuilder] using `ccg`
/// as the implementation.
pub fn add_collections_extensions<'a, H: HugrView + 'a>(
    cem: CodegenExtsBuilder<'a, H>,
    ccg: impl CollectionsCodegen + 'a,
) -> CodegenExtsBuilder<'a, H> {
    cem.custom_type((collections::EXTENSION_ID, collections::LIST_TYPENAME), {
        let ccg = ccg.clone();
        move |ts, _hugr_type| Ok(ccg.list_type(ts).as_basic_type_enum())
    })
    .custom_const::<ListValue>({
        let ccg = ccg.clone();
        move |ctx, k| {
            let ty = ctx.llvm_type(&k.get_type())?;
            let elems = k
                .get_contents()
                .iter()
                .map(|v| emit_value(ctx, v))
                .try_collect()?;
            ccg.emit_const_list(ctx, elems, ty)
        }
    })
    .simple_extension_op::<ListOp>(move |ctx, args, op| emit_list_op(ctx, &ccg, args, op))
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{Dataflow, DataflowSubContainer},
        extension::{
            prelude::{self, ConstUsize, QB_T, USIZE_T},
            ExtensionRegistry,
        },
        ops::{DataflowOpTrait, NamedOp, Value},
        std_extensions::collections::{self, list_type, ListOp, ListValue},
    };
    use rstest::rstest;

    use crate::{
        check_emission,
        custom::CodegenExtsBuilder,
        emit::test::SimpleHugrConfig,
        test::{llvm_ctx, TestContext},
    };

    #[rstest]
    #[case::push(ListOp::push)]
    #[case::pop(ListOp::pop)]
    #[case::get(ListOp::get)]
    #[case::set(ListOp::set)]
    #[case::insert(ListOp::insert)]
    #[case::length(ListOp::length)]
    fn test_collections_emission(mut llvm_ctx: TestContext, #[case] op: ListOp) {
        let ext_op = collections::EXTENSION
            .instantiate_extension_op(
                op.name().as_ref(),
                [QB_T.into()],
                &collections::COLLECTIONS_REGISTRY,
            )
            .unwrap();
        let es = ExtensionRegistry::try_new([
            collections::EXTENSION.to_owned(),
            prelude::PRELUDE.to_owned(),
        ])
        .unwrap();
        let hugr = SimpleHugrConfig::new()
            .with_ins(ext_op.signature().input().clone())
            .with_outs(ext_op.signature().output().clone())
            .with_extensions(es)
            .finish(|mut hugr_builder| {
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, hugr_builder.input_wires())
                    .unwrap()
                    .outputs();
                hugr_builder.finish_with_outputs(outputs).unwrap()
            });
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_collections_extensions);
        check_emission!(op.name().as_str(), hugr, llvm_ctx);
    }

    #[rstest]
    fn test_const_list_emmission(mut llvm_ctx: TestContext) {
        let elem_ty = USIZE_T;
        let contents = (1..4).map(|i| Value::extension(ConstUsize::new(i)));
        let es = ExtensionRegistry::try_new([
            collections::EXTENSION.to_owned(),
            prelude::PRELUDE.to_owned(),
        ])
        .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_ins(vec![])
            .with_outs(vec![list_type(elem_ty.clone())])
            .with_extensions(es)
            .finish(|mut hugr_builder| {
                let list = hugr_builder.add_load_value(ListValue::new(elem_ty, contents));
                hugr_builder.finish_with_outputs(vec![list]).unwrap()
            });

        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_collections_extensions);
        check_emission!("const", hugr, llvm_ctx);
    }
}

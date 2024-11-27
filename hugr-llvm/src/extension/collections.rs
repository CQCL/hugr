use anyhow::{anyhow, bail, Ok, Result};
use hugr_core::{
    ops::{ExtensionOp, NamedOp},
    std_extensions::collections::{self, ListOp, ListValue},
    types::{SumType, Type, TypeArg},
    HugrView,
};
use inkwell::values::FunctionValue;
use inkwell::{
    types::{BasicType, BasicTypeEnum, FunctionType},
    values::{BasicValueEnum, PointerValue},
    AddressSpace,
};

use crate::emit::func::{build_ok_or_else, build_option};
use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{emit_value, func::EmitFuncContext, EmitOpArgs},
    types::TypingSession,
};

/// Runtime functions that implement operations on lists.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum CollectionsRtFunc {
    New,
    Push,
    Pop,
    Get,
    Set,
    Insert,
    Length,
}

impl CollectionsRtFunc {
    /// The signature of a given [CollectionsRtFunc].
    ///
    /// Requires a [CollectionsCodegen] to determine the type of lists.
    pub fn signature<'c>(
        self,
        ts: TypingSession<'c, '_>,
        ccg: &(impl CollectionsCodegen + 'c),
    ) -> FunctionType<'c> {
        let iwc = ts.iw_context();
        match self {
            CollectionsRtFunc::New => ccg.list_type(ts).fn_type(
                &[
                    iwc.i64_type().into(), // Capacity
                    iwc.i64_type().into(), // Single element size in bytes
                    iwc.i64_type().into(), // Element alignment
                    // Pointer to element destructor
                    iwc.i8_type().ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            CollectionsRtFunc::Push => iwc.void_type().fn_type(
                &[
                    ccg.list_type(ts).into(),
                    iwc.i8_type().ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            CollectionsRtFunc::Pop => iwc.bool_type().fn_type(
                &[
                    ccg.list_type(ts).into(),
                    iwc.i8_type().ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            CollectionsRtFunc::Get | CollectionsRtFunc::Set | CollectionsRtFunc::Insert => {
                iwc.bool_type().fn_type(
                    &[
                        ccg.list_type(ts).into(),
                        iwc.i64_type().into(),
                        iwc.i8_type().ptr_type(AddressSpace::default()).into(),
                    ],
                    false,
                )
            }
            CollectionsRtFunc::Length => iwc.i64_type().fn_type(&[ccg.list_type(ts).into()], false),
        }
    }

    /// Returns the extern function corresponding to this [CollectionsRtFunc].
    ///
    /// Requires a [CollectionsCodegen] to determine the function signature.
    pub fn get_extern<'c, H: HugrView>(
        self,
        ctx: &EmitFuncContext<'c, '_, H>,
        ccg: &(impl CollectionsCodegen + 'c),
    ) -> Result<FunctionValue<'c>> {
        ctx.get_extern_func(
            ccg.rt_func_name(self),
            self.signature(ctx.typing_session(), ccg),
        )
    }
}

impl From<ListOp> for CollectionsRtFunc {
    fn from(op: ListOp) -> Self {
        match op {
            ListOp::get => CollectionsRtFunc::Get,
            ListOp::set => CollectionsRtFunc::Set,
            ListOp::push => CollectionsRtFunc::Push,
            ListOp::pop => CollectionsRtFunc::Pop,
            ListOp::insert => CollectionsRtFunc::Insert,
            ListOp::length => CollectionsRtFunc::Length,
            _ => todo!(),
        }
    }
}

/// A helper trait for customising the lowering of [hugr_core::std_extensions::collections]
/// types, [hugr_core::ops::constant::CustomConst]s, and ops.
pub trait CollectionsCodegen: Clone {
    /// Return the llvm type of [hugr_core::std_extensions::collections::LIST_TYPENAME].
    fn list_type<'c>(&self, session: TypingSession<'c, '_>) -> BasicTypeEnum<'c> {
        session
            .iw_context()
            .i8_type()
            .ptr_type(AddressSpace::default())
            .into()
    }

    /// Return the name of a given [CollectionsRtFunc].
    fn rt_func_name(&self, func: CollectionsRtFunc) -> String {
        match func {
            CollectionsRtFunc::New => "__rt__list__new",
            CollectionsRtFunc::Push => "__rt__list__push",
            CollectionsRtFunc::Pop => "__rt__list__pop",
            CollectionsRtFunc::Get => "__rt__list__get",
            CollectionsRtFunc::Set => "__rt__list__set",
            CollectionsRtFunc::Insert => "__rt__list__insert",
            CollectionsRtFunc::Length => "__rt__list__length",
        }
        .into()
    }
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
        builder
            .custom_type((collections::EXTENSION_ID, collections::LIST_TYPENAME), {
                let ccg = self.0.clone();
                move |ts, _hugr_type| Ok(ccg.list_type(ts).as_basic_type_enum())
            })
            .custom_const::<ListValue>({
                let ccg = self.0.clone();
                move |ctx, k| emit_list_value(ctx, &ccg, k)
            })
            .simple_extension_op::<ListOp>(move |ctx, args, op| {
                emit_list_op(ctx, &self.0, args, op)
            })
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
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &(impl CollectionsCodegen + 'c),
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: ListOp,
) -> Result<()> {
    let hugr_elem_ty = match args.node().args() {
        [TypeArg::Type { ty }] => ty.clone(),
        _ => {
            bail!("Collections: invalid type args for list op");
        }
    };
    let elem_ty = ctx.llvm_type(&hugr_elem_ty)?;
    let func = CollectionsRtFunc::get_extern(op.into(), ctx, ccg)?;
    match op {
        ListOp::push => {
            let [list, elem] = args.inputs.try_into().unwrap();
            let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
            ctx.builder()
                .build_call(func, &[list.into(), elem_ptr.into()], "")?;
            args.outputs.finish(ctx.builder(), vec![list])?;
        }
        ListOp::pop => {
            let [list] = args.inputs.try_into().unwrap();
            let out_ptr = build_alloca_i8_ptr(ctx, elem_ty, None)?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), out_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let elem = build_load_i8_ptr(ctx, out_ptr, elem_ty)?;
            let elem_opt = build_option(ctx, ok, elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![list, elem_opt])?;
        }
        ListOp::get => {
            let [list, idx] = args.inputs.try_into().unwrap();
            let out_ptr = build_alloca_i8_ptr(ctx, elem_ty, None)?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), idx.into(), out_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let elem = build_load_i8_ptr(ctx, out_ptr, elem_ty)?;
            let elem_opt = build_option(ctx, ok, elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![elem_opt])?;
        }
        ListOp::set => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let old_elem = build_load_i8_ptr(ctx, elem_ptr, elem.get_type())?;
            let ok_or =
                build_ok_or_else(ctx, ok, elem, hugr_elem_ty.clone(), old_elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![list, ok_or])?;
        }
        ListOp::insert => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            let unit =
                ctx.llvm_sum_type(SumType::new_unary(1))?
                    .build_tag(ctx.builder(), 0, vec![])?;
            let ok_or = build_ok_or_else(ctx, ok, unit, Type::UNIT, elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![list, ok_or])?;
        }
        ListOp::length => {
            let [list] = args.inputs.try_into().unwrap();
            let length = ctx
                .builder()
                .build_call(func, &[list.into()], "")?
                .try_as_basic_value()
                .unwrap_left()
                .into_int_value();
            args.outputs
                .finish(ctx.builder(), vec![list, length.into()])?;
        }
        _ => return Err(anyhow!("Collections: unimplemented op: {}", op.name())),
    }
    Ok(())
}

fn emit_list_value<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &(impl CollectionsCodegen + 'c),
    val: &ListValue,
) -> Result<BasicValueEnum<'c>> {
    let elem_ty = ctx.llvm_type(val.get_element_type())?;
    let iwc = ctx.typing_session().iw_context();
    let capacity = iwc
        .i64_type()
        .const_int(val.get_contents().len() as u64, false);
    let elem_size = elem_ty.size_of().unwrap();
    let alignment = iwc.i64_type().const_int(8, false);
    // TODO: Lookup destructor for elem_ty
    let destructor = iwc.i8_type().ptr_type(AddressSpace::default()).const_null();
    let list = ctx
        .builder()
        .build_call(
            CollectionsRtFunc::New.get_extern(ctx, ccg)?,
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
    // Push elements onto the list
    let rt_push = CollectionsRtFunc::Push.get_extern(ctx, ccg)?;
    for v in val.get_contents() {
        let elem = emit_value(ctx, v)?;
        let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
        ctx.builder()
            .build_call(rt_push, &[list.into(), elem_ptr.into()], "")?;
    }
    Ok(list)
}

/// Helper function to allocate space on the stack for a given type.
///
/// Optionally also stores a value at that location.
///
/// Returns an i8 pointer to the allocated memory.
fn build_alloca_i8_ptr<'c, H: HugrView>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ty: BasicTypeEnum<'c>,
    value: Option<BasicValueEnum<'c>>,
) -> Result<PointerValue<'c>> {
    let builder = ctx.builder();
    let ptr = builder.build_alloca(ty, "")?;
    if let Some(val) = value {
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

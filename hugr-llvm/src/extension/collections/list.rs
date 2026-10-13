use anyhow::{Ok, Result, bail};
use hugr_core::{
    HugrView, Node,
    extension::simple_op::MakeExtensionOp as _,
    ops::ExtensionOp,
    std_extensions::collections::list::{self, ListOp, ListValue},
};
use inkwell::values::FunctionValue;
use inkwell::{
    AddressSpace,
    types::{BasicType, BasicTypeEnum, FunctionType},
    values::{BasicValueEnum, IntValue, PointerValue},
};

use crate::emit::func::{build_ok_or_else, build_option};
use crate::{
    custom::{CodegenExtension, CodegenExtsBuilder},
    emit::{EmitOpArgs, emit_value, func::EmitFuncContext},
    types::TypingSession,
};

/// Runtime functions that implement operations on lists.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
#[non_exhaustive]
pub enum ListRtFunc {
    New,
    Push,
    Pop,
    Get,
    Set,
    Insert,
    Length,
    DiscardEmpty,
}

impl ListRtFunc {
    /// The signature of a given [`ListRtFunc`].
    ///
    /// Requires a [`ListCodegen`] to determine the type of lists.
    pub fn signature<'c>(
        self,
        ts: TypingSession<'c, '_>,
        ccg: &(impl ListCodegen + 'c),
    ) -> FunctionType<'c> {
        let iwc = ts.iw_context();
        match self {
            ListRtFunc::New => ccg.list_type(ts).fn_type(
                &[
                    iwc.i64_type().into(), // Capacity
                    iwc.i64_type().into(), // Single element size in bytes
                    iwc.i64_type().into(), // Element alignment
                    // Pointer to element destructor
                    iwc.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            ListRtFunc::DiscardEmpty => iwc.void_type().fn_type(&[ccg.list_type(ts).into()], false),
            ListRtFunc::Push => iwc.void_type().fn_type(
                &[
                    ccg.list_type(ts).into(),
                    iwc.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            ListRtFunc::Pop => iwc.bool_type().fn_type(
                &[
                    ccg.list_type(ts).into(),
                    iwc.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            ListRtFunc::Get | ListRtFunc::Set | ListRtFunc::Insert => iwc.bool_type().fn_type(
                &[
                    ccg.list_type(ts).into(),
                    iwc.i64_type().into(),
                    iwc.ptr_type(AddressSpace::default()).into(),
                ],
                false,
            ),
            ListRtFunc::Length => iwc.i64_type().fn_type(&[ccg.list_type(ts).into()], false),
        }
    }

    /// Returns the extern function corresponding to this [`ListRtFunc`].
    ///
    /// Requires a [`ListCodegen`] to determine the function signature.
    pub fn get_extern<'c, H: HugrView<Node = Node>>(
        self,
        ctx: &EmitFuncContext<'c, '_, H>,
        ccg: &(impl ListCodegen + 'c),
    ) -> Result<FunctionValue<'c>> {
        ctx.get_extern_func(
            ccg.rt_func_name(self),
            self.signature(ctx.typing_session(), ccg),
        )
    }
}

impl From<ListOp> for ListRtFunc {
    fn from(op: ListOp) -> Self {
        match op {
            ListOp::new => ListRtFunc::New,
            ListOp::discard_empty => ListRtFunc::DiscardEmpty,
            ListOp::get => ListRtFunc::Get,
            ListOp::set => ListRtFunc::Set,
            ListOp::push => ListRtFunc::Push,
            ListOp::pop => ListRtFunc::Pop,
            ListOp::insert => ListRtFunc::Insert,
            ListOp::length => ListRtFunc::Length,
            _ => unreachable!("unknown list operation"),
        }
    }
}

/// A helper trait for customizing the lowering of [`hugr_core::std_extensions::collections::list`]
/// types, [`hugr_core::ops::constant::CustomConst`]s, and ops.
pub trait ListCodegen: Clone {
    /// Return the llvm type of [`hugr_core::std_extensions::collections::list::LIST_TYPENAME`].
    fn list_type<'c>(&self, session: TypingSession<'c, '_>) -> BasicTypeEnum<'c> {
        session
            .iw_context()
            .ptr_type(AddressSpace::default())
            .into()
    }

    /// Allocate bytes using the same customization pattern as ArrayCodegen.
    fn emit_allocate_list<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        size: IntValue<'c>,
    ) -> Result<PointerValue<'c>> {
        Ok(crate::emit::libc::emit_libc_malloc(ctx, size.into())?.into_pointer_value())
    }

    /// Free a pointer allocated by emit_allocate_list, including replaced buffers.
    fn emit_free_list<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        crate::emit::libc::emit_libc_free(ctx, ptr.into())
    }

    /// Prelude implementation used for every list panic. Register the same
    /// implementation for prelude types when it customizes error representation.
    fn prelude_codegen(&self) -> impl crate::extension::prelude::PreludeCodegen {
        crate::extension::prelude::DefaultPreludeCodegen
    }

    /// Define runtime functions. Custom external runtimes may override this hook.
    fn define_runtime<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
    ) -> Result<()> {
        define_list_runtime(ctx, self)
    }

    /// Return the name of a given [`ListRtFunc`].
    fn rt_func_name(&self, func: ListRtFunc) -> String {
        match func {
            ListRtFunc::New => "__rt__list__new",
            ListRtFunc::DiscardEmpty => "__rt__list__discard_empty",
            ListRtFunc::Push => "__rt__list__push",
            ListRtFunc::Pop => "__rt__list__pop",
            ListRtFunc::Get => "__rt__list__get",
            ListRtFunc::Set => "__rt__list__set",
            ListRtFunc::Insert => "__rt__list__insert",
            ListRtFunc::Length => "__rt__list__length",
        }
        .into()
    }
}

/// Default list lowering with an embedded byte-buffer runtime.
///
/// Allocation and freeing use the ListCodegen hooks (libc by default). Every
/// failure uses the selected PreludeCodegen; discard_empty frees empty handles
/// of arbitrary element types. The byte-copy runtime currently uses 64-bit sizes.
#[derive(Default, Clone)]
pub struct DefaultListCodegen;

impl ListCodegen for DefaultListCodegen {}

fn define_list_runtime<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &impl ListCodegen,
) -> Result<()> {
    use crate::emit::func::get_or_define_function;
    use crate::extension::prelude::PreludeCodegen;
    use hugr_core::extension::prelude::ConstError;
    let name = ccg.rt_func_name(ListRtFunc::New);
    if ctx
        .get_current_module()
        .get_function(&name)
        .is_some_and(|f| f.count_basic_blocks() != 0)
    {
        return Ok(());
    }
    let iwc = ctx.iw_context();
    let ptr_ty = iwc.ptr_type(AddressSpace::default());
    get_or_define_function(
        ctx,
        "__hugr_list_allocate",
        &[iwc.i64_type().into()],
        Some(ptr_ty.into()),
        |ctx, [size]| {
            Ok(Some(
                ccg.emit_allocate_list(ctx, size.into_int_value())?.into(),
            ))
        },
    )?;
    get_or_define_function(
        ctx,
        "__hugr_list_free",
        &[ptr_ty.into()],
        None,
        |ctx, [ptr]| {
            ccg.emit_free_list(ctx, ptr.into_pointer_value())?;
            Ok(None)
        },
    )?;
    for (name, message) in [
        (
            "__hugr_list_nonempty_failure",
            "Cannot discard a nonempty list",
        ),
        ("__hugr_list_allocation_failure", "List allocation failed"),
        ("__hugr_list_capacity_failure", "List capacity overflow"),
    ] {
        get_or_define_function(ctx, name, &[], None, |ctx, []| {
            let pcg = ccg.prelude_codegen();
            let err = pcg.emit_const_error(ctx, &ConstError::new(1, message))?;
            pcg.emit_panic(ctx, err)?;
            ctx.builder().build_unreachable()?;
            Ok(None)
        })?;
    }
    let hooks = [
        "__hugr_list_allocate",
        "__hugr_list_free",
        "__hugr_list_nonempty_failure",
        "__hugr_list_allocation_failure",
        "__hugr_list_capacity_failure",
    ];
    // Source runtime declarations must resolve these destination definitions;
    // internal linkage would let LLVM rename them instead of resolving calls.
    for name in hooks {
        ctx.get_current_module()
            .get_function(name)
            .unwrap()
            .set_linkage(inkwell::module::Linkage::External);
    }
    let buffer = inkwell::memory_buffer::MemoryBuffer::create_from_memory_range_copy(
        // Inkwell 0.9 requires a trailing NUL; 0.10 also accepts it.
        concat!(include_str!("list_runtime.ll"), "\0").as_bytes(),
        "hugr list runtime",
    );
    let runtime = iwc
        .create_module_from_ir(buffer)
        .map_err(|err| anyhow::anyhow!("Invalid list runtime: {err}"))?;
    let ops = [
        ListRtFunc::New,
        ListRtFunc::Push,
        ListRtFunc::Pop,
        ListRtFunc::Get,
        ListRtFunc::Set,
        ListRtFunc::Insert,
        ListRtFunc::Length,
        ListRtFunc::DiscardEmpty,
    ];
    for op in ops {
        let default_name = DefaultListCodegen.rt_func_name(op);
        runtime
            .get_function(&default_name)
            .unwrap()
            .as_global_value()
            .set_name(&ccg.rt_func_name(op));
    }
    ctx.get_current_module()
        .link_in_module(runtime)
        .map_err(|err| anyhow::anyhow!("Cannot link list runtime: {err}"))?;
    for name in hooks {
        let hook = ctx.get_current_module().get_function(name).unwrap();
        anyhow::ensure!(
            hook.count_basic_blocks() != 0,
            "Missing list callback definition {name}"
        );
        hook.set_linkage(inkwell::module::Linkage::Internal);
    }
    // Link entry points externally so LLVM retains them, then make them private.
    for op in ops {
        ctx.get_current_module()
            .get_function(&ccg.rt_func_name(op))
            .unwrap()
            .set_linkage(inkwell::module::Linkage::Internal);
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct ListCodegenExtension<CCG>(CCG);

impl<CCG: ListCodegen> ListCodegenExtension<CCG> {
    pub fn new(ccg: CCG) -> Self {
        Self(ccg)
    }
}

impl<CCG: ListCodegen> From<CCG> for ListCodegenExtension<CCG> {
    fn from(ccg: CCG) -> Self {
        Self::new(ccg)
    }
}

impl<CCG: ListCodegen> CodegenExtension for ListCodegenExtension<CCG> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type((list::EXTENSION_ID, list::LIST_TYPENAME), {
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

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [`ListCodegenExtension`] to the given [`CodegenExtsBuilder`] using `ccg`
    /// as the implementation.
    #[must_use]
    pub fn add_default_list_extensions(self) -> Self {
        self.add_list_extensions(DefaultListCodegen)
    }

    /// Add a [`ListCodegenExtension`] to the given [`CodegenExtsBuilder`] using
    /// [`DefaultListCodegen`] as the implementation.
    pub fn add_list_extensions(self, ccg: impl ListCodegen + 'a) -> Self {
        self.add_extension(ListCodegenExtension::from(ccg))
    }
}

fn emit_list_op<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &(impl ListCodegen + 'c),
    args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    op: ListOp,
) -> Result<()> {
    let hugr_elem_ty = match args.node().args() {
        [ty] => ty.clone().try_into()?,
        _ => {
            bail!("Collections: invalid type args for list op");
        }
    };
    let elem_ty = ctx.llvm_type(&hugr_elem_ty)?;
    ccg.define_runtime(ctx)?;
    let func = ListRtFunc::get_extern(op.into(), ctx, ccg)?;
    match op {
        ListOp::new => {
            let iwc = ctx.typing_session().iw_context();
            let list = ctx
                .builder()
                .build_call(
                    func,
                    &[
                        iwc.i64_type().const_zero().into(),
                        elem_ty.size_of().unwrap().into(),
                        iwc.i64_type().const_int(1, false).into(),
                        iwc.ptr_type(AddressSpace::default()).const_null().into(),
                    ],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_basic();
            args.outputs.finish(ctx.builder(), vec![list])?;
        }
        ListOp::discard_empty => {
            let [list] = args.inputs.try_into().unwrap();
            ctx.builder().build_call(func, &[list.into()], "")?;
            args.outputs.finish(ctx.builder(), vec![])?;
        }
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
                .unwrap_basic()
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
                .unwrap_basic()
                .into_int_value();
            let elem = build_load_i8_ptr(ctx, out_ptr, elem_ty)?;
            let elem_opt = build_option(ctx, ok, elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![list, elem_opt])?;
        }
        ListOp::set => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            let old_elem = build_load_i8_ptr(ctx, elem_ptr, elem.get_type())?;
            let ok_or =
                build_ok_or_else(ctx, ok, old_elem, hugr_elem_ty.clone(), elem, hugr_elem_ty)?;
            args.outputs.finish(ctx.builder(), vec![list, ok_or])?;
        }
        ListOp::insert => {
            let [list, idx, elem] = args.inputs.try_into().unwrap();
            let elem_ptr = build_alloca_i8_ptr(ctx, elem_ty, Some(elem))?;
            let ok = ctx
                .builder()
                .build_call(func, &[list.into(), idx.into(), elem_ptr.into()], "")?
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            let result_ty = ctx.llvm_sum_type(hugr_core::extension::prelude::either_type(
                [hugr_elem_ty],
                [],
            ))?;
            let success = result_ty.build_tag(ctx.builder(), 1, vec![])?;
            let failure = result_ty.build_tag(ctx.builder(), 0, vec![elem])?;
            let ok_or = ctx.builder().build_select(ok, success, failure, "")?;
            args.outputs.finish(ctx.builder(), vec![list, ok_or])?;
        }
        ListOp::length => {
            let [list] = args.inputs.try_into().unwrap();
            let length = ctx
                .builder()
                .build_call(func, &[list.into()], "")?
                .try_as_basic_value()
                .unwrap_basic()
                .into_int_value();
            args.outputs
                .finish(ctx.builder(), vec![list, length.into()])?;
        }
        _ => bail!("Collections: unimplemented op: {}", op.op_id()),
    }
    Ok(())
}

fn emit_list_value<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ccg: &(impl ListCodegen + 'c),
    val: &ListValue,
) -> Result<BasicValueEnum<'c>> {
    let elem_ty = ctx.llvm_type(val.get_element_type())?;
    let iwc = ctx.typing_session().iw_context();
    let capacity = iwc
        .i64_type()
        .const_int(val.get_contents().len() as u64, false);
    let elem_size = elem_ty.size_of().unwrap();
    let alignment = iwc.i64_type().const_int(8, false);
    // Empty disposal needs no element destructor. Elements must be transferred
    // explicitly before discarding the emptied handle.
    let destructor = iwc.ptr_type(AddressSpace::default()).const_null();
    ccg.define_runtime(ctx)?;
    let rt_new = ListRtFunc::New.get_extern(ctx, ccg)?;
    let list = ctx
        .builder()
        .build_call(
            rt_new,
            &[
                capacity.into(),
                elem_size.into(),
                alignment.into(),
                destructor.into(),
            ],
            "",
        )?
        .try_as_basic_value()
        .unwrap_basic();
    // Push elements onto the list
    let rt_push = ListRtFunc::Push.get_extern(ctx, ccg)?;
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
/// Returns a pointer to the allocated memory.
fn build_alloca_i8_ptr<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ty: BasicTypeEnum<'c>,
    value: Option<BasicValueEnum<'c>>,
) -> Result<PointerValue<'c>> {
    let builder = ctx.builder();
    let ptr = builder.build_alloca(ty, "")?;
    // Initialize failed get/pop/set output storage: never load an undefined value.
    builder.build_store(ptr, value.unwrap_or_else(|| ty.const_zero()))?;
    Ok(ptr)
}

/// Helper function to load a value from a pointer.
fn build_load_i8_ptr<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    i8_ptr: PointerValue<'c>,
    ty: BasicTypeEnum<'c>,
) -> Result<BasicValueEnum<'c>> {
    let builder = ctx.builder();
    let val = builder.build_load(ty, i8_ptr, "")?;
    Ok(val)
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{Dataflow, DataflowHugr},
        extension::{
            ExtensionRegistry,
            prelude::{self, ConstUsize, qb_t, usize_t},
        },
        ops::{DataflowOpTrait, Value},
        std_extensions::collections::list::{self, ListOp, ListValue, list_type},
    };
    use rstest::rstest;

    #[derive(Clone)]
    struct ExternalListCodegen;
    impl super::ListCodegen for ExternalListCodegen {
        fn define_runtime<'c, H: hugr_core::HugrView<Node = hugr_core::Node>>(
            &self,
            _: &mut crate::emit::EmitFuncContext<'c, '_, H>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    use crate::{
        check_emission,
        custom::CodegenExtsBuilder,
        emit::test::SimpleHugrConfig,
        test::{TestContext, exec_ctx, llvm_ctx},
    };

    #[rstest]
    #[case::new(ListOp::new)]
    #[case::discard_empty(ListOp::discard_empty)]
    #[case::push(ListOp::push)]
    #[case::pop(ListOp::pop)]
    #[case::get(ListOp::get)]
    #[case::set(ListOp::set)]
    #[case::insert(ListOp::insert)]
    #[case::length(ListOp::length)]
    fn test_list_emission(mut llvm_ctx: TestContext, #[case] op: ListOp) {
        use hugr_core::extension::simple_op::MakeExtensionOp as _;

        let ext_op = list::EXTENSION
            .instantiate_extension_op(
                op.op_id().as_ref(),
                [if matches!(op, ListOp::get | ListOp::discard_empty) {
                    usize_t()
                } else {
                    qb_t()
                }
                .into()],
            )
            .unwrap();
        let es = ExtensionRegistry::new([list::EXTENSION.to_owned(), prelude::PRELUDE.to_owned()]);
        es.validate().unwrap();
        let mut hugr = SimpleHugrConfig::new()
            .with_ins(ext_op.signature().input().clone())
            .with_outs(ext_op.signature().output().clone())
            .with_extensions(es)
            .finish(|mut hugr_builder| {
                let outputs = hugr_builder
                    .add_dataflow_op(ext_op, hugr_builder.input_wires())
                    .unwrap()
                    .outputs();
                hugr_builder.finish_hugr_with_outputs(outputs).unwrap()
            });
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        llvm_ctx.add_extensions(|cge| cge.add_list_extensions(ExternalListCodegen));
        check_emission!(op.op_id().as_str(), hugr, llvm_ctx);
    }

    // Convert a result sum into an observable integer. Empty rows return 0;
    // populated rows return their element. This pins both tags and payloads.
    fn sum_value(
        builder: &mut crate::emit::test::DFGW,
        value: hugr_core::Wire,
        sum: hugr_core::types::SumType,
    ) -> hugr_core::Wire {
        use hugr_core::builder::{DataflowSubContainer, SubContainer};
        let zero = builder.add_load_value(ConstUsize::new(0));
        let rows: Vec<hugr_core::types::TypeRow> = sum
            .variants()
            .map(|row| row.clone().try_into().unwrap())
            .collect();
        let mut cond = builder
            .conditional_builder((rows, value), [], [usize_t()].into())
            .unwrap();
        for tag in 0..sum.num_variants() {
            let case = cond.case_builder(tag).unwrap();
            let mut input = case.input_wires();
            case.finish_with_outputs([input.next().unwrap_or(zero)])
                .unwrap();
        }
        cond.finish_sub_container().unwrap().out_wire(0)
    }

    #[rstest]
    #[case::get_first(ListOp::get, 0, 2, false, 10)]
    #[case::get_last(ListOp::get, 1, 2, false, 20)]
    #[case::get_end(ListOp::get, 2, 2, false, 0)]
    #[case::get_huge(ListOp::get, u64::MAX, 2, false, 0)]
    #[case::get_empty(ListOp::get, 0, 0, false, 0)]
    #[case::get_keeps_list(ListOp::get, 0, 2, true, 2)]
    #[case::pop(ListOp::pop, 0, 2, false, 20)]
    #[case::pop_empty(ListOp::pop, 0, 0, false, 0)]
    #[case::pop_keeps_list(ListOp::pop, 0, 2, true, 1)]
    #[case::set_old(ListOp::set, 1, 2, false, 20)]
    #[case::set_invalid(ListOp::set, 2, 2, false, 99)]
    #[case::set_huge(ListOp::set, u64::MAX, 2, false, 99)]
    #[case::set_new(ListOp::set, 1, 2, true, 99)]
    #[case::set_failed_unchanged(ListOp::set, 2, 2, true, 10)]
    #[case::insert_front(ListOp::insert, 0, 2, true, 99)]
    #[case::insert_middle(ListOp::insert, 1, 2, true, 99)]
    #[case::insert_end(ListOp::insert, 2, 2, true, 99)]
    #[case::insert_empty(ListOp::insert, 0, 0, true, 99)]
    #[case::insert_success(ListOp::insert, 2, 2, false, 0)]
    #[case::insert_invalid(ListOp::insert, 3, 2, false, 99)]
    #[case::insert_huge(ListOp::insert, u64::MAX, 2, false, 99)]
    #[case::insert_failed_unchanged(ListOp::insert, 3, 2, true, 10)]
    #[case::growth(ListOp::length, 0, 100, false, 100)]
    #[case::length_empty(ListOp::length, 0, 0, false, 0)]
    fn exec_list(
        mut exec_ctx: TestContext,
        #[case] op: ListOp,
        #[case] index: u64,
        #[case] initial_len: u64,
        #[case] observe_list: bool,
        #[case] expected: u64,
    ) {
        use hugr_core::extension::prelude::{either_type, option_type};
        let hugr = SimpleHugrConfig::new()
            .with_outs([usize_t()])
            .finish(|mut b| {
                let mut list = b
                    .add_dataflow_op(
                        ListOp::new.with_type(usize_t()).to_extension_op().unwrap(),
                        [],
                    )
                    .unwrap()
                    .out_wire(0);
                for i in 1..=initial_len {
                    let elem = b.add_load_value(ConstUsize::new(i * 10));
                    list = b
                        .add_dataflow_op(
                            ListOp::push.with_type(usize_t()).to_extension_op().unwrap(),
                            [list, elem],
                        )
                        .unwrap()
                        .out_wire(0);
                }
                let idx = b.add_load_value(ConstUsize::new(index));
                let elem = b.add_load_value(ConstUsize::new(99));
                let inputs = match op {
                    ListOp::get => vec![list, idx],
                    ListOp::set | ListOp::insert => vec![list, idx, elem],
                    _ => vec![list],
                };
                let result = b
                    .add_dataflow_op(op.with_type(usize_t()).to_extension_op().unwrap(), inputs)
                    .unwrap();
                list = result.out_wire(0);
                let value = result.out_wire(1);
                let result_value = match op {
                    ListOp::get | ListOp::pop => sum_value(&mut b, value, option_type([usize_t()])),
                    ListOp::set => sum_value(&mut b, value, either_type([usize_t()], [usize_t()])),
                    ListOp::insert => sum_value(&mut b, value, either_type([usize_t()], [])),
                    _ => value,
                };
                let out = if observe_list {
                    if matches!(op, ListOp::get | ListOp::pop) {
                        let result = b
                            .add_dataflow_op(
                                ListOp::length
                                    .with_type(usize_t())
                                    .to_extension_op()
                                    .unwrap(),
                                [list],
                            )
                            .unwrap();
                        list = result.out_wire(0);
                        result.out_wire(1)
                    } else {
                        let read_index = if index < initial_len
                            || (op == ListOp::insert && index == initial_len)
                        {
                            index
                        } else {
                            0
                        };
                        let idx = b.add_load_value(ConstUsize::new(read_index));
                        let result = b
                            .add_dataflow_op(
                                ListOp::get.with_type(usize_t()).to_extension_op().unwrap(),
                                [list, idx],
                            )
                            .unwrap();
                        list = result.out_wire(0);
                        sum_value(&mut b, result.out_wire(1), option_type([usize_t()]))
                    }
                } else {
                    result_value
                };
                for _ in 0..=initial_len {
                    list = b
                        .add_dataflow_op(
                            ListOp::pop.with_type(usize_t()).to_extension_op().unwrap(),
                            [list],
                        )
                        .unwrap()
                        .out_wire(0);
                }
                b.add_dataflow_op(
                    ListOp::discard_empty
                        .with_type(usize_t())
                        .to_extension_op()
                        .unwrap(),
                    [list],
                )
                .unwrap();
                b.finish_hugr_with_outputs([out]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_list_extensions()
        });
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), expected);
    }

    #[derive(Clone)]
    struct ConfiguredListCodegen {
        fail_at: u64,
        verify_counts: bool,
    }

    fn increment_counter<'c, H: hugr_core::HugrView<Node = hugr_core::Node>>(
        ctx: &mut crate::emit::EmitFuncContext<'c, '_, H>,
        name: &str,
    ) -> anyhow::Result<inkwell::values::IntValue<'c>> {
        let ty = ctx.iw_context().i64_type();
        let global = ctx.get_global(name, ty, false)?;
        if global.get_initializer().is_none() {
            global.set_initializer(&ty.const_zero());
        }
        let old = ctx
            .builder()
            .build_load(ty, global.as_pointer_value(), "")?
            .into_int_value();
        let next = ctx
            .builder()
            .build_int_add(old, ty.const_int(1, false), "")?;
        ctx.builder().build_store(global.as_pointer_value(), next)?;
        Ok(next)
    }

    impl super::ListCodegen for ConfiguredListCodegen {
        fn prelude_codegen(&self) -> impl crate::extension::prelude::PreludeCodegen {
            crate::emit::test::PanicTestPreludeCodegen
        }
        fn emit_allocate_list<'c, H: hugr_core::HugrView<Node = hugr_core::Node>>(
            &self,
            ctx: &mut crate::emit::EmitFuncContext<'c, '_, H>,
            size: inkwell::values::IntValue<'c>,
        ) -> anyhow::Result<inkwell::values::PointerValue<'c>> {
            let count = increment_counter(ctx, "custom.list.allocations")?;
            let fail = ctx.builder().build_int_compare(
                inkwell::IntPredicate::EQ,
                count,
                count.get_type().const_int(self.fail_at, false),
                "",
            )?;
            let failure = ctx.new_basic_block("allocation.failure", None);
            let success = ctx.new_basic_block("allocation.success", None);
            let done = ctx.new_basic_block("allocation.done", None);
            ctx.builder()
                .build_conditional_branch(fail, failure, success)?;
            ctx.builder().position_at_end(success);
            let ptr = crate::emit::libc::emit_libc_malloc(ctx, size.into())?.into_pointer_value();
            ctx.builder().build_unconditional_branch(done)?;
            ctx.builder().position_at_end(failure);
            let null = ptr.get_type().const_null();
            ctx.builder().build_unconditional_branch(done)?;
            ctx.builder().position_at_end(done);
            let result = ctx.builder().build_phi(ptr.get_type(), "")?;
            result.add_incoming(&[(&ptr, success), (&null, failure)]);
            Ok(result.as_basic_value().into_pointer_value())
        }
        fn emit_free_list<'c, H: hugr_core::HugrView<Node = hugr_core::Node>>(
            &self,
            ctx: &mut crate::emit::EmitFuncContext<'c, '_, H>,
            ptr: inkwell::values::PointerValue<'c>,
        ) -> anyhow::Result<()> {
            let frees = increment_counter(ctx, "custom.list.frees")?;
            crate::emit::libc::emit_libc_free(ctx, ptr.into())?;
            if self.verify_counts {
                use crate::extension::prelude::PreludeCodegen;
                let ty = ctx.iw_context().i64_type();
                let last = ctx.builder().build_int_compare(
                    inkwell::IntPredicate::EQ,
                    frees,
                    ty.const_int(12, false),
                    "",
                )?;
                let verify = ctx.new_basic_block("verify.allocations", None);
                let done = ctx.new_basic_block("free.done", None);
                ctx.builder().build_conditional_branch(last, verify, done)?;
                ctx.builder().position_at_end(verify);
                let allocations = ctx
                    .builder()
                    .build_load(
                        ty,
                        ctx.get_current_module()
                            .get_global("custom.list.allocations")
                            .unwrap()
                            .as_pointer_value(),
                        "",
                    )?
                    .into_int_value();
                let ok = ctx.builder().build_int_compare(
                    inkwell::IntPredicate::EQ,
                    allocations,
                    ty.const_int(12, false),
                    "",
                )?;
                let pcg = self.prelude_codegen();
                let matched = pcg.emit_const_error(
                    ctx,
                    &hugr_core::extension::prelude::ConstError::new(
                        1,
                        "12 allocations and 12 frees",
                    ),
                )?;
                let mismatch = pcg.emit_const_error(
                    ctx,
                    &hugr_core::extension::prelude::ConstError::new(1, "Allocator count mismatch"),
                )?;
                let err = ctx.builder().build_select(ok, matched, mismatch, "")?;
                pcg.emit_panic(ctx, err)?;
                ctx.builder().build_unreachable()?;
                ctx.builder().position_at_end(done);
            }
            Ok(())
        }
    }

    fn empty_list_graph(
        elem_ty: hugr_core::types::Type,
        len: u64,
        drain: bool,
        include_constant: bool,
    ) -> hugr_core::Hugr {
        SimpleHugrConfig::new()
            .with_outs([usize_t()])
            .finish(|mut b| {
                let mut list = b
                    .add_dataflow_op(
                        ListOp::new
                            .with_type(elem_ty.clone())
                            .to_extension_op()
                            .unwrap(),
                        [],
                    )
                    .unwrap()
                    .out_wire(0);
                for i in 0..len {
                    let elem = b.add_load_value(ConstUsize::new(i));
                    list = b
                        .add_dataflow_op(
                            ListOp::push
                                .with_type(elem_ty.clone())
                                .to_extension_op()
                                .unwrap(),
                            [list, elem],
                        )
                        .unwrap()
                        .out_wire(0);
                }
                if drain {
                    for _ in 0..len {
                        list = b
                            .add_dataflow_op(
                                ListOp::pop
                                    .with_type(elem_ty.clone())
                                    .to_extension_op()
                                    .unwrap(),
                                [list],
                            )
                            .unwrap()
                            .out_wire(0);
                    }
                }
                b.add_dataflow_op(
                    ListOp::discard_empty
                        .with_type(elem_ty)
                        .to_extension_op()
                        .unwrap(),
                    [list],
                )
                .unwrap();
                if include_constant {
                    let mut konst = b.add_load_value(ListValue::new(
                        usize_t(),
                        (0..3).map(|i| ConstUsize::new(i).into()),
                    ));
                    for _ in 0..3 {
                        konst = b
                            .add_dataflow_op(
                                ListOp::pop.with_type(usize_t()).to_extension_op().unwrap(),
                                [konst],
                            )
                            .unwrap()
                            .out_wire(0);
                    }
                    b.add_dataflow_op(
                        ListOp::discard_empty
                            .with_type(usize_t())
                            .to_extension_op()
                            .unwrap(),
                        [konst],
                    )
                    .unwrap();
                }
                let out = b.add_load_value(ConstUsize::new(42));
                b.finish_hugr_with_outputs([out]).unwrap()
            })
    }

    #[rstest]
    fn exec_discard_empty_linear(mut exec_ctx: TestContext) {
        let hugr = empty_list_graph(qb_t(), 0, true, false);
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_list_extensions()
        });
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), 42);
    }

    #[rstest]
    fn exec_linear_after_draining(mut exec_ctx: TestContext) {
        use crate::{
            emit::test::{Emission, TEST_EMIT_DEBUG},
            utils::fat::FatExt,
        };
        use hugr_core::extension::prelude::{UnwrapBuilder, option_type};
        let hugr = SimpleHugrConfig::new()
            .with_ins([qb_t()])
            .with_outs([qb_t()])
            .finish(|mut b| {
                let qubit = b.input_wires().next().unwrap();
                let list = b
                    .add_dataflow_op(ListOp::new.with_type(qb_t()).to_extension_op().unwrap(), [])
                    .unwrap()
                    .out_wire(0);
                let list = b
                    .add_dataflow_op(
                        ListOp::push.with_type(qb_t()).to_extension_op().unwrap(),
                        [list, qubit],
                    )
                    .unwrap()
                    .out_wire(0);
                let pop = b
                    .add_dataflow_op(
                        ListOp::pop.with_type(qb_t()).to_extension_op().unwrap(),
                        [list],
                    )
                    .unwrap();
                b.add_dataflow_op(
                    ListOp::discard_empty
                        .with_type(qb_t())
                        .to_extension_op()
                        .unwrap(),
                    [pop.out_wire(0)],
                )
                .unwrap();
                let [value] = b
                    .build_unwrap_sum(1, option_type([qb_t()]), pop.out_wire(1))
                    .unwrap();
                b.finish_hugr_with_outputs([value]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_list_extensions()
        });
        let emission = Emission::emit_hugr(
            hugr.fat_root().unwrap(),
            exec_ctx.get_emit_hugr(),
            TEST_EMIT_DEBUG,
        )
        .unwrap();
        emission.verify().unwrap();
        emission
            .module()
            .get_function("main")
            .unwrap()
            .set_linkage(inkwell::module::Linkage::External);
        let ee = emission
            .module()
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .unwrap();
        // Default qubits lower to i16. This exercises linear token ownership,
        // not physical qubit allocation or a quantum runtime.
        unsafe {
            assert_eq!(
                ee.get_function::<unsafe extern "C" fn(u16) -> u16>("main")
                    .unwrap()
                    .call(0xbeef),
                0xbeef
            );
        }
    }

    #[rstest]
    #[case::byte_size(false)]
    #[case::capacity_doubling(true)]
    fn exec_capacity_panic(mut exec_ctx: TestContext, #[case] growth: bool) {
        use crate::emit::{EmitFuncContext, test::Emission};
        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(crate::emit::test::PanicTestPreludeCodegen)
        });
        let module_ctx = exec_ctx.get_emit_module_context();
        let iwc = exec_ctx.iw_context();
        let main =
            module_ctx
                .module()
                .add_function("main", iwc.i64_type().fn_type(&[], false), None);
        let mut ctx = EmitFuncContext::new(module_ctx, main).unwrap();
        super::ListCodegen::define_runtime(
            &ConfiguredListCodegen {
                fail_at: 0,
                verify_counts: false,
            },
            &mut ctx,
        )
        .unwrap();
        let new = super::ListRtFunc::New
            .get_extern(
                &ctx,
                &ConfiguredListCodegen {
                    fail_at: 0,
                    verify_counts: false,
                },
            )
            .unwrap();
        let list = ctx
            .builder()
            .build_call(
                new,
                &[
                    iwc.i64_type()
                        .const_int(if growth { 0 } else { u64::MAX }, false)
                        .into(),
                    iwc.i64_type()
                        .const_int(if growth { 0 } else { 2 }, false)
                        .into(),
                    iwc.i64_type().const_int(1, false).into(),
                    iwc.ptr_type(inkwell::AddressSpace::default())
                        .const_null()
                        .into(),
                ],
                "",
            )
            .unwrap()
            .try_as_basic_value()
            .unwrap_basic()
            .into_pointer_value();
        if growth {
            // A zero-sized element permits a tiny buffer. Force a full capacity
            // at the doubling limit to exercise overflow before any slot access.
            let header = iwc.get_struct_type("hugr.list").unwrap();
            for field in [1, 2] {
                let ptr = ctx
                    .builder()
                    .build_struct_gep(header, list, field, "")
                    .unwrap();
                ctx.builder()
                    .build_store(ptr, iwc.i64_type().const_int(1 << 63, false))
                    .unwrap();
            }
            let push = super::ListRtFunc::Push
                .get_extern(
                    &ctx,
                    &ConfiguredListCodegen {
                        fail_at: 0,
                        verify_counts: false,
                    },
                )
                .unwrap();
            ctx.builder()
                .build_call(
                    push,
                    &[
                        list.into(),
                        iwc.ptr_type(inkwell::AddressSpace::default())
                            .const_null()
                            .into(),
                    ],
                    "",
                )
                .unwrap();
        }
        ctx.builder()
            .build_return(Some(&iwc.i64_type().const_zero()))
            .unwrap();
        let (module_ctx, _) = ctx.finish().unwrap();
        let (module, _) = module_ctx.finish();
        let emission = Emission::new(module);
        emission.verify().unwrap();
        assert_eq!(
            emission.exec_panicking("main").unwrap(),
            "List capacity overflow"
        );
    }

    #[rstest]
    #[case::nonempty(0, 1, false, "Cannot discard a nonempty list")]
    #[case::buffer_failure(1, 0, false, "List allocation failed")]
    #[case::header_failure(2, 0, false, "List allocation failed")]
    #[case::growth_failure(3, 1, true, "List allocation failed")]
    fn exec_custom_panic(
        mut exec_ctx: TestContext,
        #[case] fail_at: u64,
        #[case] len: u64,
        #[case] drain: bool,
        #[case] expected: &str,
    ) {
        let hugr = empty_list_graph(usize_t(), len, drain, false);
        exec_ctx.add_extensions(move |cge| {
            cge.add_prelude_extensions(crate::emit::test::PanicTestPreludeCodegen)
                .add_list_extensions(ConfiguredListCodegen {
                    fail_at,
                    verify_counts: false,
                })
        });
        assert_eq!(exec_ctx.exec_hugr_panicking(hugr, "main"), expected);
    }

    #[rstest]
    fn exec_custom_allocator(mut exec_ctx: TestContext) {
        let hugr = empty_list_graph(usize_t(), 100, true, true);
        exec_ctx.add_extensions(|cge| {
            cge.add_prelude_extensions(crate::emit::test::PanicTestPreludeCodegen)
                .add_list_extensions(ConfiguredListCodegen {
                    fail_at: 0,
                    verify_counts: true,
                })
        });
        // The selected free hook verifies the allocation count after freeing the
        // final handle, then reports through the non-unwinding panic harness.
        // New/growth allocate 10 times; the constant list allocates twice.
        assert_eq!(
            exec_ctx.exec_hugr_panicking(hugr, "main"),
            "12 allocations and 12 frees"
        );
    }

    #[rstest]
    fn exec_const_list(mut exec_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs([usize_t()])
            .finish(|mut b| {
                let list = b.add_load_value(ListValue::new(
                    usize_t(),
                    [ConstUsize::new(10).into(), ConstUsize::new(20).into()],
                ));
                let result = b
                    .add_dataflow_op(
                        ListOp::length
                            .with_type(usize_t())
                            .to_extension_op()
                            .unwrap(),
                        [list],
                    )
                    .unwrap();
                let mut list = result.out_wire(0);
                for _ in 0..2 {
                    list = b
                        .add_dataflow_op(
                            ListOp::pop.with_type(usize_t()).to_extension_op().unwrap(),
                            [list],
                        )
                        .unwrap()
                        .out_wire(0);
                }
                b.add_dataflow_op(
                    ListOp::discard_empty
                        .with_type(usize_t())
                        .to_extension_op()
                        .unwrap(),
                    [list],
                )
                .unwrap();
                b.finish_hugr_with_outputs([result.out_wire(1)]).unwrap()
            });
        exec_ctx.add_extensions(|cge| {
            cge.add_default_prelude_extensions()
                .add_default_list_extensions()
        });
        assert_eq!(exec_ctx.exec_hugr_u64(hugr, "main"), 2);
    }

    #[test]
    fn runtime_handles_arbitrary_bytes_and_zero_sized_elements() {
        use inkwell::{context::Context, memory_buffer::MemoryBuffer, module::Linkage};
        use std::ffi::c_void;
        let context = Context::create();
        let buffer = MemoryBuffer::create_from_memory_range_copy(
            concat!(include_str!("list_runtime.ll"), "\0").as_bytes(),
            "list runtime",
        );
        let module = context.create_module_from_ir(buffer).unwrap();
        module.verify().unwrap();
        for name in [
            "new",
            "push",
            "pop",
            "get",
            "set",
            "insert",
            "length",
            "discard_empty",
        ] {
            module
                .get_function(&format!("__rt__list__{name}"))
                .unwrap()
                .set_linkage(Linkage::External);
        }
        inkwell::targets::Target::initialize_native(&Default::default()).unwrap();
        let ee = module
            .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
            .unwrap();
        unsafe extern "C" {
            fn malloc(size: usize) -> *mut c_void;
            fn free(ptr: *mut c_void);
            fn abort() -> !;
        }
        ee.add_global_mapping(
            &module.get_function("__hugr_list_allocate").unwrap(),
            malloc as *const () as usize,
        );
        ee.add_global_mapping(
            &module.get_function("__hugr_list_free").unwrap(),
            free as *const () as usize,
        );
        for name in [
            "__hugr_list_allocation_failure",
            "__hugr_list_capacity_failure",
            "__hugr_list_nonempty_failure",
        ] {
            ee.add_global_mapping(
                &module.get_function(name).unwrap(),
                abort as *const () as usize,
            );
        }
        // The function types match ListRtFunc's ABI. Every list is used uniquely
        // and freed once; all buffers passed below contain size bytes.
        unsafe {
            let new = ee
                .get_function::<unsafe extern "C" fn(u64, u64, u64, *const c_void) -> *mut c_void>(
                    "__rt__list__new",
                )
                .unwrap();
            let push = ee
                .get_function::<unsafe extern "C" fn(*mut c_void, *const u8)>("__rt__list__push")
                .unwrap();
            let pop = ee
                .get_function::<unsafe extern "C" fn(*mut c_void, *mut u8) -> bool>(
                    "__rt__list__pop",
                )
                .unwrap();
            let get = ee
                .get_function::<unsafe extern "C" fn(*mut c_void, u64, *mut u8) -> bool>(
                    "__rt__list__get",
                )
                .unwrap();
            let set = ee
                .get_function::<unsafe extern "C" fn(*mut c_void, u64, *mut u8) -> bool>(
                    "__rt__list__set",
                )
                .unwrap();
            let insert = ee
                .get_function::<unsafe extern "C" fn(*mut c_void, u64, *const u8) -> bool>(
                    "__rt__list__insert",
                )
                .unwrap();
            let length = ee
                .get_function::<unsafe extern "C" fn(*mut c_void) -> u64>("__rt__list__length")
                .unwrap();
            let discard_empty = ee
                .get_function::<unsafe extern "C" fn(*mut c_void)>("__rt__list__discard_empty")
                .unwrap();
            for size in [0, 1, 3, 32] {
                let list = new.call(0, size, 16, std::ptr::null());
                let first = [42; 32];
                let second = [99; 32];
                let mut output = [0; 32];
                assert!(!pop.call(list, output.as_mut_ptr()));
                for _ in 0..100 {
                    push.call(list, first.as_ptr());
                }
                assert_eq!(length.call(list), 100);
                assert!(insert.call(list, 100, second.as_ptr()));
                assert!(get.call(list, 100, output.as_mut_ptr()));
                assert_eq!(&output[..size as usize], &second[..size as usize]);
                output = second;
                assert!(set.call(list, 0, output.as_mut_ptr()));
                assert_eq!(&output[..size as usize], &first[..size as usize]);
                assert!(!get.call(list, u64::MAX, output.as_mut_ptr()));
                assert!(!insert.call(list, u64::MAX, first.as_ptr()));
                assert!(insert.call(list, 0, second.as_ptr()));
                assert!(get.call(list, 2, output.as_mut_ptr()));
                assert_eq!(&output[..size as usize], &first[..size as usize]);
                assert!(insert.call(list, 50, second.as_ptr()));
                assert!(get.call(list, 51, output.as_mut_ptr()));
                assert_eq!(&output[..size as usize], &first[..size as usize]);
                assert_eq!(length.call(list), 103);
                assert!(!get.call(list, 103, output.as_mut_ptr()));
                for _ in 0..103 {
                    assert!(pop.call(list, output.as_mut_ptr()));
                }
                assert_eq!(length.call(list), 0);
                assert!(!pop.call(list, output.as_mut_ptr()));
                discard_empty.call(list);
                let empty = new.call(0, size, 1, std::ptr::null());
                discard_empty.call(empty);
            }
        }
    }

    #[rstest]
    fn test_const_list_emission(mut llvm_ctx: TestContext) {
        let elem_ty = usize_t();
        let contents = (1..4).map(|i| Value::extension(ConstUsize::new(i)));
        let es = ExtensionRegistry::new([list::EXTENSION.to_owned(), prelude::PRELUDE.to_owned()]);
        es.validate().unwrap();

        let mut hugr = SimpleHugrConfig::new()
            .with_ins(vec![])
            .with_outs(vec![list_type(elem_ty.clone())])
            .with_extensions(es)
            .finish(|mut hugr_builder| {
                let list = hugr_builder.add_load_value(ListValue::new(elem_ty, contents));
                hugr_builder.finish_hugr_with_outputs(vec![list]).unwrap()
            });

        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        llvm_ctx.add_extensions(|cge| cge.add_list_extensions(ExternalListCodegen));
        check_emission!("const", hugr, llvm_ctx);
    }
}

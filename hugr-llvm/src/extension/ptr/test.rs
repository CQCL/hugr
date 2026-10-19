use super::*;
use crate::{
    emit::{
        EmitDebugInfo,
        test::{Emission, SimpleHugrConfig},
    },
    extension::{DefaultPreludeCodegen, collections::array::DefaultArrayCodegen},
    test::{TestContext, exec_ctx},
    utils::{IntOpBuilder, LogicOpBuilder, fat::FatExt},
};
use hugr_core::{
    Hugr,
    builder::{Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder},
    extension::prelude::{UnwrapBuilder, bool_t},
    std_extensions::{
        STD_REG,
        arithmetic::int_types::{ConstInt, int_type},
        ptr::PtrOpBuilder,
    },
    types::Type,
};
use inkwell::types::BasicType;
use rstest::rstest;

fn configure(ctx: &mut TestContext) {
    ctx.add_extensions(|b| {
        b.add_default_prelude_extensions()
            .add_default_int_extensions()
            .add_logic_extensions()
            .add_default_ptr_extensions(DefaultPreludeCodegen, DefaultArrayCodegen)
    });
}

fn lifecycle() -> Hugr {
    let ty = int_type(6);
    SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_outs([bool_t()])
        .finish(|mut b| {
            let one = b.add_load_value(ConstInt::new_u(6, 1).unwrap());
            let two = b.add_load_value(ConstInt::new_u(6, 2).unwrap());
            let three = b.add_load_value(ConstInt::new_u(6, 3).unwrap());
            let ptr = b.add_new_ptr(one).unwrap();
            let (a, other) = b.add_dup_ptr(ptr, ty.clone()).unwrap();
            let a = b.add_write_ptr(a, two).unwrap();
            let (a, read) = b.add_read_ptr(a, ty.clone()).unwrap();
            let (a, old) = b.add_swap_ptr(a, three).unwrap();
            let none = b.add_free_ptr(a, ty.clone()).unwrap();
            b.build_unwrap_sum::<0>(0, option_type([ty.clone()]), none)
                .unwrap();
            let some = b.add_free_ptr(other, ty.clone()).unwrap();
            // The two handles alone do not order their Free operations.
            b.set_order(&none.node(), &some.node());
            let [last] = b.build_unwrap_sum(1, option_type([ty]), some).unwrap();
            let read_ok = b.add_ieq(6, read, two).unwrap();
            let old_ok = b.add_ieq(6, old, two).unwrap();
            let last_ok = b.add_ieq(6, last, three).unwrap();
            let ok = b.add_and(read_ok, old_ok).unwrap();
            let ok = b.add_and(ok, last_ok).unwrap();
            b.finish_hugr_with_outputs([ok]).unwrap()
        })
}

#[rstest]
#[case(false)]
#[case(true)]
fn exec_lifecycle(mut exec_ctx: TestContext, #[case] custom_mutex: bool) {
    configure(&mut exec_ctx);
    if custom_mutex {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(SpinCodegen));
    }
    assert!(exec_ctx.exec_hugr::<bool>(lifecycle(), "main"));
}

#[rstest]
#[case(false)]
#[case(true)]
fn exec_map(mut exec_ctx: TestContext, #[case] extra: bool) {
    configure(&mut exec_ctx);
    let ty = int_type(6);
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_outs([ty.clone()])
        .finish(|mut b| {
            let mut row = vec![ty.clone()];
            if extra {
                row.push(ty.clone());
            }
            let mut mb = b.module_root_builder();
            let mut callback = mb
                .define_function("update", Signature::new_endo(row))
                .unwrap();
            let ins = callback.input_wires().collect::<Vec<_>>();
            let delta = if extra {
                ins[1]
            } else {
                callback.add_load_value(ConstInt::new_u(6, 4).unwrap())
            };
            let next = callback.add_iadd(6, ins[0], delta).unwrap();
            let out = if extra {
                vec![next, ins[0]]
            } else {
                vec![next]
            };
            let callback = callback.finish_with_outputs(out).unwrap();
            let function = b.load_func(callback.handle(), &[]).unwrap();
            let value = b.add_load_value(ConstInt::new_u(6, 10).unwrap());
            let ptr = b.add_new_ptr(value).unwrap();
            let inputs = if extra {
                vec![b.add_load_value(ConstInt::new_u(6, 4).unwrap())]
            } else {
                vec![]
            };
            let outputs = if extra { vec![ty.clone()] } else { vec![] };
            let (ptr, extras) = b
                .add_map_ptr(ptr, function, ty.clone(), inputs, outputs)
                .unwrap();
            let freed = b.add_free_ptr(ptr, ty.clone()).unwrap();
            let [value] = b.build_unwrap_sum(1, option_type([ty]), freed).unwrap();
            let value = if extra {
                b.add_iadd(6, value, extras[0]).unwrap()
            } else {
                value
            };
            b.finish_hugr_with_outputs([value]).unwrap()
        });
    assert_eq!(
        exec_ctx.exec_hugr_u64(hugr, "main"),
        if extra { 24 } else { 14 }
    );
}

#[rstest]
#[case(false)]
#[case(true)]
fn exec_linear_nested_pointer(mut exec_ctx: TestContext, #[case] runtime: bool) {
    configure(&mut exec_ctx);
    if runtime {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    }
    let ty = int_type(6);
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_outs([ty.clone()])
        .finish(|mut b| {
            let value = b.add_load_value(ConstInt::new_u(6, 42).unwrap());
            let inner = b.add_new_ptr(value).unwrap();
            let outer = b.add_new_ptr(inner).unwrap();
            let inner_ty = ptr::ptr_type(ty.clone());
            let (lhs, rhs) = b.add_dup_ptr(outer, inner_ty.clone()).unwrap();
            let (lhs, rhs, equal) = b.add_eq_ptr(lhs, rhs, inner_ty.clone()).unwrap();
            b.build_unwrap_sum::<0>(1, hugr_core::types::SumType::new_unary(2), equal)
                .unwrap();
            let first = b.add_free_ptr(lhs, inner_ty.clone()).unwrap();
            b.build_unwrap_sum::<0>(0, option_type([inner_ty.clone()]), first)
                .unwrap();
            let outer = b.add_free_ptr(rhs, inner_ty.clone()).unwrap();
            b.set_order(&first.node(), &outer.node());
            let [inner] = b
                .build_unwrap_sum(1, option_type([inner_ty]), outer)
                .unwrap();
            let inner = b.add_free_ptr(inner, ty.clone()).unwrap();
            let [value] = b.build_unwrap_sum(1, option_type([ty]), inner).unwrap();
            b.finish_hugr_with_outputs([value]).unwrap()
        });
    assert_eq!(
        if runtime {
            exec_runtime_u64(&exec_ctx, &hugr)
        } else {
            exec_ctx.exec_hugr_u64(hugr, "main")
        },
        42
    );
}

#[rstest]
#[case(false)]
#[case(true)]
fn exec_zero_sized_value(mut exec_ctx: TestContext, #[case] runtime: bool) {
    configure(&mut exec_ctx);
    if runtime {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    }
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .finish(|mut b| {
            let unit = b.make_tuple([]).unwrap();
            let ptr = b.add_new_ptr(unit).unwrap();
            let freed = b.add_free_ptr(ptr, Type::UNIT).unwrap();
            b.build_unwrap_sum::<1>(1, option_type([Type::UNIT]), freed)
                .unwrap();
            b.finish_hugr_with_outputs([]).unwrap()
        });
    if runtime {
        with_runtime_engine(&exec_ctx, &hugr, |engine| {
            // This entry has no arguments or results.
            unsafe {
                engine
                    .get_function::<unsafe extern "C" fn()>("main")
                    .unwrap()
                    .call()
            }
        });
    } else {
        exec_ctx.exec_hugr::<()>(hugr, "main");
    }
}

fn emit<'c>(ctx: &'c TestContext, hugr: &'c Hugr) -> Emission<'c> {
    let emission = Emission::emit_hugr(
        hugr.fat_root().unwrap(),
        ctx.get_emit_hugr(),
        EmitDebugInfo::Exclude,
    )
    .unwrap();
    emission.verify().unwrap();
    emission
}

#[rstest]
#[case(false)]
#[case(true)]
fn exec_map_linear_payload_and_extra(mut exec_ctx: TestContext, #[case] runtime: bool) {
    configure(&mut exec_ctx);
    if runtime {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    }
    let int = int_type(6);
    let inner = ptr::ptr_type(int.clone());
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_outs([int.clone()])
        .finish(|mut b| {
            let mut mb = b.module_root_builder();
            let callback = mb
                .define_function(
                    "exchange",
                    Signature::new_endo([inner.clone(), inner.clone()]),
                )
                .unwrap();
            let [old, new] = callback.input_wires_arr();
            let callback = callback.finish_with_outputs([new, old]).unwrap();
            let function = b.load_func(callback.handle(), &[]).unwrap();
            let a = b.add_load_value(ConstInt::new_u(6, 10).unwrap());
            let c = b.add_load_value(ConstInt::new_u(6, 20).unwrap());
            let a = b.add_new_ptr(a).unwrap();
            let c = b.add_new_ptr(c).unwrap();
            let outer = b.add_new_ptr(a).unwrap();
            let (outer, extra) = b
                .add_map_ptr(outer, function, inner.clone(), [c], [inner.clone()])
                .unwrap();
            let old = b.add_free_ptr(extra[0], int.clone()).unwrap();
            let [old] = b
                .build_unwrap_sum(1, option_type([int.clone()]), old)
                .unwrap();
            let outer = b.add_free_ptr(outer, inner.clone()).unwrap();
            let [current] = b.build_unwrap_sum(1, option_type([inner]), outer).unwrap();
            let current = b.add_free_ptr(current, int.clone()).unwrap();
            let [current] = b.build_unwrap_sum(1, option_type([int]), current).unwrap();
            let sum = b.add_iadd(6, old, current).unwrap();
            b.finish_hugr_with_outputs([sum]).unwrap()
        });
    assert_eq!(
        if runtime {
            exec_runtime_u64(&exec_ctx, &hugr)
        } else {
            exec_ctx.exec_hugr_u64(hugr, "main")
        },
        30
    );
}

#[rstest]
fn custom_mutex_makes_concurrent_map_exclusive(mut exec_ctx: TestContext) {
    use std::{cell::UnsafeCell, sync::atomic::AtomicU8};
    configure(&mut exec_ctx);
    exec_ctx.add_extensions(|b| b.add_ptr_extensions(SpinCodegen));
    let ty = int_type(6);
    let ptr_ty = ptr::ptr_type(ty.clone());
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_ins([ptr_ty.clone()])
        .with_outs([ptr_ty])
        .finish(|mut b| {
            let mut mb = b.module_root_builder();
            let mut callback = mb
                .define_function("increment", Signature::new_endo([ty.clone()]))
                .unwrap();
            let [value] = callback.input_wires_arr();
            let one = callback.add_load_value(ConstInt::new_u(6, 1).unwrap());
            let value = callback.add_iadd(6, value, one).unwrap();
            let callback = callback.finish_with_outputs([value]).unwrap();
            let function = b.load_func(callback.handle(), &[]).unwrap();
            let [ptr] = b.input_wires_arr();
            let (ptr, _) = b.add_map_ptr(ptr, function, ty, [], []).unwrap();
            b.finish_hugr_with_outputs([ptr]).unwrap()
        });
    let emission = emit(&exec_ctx, &hugr);
    let function = emission.module().get_function("main").unwrap();
    function.set_linkage(inkwell::module::Linkage::External);
    let engine = emission
        .module()
        .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
        .unwrap();
    let address = engine.get_function_address("main").unwrap();
    // Mirror the test spin-mutex cell layout. Each worker owns one of four handles;
    // storage is owned by this test and stays alive until every worker joins.
    #[repr(C)]
    struct Cell {
        mutex: AtomicU8,
        count: u64,
        locked: u8,
        value: UnsafeCell<u64>,
    }
    let cell = Cell {
        count: 4,
        locked: 0,
        mutex: AtomicU8::new(0),
        value: UnsafeCell::new(0),
    };
    let ptr = (&cell as *const Cell) as usize;
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(move || {
                // The emitted function takes and returns an opaque cell pointer.
                // Its mutex protects the only accesses to value while threads run.
                let map: unsafe extern "C" fn(usize) -> usize =
                    unsafe { std::mem::transmute(address) };
                for _ in 0..2000 {
                    assert_eq!(unsafe { map(ptr) }, ptr);
                }
            });
        }
    });
    // All workers have joined, so no concurrent access to the value remains.
    assert_eq!(unsafe { *cell.value.get() }, 8000);
    assert_eq!(cell.count, 4);
}

#[derive(Clone)]
struct HookCodegen;

fn hook_void<H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<H>,
    name: &str,
    ptr: PointerValue,
) -> Result<()> {
    let function = ctx.get_extern_func(
        name,
        ctx.iw_context()
            .void_type()
            .fn_type(&[ctx.llvm_ptr_type().into()], false),
    )?;
    ctx.builder().build_call(function, &[ptr.into()], "")?;
    Ok(())
}

impl PtrCodegen for HookCodegen {
    fn emit_new<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let ty = value.get_type();
        let source = value_buffer(ctx, ty, "test.source")?;
        ctx.builder().build_store(source, value)?;
        let i64_t = ctx.iw_context().i64_type();
        let function = ctx.get_extern_func(
            "test_ptr_create",
            ctx.llvm_ptr_type().fn_type(
                &[i64_t.into(), i64_t.into(), ctx.llvm_ptr_type().into()],
                false,
            ),
        )?;
        Ok(ctx
            .builder()
            .build_call(
                function,
                &[
                    ty.size_of().unwrap().into(),
                    ty.get_alignment().into(),
                    source.into(),
                ],
                "",
            )?
            .try_as_basic_value()
            .unwrap_basic()
            .into_pointer_value())
    }

    fn emit_dup<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        _value_ty: BasicTypeEnum<'c>,
    ) -> Result<()> {
        hook_void(ctx, "test_ptr_dup", ptr)
    }

    fn emit_free<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        _value_ty: BasicTypeEnum<'c>,
        destination: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        let function = ctx.get_extern_func(
            "test_ptr_release",
            ctx.iw_context().bool_type().fn_type(
                &[ctx.llvm_ptr_type().into(), ctx.llvm_ptr_type().into()],
                false,
            ),
        )?;
        Ok(ctx
            .builder()
            .build_call(function, &[ptr.into(), destination.into()], "test.final")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value())
    }

    fn emit_get_ptr<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        _value_ty: BasicTypeEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let function = ctx.get_extern_func(
            "test_ptr_get_ptr",
            ctx.llvm_ptr_type()
                .fn_type(&[ctx.llvm_ptr_type().into()], false),
        )?;
        Ok(ctx
            .builder()
            .build_call(function, &[ptr.into()], "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_pointer_value())
    }

    fn emit_lock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        hook_void(ctx, "test_ptr_lock", ptr)
    }

    fn emit_unlock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        hook_void(ctx, "test_ptr_unlock", ptr)
    }
}

struct RuntimeAllocation {
    layout: std::alloc::Layout,
    offset: usize,
    tail: usize,
    size: usize,
    refs: u64,
    locked: bool,
}
const GUARD: u64 = 0x1234_5678_9abc_def0;
impl RuntimeAllocation {
    fn check_guards(&self, ptr: *mut u8) {
        // These aligned guard words are initialized on allocation and stay live
        // until final release. They surround the value-only payload.
        assert_eq!(unsafe { ptr.cast::<u64>().read() }, GUARD);
        assert_eq!(unsafe { ptr.add(self.tail).cast::<u64>().read() }, GUARD);
    }
}
thread_local! {
    static EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
    // The count lives only in Rust metadata, completely outside the LLVM payload.
    static ALLOCATIONS: std::cell::RefCell<std::collections::BTreeMap<usize, RuntimeAllocation>> = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}
fn event(name: &'static str) {
    EVENTS.with_borrow_mut(|events| events.push(name));
}
extern "C" fn hook_create(size: u64, align: u64, source: *const u8) -> *mut u8 {
    let value = std::alloc::Layout::from_size_align(size as usize, align as usize).unwrap();
    let (layout, offset) = std::alloc::Layout::new::<u64>().extend(value).unwrap();
    let (layout, tail) = layout.extend(std::alloc::Layout::new::<u64>()).unwrap();
    // The JIT supplies a typed, aligned source, and creation transfers its value.
    // Raw byte copying preserves any uninitialized padding without inspecting it.
    let ptr = unsafe { std::alloc::alloc(layout) };
    assert!(!ptr.is_null());
    unsafe {
        ptr.cast::<u64>().write(GUARD);
        ptr.add(tail).cast::<u64>().write(GUARD);
        if size != 0 {
            std::ptr::copy_nonoverlapping(source, ptr.add(offset), size as usize);
        }
    }
    let old = ALLOCATIONS.with_borrow_mut(|allocs| {
        allocs.insert(
            ptr as usize,
            RuntimeAllocation {
                layout,
                offset,
                tail,
                size: size as usize,
                refs: 1,
                locked: false,
            },
        )
    });
    assert!(old.is_none());
    event("new");
    ptr
}
extern "C" fn hook_dup(ptr: *mut u8) {
    ALLOCATIONS.with_borrow_mut(|allocs| {
        let allocation = allocs.get_mut(&(ptr as usize)).unwrap();
        allocation.check_guards(ptr);
        assert!(!allocation.locked, "Dup entered with payload lock held");
        allocation.refs = allocation.refs.checked_add(1).unwrap();
    });
    event("dup");
}
extern "C" fn hook_release(ptr: *mut u8, destination: *mut u8) -> bool {
    let last = ALLOCATIONS.with_borrow_mut(|allocs| {
        let allocation = allocs.get_mut(&(ptr as usize)).unwrap();
        allocation.check_guards(ptr);
        assert!(!allocation.locked, "Free entered with payload lock held");
        allocation.refs = allocation.refs.checked_sub(1).unwrap();
        if allocation.refs == 0 {
            allocs.remove(&(ptr as usize))
        } else {
            None
        }
    });
    let Some(allocation) = last else {
        event("release.shared");
        return false;
    };
    assert!(!destination.is_null());
    // Move the final value into the disjoint typed destination before destroying
    // storage. There is no count in this payload and no destructor to invoke.
    unsafe {
        if allocation.size != 0 {
            std::ptr::copy_nonoverlapping(ptr.add(allocation.offset), destination, allocation.size);
        }
    }
    event("extract");
    event("destroy");
    unsafe {
        std::alloc::dealloc(ptr, allocation.layout);
    }
    event("free");
    true
}
extern "C" fn hook_get_ptr(ptr: *mut u8) -> *mut u8 {
    let offset = ALLOCATIONS.with_borrow(|allocs| {
        let allocation = &allocs[&(ptr as usize)];
        allocation.check_guards(ptr);
        assert!(allocation.locked, "Projection outside payload lock");
        allocation.offset
    });
    event("get_ptr");
    // The value's aligned storage lies between the two runtime guard words.
    unsafe { ptr.add(offset) }
}
extern "C" fn hook_lock(ptr: *mut u8) {
    ALLOCATIONS.with_borrow_mut(|allocs| {
        let allocation = allocs.get_mut(&(ptr as usize)).unwrap();
        allocation.check_guards(ptr);
        assert!(!allocation.locked, "Recursive payload lock");
        allocation.locked = true;
    });
    event("lock");
}
extern "C" fn hook_unlock(ptr: *mut u8) {
    ALLOCATIONS.with_borrow_mut(|allocs| {
        let allocation = allocs.get_mut(&(ptr as usize)).unwrap();
        allocation.check_guards(ptr);
        assert!(
            allocation.locked,
            "Unlock without a lock or after final release"
        );
        allocation.locked = false;
    });
    event("unlock");
}

fn with_runtime_engine<R>(
    ctx: &TestContext,
    hugr: &Hugr,
    run: impl FnOnce(&inkwell::execution_engine::ExecutionEngine<'_>) -> R,
) -> R {
    let emission = emit(ctx, hugr);
    emission
        .module()
        .get_function("main")
        .unwrap()
        .set_linkage(inkwell::module::Linkage::External);
    let engine = emission
        .module()
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .unwrap();
    let ir = emission.module().print_to_string().to_string();
    assert!(!ir.contains("ptr.count"));
    assert!(!ir.contains("ptr.refcount.overflow"));
    assert!(!ir.contains("ptr.last"));
    for (name, address) in [
        ("test_ptr_create", hook_create as *const () as usize),
        ("test_ptr_dup", hook_dup as *const () as usize),
        ("test_ptr_release", hook_release as *const () as usize),
        ("test_ptr_get_ptr", hook_get_ptr as *const () as usize),
        ("test_ptr_lock", hook_lock as *const () as usize),
        ("test_ptr_unlock", hook_unlock as *const () as usize),
    ] {
        if let Some(function) = emission.module().get_function(name) {
            engine.add_global_mapping(&function, address);
        }
    }
    EVENTS.with_borrow_mut(Vec::clear);
    ALLOCATIONS.with_borrow(|allocs| assert!(allocs.is_empty()));
    let result = run(&engine);
    ALLOCATIONS.with_borrow(|allocs| assert!(allocs.is_empty()));
    result
}
fn exec_runtime_bool(ctx: &TestContext, hugr: &Hugr) -> bool {
    with_runtime_engine(ctx, hugr, |engine| {
        // This test entry has no arguments and returns a HUGR boolean.
        unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> bool>("main")
                .unwrap()
                .call()
        }
    })
}
fn exec_runtime_u64(ctx: &TestContext, hugr: &Hugr) -> u64 {
    with_runtime_engine(ctx, hugr, |engine| {
        // This test entry has no arguments and returns a 64-bit integer.
        unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> u64>("main")
                .unwrap()
                .call()
        }
    })
}

#[rstest]
fn runtime_backend_owns_count_and_lifecycle(mut exec_ctx: TestContext) {
    configure(&mut exec_ctx);
    exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    assert!(exec_runtime_bool(&exec_ctx, &lifecycle()));
    EVENTS.with_borrow(|events| {
        assert_eq!(
            events,
            &[
                "new",
                "dup",
                "lock",
                "get_ptr",
                "unlock",
                "lock",
                "get_ptr",
                "unlock",
                "lock",
                "get_ptr",
                "unlock",
                "release.shared",
                "extract",
                "destroy",
                "free",
            ]
        )
    });
}

#[derive(Clone)]
struct SpinCodegen;
impl PtrCodegen for SpinCodegen {
    fn emit_new<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        value: BasicValueEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let layout = counted_layout(ctx.iw_context(), value.get_type());
        let storage = ctx
            .iw_context()
            .struct_type(&[ctx.iw_context().i8_type().into(), layout.into()], false);
        let ptr = emit_alloc_checked(
            ctx,
            &DefaultArrayCodegen,
            &DefaultPreludeCodegen,
            &ConstError::new(2, "Pointer allocation failed"),
            storage,
        )?;
        ctx.builder()
            .build_store(ptr, ctx.iw_context().i8_type().const_zero())?;
        let payload = spin_payload(ctx, ptr, value.get_type())?;
        emit_counted_init(ctx, payload, value)?;
        Ok(ptr)
    }

    fn emit_dup<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<()> {
        self.emit_lock(ctx, ptr)?;
        let payload = spin_payload(ctx, ptr, value_ty)?;
        let (count, _) = counted_fields(ctx, payload, value_ty)?;
        emit_counted_dup(
            ctx,
            &DefaultPreludeCodegen,
            &ConstError::new(2, "Pointer reference count overflow"),
            count,
        )?;
        self.emit_unlock(ctx, ptr)
    }

    fn emit_free<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
        destination: PointerValue<'c>,
    ) -> Result<IntValue<'c>> {
        self.emit_lock(ctx, ptr)?;
        let payload = spin_payload(ctx, ptr, value_ty)?;
        let (count, value) = counted_fields(ctx, payload, value_ty)?;
        let last = emit_counted_take(ctx, count, value, value_ty, destination)?;
        self.emit_unlock(ctx, ptr)?;
        emit_free_if_last(ctx, &DefaultArrayCodegen, ptr, last)?;
        Ok(last)
    }

    fn emit_get_ptr<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
        value_ty: BasicTypeEnum<'c>,
    ) -> Result<PointerValue<'c>> {
        let payload = spin_payload(ctx, ptr, value_ty)?;
        Ok(counted_fields(ctx, payload, value_ty)?.1)
    }

    fn emit_lock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        mutex: PointerValue<'c>,
    ) -> Result<()> {
        let spin = ctx.new_basic_block("test.lock", None);
        let acquired = ctx.new_basic_block("test.locked", None);
        ctx.builder().build_unconditional_branch(spin)?;
        ctx.builder().position_at_end(spin);
        let byte = ctx.iw_context().i8_type();
        let old = ctx.builder().build_atomicrmw(
            inkwell::AtomicRMWBinOp::Xchg,
            mutex,
            byte.const_int(1, false),
            inkwell::AtomicOrdering::Acquire,
        )?;
        let available =
            ctx.builder()
                .build_int_compare(IntPredicate::EQ, old, byte.const_zero(), "")?;
        ctx.builder()
            .build_conditional_branch(available, acquired, spin)?;
        ctx.builder().position_at_end(acquired);
        Ok(())
    }
    fn emit_unlock<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        mutex: PointerValue<'c>,
    ) -> Result<()> {
        ctx.builder()
            .build_store(mutex, ctx.iw_context().i8_type().const_zero())?
            .set_atomic_ordering(inkwell::AtomicOrdering::Release)
            .unwrap();
        Ok(())
    }
}

fn spin_payload<'c, H: HugrView<Node = Node>>(
    ctx: &mut EmitFuncContext<'c, '_, H>,
    ptr: PointerValue<'c>,
    value_ty: BasicTypeEnum<'c>,
) -> Result<PointerValue<'c>> {
    let layout = counted_layout(ctx.iw_context(), value_ty);
    let storage = ctx
        .iw_context()
        .struct_type(&[ctx.iw_context().i8_type().into(), layout.into()], false);
    Ok(ctx
        .builder()
        .build_struct_gep(storage, ptr, 1, "test.payload")?)
}

#[rstest]
fn default_lock_checks_are_non_atomic(mut exec_ctx: TestContext) {
    configure(&mut exec_ctx);
    let hugr = lifecycle();
    let emission = emit(&exec_ctx, &hugr);
    let ir = emission.module().print_to_string().to_string();
    assert!(ir.contains("@malloc("));
    assert!(ir.contains("@free("));
    assert!(!ir.contains("atomic"));
    assert!(!ir.contains("@aligned_alloc("));
}

#[rstest]
#[case(false, 0)]
#[case(true, 0)]
#[case(false, 1)]
#[case(true, 1)]
#[case(false, 2)]
#[case(true, 2)]
fn exec_eq_identity_and_handles(
    mut exec_ctx: TestContext,
    #[case] aliases: bool,
    #[case] backend: u8,
) {
    configure(&mut exec_ctx);
    if backend == 1 {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(SpinCodegen));
    }
    if backend == 2 {
        exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    }
    let ty = int_type(6);
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_outs([bool_t()])
        .finish(|mut b| {
            let seven = b.add_load_value(ConstInt::new_u(6, 7).unwrap());
            let eleven = b.add_load_value(ConstInt::new_u(6, 11).unwrap());
            let lhs = b.add_new_ptr(seven).unwrap();
            let (lhs, rhs) = if aliases {
                b.add_dup_ptr(lhs, ty.clone()).unwrap()
            } else {
                (lhs, b.add_new_ptr(seven).unwrap())
            };
            let (lhs, lhs_witness) = b.add_dup_ptr(lhs, ty.clone()).unwrap();
            let (rhs, rhs_witness) = b.add_dup_ptr(rhs, ty.clone()).unwrap();
            let (lhs, rhs, equal) = b.add_eq_ptr(lhs, rhs, ty.clone()).unwrap();
            // Independent aliases pin the identity of each output, detecting an
            // accidental output swap even when both cells have equal payloads.
            let (lhs, lhs_witness, lhs_preserved) =
                b.add_eq_ptr(lhs, lhs_witness, ty.clone()).unwrap();
            let (rhs, rhs_witness, rhs_preserved) =
                b.add_eq_ptr(rhs, rhs_witness, ty.clone()).unwrap();
            let identity_ok = if aliases {
                equal
            } else {
                b.add_not(equal).unwrap()
            };
            // Writing through the first result must preserve the second handle's
            // identity. Distinct cells initially contain equal payloads.
            let lhs = b.add_write_ptr(lhs, eleven).unwrap();
            let (rhs, value) = b.add_read_ptr(rhs, ty.clone()).unwrap();
            b.set_order(&lhs.node(), &rhs.node());
            let expected = if aliases { eleven } else { seven };
            let value_ok = b.add_ieq(6, value, expected).unwrap();
            let lhs_witness = b.add_free_ptr(lhs_witness, ty.clone()).unwrap();
            let rhs_witness = b.add_free_ptr(rhs_witness, ty.clone()).unwrap();
            for witness in [lhs_witness, rhs_witness] {
                b.build_unwrap_sum::<0>(0, option_type([ty.clone()]), witness)
                    .unwrap();
            }
            let first = b.add_free_ptr(lhs, ty.clone()).unwrap();
            let last = b.add_free_ptr(rhs, ty.clone()).unwrap();
            b.set_order(&lhs_witness.node(), &first.node());
            b.set_order(&rhs_witness.node(), &first.node());
            b.set_order(&first.node(), &last.node());
            let mut ok = b.add_and(identity_ok, value_ok).unwrap();
            ok = b.add_and(ok, lhs_preserved).unwrap();
            ok = b.add_and(ok, rhs_preserved).unwrap();
            if aliases {
                b.build_unwrap_sum::<0>(0, option_type([ty.clone()]), first)
                    .unwrap();
            } else {
                let [value] = b
                    .build_unwrap_sum(1, option_type([ty.clone()]), first)
                    .unwrap();
                let first_ok = b.add_ieq(6, value, eleven).unwrap();
                ok = b.add_and(ok, first_ok).unwrap();
            }
            let [value] = b.build_unwrap_sum(1, option_type([ty]), last).unwrap();
            let last_ok = b.add_ieq(6, value, expected).unwrap();
            ok = b.add_and(ok, last_ok).unwrap();
            b.finish_hugr_with_outputs([ok]).unwrap()
        });
    assert!(if backend == 2 {
        exec_runtime_bool(&exec_ctx, &hugr)
    } else {
        exec_ctx.exec_hugr::<bool>(hugr, "main")
    });
}

#[rstest]
fn eq_emits_no_runtime_hooks_or_payload_access(mut exec_ctx: TestContext) {
    configure(&mut exec_ctx);
    exec_ctx.add_extensions(|b| b.add_ptr_extensions(HookCodegen));
    // A nested pointer is a linear payload; Eq needs no lowering of its contents.
    let ty = ptr::ptr_type(int_type(6));
    let pointer = ptr::ptr_type(ty.clone());
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .with_ins([pointer.clone(), pointer.clone()])
        .with_outs([pointer.clone(), pointer, bool_t()])
        .finish(|mut b| {
            let [lhs, rhs] = b.input_wires_arr();
            let (lhs, rhs, equal) = b.add_eq_ptr(lhs, rhs, ty).unwrap();
            b.finish_hugr_with_outputs([lhs, rhs, equal]).unwrap()
        });
    let emission = emit(&exec_ctx, &hugr);
    let ir = emission.module().print_to_string().to_string();
    assert!(ir.contains("icmp eq ptr"));
    assert!(!ir.contains("@test_ptr_"));
    assert!(!ir.contains("getelementptr"));
    assert!(!ir.contains("atomic"));
}

#[derive(Clone)]
struct TestHeap;
impl ArrayCodegen for TestHeap {
    fn emit_allocate_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        size: IntValue<'c>,
    ) -> Result<PointerValue<'c>> {
        let f = ctx.get_extern_func(
            "test_heap_alloc",
            ctx.llvm_ptr_type()
                .fn_type(&[size.get_type().into()], false),
        )?;
        Ok(ctx
            .builder()
            .build_call(f, &[size.into()], "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_pointer_value())
    }
    fn emit_free_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        hook_void(ctx, "test_heap_free", ptr)
    }
}

thread_local! { static HEAP_EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) }; }
unsafe extern "C" {
    fn malloc(size: usize) -> *mut std::ffi::c_void;
    fn free(ptr: *mut std::ffi::c_void);
}
extern "C" fn test_heap_alloc(size: usize) -> *mut std::ffi::c_void {
    HEAP_EVENTS.with_borrow_mut(|events| events.push("allocate"));
    // Forward the emitted allocation size to the platform allocator.
    unsafe { malloc(size) }
}
extern "C" fn test_heap_free(ptr: *mut std::ffi::c_void) {
    HEAP_EVENTS.with_borrow_mut(|events| events.push("free"));
    // The lowering releases exactly the live allocation returned above.
    unsafe { free(ptr) }
}

#[rstest]
fn configured_heap_owns_allocation_and_final_free(mut exec_ctx: TestContext) {
    configure(&mut exec_ctx);
    exec_ctx.add_extensions(|b| b.add_default_ptr_extensions(DefaultPreludeCodegen, TestHeap));
    let hugr = lifecycle();
    let emission = emit(&exec_ctx, &hugr);
    emission
        .module()
        .get_function("main")
        .unwrap()
        .set_linkage(inkwell::module::Linkage::External);
    let engine = emission
        .module()
        .create_jit_execution_engine(inkwell::OptimizationLevel::None)
        .unwrap();
    engine.add_global_mapping(
        &emission.module().get_function("test_heap_alloc").unwrap(),
        test_heap_alloc as *const () as usize,
    );
    engine.add_global_mapping(
        &emission.module().get_function("test_heap_free").unwrap(),
        test_heap_free as *const () as usize,
    );
    HEAP_EVENTS.with_borrow_mut(Vec::clear);
    // main has no inputs and returns the HUGR boolean success witness.
    let result = unsafe {
        engine
            .get_function::<unsafe extern "C" fn() -> bool>("main")
            .unwrap()
            .call()
    };
    assert!(result);
    HEAP_EVENTS.with_borrow(|events| assert_eq!(events, &["allocate", "free"]));
}

// Stack storage avoids leaking a heap allocation when the panic test jumps out.
#[derive(Clone)]
struct PanicHeap {
    fail: bool,
}
impl ArrayCodegen for PanicHeap {
    fn emit_allocate_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        size: IntValue<'c>,
    ) -> Result<PointerValue<'c>> {
        if self.fail {
            return Ok(ctx.llvm_ptr_type().const_null());
        }
        Ok(ctx
            .builder()
            .build_array_alloca(ctx.iw_context().i64_type(), size, "test.storage")?)
    }
    fn emit_free_array<'c, H: HugrView<Node = Node>>(
        &self,
        _ctx: &mut EmitFuncContext<'c, '_, H>,
        _ptr: PointerValue<'c>,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct SignalCheckingPrelude {
    expected: u32,
}
impl PreludeCodegen for SignalCheckingPrelude {
    fn emit_panic<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        error: BasicValueEnum,
    ) -> Result<()> {
        use crate::emit::test::PanicTestPreludeCodegen;
        let signal = ctx
            .builder()
            .build_extract_value(error.into_struct_value(), 0, "test.signal")?
            .into_int_value();
        let mismatch = ctx.builder().build_int_compare(
            IntPredicate::NE,
            signal,
            signal.get_type().const_int(self.expected.into(), false),
            "",
        )?;
        panic_if(
            ctx,
            &PanicTestPreludeCodegen,
            mismatch,
            &ConstError::new(99, "Unexpected pointer error signal"),
        )?;
        PanicTestPreludeCodegen.emit_panic(ctx, error)
    }
}

#[rstest]
#[case(0, "Pointer cell is already locked")]
#[case(1, "Pointer cell is not locked")]
#[case(2, "Pointer cell is not locked")]
#[case(3, "Pointer allocation failed")]
#[case(4, "Pointer reference count overflow")]
fn configured_pointer_errors_panic(
    mut exec_ctx: TestContext,
    #[case] misuse: u8,
    #[case] message: &str,
    #[values(false, true)] configured: bool,
) {
    configure(&mut exec_ctx);
    let expected = if configured {
        match misuse {
            0 => 42,
            1 | 2 => 43,
            3 => 40,
            _ => 41,
        }
    } else {
        2
    };
    exec_ctx.add_extensions(move |b| {
        b.add_prelude_extensions(SignalCheckingPrelude { expected })
            .simple_extension_op::<PtrOpDef>(move |ctx, args, _| {
                let op = PtrOp::from_extension_op(args.node().as_ref())?;
                let mut cg = DefaultPtrCodegen::new(
                    SignalCheckingPrelude { expected },
                    PanicHeap { fail: misuse == 3 },
                );
                if configured {
                    cg = cg
                        .with_allocation_error(ConstError::new(
                            40,
                            "custom Pointer allocation failed",
                        ))
                        .with_refcount_overflow_error(ConstError::new(
                            41,
                            "custom Pointer reference count overflow",
                        ))
                        .with_already_locked_error(ConstError::new(
                            42,
                            "custom Pointer cell is already locked",
                        ))
                        .with_not_locked_error(ConstError::new(
                            43,
                            "custom Pointer cell is not locked",
                        ));
                }
                if op.def == PtrOpDef::New {
                    let ptr = cg.emit_new(ctx, args.inputs[0])?;
                    match misuse {
                        0 => {
                            cg.emit_lock(ctx, ptr)?;
                            cg.emit_lock(ctx, ptr)?;
                        }
                        1 => cg.emit_unlock(ctx, ptr)?,
                        2 => {
                            cg.emit_lock(ctx, ptr)?;
                            cg.emit_unlock(ctx, ptr)?;
                            cg.emit_unlock(ctx, ptr)?;
                        }
                        4 => {
                            let (count, _) = counted_fields(ctx, ptr, args.inputs[0].get_type())?;
                            ctx.builder()
                                .build_store(count, ctx.iw_context().i64_type().const_all_ones())?;
                            cg.emit_dup(ctx, ptr, args.inputs[0].get_type())?;
                        }
                        _ => {}
                    }
                    args.outputs.finish(ctx.builder(), [ptr.into()])
                } else {
                    emit_ptr_op(&cg, ctx, op, args.inputs, args.outputs)
                }
            })
    });
    let expected_message = if configured {
        format!("custom {message}")
    } else {
        message.to_owned()
    };
    assert_eq!(
        exec_ctx.exec_hugr_panicking(lifecycle(), "main"),
        expected_message
    );
}

#[rstest]
fn reentrant_map_access_panics(mut exec_ctx: TestContext) {
    use crate::emit::test::PanicTestPreludeCodegen;
    configure(&mut exec_ctx);
    exec_ctx.add_extensions(|b| {
        b.add_prelude_extensions(PanicTestPreludeCodegen)
            .add_default_ptr_extensions(PanicTestPreludeCodegen, PanicHeap { fail: false })
    });
    let int = int_type(6);
    let ptr_ty = ptr::ptr_type(int.clone());
    let hugr = SimpleHugrConfig::new()
        .with_extensions(STD_REG.to_owned())
        .finish(|mut b| {
            let mut mb = b.module_root_builder();
            let mut callback = mb
                .define_function(
                    "reenter",
                    Signature::new_endo([int.clone(), ptr_ty.clone()]),
                )
                .unwrap();
            let [value, alias] = callback.input_wires_arr();
            let (alias, _) = callback.add_read_ptr(alias, int.clone()).unwrap();
            let callback = callback.finish_with_outputs([value, alias]).unwrap();
            let function = b.load_func(callback.handle(), &[]).unwrap();
            let value = b.add_load_value(ConstInt::new_u(6, 42).unwrap());
            let handle = b.add_new_ptr(value).unwrap();
            let (handle, alias) = b.add_dup_ptr(handle, int.clone()).unwrap();
            let (handle, aliases) = b
                .add_map_ptr(handle, function, int.clone(), [alias], [ptr_ty])
                .unwrap();
            let a = b.add_free_ptr(handle, int.clone()).unwrap();
            let c = b.add_free_ptr(aliases[0], int).unwrap();
            b.set_order(&a.node(), &c.node());
            b.finish_hugr_with_outputs([]).unwrap()
        });
    assert_eq!(
        exec_ctx.exec_hugr_panicking(hugr, "main"),
        "Pointer cell is already locked"
    );
}

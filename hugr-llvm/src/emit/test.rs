use inkwell::execution_engine::ExecutionEngine;
use std::ffi::CStr;

use crate::emit::EmitFuncContext;
use crate::extension::PreludeCodegen;
use crate::types::HugrFuncType;
use crate::utils::fat::FatNode;
use anyhow::{Result, anyhow};
use hugr_core::builder::{BuildHandle, DFGWrapper, FunctionBuilder};
use hugr_core::extension::ExtensionRegistry;
use hugr_core::ops::handle::FuncID;
use hugr_core::types::TypeRow;
use hugr_core::{Hugr, HugrView, Node};
use inkwell::AddressSpace;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassManager;
use inkwell::values::{BasicValueEnum, GenericValue, GlobalValue, PointerValue};

use super::EmitHugr;

mod panic_runtime;

#[allow(clippy::upper_case_acronyms)]
pub type DFGW = DFGWrapper<Hugr, BuildHandle<FuncID<true>>>;

pub struct SimpleHugrConfig {
    ins: TypeRow,
    outs: TypeRow,
    extensions: ExtensionRegistry,
}

/// A wrapper for a module into which our tests will emit hugr.
pub struct Emission<'c> {
    module: Module<'c>,
}

impl<'c> Emission<'c> {
    /// Create an `Emission` from a HUGR.
    pub fn emit_hugr<'a: 'c, H: HugrView<Node = Node>>(
        hugr: FatNode<'c, hugr_core::ops::Module, H>,
        eh: EmitHugr<'c, 'a, H>,
    ) -> Result<Self> where {
        let module = eh.emit_module(hugr)?.finish();
        Ok(Self { module })
    }

    /// Create an `Emission` from an LLVM Module.
    pub fn new(module: Module<'c>) -> Self {
        Self { module }
    }

    // Verify the inner Module.
    pub fn verify(&self) -> Result<()> {
        self.module
            .verify()
            .map_err(|err| anyhow!("Failed to verify module: {err}"))
    }

    /// Return the inner module.
    pub fn module(&self) -> &Module<'c> {
        &self.module
    }

    /// Run passes on the inner module.
    pub fn opt(&self, go: impl FnOnce() -> PassManager<Module<'c>>) {
        go().run_on(&self.module);
    }

    // Print the inner module to stderr.
    pub fn print_module(&self) {
        self.module.print_to_stderr();
    }

    /// JIT and execute the function named `entry` in the inner module.
    ///
    /// That function must take no arguments and return an `i64`.
    pub fn exec_u64(&self, entry: impl AsRef<str>) -> Result<u64> {
        let gv = self.exec_impl(entry)?;
        Ok(gv.as_int(false))
    }

    /// JIT and execute the function named `entry` in the inner module.
    ///
    /// That function must take no arguments and return an `i64`.
    pub fn exec_i64(&self, entry: impl AsRef<str>) -> Result<i64> {
        let gv = self.exec_impl(entry)?;
        Ok(gv.as_int(true) as i64)
    }

    /// JIT and execute the function named `entry` in the inner module.
    ///
    /// That function must take no arguments and return an `f64`.
    pub fn exec_f64(&self, entry: impl AsRef<str>) -> Result<f64> {
        let gv = self.exec_impl(entry)?;
        Ok(gv.as_float(&self.module.get_context().f64_type()))
    }

    pub(crate) fn exec_impl(&self, entry: impl AsRef<str>) -> Result<GenericValue<'c>> {
        let entry_fv = self
            .module
            .get_function(entry.as_ref())
            .ok_or_else(|| anyhow!("Function {} not found in module", entry.as_ref()))?;

        entry_fv.set_linkage(inkwell::module::Linkage::External);

        let ee = self
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .map_err(|err| anyhow!("Failed to create execution engine: {err}"))?;
        let fv = ee.get_function_value(entry.as_ref())?;
        Ok(unsafe { ee.run_function(fv, &[]) })
    }

    /// JIT and execute the function named `entry` in the inner module.
    ///
    /// Safely handles panics ocurring in the program and returns the produced
    /// panic message, or an empty string if no panic ocurred.
    ///
    /// For this to work, [`Emission::exec_panicking`] must be used together with the
    /// [`PanicTestPreludeCodegen`].
    pub fn exec_panicking(&self, entry: impl AsRef<str>) -> Result<String> {
        // The default lowering for `panic` ops in `DefaultPreludeCodegen` just aborts.
        // This kills the entire process and cannot be caught, so we have to use something
        // else when trying to test panic behaviour. Unfortunately, we also cannot rely
        // on Rust's builtin panic mechanism since we cannot unwind through JIT frames.
        // Thus, we need to roll our own panic runtime for testing. See the `panic_runtime`
        // module for details.
        let entry_fv = self
            .module
            .get_function(entry.as_ref())
            .ok_or_else(|| anyhow!("Function {} not found in module", entry.as_ref()))?;
        entry_fv.set_linkage(inkwell::module::Linkage::External);
        assert_eq!(
            entry_fv.get_type().count_param_types(),
            0,
            "Entry not allowed to take arguments"
        );

        // Set up JIT execution engine
        self.module.verify().unwrap();
        let ee = self
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .map_err(|err| anyhow!("Failed to create execution engine: {err}"))?;

        // Create buffers for SJLJ jumping and for the panic message and link them
        // to the execution engine
        let jum_buf_size = unsafe { panic_runtime::jmp_buf_size() };
        let mut panic_jmp_buf =
            alloc_shared_buffer(PANIC_JMP_BUFFER, jum_buf_size, &self.module, &ee);
        let mut panic_msg_buf =
            alloc_shared_buffer(PANIC_MSG_BUFFER, PANIC_MSG_BUFFER_LEN, &self.module, &ee);

        // Link the `panic_exit` function into the execution engine
        let panic_exit_func = self
            .module
            .get_function(PANIC_EXIT)
            .ok_or_else(|| anyhow!("exec_panicking requires using UnwindingPreludeCodegen"))?;
        ee.add_global_mapping(&panic_exit_func, panic_runtime::panic_exit as usize);

        // Invoke the entry function using the panic runtime trampoline
        let entry_ptr = unsafe { ee.get_function(entry.as_ref()).unwrap() };
        unsafe {
            panic_runtime::trampoline(panic_jmp_buf.as_mut_ptr().cast(), Some(entry_ptr.as_raw()));
        }

        // Read panic message from buffer
        panic_msg_buf[PANIC_MSG_BUFFER_LEN - 1] = 0;
        let panic_msg = CStr::from_bytes_until_nul(&panic_msg_buf).unwrap();
        Ok(panic_msg.to_string_lossy().to_string())
    }
}

impl SimpleHugrConfig {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ins: Default::default(),
            outs: Default::default(),
            extensions: Default::default(),
        }
    }

    pub fn with_ins(mut self, ins: impl Into<TypeRow>) -> Self {
        self.ins = ins.into();
        self
    }

    pub fn with_outs(mut self, outs: impl Into<TypeRow>) -> Self {
        self.outs = outs.into();
        self
    }

    pub fn with_extensions(mut self, extensions: ExtensionRegistry) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn finish(self, make: impl FnOnce(DFGW) -> Hugr) -> Hugr {
        self.finish_with_exts(|builder, _| make(builder))
    }

    pub fn finish_with_exts(self, make: impl FnOnce(DFGW, &ExtensionRegistry) -> Hugr) -> Hugr {
        let func_b = FunctionBuilder::new("main", HugrFuncType::new(self.ins, self.outs)).unwrap();
        make(func_b, &self.extensions)
    }
}

impl Default for SimpleHugrConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Name of the LLVM global that holds the buffer to write panic messages.
const PANIC_MSG_BUFFER: &str = "__panic_msg_buf";

/// Name of the LLVM global that holds the buffer for the panic runtime
const PANIC_JMP_BUFFER: &str = "__panic_jmp_buf";

/// Name of the LLVM function that is linked to [`panic_runtime::panic_exit`]
/// to exit the program.
const PANIC_EXIT: &str = "__panic_exit";

/// Length of the panic message buffer.
const PANIC_MSG_BUFFER_LEN: usize = 1024;

/// Retrieves a named global buffer from an LLVM module or creates one
/// if it doesn't exist yet.
fn get_global_buffer<'c>(name: &str, size: usize, module: &Module<'c>) -> GlobalValue<'c> {
    module.get_global(name).unwrap_or_else(|| {
        module.add_global(
            module.get_context().i8_type().array_type(size as u32),
            None,
            name,
        )
    })
}

/// Allocates a buffer and links it to a global variable with the given name
/// in the execution engine.
fn alloc_shared_buffer(name: &str, size: usize, module: &Module, ee: &ExecutionEngine) -> Vec<u8> {
    let mut buf = vec![0; size];
    let global = get_global_buffer(name, size, module);
    global.set_linkage(Linkage::External);
    ee.add_global_mapping(&global, buf.as_mut_ptr() as usize);
    buf
}

/// Builds an `i8*` [`PointerValue`] to a global buffer with the given name.
fn get_buffer_ptr<'c, H: HugrView<Node = Node>>(
    name: &str,
    size: usize,
    ctx: &mut EmitFuncContext<'c, '_, H>,
) -> Result<PointerValue<'c>> {
    let global = get_global_buffer(name, size, ctx.get_current_module());
    let ptr = ctx.builder().build_bit_cast(
        global.as_pointer_value(),
        ctx.iw_context().i8_type().ptr_type(AddressSpace::default()),
        "",
    )?;
    Ok(ptr.into_pointer_value())
}

/// Prelude codegen that exits the current thread on panic instead of aborting.
#[derive(Clone)]
pub struct PanicTestPreludeCodegen;

impl PreludeCodegen for PanicTestPreludeCodegen {
    fn emit_panic<H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<H>,
        err: BasicValueEnum,
    ) -> Result<()> {
        // Emit a `panic_exit(jmp_buf, msg_buf, msg, msg_buf_len)` runtime call
        let usize_ty = self.usize_type(&ctx.typing_session());
        let i8_ptr_ty = ctx.iw_context().i8_type().ptr_type(AddressSpace::default());
        let sig = ctx.iw_context().void_type().fn_type(
            &[
                i8_ptr_ty.into(),
                i8_ptr_ty.into(),
                i8_ptr_ty.into(),
                usize_ty.into(),
            ],
            false,
        );
        let panic_exit = ctx.get_extern_func(PANIC_EXIT, sig)?;

        let jmp_buf_size = unsafe { panic_runtime::jmp_buf_size() };
        let jmp_buf_ptr = get_buffer_ptr(PANIC_JMP_BUFFER, jmp_buf_size, ctx)?;
        let msg_buf_ptr = get_buffer_ptr(PANIC_MSG_BUFFER, PANIC_MSG_BUFFER_LEN, ctx)?;
        let msg_ptr = ctx
            .builder()
            .build_extract_value(err.into_struct_value(), 1, "")?
            .into_pointer_value();
        let msg_buf_len = usize_ty.const_int(PANIC_MSG_BUFFER_LEN as u64, false);

        ctx.builder().build_call(
            panic_exit,
            &[
                jmp_buf_ptr.into(),
                msg_buf_ptr.into(),
                msg_ptr.into(),
                msg_buf_len.into(),
            ],
            "",
        )?;
        Ok(())
    }
}

#[doc(hidden)]
pub use hugr_core;
#[doc(hidden)]
pub use inkwell;
#[doc(hidden)]
pub use insta;

/// A macro used to check the emission of a Hugr module,
/// and to assert the correctness of the emitted LLVM IR using [insta].
///
/// Call with
/// ```ignore
/// check_emission!(hugr, llvm_ctx);
/// ```
/// or
/// ```ignore
/// check_emission!("snapshot_name", hugr, llvm_ctx);
/// ```
#[macro_export]
macro_rules! check_emission {
    // Call the macro with a snapshot name.
    ($snapshot_name:expr, $hugr: ident, $test_ctx:ident) => {{
        let root = $crate::utils::fat::FatExt::fat_root(&$hugr).unwrap();
        let emission =
            $crate::emit::test::Emission::emit_hugr(root, $test_ctx.get_emit_hugr()).unwrap();

        let mut settings = $crate::emit::test::insta::Settings::clone_current();
        let new_suffix = settings
            .snapshot_suffix()
            .map_or("pre-mem2reg".into(), |x| format!("pre-mem2reg@{x}"));
        settings.set_snapshot_suffix(new_suffix);
        settings.bind(|| {
            let mod_str = emission.module().to_string();
            if $snapshot_name == "" {
                $crate::emit::test::insta::assert_snapshot!(mod_str)
            } else {
                $crate::emit::test::insta::assert_snapshot!($snapshot_name, mod_str)
            }
        });

        emission.verify().unwrap();

        emission.opt(|| {
            let pb = $crate::emit::test::inkwell::passes::PassManager::create(());
            pb.add_promote_memory_to_register_pass();
            pb
        });

        let mod_str = emission.module().to_string();
        if $snapshot_name == "" {
            $crate::emit::test::insta::assert_snapshot!(mod_str)
        } else {
            $crate::emit::test::insta::assert_snapshot!($snapshot_name, mod_str)
        }
        emission
    }};
    // Use the default snapshot name.
    ($hugr: ident, $test_ctx:ident) => {
        check_emission!("", $hugr, $test_ctx)
    };
}

#[cfg(test)]
mod test_fns {
    use super::*;
    use crate::custom::CodegenExtsBuilder;
    use crate::types::{HugrFuncType, HugrSumType};

    use hugr_core::builder::{Container, Dataflow, HugrBuilder, ModuleBuilder, SubContainer};
    use hugr_core::builder::{DataflowHugr, DataflowSubContainer};
    use hugr_core::extension::prelude::{ConstError, ConstUsize, EXIT_OP_ID, bool_t, usize_t};
    use hugr_core::extension::{PRELUDE, PRELUDE_REGISTRY};
    use hugr_core::ops::constant::CustomConst;
    use hugr_core::types::Term;

    use hugr_core::ops::{CallIndirect, Tag, Value};
    use hugr_core::std_extensions::STD_REG;
    use hugr_core::std_extensions::arithmetic::int_ops::{self};
    use hugr_core::std_extensions::arithmetic::int_types::{self, ConstInt};
    use hugr_core::types::{Signature, Type, TypeRow};
    use hugr_core::{Hugr, type_row};

    use itertools::Itertools;
    use rstest::{fixture, rstest};
    use std::iter;

    use crate::test::*;
    #[rstest]
    fn emit_hugr_tag(llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(Type::new_unit_sum(3))
            .finish(|mut builder: DFGW| {
                let tag = builder
                    .add_dataflow_op(
                        Tag::new(1, vec![vec![].into(), vec![].into(), vec![].into()]),
                        builder.input_wires(),
                    )
                    .unwrap();
                builder.finish_hugr_with_outputs(tag.outputs()).unwrap()
            });
        let _ = check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_dfg(llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_ins(Type::UNIT)
            .with_outs(Type::UNIT)
            .finish(|mut builder: DFGW| {
                let dfg = {
                    let b = builder
                        .dfg_builder(HugrFuncType::new_endo(Type::UNIT), builder.input_wires())
                        .unwrap();
                    let w = b.input_wires();
                    b.finish_with_outputs(w).unwrap()
                };
                builder.finish_hugr_with_outputs(dfg.outputs()).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_conditional(llvm_ctx: TestContext) {
        let hugr = {
            let input_v_rows: Vec<TypeRow> =
                (1..4).map(Type::new_unit_sum).map_into().collect_vec();
            let output_v_rows = {
                let mut r = input_v_rows.clone();
                r.reverse();
                r
            };

            SimpleHugrConfig::new()
                .with_ins(vec![Type::new_sum(input_v_rows.clone()), Type::UNIT])
                .with_outs(vec![Type::new_sum(output_v_rows.clone()), Type::UNIT])
                .finish(|mut builder: DFGW| {
                    let cond = {
                        let [sum_input, other_input] = builder.input_wires_arr();
                        let mut cond_b = builder
                            .conditional_builder(
                                (input_v_rows.clone(), sum_input),
                                [(Type::UNIT, other_input)],
                                vec![Type::new_sum(output_v_rows.clone()), Type::UNIT].into(),
                            )
                            .unwrap();
                        for i in 0..3 {
                            let mut case_b = cond_b.case_builder(i).unwrap();
                            let [case_input, other_input] = case_b.input_wires_arr();
                            let tag = case_b
                                .add_dataflow_op(
                                    Tag::new(2 - i, output_v_rows.clone()),
                                    [case_input],
                                )
                                .unwrap();
                            case_b
                                .finish_with_outputs([tag.out_wire(0), other_input])
                                .unwrap();
                        }
                        cond_b.finish_sub_container().unwrap()
                    };
                    let [o1, o2] = cond.outputs_arr();
                    builder.finish_hugr_with_outputs([o1, o2]).unwrap()
                })
        };
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_load_constant(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);

        let v = Value::tuple([
            Value::unit_sum(2, 4).unwrap(),
            ConstInt::new_s(4, -24).unwrap().into(),
        ]);

        let hugr = SimpleHugrConfig::new()
            .with_outs(v.get_type())
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder: DFGW| {
                let konst = builder.add_load_value(v);
                builder.finish_hugr_with_outputs([konst]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_call(llvm_ctx: TestContext) {
        fn build_recursive(mod_b: &mut ModuleBuilder<Hugr>, name: &str, io: TypeRow) {
            let f_id = mod_b
                .declare(name, HugrFuncType::new_endo(io).into())
                .unwrap();
            let mut func_b = mod_b.define_declaration(&f_id).unwrap();
            let call = func_b.call(&f_id, &[], func_b.input_wires()).unwrap();
            func_b.finish_with_outputs(call.outputs()).unwrap();
        }

        let mut mod_b = ModuleBuilder::new();
        build_recursive(&mut mod_b, "main_void", type_row![]);
        build_recursive(&mut mod_b, "main_unary", vec![bool_t()].into());
        build_recursive(&mut mod_b, "main_binary", vec![bool_t(), bool_t()].into());
        let hugr = mod_b.finish_hugr().unwrap();
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_call_indirect(llvm_ctx: TestContext) {
        fn build_recursive(mod_b: &mut ModuleBuilder<Hugr>, name: &str, io: TypeRow) {
            let signature = HugrFuncType::new_endo(io);
            let f_id = mod_b.declare(name, signature.clone().into()).unwrap();
            let mut func_b = mod_b.define_declaration(&f_id).unwrap();
            let func = func_b.load_func(&f_id, &[]).unwrap();
            let inputs = iter::once(func).chain(func_b.input_wires());
            let call_indirect = func_b
                .add_dataflow_op(CallIndirect { signature }, inputs)
                .unwrap();
            func_b.finish_with_outputs(call_indirect.outputs()).unwrap();
        }

        let mut mod_b = ModuleBuilder::new();
        build_recursive(&mut mod_b, "main_void", type_row![]);
        build_recursive(&mut mod_b, "main_unary", vec![bool_t()].into());
        build_recursive(&mut mod_b, "main_binary", vec![bool_t(), bool_t()].into());
        let hugr = mod_b.finish_hugr().unwrap();
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn emit_hugr_custom_op(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);
        let v1 = ConstInt::new_s(4, -24).unwrap();
        let v2 = ConstInt::new_s(4, 24).unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_outs(v1.get_type())
            .with_extensions(STD_REG.to_owned())
            .finish(|mut builder: DFGW| {
                let k1 = builder.add_load_value(v1);
                let k2 = builder.add_load_value(v2);
                let ext_op = int_ops::EXTENSION
                    .instantiate_extension_op("iadd", [4.into()])
                    .unwrap();
                let add = builder.add_dataflow_op(ext_op, [k1, k2]).unwrap();
                builder.finish_hugr_with_outputs(add.outputs()).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn get_external_func(llvm_ctx: TestContext) {
        let emc = llvm_ctx.get_emit_module_context();
        let func_type1 = emc.iw_context().i32_type().fn_type(&[], false);
        let func_type2 = emc.iw_context().f64_type().fn_type(&[], false);
        let foo1 = emc.get_extern_func("foo", func_type1).unwrap();
        assert_eq!(foo1.get_name().to_str().unwrap(), "foo");
        let foo2 = emc.get_extern_func("foo", func_type1).unwrap();
        assert_eq!(foo1, foo2);
        assert!(emc.get_extern_func("foo", func_type2).is_err());
    }

    #[rstest]
    fn diverse_module_children(llvm_ctx: TestContext) {
        let hugr = {
            let mut builder = ModuleBuilder::new();
            let _ = {
                let fbuilder = builder
                    .define_function("f1", HugrFuncType::new_endo(type_row![]))
                    .unwrap();
                fbuilder.finish_sub_container().unwrap()
            };
            let _ = {
                let fbuilder = builder
                    .define_function("f2", HugrFuncType::new_endo(type_row![]))
                    .unwrap();
                fbuilder.finish_sub_container().unwrap()
            };
            let _ = builder.add_constant(Value::true_val());
            let _ = builder
                .declare("decl", HugrFuncType::new_endo(type_row![]).into())
                .unwrap();
            builder.finish_hugr().unwrap()
        };
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn diverse_cfg_children(llvm_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(bool_t())
            .finish(|mut builder: DFGW| {
                let [r] = {
                    let mut builder = builder.cfg_builder([], vec![bool_t()].into()).unwrap();
                    let konst = builder.add_constant(Value::false_val());
                    let entry = {
                        let mut builder = builder
                            .entry_builder([type_row![]], vec![bool_t()].into())
                            .unwrap();
                        let control = builder.add_load_value(Value::unary_unit_sum());
                        let r = builder.load_const(&konst);
                        builder.finish_with_outputs(control, [r]).unwrap()
                    };
                    let exit = builder.exit_block();
                    builder.branch(&entry, 0, &exit).unwrap();
                    builder.finish_sub_container().unwrap().outputs_arr()
                };
                builder.finish_hugr_with_outputs([r]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn load_function(llvm_ctx: TestContext) {
        let hugr = {
            let mut builder = ModuleBuilder::new();
            let target_sig = Signature::new_endo(type_row![]);
            let target_func = builder
                .declare("target_func", target_sig.clone().into())
                .unwrap();
            let _ = {
                let mut builder = builder
                    .define_function(
                        "main",
                        Signature::new(type_row![], Type::new_function(target_sig)),
                    )
                    .unwrap();
                let r = builder.load_func(&target_func, &[]).unwrap();
                builder.finish_with_outputs([r]).unwrap()
            };
            builder.finish_hugr().unwrap()
        };

        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    fn tail_loop_simple(mut llvm_ctx: TestContext) {
        let hugr = {
            let just_input = usize_t();
            let just_output = Type::UNIT;
            let input_v = TypeRow::from(vec![just_input.clone()]);
            let output_v = TypeRow::from(vec![just_output.clone()]);

            SimpleHugrConfig::new()
                .with_extensions(PRELUDE_REGISTRY.clone())
                .with_ins(input_v)
                .with_outs(output_v)
                .finish(|mut builder: DFGW| {
                    let [just_in_w] = builder.input_wires_arr();
                    let mut tail_b = builder
                        .tail_loop_builder(
                            [(just_input.clone(), just_in_w)],
                            [],
                            vec![just_output.clone()].into(),
                        )
                        .unwrap();

                    let input = tail_b.input();
                    let [inp_w] = input.outputs_arr();

                    let loop_sig = tail_b.loop_signature().unwrap().clone();

                    // builder.add_dataflow_op(ops::Noop, input_wires)

                    let sum_inp_w = tail_b.make_continue(loop_sig.clone(), [inp_w]).unwrap();

                    let outs @ [_] = tail_b
                        .finish_with_outputs(sum_inp_w, [])
                        .unwrap()
                        .outputs_arr();
                    builder.finish_hugr_with_outputs(outs).unwrap()
                })
        };
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);

        check_emission!(hugr, llvm_ctx);
    }

    #[fixture]
    fn terminal_loop(#[default(3)] iters: u64, #[default(7)] input: u64) -> Hugr {
        /*
        Computes roughly the following:
        ```python
        def terminal_loop(counter: int, val: int) -> int:
            while True:
                val = val * 2
                if counter == 0:
                    break
                else:
                    counter -= 1
            return val
        ```
         */
        let int_ty = int_types::int_type(6);
        let just_input = int_ty.clone();
        let other_ty = int_ty.clone();

        let mut registry = PRELUDE_REGISTRY.clone();
        registry.register(int_ops::EXTENSION.clone()).unwrap();
        registry.register(int_types::EXTENSION.clone()).unwrap();

        SimpleHugrConfig::new()
            .with_extensions(registry)
            .with_outs(int_ty.clone())
            .finish(|mut builder: DFGW| {
                let just_in_w = builder.add_load_value(ConstInt::new_u(6, iters).unwrap());
                let other_w = builder.add_load_value(ConstInt::new_u(6, input).unwrap());

                let tail_l = {
                    let mut tail_b = builder
                        .tail_loop_builder(
                            [(just_input.clone(), just_in_w)],
                            [(other_ty.clone(), other_w)],
                            type_row![],
                        )
                        .unwrap();
                    let [loop_int_w, other_w] = tail_b.input_wires_arr();

                    let zero = ConstInt::new_u(6, 0).unwrap();
                    let zero_w = tail_b.add_load_value(zero);
                    let [eq_0] = tail_b
                        .add_dataflow_op(
                            int_ops::IntOpDef::ieq.with_log_width(6),
                            [loop_int_w, zero_w],
                        )
                        .unwrap()
                        .outputs_arr();

                    let loop_sig = tail_b.loop_signature().unwrap().clone();

                    let two = ConstInt::new_u(6, 2).unwrap();
                    let two_w = tail_b.add_load_value(two);

                    let [other_mul_2] = tail_b
                        .add_dataflow_op(
                            int_ops::IntOpDef::imul.with_log_width(6),
                            [other_w, two_w],
                        )
                        .unwrap()
                        .outputs_arr();
                    let cond = {
                        let mut cond_b = tail_b
                            .conditional_builder(
                                ([type_row![], type_row![]], eq_0),
                                vec![(just_input.clone(), loop_int_w)],
                                vec![
                                    HugrSumType::new(vec![vec![just_input.clone()], vec![]]).into(),
                                ]
                                .into(),
                            )
                            .unwrap();

                        // If the check is false, we subtract 1 and continue
                        let _false_case = {
                            let mut false_case_b = cond_b.case_builder(0).unwrap();
                            let [counter] = false_case_b.input_wires_arr();
                            let one = ConstInt::new_u(6, 1).unwrap();
                            let one_w = false_case_b.add_load_value(one);

                            let [counter] = false_case_b
                                .add_dataflow_op(
                                    int_ops::IntOpDef::isub.with_log_width(6),
                                    [counter, one_w],
                                )
                                .unwrap()
                                .outputs_arr();
                            let tag_continue = false_case_b
                                .make_continue(loop_sig.clone(), [counter])
                                .unwrap();

                            false_case_b.finish_with_outputs([tag_continue]).unwrap()
                        };
                        let _true_case = {
                            // In the true case, we break and output true along with the "other" input wire
                            let mut true_case_b = cond_b.case_builder(1).unwrap();

                            let [_counter] = true_case_b.input_wires_arr();

                            let tagged_break =
                                true_case_b.make_break(loop_sig.clone(), []).unwrap();
                            true_case_b.finish_with_outputs([tagged_break]).unwrap()
                        };

                        cond_b.finish_sub_container().unwrap()
                    };
                    tail_b
                        .finish_with_outputs(cond.out_wire(0), [other_mul_2])
                        .unwrap()
                };
                let [out_int] = tail_l.outputs_arr();
                builder
                    .finish_hugr_with_outputs([out_int])
                    .unwrap_or_else(|e| panic!("{e}"))
            })
    }

    #[rstest]
    fn tail_loop(mut llvm_ctx: TestContext, #[with(3, 7)] terminal_loop: Hugr) {
        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);
        check_emission!(terminal_loop, llvm_ctx);
    }

    #[rstest]
    #[case(3, 7)]
    #[case(2, 1)]
    #[case(20, 0)]
    fn tail_loop_exec(
        mut exec_ctx: TestContext,
        #[case] iters: u64,
        #[case] input: u64,
        #[with(iters, input)] terminal_loop: Hugr,
    ) {
        exec_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);
        assert_eq!(
            input << (iters + 1),
            exec_ctx.exec_hugr_u64(terminal_loop, "main")
        );
    }

    #[rstest]
    fn test_exec(mut exec_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder: DFGW| {
                let konst = builder.add_load_value(ConstUsize::new(42));
                builder.finish_hugr_with_outputs([konst]).unwrap()
            });
        exec_ctx.add_extensions(CodegenExtsBuilder::add_default_prelude_extensions);
        assert_eq!(42, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    #[case::basic("Basic panic message")]
    #[case::empty("")]
    #[case::unicode("We ❤️ ünîçödè ( ͡° ͜ʖ ͡°)")]
    #[case::long(&"x".repeat(PANIC_MSG_BUFFER_LEN + 100))]
    fn test_exec_panic(mut exec_ctx: TestContext, #[case] msg: &str) {
        let panic_op = PRELUDE
            .instantiate_extension_op(&EXIT_OP_ID, [Term::new_list([]), Term::new_list([])])
            .unwrap();

        let hugr = SimpleHugrConfig::new()
            .with_extensions(PRELUDE_REGISTRY.to_owned())
            .finish(|mut builder: DFGW| {
                let err = builder.add_load_value(ConstError::new(2, msg));
                builder.add_dataflow_op(panic_op, [err]).unwrap();
                builder.finish_hugr_with_outputs([]).unwrap()
            });
        exec_ctx.add_extensions(|b| b.add_prelude_extensions(PanicTestPreludeCodegen));
        let msg_trunc = if msg.len() >= PANIC_MSG_BUFFER_LEN {
            &msg[..PANIC_MSG_BUFFER_LEN - 1]
        } else {
            msg
        };
        assert_eq!(exec_ctx.exec_hugr_panicking(hugr, "main"), msg_trunc);
    }
}

use anyhow::{Result, anyhow};
use delegate::delegate;
use hugr_core::{
    HugrView, Node, Visibility,
    ops::{FuncDecl, FuncDefn, OpType},
    types::PolyFuncType,
};
use inkwell::{
    builder::Builder,
    context::Context,
    intrinsics::Intrinsic,
    module::{Linkage, Module},
    types::{AnyType, BasicType, BasicTypeEnum, FunctionType},
    values::{BasicValueEnum, CallSiteValue, FunctionValue, GlobalValue},
};
use std::{collections::HashSet, rc::Rc};

use crate::types::{HugrFuncType, HugrSumType, HugrType, TypingSession};

use crate::{custom::CodegenExtsMap, types::LLVMSumType, utils::fat::FatNode};

pub mod args;
pub mod func;
pub mod libc;
pub mod namer;
pub mod ops;

pub use args::EmitOpArgs;
pub use func::{EmitFuncContext, RowPromise};
pub use namer::Namer;
pub use ops::emit_value;

/// A context holding data required for emitting HUGRs into an LLVM module.
/// This includes the module itself, a set of extensions for lowering custom
/// elements, and policy for naming various HUGR elements.
///
/// `'c` names the lifetime of the LLVM context, while `'a` names the lifetime
/// of other internal references.
pub struct EmitModuleContext<'c, 'a, H>
where
    'a: 'c,
{
    iw_context: &'c Context,
    module: Module<'c>,
    extensions: Rc<CodegenExtsMap<'a, H>>,
    namer: Rc<Namer>,
}

impl<'c, 'a, H> EmitModuleContext<'c, 'a, H> {
    delegate! {
        to self.typing_session() {
            /// Convert a [HugrType] into an LLVM [Type](BasicTypeEnum).
            pub fn llvm_type(&self, hugr_type: &HugrType) -> Result<BasicTypeEnum<'c>>;
            /// Convert a [HugrFuncType] into an LLVM [FunctionType].
            pub fn llvm_func_type(&self, hugr_type: &HugrFuncType) -> Result<FunctionType<'c>>;
            /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
            pub fn llvm_sum_type(&self, sum_type: HugrSumType) -> Result<LLVMSumType<'c>>;
        }

        to self.namer {
            /// Mangle the name of a [FuncDefn]  or a [FuncDecl].
            pub fn name_func(&self, name: impl AsRef<str>, node: Node) -> String;
        }
    }

    pub fn iw_context(&self) -> &'c Context {
        self.iw_context
    }

    /// Creates a new  `EmitModuleContext`. We take ownership of the [Module],
    /// and return it in [`EmitModuleContext::finish`].
    pub fn new(
        iw_context: &'c Context,
        module: Module<'c>,
        namer: Rc<Namer>,
        extensions: Rc<CodegenExtsMap<'a, H>>,
    ) -> Self {
        Self {
            iw_context,
            module,
            extensions,
            namer,
        }
    }

    /// Returns a reference to the inner [Module]. Note that this type has
    /// "interior mutability", and this reference can be used to add functions
    /// and globals to the [Module].
    pub fn module(&self) -> &Module<'c> {
        &self.module
    }

    /// Returns a reference to the inner [`CodegenExtsMap`].
    pub fn extensions(&self) -> Rc<CodegenExtsMap<'a, H>> {
        self.extensions.clone()
    }

    /// Returns a [`TypingSession`] constructed from it's members.
    pub fn typing_session(&self) -> TypingSession<'c, 'a> {
        self.extensions
            .type_converter
            .clone()
            .session(self.iw_context)
    }

    fn get_func_impl(
        &self,
        name: impl AsRef<str>,
        func_ty: FunctionType<'c>,
        linkage: Option<Linkage>,
    ) -> Result<FunctionValue<'c>> {
        let func = self
            .module()
            .get_function(name.as_ref())
            .unwrap_or_else(|| self.module.add_function(name.as_ref(), func_ty, linkage));
        if func.get_type() != func_ty {
            Err(anyhow!(
                "Function '{}' has wrong type: expected: {func_ty} actual: {}",
                name.as_ref(),
                func.get_type()
            ))?;
        }
        Ok(func)
    }

    fn get_hugr_func_impl(
        &self,
        name: impl AsRef<str>,
        node: Node,
        func_ty: &PolyFuncType,
        visibility: &Visibility,
    ) -> Result<FunctionValue<'c>> {
        let func_ty = (func_ty.params().is_empty())
            .then_some(func_ty.body())
            .ok_or(anyhow!("function has type params"))?;
        let llvm_func_ty = self.llvm_func_type(func_ty)?;
        let name = self.name_func(name, node);
        match visibility {
            Visibility::Public => self.get_func_impl(name, llvm_func_ty, Some(Linkage::External)),
            Visibility::Private => self.get_func_impl(name, llvm_func_ty, Some(Linkage::Private)),
            _ => self.get_func_impl(name, llvm_func_ty, None),
        }
    }

    /// Adds or gets the [`FunctionValue`] in the [Module] corresponding to the given [`FuncDefn`].
    ///
    /// The name of the result is mangled by [`EmitModuleContext::name_func`].
    pub fn get_func_defn<'hugr>(
        &self,
        node: FatNode<'hugr, FuncDefn, H>,
    ) -> Result<FunctionValue<'c>>
    where
        H: HugrView<Node = Node>,
    {
        self.get_hugr_func_impl(
            node.func_name(),
            node.node(),
            node.signature(),
            node.visibility(),
        )
    }

    /// Adds or gets the [`FunctionValue`] in the [Module] corresponding to the given [`FuncDecl`].
    ///
    /// The name of the result is mangled by [`EmitModuleContext::name_func`].
    pub fn get_func_decl<'hugr>(
        &self,
        node: FatNode<'hugr, FuncDecl, H>,
    ) -> Result<FunctionValue<'c>>
    where
        H: HugrView<Node = Node>,
    {
        self.get_hugr_func_impl(
            node.func_name(),
            node.node(),
            node.signature(),
            node.visibility(),
        )
    }

    /// Adds or get the [`FunctionValue`] in the [Module] with the given symbol
    /// and function type.
    ///
    /// The name undergoes no mangling. The [`FunctionValue`] will have
    /// [`Linkage::External`].
    ///
    /// If this function is called multiple times with the same arguments it
    /// will return the same [`FunctionValue`].
    ///
    /// If a function with the given name exists but the type does not match
    /// then an Error is returned.
    pub fn get_extern_func(
        &self,
        symbol: impl AsRef<str>,
        typ: FunctionType<'c>,
    ) -> Result<FunctionValue<'c>> {
        self.get_func_impl(symbol, typ, Some(Linkage::External))
    }

    /// Adds or gets the [`GlobalValue`] in the [Module] corresponding to the
    /// given symbol and LLVM type.
    ///
    /// The name will not be mangled.
    ///
    /// If a global with the given name exists but the type or constant-ness
    /// does not match then an error will be returned.
    pub fn get_global(
        &self,
        symbol: impl AsRef<str>,
        typ: impl BasicType<'c>,
        constant: bool,
    ) -> Result<GlobalValue<'c>> {
        let symbol = symbol.as_ref();
        let typ = typ.as_basic_type_enum();
        if let Some(global) = self.module().get_global(symbol) {
            let global_type = {
                // TODO This is exposed as `get_value_type` on the master branch
                // of inkwell, will be in the next release. When it's released
                // use `get_value_type`.
                use inkwell::types::AnyTypeEnum;
                use inkwell::values::AsValueRef;
                unsafe {
                    AnyTypeEnum::new(inkwell::llvm_sys::core::LLVMGlobalGetValueType(
                        global.as_value_ref(),
                    ))
                }
            };
            if global_type != typ.as_any_type_enum() {
                Err(anyhow!(
                    "Global '{symbol}' has wrong type: expected: {typ} actual: {global_type}"
                ))?;
            }
            if global.is_constant() != constant {
                Err(anyhow!(
                    "Global '{symbol}' has wrong constant-ness: expected: {constant} actual: {}",
                    global.is_constant()
                ))?;
            }
            Ok(global)
        } else {
            let global = self.module().add_global(typ, None, symbol.as_ref());
            global.set_constant(constant);
            Ok(global)
        }
    }

    /// Consumes the `EmitModuleContext` and returns the internal [Module].
    pub fn finish(self) -> Module<'c> {
        self.module
    }
}

type EmissionSet = HashSet<Node>;

/// Emits [`HugrView`]s into an LLVM [Module].
pub struct EmitHugr<'c, 'a, H>
where
    'a: 'c,
{
    emitted: EmissionSet,
    module_context: EmitModuleContext<'c, 'a, H>,
}

impl<'c, 'a, H: HugrView<Node = Node>> EmitHugr<'c, 'a, H> {
    delegate! {
        to self.module_context {
            /// Returns a reference to the inner [Context].
            pub fn iw_context(&self) -> &'c Context;
            /// Returns a reference to the inner [Module]. Note that this type has
            /// "interior mutability", and this reference can be used to add functions
            /// and globals to the [Module].
            pub fn module(&self) -> &Module<'c>;
        }
    }

    /// Creates a new  `EmitHugr`. We take ownership of the [Module], and return it in [`Self::finish`].
    pub fn new(
        iw_context: &'c Context,
        module: Module<'c>,
        namer: Rc<Namer>,
        extensions: Rc<CodegenExtsMap<'a, H>>,
    ) -> Self {
        assert_eq!(iw_context, &module.get_context());
        Self {
            emitted: Default::default(),
            module_context: EmitModuleContext::new(iw_context, module, namer, extensions),
        }
    }

    /// Emits a `FuncDefn` into the inner [Module].
    ///
    /// `node` need not be a child of a hugr [Module](hugr_core::ops::Module), but it will
    /// be emitted as a top-level function in the inner [Module]. Indeed, there
    /// are only top-level functions in LLVM IR.
    ///
    /// Any child [`FuncDefn`] will also be emitted.
    ///
    /// It is safe to emit the same node multiple times: the second and further
    /// emissions will be no-ops.
    ///
    /// If any LLVM IR declaration which is to be emitted already exists in the
    /// [Module] and it differs from what would be emitted, then we fail.
    pub fn emit_func(mut self, node: FatNode<'_, FuncDefn, H>) -> Result<Self> {
        let mut worklist: EmissionSet = [node.node()].into_iter().collect();
        let pop = |wl: &mut EmissionSet| wl.iter().next().copied().map(|x| wl.take(&x).unwrap());

        while let Some(next_node) = pop(&mut worklist) {
            use crate::utils::fat::FatExt as _;
            let Some(func) = node.hugr().try_fat(next_node) else {
                panic!(
                    "emit_func: node in worklist was not a FuncDefn: {:?}",
                    node.hugr().get_optype(next_node)
                )
            };
            let (new_self, new_tasks) = self.emit_func_impl(func)?;
            self = new_self;
            worklist.extend(new_tasks.into_iter());
        }
        Ok(self)
    }

    /// Emits all children of a hugr [Module](hugr_core::ops::Module).
    ///
    /// Note that type aliases are not supported, and that [`hugr_core::ops::Const`]
    /// and [`hugr_core::ops::FuncDecl`] nodes are not emitted directly, but instead by
    /// emission of ops with static edges from them. So [`FuncDefn`] are the only
    /// interesting children.
    pub fn emit_module(mut self, node: FatNode<'_, hugr_core::ops::Module, H>) -> Result<Self> {
        for c in node.children() {
            match c.as_ref() {
                OpType::FuncDefn(fd) => {
                    let fat_ot = c.into_ot(fd);
                    self = self.emit_func(fat_ot)?;
                }
                // FuncDecls are allowed, but we don't need to do anything here.
                OpType::FuncDecl(_) => (),
                // Consts are allowed, but we don't need to do anything here.
                OpType::Const(_) => (),
                _ => Err(anyhow!("Module has invalid child: {c}"))?,
            }
        }
        Ok(self)
    }

    fn emit_func_impl(mut self, node: FatNode<'_, FuncDefn, H>) -> Result<(Self, EmissionSet)> {
        if !self.emitted.insert(node.node()) {
            return Ok((self, EmissionSet::default()));
        }
        let func = self.module_context.get_func_defn(node)?;
        let mut func_ctx = EmitFuncContext::new(self.module_context, func)?;
        let ret_rmb = func_ctx.new_row_mail_box(node.signature().body().output.iter(), "ret")?;
        ops::emit_dataflow_parent(
            &mut func_ctx,
            EmitOpArgs {
                node,
                inputs: func.get_params(),
                outputs: ret_rmb.promise(),
            },
        )?;
        let builder = func_ctx.builder();
        match &ret_rmb.read::<Vec<_>>(builder, [])?[..] {
            [] => builder.build_return(None)?,
            [x] => builder.build_return(Some(x))?,
            xs => builder.build_aggregate_return(xs)?,
        };
        let (mctx, todos) = func_ctx.finish()?;
        self.module_context = mctx;
        Ok((self, todos))
    }

    /// Consumes the `EmitHugr` and returns the internal [Module].
    pub fn finish(self) -> Module<'c> {
        self.module_context.finish()
    }
}

/// Extract all return values from the result of a `call`.
///
/// LLVM only supports functions with exactly zero or one return value.
/// For functions with multiple return values, we return a struct containing
/// all the return values.
///
/// `inkwell` provides a helper [`Builder::build_aggregate_return`] to construct
/// the return value, see `EmitHugr::emit_func_impl`. This function performs the
/// inverse.
pub fn deaggregate_call_result<'c>(
    builder: &Builder<'c>,
    call_result: CallSiteValue<'c>,
    num_results: usize,
) -> Result<Vec<BasicValueEnum<'c>>> {
    let call_result = call_result.try_as_basic_value();
    Ok(match num_results as u32 {
        0 => {
            call_result.expect_right("void");
            vec![]
        }
        1 => vec![call_result.expect_left("non-void")],
        n => {
            let return_struct = call_result.expect_left("non-void").into_struct_value();
            (0..n)
                .map(|i| builder.build_extract_value(return_struct, i, ""))
                .collect::<Result<Vec<_>, _>>()?
        }
    })
}

pub fn get_intrinsic<'c>(
    module: &Module<'c>,
    name: impl AsRef<str>,
    args: impl AsRef<[BasicTypeEnum<'c>]>,
) -> Result<FunctionValue<'c>> {
    let (name, args) = (name.as_ref(), args.as_ref());
    let intrinsic = Intrinsic::find(name).ok_or(anyhow!("Failed to find intrinsic: '{name}'"))?;
    intrinsic
        .get_declaration(module, args.as_ref())
        .ok_or(anyhow!(
            "failed to get_declaration for intrinsic '{name}' with args '{args:?}'"
        ))
}

#[cfg(any(test, feature = "test-utils"))]
pub mod test;

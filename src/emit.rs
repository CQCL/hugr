use anyhow::{anyhow, Result};
use delegate::delegate;
use hugr::{
    ops::{FuncDecl, FuncDefn, NamedOp as _, OpType},
    types::PolyFuncType,
    HugrView, Node, NodeIndex,
};
use inkwell::{
    context::Context,
    module::{Linkage, Module},
    types::{BasicTypeEnum, FunctionType},
    values::{BasicValueEnum, FunctionValue},
};
use std::{collections::HashSet, hash::Hash, rc::Rc};

use crate::types::TypingSession;

use crate::{
    custom::CodegenExtsMap,
    fat::FatNode,
    types::{LLVMSumType, TypeConverter},
};

use self::func::{EmitFuncContext, RowPromise};

pub mod func;
mod ops;

/// A type used whenever emission is delegated to a function
pub struct EmitOpArgs<'c, OT, H> {
    /// This [HugrView] and [Node] we are emitting
    pub node: FatNode<'c, OT, H>,
    /// The values that should be used for all Value input ports of the node
    pub inputs: Vec<BasicValueEnum<'c>>,
    /// The results of the node should be put here
    pub outputs: RowPromise<'c>,
}

impl<'c, OT, H> EmitOpArgs<'c, OT, H> {
    /// Get the internal [FatNode]
    pub fn node(&self) -> FatNode<'c, OT, H> {
        self.node.clone()
    }
}

impl<'c, H: HugrView> EmitOpArgs<'c, OpType, H> {
    /// Attempt to specialise the internal [FatNode].
    pub fn try_into_ot<OT: 'c>(self) -> Result<EmitOpArgs<'c, OT, H>, Self>
    where
        &'c OpType: TryInto<&'c OT>,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
        } = self;
        match node.try_into_ot() {
            Some(new_node) => Ok(EmitOpArgs {
                node: new_node,
                inputs,
                outputs,
            }),
            None => Err(EmitOpArgs {
                node,
                inputs,
                outputs,
            }),
        }
    }

    /// Specialise the internal [FatNode].
    ///
    /// Panics if `OT` is not the `get_optype` of the internal [Node].
    pub fn into_ot<OTInto: PartialEq + 'c>(self, ot: &'c OTInto) -> EmitOpArgs<'c, OTInto, H>
    where
        &'c OpType: TryInto<&'c OTInto>,
        <&'c OpType as TryInto<&'c OTInto>>::Error: std::fmt::Debug,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
        } = self;
        EmitOpArgs {
            node: node.into_ot(ot),
            inputs,
            outputs,
        }
    }
}

/// A trait used to abstract over emission.
///
/// In particular a `Box<dyn EmitOp>` is returned by
/// [crate::custom::CodegenExtension::emitter].
///
/// Any non-trivial implementation will need to contain an [&mut
/// EmitFuncContext](EmitFuncContext) in `Self` in order to be able to implement
/// this trait.
pub trait EmitOp<'c, OT, H: HugrView> {
    /// Emit the node in `args` using the inputs, and [finishing](RowPromise::finish) the outputs.
    fn emit(&mut self, args: EmitOpArgs<'c, OT, H>) -> Result<()>;
}

/// A trivial implementation of [EmitOp] used for [crate::custom::CodegenExtension]s that do
/// not support emitting any ops.
pub struct NullEmitLlvm;

impl<OT, H: HugrView> EmitOp<'_, OT, H> for NullEmitLlvm {
    fn emit(&mut self, _args: EmitOpArgs<OT, H>) -> Result<()> {
        Err(anyhow!("NullEmitLLVM"))
    }
}

/// A type with features for mangling the naming of symbols.
///
/// TODO This is mostly a placeholder
#[derive(Clone)]
pub struct Namer {
    prefix: String,
}

impl Namer {
    /// Create a new `Namer` that for each symbol:
    /// * prefix  `prefix`
    /// * postfixes ".{node.index()}"
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    /// Mangle the the name of a [FuncDefn] or [FuncDecl].
    pub fn name_func(&self, name: impl AsRef<str>, node: Node) -> String {
        format!("{}{}.{}", &self.prefix, name.as_ref(), node.index())
    }
}

impl Default for Namer {
    fn default() -> Self {
        Self::new("_hl.")
    }
}

pub struct EmitModuleContext<'c, H: HugrView> {
    module: Module<'c>,
    extension_context: Rc<CodegenExtsMap<'c, H>>,
    typer: Rc<TypeConverter<'c>>,
    namer: Rc<Namer>,
}

impl<'c, H: HugrView> EmitModuleContext<'c, H> {
    delegate! {
        to self.typer {
            /// Returns a reference to the inner [Context].
            pub fn iw_context(&self) -> &'c Context;
        }
        to self.typer.clone() {
            /// Convert hugr [hugr::types::Type] into an LLVM [Type](BasicTypeEnum).
            pub fn llvm_type(&self, [self.extensions()], hugr_type: &hugr::types::Type) -> Result<BasicTypeEnum<'c>>;
            /// Convert a hugr (FunctionType)[hugr::types::FunctionType] into an LLVM [FunctionType].
            pub fn llvm_func_type(&self, [self.extensions()], hugr_type: &hugr::types::FunctionType) -> Result<FunctionType<'c>>;
            /// Convert a hugr [hugr::types::SumType] into an LLVM [LLVMSumType].
            pub fn llvm_sum_type(&self, [self.extensions()], sum_ty: hugr::types::SumType) -> Result<LLVMSumType<'c>>;
        }

        to self.namer {
            /// Mangle the name of a [FuncDefn]  or a [FuncDecl].
            pub fn name_func(&self, name: impl AsRef<str>, node: Node) -> String;
        }
    }

    /// Creates a new  `EmitModuleContext`. We take ownership of the [Module],
    /// and return it in [EmitModuleContext::finish].
    pub fn new(
        module: Module<'c>,
        namer: Rc<Namer>,
        extension_context: Rc<CodegenExtsMap<'c, H>>,
        typer: Rc<TypeConverter<'c>>,
    ) -> Self {
        Self {
            module,
            namer,
            extension_context,
            typer,
        }
    }

    /// Returns a reference to the inner [Module]. Note that this type has
    /// "interior mutability", and this reference can be used to add functions
    /// and globals to the [Module].
    pub fn module(&self) -> &Module<'c> {
        &self.module
    }

    /// Returns a reference to the inner [CodegenExtsMap].
    pub fn extensions(&self) -> Rc<CodegenExtsMap<'c, H>> {
        self.extension_context.clone()
    }

    /// Returns a [TypingSession] constructed from it's members.
    pub fn typing_session(&self) -> TypingSession<'c, H> {
        self.typer.clone().session::<H>(self.extensions())
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
            ))?
        }
        Ok(func)
    }

    fn get_hugr_func_impl(
        &self,
        name: impl AsRef<str>,
        node: Node,
        func_ty: &PolyFuncType,
    ) -> Result<FunctionValue<'c>> {
        let func_ty = (func_ty.params().is_empty())
            .then_some(func_ty.body())
            .ok_or(anyhow!("function has type params"))?;
        let llvm_func_ty = self.llvm_func_type(func_ty)?;
        let name = self.name_func(name, node);
        self.get_func_impl(name, llvm_func_ty, None)
    }

    /// Adds or gets the [FunctionValue] in the [Module] corresponding to the given [FuncDefn].
    ///
    /// The name of the result is mangled by [EmitModuleContext::name_func].
    pub fn get_func_defn(&self, node: FatNode<'c, FuncDefn, H>) -> Result<FunctionValue<'c>> {
        self.get_hugr_func_impl(&node.name, node.node(), &node.signature)
    }

    /// Adds or gets the [FunctionValue] in the [Module] corresponding to the given [FuncDecl].
    ///
    /// The name of the result is mangled by [EmitModuleContext::name_func].
    pub fn get_func_decl(&self, node: FatNode<'c, FuncDecl, H>) -> Result<FunctionValue<'c>> {
        self.get_hugr_func_impl(&node.name, node.node(), &node.signature)
    }

    /// Adds or get the [FunctionValue] in the [Module] with the given symbol
    /// and function type.
    ///
    /// The name undergoes no mangling. The [FunctionValue] will have
    /// [Linkage::External].
    ///
    /// If this function is called multiple times with the same arguments it
    /// will return the same [FunctionValue].
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

    /// Consumes the `EmitModuleContext` and returns the internal [Module].
    pub fn finish(self) -> Module<'c> {
        self.module
    }
}

/// TODO
type EmissionSet<'c, H> = HashSet<Emission<'c, H>>;

/// An enum with a constructor for each [OpType] which can be emitted by [EmitHugr].
pub enum Emission<'c, H> {
    FuncDefn(FatNode<'c, FuncDefn, H>),
    FuncDecl(FatNode<'c, FuncDecl, H>),
}

impl<'c, H> From<FatNode<'c, FuncDefn, H>> for Emission<'c, H> {
    fn from(value: FatNode<'c, FuncDefn, H>) -> Self {
        Self::FuncDefn(value)
    }
}

impl<'c, H> From<FatNode<'c, FuncDecl, H>> for Emission<'c, H> {
    fn from(value: FatNode<'c, FuncDecl, H>) -> Self {
        Self::FuncDecl(value)
    }
}

impl<'c, H> PartialEq for Emission<'c, H> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::FuncDefn(l0), Self::FuncDefn(r0)) => l0 == r0,
            (Self::FuncDecl(l0), Self::FuncDecl(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<'c, H> Eq for Emission<'c, H> {}

impl<'c, H> Hash for Emission<'c, H> {
    fn hash<HA: std::hash::Hasher>(&self, state: &mut HA) {
        match self {
            Emission::FuncDefn(x) => x.hash(state),
            Emission::FuncDecl(x) => x.hash(state),
        }
    }
}

impl<'c, H> Clone for Emission<'c, H> {
    fn clone(&self) -> Self {
        match self {
            Self::FuncDefn(arg0) => Self::FuncDefn(arg0.clone()),
            Self::FuncDecl(arg0) => Self::FuncDecl(arg0.clone()),
        }
    }
}

/// Emits [HugrView]s into an LLVM [Module].
pub struct EmitHugr<'c, H: HugrView> {
    // funcs: HashMap<Node, FunctionValue<'c>>,
    // globals: HashMap<Node, GlobalValue<'c>>,
    emitted: EmissionSet<'c, H>,
    module_context: EmitModuleContext<'c, H>,
}

impl<'c, H: HugrView> EmitHugr<'c, H> {
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

    /// Creates a new  `EmitHugr`. We take ownership of the [Module], and return it in [Self::finish].
    pub fn new(
        iw_context: &'c Context,
        module: Module<'c>,
        exts: Rc<CodegenExtsMap<'c, H>>,
    ) -> Self {
        assert_eq!(iw_context, &module.get_context());
        Self {
            // todos: Default::default(),
            emitted: Default::default(),
            module_context: EmitModuleContext::new(
                module,
                Default::default(),
                exts,
                TypeConverter::new(iw_context),
            ),
        }
    }

    /// Emits a global node (either a [FuncDefn] or [FuncDecl]) into the inner [Module].
    ///
    /// `node` need not be a child of a hugr [Module](hugr::ops::Module), but it will
    /// be emitted as a top-level function in the inner [Module]. Indeed, there
    /// are only top-level functions in LLVM IR.
    ///
    /// Any child [FuncDefn] (or [FuncDecl], although those are currently
    /// prohibited by hugr validation) will also be emitted.
    ///
    /// It is safe to emit the same node multiple times, it will be detected and
    /// omitted.
    ///
    /// If any LLVM IR declaration which is to be emitted already exists in the
    /// [Module] and it differs from what would be emitted, then we fail.
    pub fn emit_global(mut self, node: impl Into<Emission<'c, H>>) -> Result<Self> {
        let mut worklist: EmissionSet<'c, H> = [node.into()].into_iter().collect();
        let pop =
            |wl: &mut EmissionSet<'c, H>| wl.iter().next().cloned().map(|x| wl.take(&x).unwrap());

        while let Some(x) = pop(&mut worklist) {
            let (new_self, new_tasks) = self.emit_global_impl(x)?;
            self = new_self;
            worklist.extend(new_tasks.into_iter());
        }
        Ok(self)
    }

    /// Emits all children of a hugr [Module](hugr::ops::Module).
    ///
    /// Note that type aliases are not supported, and [hugr::ops::Const] nodes
    /// are not emitted directly, but instead by [hugr::ops::LoadConstant] emission. So
    /// [FuncDefn] and [FuncDecl] are the only interesting children.
    pub fn emit_module(mut self, node: FatNode<'c, hugr::ops::Module, H>) -> Result<Self> {
        println!("emit module");
        for c in node.children() {
            println!("emit child: {}", &c);
            match c.get() {
                OpType::FuncDefn(ref fd) => {
                    self = self.emit_global(c.into_ot(fd))?;
                }
                _ => todo!("emit_module: unimplemented: {}", c.name()),
            }
        }
        Ok(self)
    }

    fn emit_global_impl(mut self, em: Emission<'c, H>) -> Result<(Self, EmissionSet<'c, H>)> {
        if !self.emitted.insert(em.clone()) {
            return Ok((self, Default::default()));
        }
        match em {
            Emission::FuncDefn(f) => self.emit_func_impl(f),
            Emission::FuncDecl(_) => todo!(), // Emission::Const(_) => todo!(),
        }
    }

    fn emit_func_impl(
        mut self,
        node: FatNode<'c, FuncDefn, H>,
    ) -> Result<(Self, EmissionSet<'c, H>)> {
        let func = self.module_context.get_func_defn(node.clone())?;
        let mut func_ctx = EmitFuncContext::new(self.module_context, func)?;
        let ret_rmb = func_ctx.new_row_mail_box(node.signature.body().output.iter(), "ret")?;
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

#[cfg(test)]
mod test;

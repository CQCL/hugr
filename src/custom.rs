use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use hugr::{
    extension::ExtensionId,
    ops::{constant::CustomConst, CustomOp},
    types::CustomType,
    HugrView,
};

use anyhow::{anyhow, Result};
use inkwell::{types::BasicTypeEnum, values::BasicValueEnum};

use crate::{
    emit::{func::EmitFuncContext, EmitOpArgs},
    types::TypingSession,
};

use super::emit::EmitOp;

pub mod float;
pub mod int;
pub mod logic;
pub mod prelude;

/// The extension point for lowering HUGR Extensions to LLVM.
pub trait CodegenExtension<'c, H> {
    /// The [ExtensionId] for which this extension will lower `ExtensionOp`s and
    /// [CustomType]s.
    ///
    /// Note that a [CodegenExtsMap] will only delegate to a single
    /// `CodegenExtension` per [ExtensionId].
    fn extension(&self) -> ExtensionId;

    /// The [TypeId]s for which [dyn CustomConst](CustomConst)s should be passed
    /// to [Self::load_constant].
    ///
    /// Defaults to an empty set.
    fn supported_consts(&self) -> HashSet<TypeId> {
        Default::default()
    }

    /// Return the type of the given [CustomType], which will have an extension
    /// that matches `Self`.
    fn llvm_type(
        &self,
        context: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>>;

    /// Return an emitter that will be asked to emit `CustomOp`s that have an
    /// extension that matches `Self.`
    fn emitter<'a>(
        &'a self,
        context: &'a mut EmitFuncContext<'c, H>,
    ) -> Box<dyn EmitOp<'c, CustomOp, H> + 'a>;

    /// Emit instructions to materialise `konst`. `konst` will have a [TypeId]
    /// that matches `self.supported_consts`.
    ///
    /// If the result is `Ok(None)`, [CodegenExtsMap] may try other
    /// `CodegenExtension`s.
    fn load_constant(
        &self,
        #[allow(unused)] context: &mut EmitFuncContext<'c, H>,
        #[allow(unused)] konst: &dyn CustomConst,
    ) -> Result<Option<BasicValueEnum<'c>>> {
        Ok(None)
    }
}

/// A collection of [CodegenExtension]s.
///
/// Provides methods to delegate operations to appropriate contained
/// [CodegenExtension]s.
pub struct CodegenExtsMap<'c, H> {
    supported_consts: HashMap<TypeId, HashSet<ExtensionId>>,
    extensions: HashMap<ExtensionId, Box<dyn 'c + CodegenExtension<'c, H>>>,
}

impl<'c, H> CodegenExtsMap<'c, H> {
    /// Create a new, empty, `CodegenExtsMap`.
    pub fn new() -> Self {
        Self {
            supported_consts: Default::default(),
            extensions: Default::default(),
        }
    }

    /// Consumes a `CodegenExtsMap` and returns a new one, with `ext`
    /// incorporated.
    pub fn add_cge(mut self, ext: impl 'c + CodegenExtension<'c, H>) -> Self {
        let extension = ext.extension();
        for k in ext.supported_consts() {
            self.supported_consts
                .entry(k)
                .or_default()
                .insert(extension.clone());
        }
        self.extensions.insert(extension, Box::new(ext));
        self
    }

    /// Returns the matching inner [CodegenExtension] if it exists.
    pub fn get(&self, extension: &ExtensionId) -> Result<&dyn CodegenExtension<'c, H>> {
        let b = self
            .extensions
            .get(extension)
            .ok_or(anyhow!("CodegenExtsMap: Unknown extension: {}", extension))?;
        Ok(b.as_ref())
    }

    /// Return the type of the given [CustomType] by delegating to the
    /// appropriate inner [CodegenExtension].
    pub fn llvm_type(
        &self,
        ts: &TypingSession<'c, H>,
        hugr_type: &CustomType,
    ) -> Result<BasicTypeEnum<'c>> {
        self.get(hugr_type.extension())?.llvm_type(ts, hugr_type)
    }

    /// Emit instructions for `args` by delegating to the appropriate inner
    /// [CodegenExtension].
    pub fn emit(
        self: Rc<Self>,
        context: &mut EmitFuncContext<'c, H>,
        args: EmitOpArgs<'c, CustomOp, H>,
    ) -> Result<()>
    where
        H: HugrView,
    {
        let node = args.node();
        self.get(custom_op_extension(&node))?
            .emitter(context)
            .emit(args)
    }

    /// Emit instructions to materialise `konst` by delegating to the
    /// appropriate inner [CodegenExtension]s.
    pub fn load_constant(
        &self,
        context: &mut EmitFuncContext<'c, H>,
        konst: &dyn CustomConst,
    ) -> Result<BasicValueEnum<'c>> {
        let type_id = konst.type_id();
        self.supported_consts
            .get(&type_id)
            .into_iter()
            .flatten()
            .filter_map(|ext| {
                let cge = self.extensions.get(ext).unwrap();
                match cge.load_constant(context, konst) {
                    Err(e) => Some(Err(e)),
                    Ok(None) => None,
                    Ok(Some(v)) => Some(Ok(v)),
                }
            })
            .next()
            .unwrap_or(Err(anyhow!(
                "No extension could load constant name: {} type_id: {type_id:?}",
                konst.name()
            )))
    }
}

impl<'c, H: HugrView> Default for CodegenExtsMap<'c, H> {
    fn default() -> Self {
        Self::new()
    }
}

// TODO upstream this to hugr
fn custom_op_extension(o: &CustomOp) -> &ExtensionId {
    match o {
        CustomOp::Extension(e) => e.def().extension(),
        CustomOp::Opaque(o) => o.extension(),
    }
}

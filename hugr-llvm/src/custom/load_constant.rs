//! Provides the implementation for a collection of [`CustomConst`] callbacks.
use std::{any::TypeId, collections::BTreeMap};

use hugr_core::{HugrView, Node, ops::constant::CustomConst};
use inkwell::values::BasicValueEnum;

use anyhow::{Result, anyhow, bail, ensure};

use crate::emit::EmitFuncContext;

/// A helper trait for describing the callback used for emitting [`CustomConst`]s,
/// and for hanging documentation.
///
/// We have the appropriate `Fn` as a supertrait,
/// and there is a blanket impl for that `Fn`. We do not intend users to impl
/// this trait.
///
/// `LoadConstantFn` callbacks for `CC`, which must impl [`CustomConst`], should
/// materialise an appropriate [`BasicValueEnum`]. The type of this value must
/// match the result of [`EmitFuncContext::llvm_type`] on [`CustomConst::get_type`].
///
/// Callbacks may hold references with lifetimes older than `'a`.
pub trait LoadConstantFn<'a, H: ?Sized, CC: CustomConst + ?Sized>:
    for<'c> Fn(&mut EmitFuncContext<'c, 'a, H>, &CC) -> Result<BasicValueEnum<'c>> + 'a
{
}

impl<
    'a,
    H: ?Sized,
    CC: ?Sized + CustomConst,
    F: 'a + ?Sized + for<'c> Fn(&mut EmitFuncContext<'c, 'a, H>, &CC) -> Result<BasicValueEnum<'c>>,
> LoadConstantFn<'a, H, CC> for F
{
}

/// A collection of [`LoadConstantFn`] callbacks registered for various impls of [`CustomConst`].
/// The callbacks are keyed by the [`TypeId`] of those impls.
#[derive(Default)]
pub struct LoadConstantsMap<'a, H>(
    BTreeMap<TypeId, Box<dyn LoadConstantFn<'a, H, dyn CustomConst>>>,
);

impl<'a, H: HugrView<Node = Node>> LoadConstantsMap<'a, H> {
    /// Register a callback to emit a `CC` value.
    ///
    /// If a callback is already registered for that type, we will replace it.
    pub fn custom_const<CC: CustomConst>(&mut self, handler: impl LoadConstantFn<'a, H, CC>) {
        self.0.insert(
            TypeId::of::<CC>(),
            Box::new(move |context, konst: &dyn CustomConst| {
                let cc = konst.downcast_ref::<CC>().ok_or(anyhow!(
                    "impossible! Failed to downcast in LoadConstantsMap::custom_const"
                ))?;
                handler(context, cc)
            }),
        );
    }

    /// Emit instructions to materialise `konst` by delegating to the
    /// appropriate inner callbacks.
    pub fn emit_load_constant<'c>(
        &self,
        context: &mut EmitFuncContext<'c, 'a, H>,
        konst: &dyn CustomConst,
    ) -> Result<BasicValueEnum<'c>> {
        let type_id = konst.type_id();
        let Some(handler) = self.0.get(&type_id) else {
            bail!(
                "No extension could load constant name: {} type_id: {type_id:?}",
                konst.name()
            )
        };
        let r = handler(context, konst)?;
        let r_type = r.get_type();
        let konst_type = context.llvm_type(&konst.get_type())?;
        ensure!(
            r_type == konst_type,
            "CustomConst handler returned a value of the wrong type. Expected: {konst_type} Actual: {r_type}"
        );
        Ok(r)
    }
}

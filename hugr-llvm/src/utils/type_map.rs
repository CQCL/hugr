//! Provides a generic mapping from [`HugrType`] into some domain.
use std::collections::BTreeMap;

use hugr_core::{
    extension::ExtensionId,
    types::{CustomType, TypeEnum, TypeName, TypeRow},
};

use anyhow::{Result, bail};

use crate::types::{HugrFuncType, HugrSumType, HugrType};

pub trait TypeMapFnHelper<'c, TM: TypeMapping>:
    Fn(TM::InV<'c>, &CustomType) -> Result<TM::OutV<'c>>
{
}

impl<'c, TM: TypeMapping, F> TypeMapFnHelper<'c, TM> for F where
    F: Fn(TM::InV<'c>, &CustomType) -> Result<TM::OutV<'c>> + ?Sized
{
}

/// A helper trait to name the type of the Callback used by
/// [`TypeMap<TM>`](TypeMap).
pub trait TypeMappingFn<'a, TM: TypeMapping>: 'a {
    fn map_type<'c>(&self, inv: TM::InV<'c>, ty: &CustomType) -> Result<TM::OutV<'c>>;
}

impl<'a, TM: TypeMapping, F: for<'c> TypeMapFnHelper<'c, TM> + 'a> TypeMappingFn<'a, TM> for F {
    fn map_type<'c>(&self, inv: TM::InV<'c>, ty: &CustomType) -> Result<TM::OutV<'c>> {
        self(inv, ty)
    }
}

/// Defines a mapping from [`HugrType`] to `OutV`;
pub trait TypeMapping {
    /// Auxiliary data provided when mapping from a [`HugrType`].
    type InV<'c>: Clone;
    /// The target type of the mapping.
    type OutV<'c>;
    /// The target type when mapping from [`HugrSumType`]s. This type must be
    /// convertible to `OutV` via `sum_into_out`.
    type SumOutV<'c>;
    /// The target type when mapping from [`HugrFuncType`]s. This type must be
    /// convertible to `OutV` via `func_into_out`.
    type FuncOutV<'c>;

    /// Returns the result of the mapping on `sum_type`, with auxiliary data
    /// `inv`, and when the result of mapping all fields of all variants is
    /// given by `variants`.
    fn map_sum_type<'c>(
        &self,
        sum_type: &HugrSumType,
        inv: Self::InV<'c>,
        variants: impl IntoIterator<Item = Vec<Self::OutV<'c>>>,
    ) -> Result<Self::SumOutV<'c>>;

    /// Returns the result of the mapping on `function_type`, with auxiliary data
    /// `inv`, and when the result of mapping all inputs is given by `inputs`
    /// and the result of mapping all outputs is given by `outputs`.
    fn map_function_type<'c>(
        &self,
        function_type: &HugrFuncType,
        inv: Self::InV<'c>,
        inputs: impl IntoIterator<Item = Self::OutV<'c>>,
        outputs: impl IntoIterator<Item = Self::OutV<'c>>,
    ) -> Result<Self::FuncOutV<'c>>;

    /// Infallibly convert from the result of `map_sum_type` to the result of
    /// the mapping.
    fn sum_into_out<'c>(&self, sum: Self::SumOutV<'c>) -> Self::OutV<'c>;

    /// Infallibly convert from the result of `map_functype` to the result of
    /// the mapping.
    fn func_into_out<'c>(&self, sum: Self::FuncOutV<'c>) -> Self::OutV<'c>;

    /// Construct an appropriate result of the mapping when `hugr_type` is not a
    /// function, sum, registered custom type, or composition of same.
    fn default_out<'c>(
        &self,
        #[allow(unused)] inv: Self::InV<'c>,
        hugr_type: &HugrType,
    ) -> Result<Self::OutV<'c>> {
        bail!("Unknown type: {hugr_type}")
    }
}

pub type CustomTypeKey = (ExtensionId, TypeName);

/// An impl of `TypeMapping` together with a collection of callbacks
/// implementing the mapping.
///
/// Callbacks may hold references with lifetimes longer than `'a`
#[derive(Default)]
pub struct TypeMap<'a, TM: TypeMapping> {
    type_map: TM,
    custom_hooks: BTreeMap<CustomTypeKey, Box<dyn TypeMappingFn<'a, TM> + 'a>>,
}

impl<'a, TM: TypeMapping + 'a> TypeMap<'a, TM> {
    /// Sets the callback for the given custom type.
    ///
    /// Returns false if this callback replaces another callback, which is
    /// discarded, and true otherwise.
    pub fn set_callback(
        &mut self,
        custom_type_key: CustomTypeKey,
        hook: impl TypeMappingFn<'a, TM> + 'a,
    ) -> bool {
        self.custom_hooks
            .insert(custom_type_key, Box::new(hook))
            .is_none()
    }

    /// Map `hugr_type` using the [`TypeMapping`] `TM`, the registered callbacks,
    /// and the auxiliary data `inv`.
    pub fn map_type<'c>(&self, hugr_type: &HugrType, inv: TM::InV<'c>) -> Result<TM::OutV<'c>> {
        match hugr_type.as_type_enum() {
            TypeEnum::Extension(custom_type) => {
                let key = (custom_type.extension().clone(), custom_type.name().clone());
                let Some(handler) = self.custom_hooks.get(&key) else {
                    return self.type_map.default_out(inv, &custom_type.clone().into());
                };
                handler.map_type(inv, custom_type)
            }
            TypeEnum::Sum(sum_type) => self
                .map_sum_type(sum_type, inv)
                .map(|x| self.type_map.sum_into_out(x)),
            TypeEnum::Function(function_type) => self
                .map_function_type(&function_type.as_ref().clone().try_into()?, inv)
                .map(|x| self.type_map.func_into_out(x)),
            _ => self.type_map.default_out(inv, hugr_type),
        }
    }

    /// As `map_type`, but maps a [`HugrSumType`] to an [`TypeMapping::SumOutV`].
    pub fn map_sum_type<'c>(
        &self,
        sum_type: &HugrSumType,
        inv: TM::InV<'c>,
    ) -> Result<TM::SumOutV<'c>> {
        let inv2 = inv.clone();
        self.type_map.map_sum_type(
            sum_type,
            inv,
            (0..sum_type.num_variants())
                .map(move |i| {
                    let tr: TypeRow = sum_type.get_variant(i).unwrap().clone().try_into().unwrap();
                    tr.iter()
                        .map(|t| self.map_type(t, inv2.clone()))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        )
    }

    /// As `map_type`, but maps a [`HugrSumType`] to an [`TypeMapping::FuncOutV`].
    pub fn map_function_type<'c>(
        &self,
        func_type: &HugrFuncType,
        inv: TM::InV<'c>,
    ) -> Result<TM::FuncOutV<'c>> {
        let inputs = func_type
            .input()
            .iter()
            .map(|t| self.map_type(t, inv.clone()))
            .collect::<Result<Vec<_>>>()?;
        let outputs = func_type
            .output()
            .iter()
            .map(|t| self.map_type(t, inv.clone()))
            .collect::<Result<Vec<_>>>()?;
        self.type_map
            .map_function_type(func_type, inv, inputs, outputs)
    }
}

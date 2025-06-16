//! Classes for row variables (i.e. Type variables that can stand for multiple types)

use super::type_param::TypeParam;
use super::{Substitution, TypeBase, TypeBound, check_typevar_decl};
use crate::extension::SignatureError;

#[cfg(test)]
use proptest::prelude::{BoxedStrategy, Strategy, any};
/// Describes a row variable - a type variable bound with a list of runtime types
/// of the specified bound (checked in validation)
// The serde derives here are not used except as markers
// so that other types containing this can also #derive-serde the same way.
#[derive(
    Clone, Debug, Eq, Hash, PartialEq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display("{_0}")]
pub struct RowVariable(pub usize, pub TypeBound);

// Note that whilst 'pub' this is not re-exported outside private module `row_var`
// so is effectively sealed.
pub trait MaybeRV:
    Clone
    + std::fmt::Debug
    + std::fmt::Display
    + From<NoRV>
    + Into<RowVariable>
    + Eq
    + PartialEq
    + 'static
{
    fn as_rv(&self) -> &RowVariable;
    fn try_from_rv(rv: RowVariable) -> Result<Self, RowVariable>;
    fn bound(&self) -> TypeBound;
    fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError>;
    #[allow(private_interfaces)]
    fn substitute(&self, s: &Substitution) -> Vec<TypeBase<Self>>;
    #[cfg(test)]
    fn weight() -> u32 {
        1
    }
    #[cfg(test)]
    fn arb() -> BoxedStrategy<Self>;
}

/// Has no instances - used as parameter to [`Type`] to rule out the possibility
/// of there being any [`TypeEnum::RowVar`]s
///
/// [`TypeEnum::RowVar`]: super::TypeEnum::RowVar
/// [`Type`]: super::Type
// The serde derives here are not used except as markers
// so that other types containing this can also #derive-serde the same way.
#[derive(
    Clone, Debug, Eq, PartialEq, Hash, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub enum NoRV {}

impl From<NoRV> for RowVariable {
    fn from(value: NoRV) -> Self {
        match value {}
    }
}

impl MaybeRV for RowVariable {
    fn as_rv(&self) -> &RowVariable {
        self
    }

    fn bound(&self) -> TypeBound {
        self.1
    }

    fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        check_typevar_decl(var_decls, self.0, &TypeParam::new_list_type(self.1))
    }

    #[allow(private_interfaces)]
    fn substitute(&self, s: &Substitution) -> Vec<TypeBase<Self>> {
        s.apply_rowvar(self.0, self.1)
    }

    fn try_from_rv(rv: RowVariable) -> Result<Self, RowVariable> {
        Ok(rv)
    }

    #[cfg(test)]
    fn arb() -> BoxedStrategy<Self> {
        (any::<usize>(), any::<TypeBound>())
            .prop_map(|(i, b)| Self(i, b))
            .boxed()
    }
}

impl MaybeRV for NoRV {
    fn as_rv(&self) -> &RowVariable {
        match *self {}
    }

    fn bound(&self) -> TypeBound {
        match *self {}
    }

    fn validate(&self, _var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        match *self {}
    }

    #[allow(private_interfaces)]
    fn substitute(&self, _s: &Substitution) -> Vec<TypeBase<Self>> {
        match *self {}
    }

    fn try_from_rv(rv: RowVariable) -> Result<Self, RowVariable> {
        Err(rv)
    }

    #[cfg(test)]
    fn weight() -> u32 {
        0
    }

    #[cfg(test)]
    fn arb() -> BoxedStrategy<Self> {
        any::<usize>()
            .prop_map(|_| panic!("Should be ruled out by weight==0"))
            .boxed()
    }
}

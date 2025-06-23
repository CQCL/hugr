//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
    sync::Arc,
};

use super::{Substitution, Transformable, Type, TypeRV, TypeTransformer, type_param::TypeParam};
use crate::{extension::SignatureError, utils::display_list};
use delegate::delegate;
use itertools::Itertools;

/// Row of single types i.e. of known length, for node inputs/outputs
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TypeRow(Arc<[Type]>);

impl TypeRow {
    /// Create a new empty row.
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new().into())
    }

    /// Returns the number of types in the row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    /// Returns the type at the specified index. Returns `None` if out of bounds.
    #[must_use]
    pub fn get(&self, offset: usize) -> Option<&Type> {
        self.0.get(offset)
    }

    #[inline(always)]
    /// Returns the type at the specified index. Returns `None` if out of bounds.
    pub fn get_mut(&mut self, offset: usize) -> Option<&mut Type> {
        self.as_mut().get_mut(offset)
    }

    /// Iterates over the types in the row.
    #[inline(always)]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Type> {
        self.0.iter()
    }

    /// Borrows the type row as a slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[Type] {
        &self.0
    }

    /// Applies a substitution to the row.
    pub(crate) fn substitute(&self, s: &Substitution) -> Self {
        self.iter()
            .flat_map(|ty| ty.substitute(s))
            .collect::<Vec<_>>()
            .into()
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        self.iter().try_for_each(|t| t.validate(var_decls))
    }

    fn as_mut(&mut self) -> &mut [Type] {
        if let None = Arc::get_mut(&mut self.0) {
            *self = self.iter().cloned().collect();
        }

        Arc::get_mut(&mut self.0).unwrap()
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a Type>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }
}

impl FromIterator<Type> for TypeRow {
    fn from_iter<T: IntoIterator<Item = Type>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<Vec<Type>> for TypeRow {
    fn from(value: Vec<Type>) -> Self {
        Self(value.into())
    }
}

impl<const N: usize> From<[Type; N]> for TypeRow {
    fn from(value: [Type; N]) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for TypeRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.0.as_ref(), f)?;
        f.write_char(']')
    }
}

impl Deref for TypeRow {
    type Target = [Type];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for TypeRow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl Transformable for TypeRow {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        let mut changed = false;
        for type_ in self.as_mut() {
            changed = changed || type_.transform(tr)?;
        }
        Ok(changed)
    }
}

/// List of types, used for function signatures.
#[derive(Clone, Default, PartialEq, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRowRV(Vec<TypeRV>);

impl Display for TypeRowRV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.0.iter(), f)?;
        f.write_char(']')
    }
}

impl TypeRowRV {
    /// Create a new empty row.
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a TypeRV>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }

    /// Returns a reference to the types in the row.
    #[must_use]
    pub fn as_slice(&self) -> &[TypeRV] {
        &self.0
    }

    /// Applies a substitution to the row.
    /// For `TypeRowRV`, note this may change the length of the row.
    /// For `TypeRow`, guaranteed not to change the length of the row.
    pub(crate) fn substitute(&self, s: &Substitution) -> Self {
        self.iter()
            .flat_map(|ty| ty.substitute(s))
            .collect::<Vec<_>>()
            .into()
    }

    delegate! {
        to self.0 {
            /// Iterator over the types in the row.
            pub fn iter(&self) -> impl Iterator<Item = &TypeRV>;

            /// Returns `true` if the row contains no types.
            #[must_use] pub fn is_empty(&self) -> bool ;
        }
    }

    /// Mutable vector of the types in the row.
    pub fn to_mut(&mut self) -> &mut Vec<TypeRV> {
        &mut self.0
    }

    /// Allow access (consumption) of the contained elements
    #[must_use]
    pub fn into_owned(self) -> Vec<TypeRV> {
        self.0
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        self.iter().try_for_each(|t| t.validate(var_decls))
    }
}

impl Transformable for TypeRowRV {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        self.to_mut().transform(tr)
    }
}

impl TryFrom<TypeRowRV> for TypeRow {
    type Error = SignatureError;

    fn try_from(value: TypeRowRV) -> Result<Self, Self::Error> {
        Ok(Self::from(
            value
                .into_owned()
                .into_iter()
                .map(std::convert::TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|var| SignatureError::RowVarWhereTypeExpected { var })?,
        ))
    }
}

impl From<Vec<TypeRV>> for TypeRowRV {
    fn from(types: Vec<TypeRV>) -> Self {
        Self(types.into())
    }
}

impl From<Vec<Type>> for TypeRowRV {
    fn from(types: Vec<Type>) -> Self {
        Self(types.into_iter().map(Type::into_).collect())
    }
}

impl<const N: usize> From<[TypeRV; N]> for TypeRowRV {
    fn from(types: [TypeRV; N]) -> Self {
        Self(types.into_iter().collect())
    }
}

impl<const N: usize> From<[Type; N]> for TypeRowRV {
    fn from(types: [Type; N]) -> Self {
        Self(types.into_iter().map_into().collect())
    }
}

impl From<TypeRow> for TypeRowRV {
    fn from(value: TypeRow) -> Self {
        Self(value.iter().cloned().map(Type::into_).collect())
    }
}

impl From<&'static [TypeRV]> for TypeRowRV {
    fn from(types: &'static [TypeRV]) -> Self {
        Self(types.into())
    }
}

impl Deref for TypeRowRV {
    type Target = [TypeRV];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for TypeRowRV {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

#[cfg(test)]
mod test {
    mod proptest {
        use crate::proptest::RecursionDepth;
        use crate::types::{Type, TypeRV, TypeRow, TypeRowRV};
        use ::proptest::prelude::*;

        impl Arbitrary for TypeRowRV {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    Just(TypeRowRV::new()).boxed()
                } else {
                    vec(any_with::<TypeRV>(depth), 0..4)
                        .prop_map(|ts| ts.clone().into())
                        .boxed()
                }
            }
        }

        impl Arbitrary for TypeRow {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;

            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    Just(TypeRow::new()).boxed()
                } else {
                    vec(any_with::<Type>(depth), 0..4)
                        .prop_map(|ts| ts.clone().into())
                        .boxed()
                }
            }
        }
    }
}

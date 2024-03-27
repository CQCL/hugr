//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::{Type, TypeBound};
use crate::utils::display_list;
use crate::PortIndex;
use delegate::delegate;
use itertools::Itertools;

#[derive(
    Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, derive_more::Display,
)]
pub enum RowVarOrType {
    #[display(fmt = "{}", _0)]
    T(Type),
    /// DeBruijn index, and cache of inner TypeBound - matches a [TypeParam::List] of [TypeParam::Type]
    /// of this bound (checked in validation)
    #[display(fmt = "RowVar({})", _0)]
    RV(usize, TypeBound),
}

impl RowVarOrType {
    pub fn least_upper_bound(&self) -> TypeBound {
        match self {
            RowVarOrType::T(t) => t.least_upper_bound(),
            RowVarOrType::RV(_, b) => *b,
        }
    }
}

impl From<Type> for RowVarOrType {
    fn from(value: Type) -> Self {
        Self::T(value)
    }
}

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRowBase<T>
where
    T: 'static, //Clone + PartialEq + Eq + std::fmt::Debug + serde::Serialize + 'static,
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// The datatypes in the row.
    types: Cow<'static, [T]>,
}

pub type TypeRow = TypeRowBase<Type>;
pub type TypeRowV = TypeRowBase<RowVarOrType>;

impl<T: Display> Display for TypeRowBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl<T> TypeRowBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: impl PortIndex) -> Option<&T> {
        self.types.get(offset.index())
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get_mut(&mut self, offset: impl PortIndex) -> Option<&mut T> {
        self.types.to_mut().get_mut(offset.index())
    }

    /// Returns a reference to the types and row variables in the row.
    pub fn as_slice(&self) -> &[T] {
        &self.types
    }

    delegate! {
        to self.types {
            /// Iterator over the types and row variables in the row.
            pub fn iter(&self) -> impl Iterator<Item = &T>;

            /// Mutable vector of the types and row variables in the row.
            pub fn to_mut(&mut self) -> &mut Vec<T>;

            /// Allow access (consumption) of the contained elements
            pub fn into_owned(self) -> Vec<T>;

            /// Returns `true` if the row contains no types or row variables
            /// (so will necessarily be empty after substitution, too).
            pub fn is_empty(&self) -> bool ;

        }
    }
}

impl TypeRow {
    delegate! {
        to self.types {
            /// Returns the number of types in the row.
            pub fn len(&self) -> usize;
        }
    }
}

impl<T> TypeRowBase<T>
where
    T: Clone,
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a T>) -> Self {
        Self {
            types: self.iter().chain(rest).cloned().collect(),
        }
    }
}

impl<T> Default for TypeRowBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> From<F> for TypeRow
where
    F: Into<Cow<'static, [Type]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl From<TypeRow> for Cow<'static, [RowVarOrType]> {
    fn from(val: TypeRow) -> Self {
        val.types
            .into_owned()
            .into_iter()
            .map(RowVarOrType::from)
            .collect()
    }
}

impl<F, T> From<F> for TypeRowV
where
    RowVarOrType: From<T>,
    F: IntoIterator<Item = T>,
{
    // Note: I tried "where F: Into<Cow<'static, [Type]>>" but
    // (a) that requires `types.into().into_owned().into_iter()`
    //     - both allow use of owned data without cloning, and use of borrowed data with clone
    //     - (this way might require an explicit `.cloned()` from the caller)
    //     but I think this looks no less efficient (?)
    // (b) can't then parameterize the impl over <T> because it's unconstrained
    //     (seems F: IntoIterator<Item=T> provides a constraint but F: Into<Cow<'static, [T]>> does not)
    fn from(types: F) -> Self {
        Self {
            types: types.into_iter().map(RowVarOrType::from).collect(),
        }
    }
}

impl<T> From<T> for TypeRowBase<T>
where
    T: Clone,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn from(value: T) -> Self {
        Self {
            types: vec![value].into(),
        }
    }
}

impl<T> Deref for TypeRowBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for TypeRowBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

/// Turns a row of [Type] into an iterator of [RowVarOrType].
/// An iterator of [Type] can be obtained by `into_owned().into_iter()`
impl IntoIterator for TypeRow {
    type Item = RowVarOrType;
    type IntoIter = itertools::MapInto<<Vec<Type> as IntoIterator>::IntoIter, RowVarOrType>;

    fn into_iter(self) -> Self::IntoIter {
        self.types.into_owned().into_iter().map_into()
    }
}

impl TryInto<TypeRow> for TypeRowV {
    type Error = (usize, TypeBound);

    fn try_into(self) -> Result<TypeRow, Self::Error> {
        self.types
            .into_owned()
            .into_iter()
            .map(|rvt| match rvt {
                RowVarOrType::T(ty) => Ok(ty),
                RowVarOrType::RV(idx, bound) => Err((idx, bound)),
            })
            .collect::<Result<Vec<Type>, _>>()
            .map(TypeRow::from)
    }
}

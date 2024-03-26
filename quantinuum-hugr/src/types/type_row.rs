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

#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, derive_more::Display)]
pub enum RowVarOrType {
    #[display(fmt="{}", _0)]
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
 where T: 'static,//Clone + PartialEq + Eq + std::fmt::Debug + serde::Serialize + 'static,
   [T]: ToOwned<Owned=Vec<T>> {
    /// The datatypes in the row.
    types: Cow<'static, [T]>,
}

pub type TypeRow = TypeRowBase<Type>;
pub type TypeRowV = TypeRowBase<RowVarOrType>;

impl<T: Display> Display for TypeRowBase<T> where [T]: ToOwned<Owned=Vec<T>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl <T> TypeRowBase<T> where [T]: ToOwned<Owned = Vec<T>> {
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

    /// Returns a reference to the types in the row.
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
            
            /// Returns the number of types in the row.
            pub fn len(&self) -> usize;

            /// Returns `true` if the row contains no types.
            pub fn is_empty(&self) -> bool ;            
        }
    }
}

impl <T> TypeRowBase<T> where T: Clone, [T]: ToOwned<Owned = Vec<T>> {
    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a T>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }
}

impl Into<Cow<'static, [RowVarOrType]>> for TypeRow {
    fn into(self: TypeRow) -> Cow<'static, [RowVarOrType]> {
        self.types.into_owned().into_iter().map(RowVarOrType::from).collect()
    }
}


impl <T> Default for TypeRowBase<T> where [T]: ToOwned<Owned = Vec<T>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T,F> From<F> for TypeRowBase<T> // impl <F> From<F> for TypeRowV where F: Into<Cow<'static, [RowVarOrType]>>
where
    F: Into<Cow<'static, [T]>>,
    [T]: ToOwned<Owned = Vec<T>>
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

/*impl <T> From<T> for TypeRowBase<T> {
    fn from(t: Type) -> Self {
        Self {
            types: vec![t].into(),
        }
    }
}*/

impl <T> Deref for TypeRowBase<T> where [T]: ToOwned<Owned=Vec<T>> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <T> DerefMut for TypeRowBase<T> where [T]: ToOwned<Owned=Vec<T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

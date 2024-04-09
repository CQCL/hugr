//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::Type;
use crate::utils::display_list;
use crate::PortIndex;
use delegate::delegate;
use itertools::Itertools;

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow {
    /// The datatypes in the row.
    types: Cow<'static, [Type]>,
}

impl Display for TypeRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl TypeRow {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: impl PortIndex) -> Option<&Type> {
        self.types.get(offset.index())
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get_mut(&mut self, offset: impl PortIndex) -> Option<&mut Type> {
        self.types.to_mut().get_mut(offset.index())
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a Type>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }

    /// Returns a reference to the types in the row.
    pub fn as_slice(&self) -> &[Type] {
        &self.types
    }

    delegate! {
        to self.types {
            /// Iterator over the types in the row.
            pub fn iter(&self) -> impl Iterator<Item = &Type>;

            /// Returns the number of types in the row.
            pub fn len(&self) -> usize;

            /// Mutable vector of the types in the row.
            pub fn to_mut(&mut self) -> &mut Vec<Type>;

            /// Allow access (consumption) of the contained elements
            pub fn into_owned(self) -> Vec<Type>;

            /// Returns `true` if the row contains no types.
            pub fn is_empty(&self) -> bool ;
        }
    }
}

impl Default for TypeRow {
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

impl From<Type> for TypeRow {
    fn from(t: Type) -> Self {
        Self {
            types: vec![t].into(),
        }
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
        self.types.to_mut()
    }
}

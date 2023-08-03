//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use crate::utils::display_list;

/// Base trait for anything that can be put in a [TypeRow]
pub trait TypeRowElem: Clone + 'static {}

impl<T: Clone + 'static> TypeRowElem for T {}

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
//#[cfg_attr(feature = "pyo3", pyclass)] // TODO: expose unparameterized versions
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow<T: TypeRowElem> {
    /// The datatypes in the row.
    types: Cow<'static, [T]>,
}

impl<T: Display + TypeRowElem> Display for TypeRow<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

// TODO some of these, but not all, will probably want exposing via
// pyo3 wrappers eventually.
impl<T: TypeRowElem> TypeRow<T> {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Iterator over the types in the row.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.types.iter()
    }

    /// Returns the number of types in the row.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns `true` if the row contains no types.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.types.len() == 0
    }
    /// Mutable iterator over the types in the row.
    pub fn to_mut(&mut self) -> &mut Vec<T> {
        self.types.to_mut()
    }

    /// Allow access (consumption) of the contained elements
    pub fn into_owned(self) -> Vec<T> {
        self.types.into_owned()
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: usize) -> Option<&T> {
        self.types.get(offset)
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        self.types.to_mut().get_mut(offset)
    }

    pub(super) fn try_convert_elems<D: TypeRowElem + TryFrom<T>>(
        self,
    ) -> Result<TypeRow<D>, D::Error> {
        let elems: Vec<D> = self
            .into_owned()
            .into_iter()
            .map(D::try_from)
            .collect::<Result<_, _>>()?;
        Ok(TypeRow::from(elems))
    }

    /// Converts the elements of this TypeRow into some other type that they can `.into()`
    pub fn map_into<T2: TypeRowElem + From<T>>(self) -> TypeRow<T2> {
        TypeRow::from(
            self.into_owned()
                .into_iter()
                .map(T2::from)
                .collect::<Vec<T2>>(),
        )
    }
}

impl<T: TypeRowElem> Default for TypeRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, T: TypeRowElem> From<F> for TypeRow<T>
where
    F: Into<Cow<'static, [T]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl<T: TypeRowElem> Deref for TypeRow<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl<T: TypeRowElem> DerefMut for TypeRow<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

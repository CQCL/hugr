//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use itertools::Itertools;

use crate::utils::display_list;

use super::{leaf::TypeClass, Type, TypeTag};

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug)]
//#[cfg_attr(feature = "pyo3", pyclass)] // TODO: expose unparameterized versions
#[non_exhaustive]
pub struct TypeRow<T: TypeClass> {
    /// The datatypes in the row.
    types: Cow<'static, [Type<T>]>,
}

impl<T: Display + TypeClass> Display for TypeRow<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

// TODO some of these, but not all, will probably want exposing via
// pyo3 wrappers eventually.
impl<T: TypeClass> TypeRow<T> {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Iterator over the types in the row.
    pub fn iter(&self) -> impl Iterator<Item = &Type<T>> {
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
    pub fn to_mut(&mut self) -> &mut Vec<Type<T>> {
        self.types.to_mut()
    }

    /// Allow access (consumption) of the contained elements
    pub fn into_owned(self) -> Vec<Type<T>> {
        self.types.into_owned()
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: usize) -> Option<&Type<T>> {
        self.types.get(offset)
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get_mut(&mut self, offset: usize) -> Option<&mut Type<T>> {
        self.types.to_mut().get_mut(offset)
    }

    /// Converts the elements of this TypeRow into some other type that they can `.into()`
    pub fn map_into<T2: TypeClass>(self) -> TypeRow<T2>
    where
        Type<T2>: From<Type<T>>,
    {
        TypeRow::from(
            self.into_owned()
                .into_iter()
                .map(Into::into)
                .collect::<Vec<Type<T2>>>(),
        )
    }

    /// Return the type row of variants required to define a Sum of Tuples type
    /// given the rows of each tuple
    pub fn predicate_variants_row(variant_rows: impl IntoIterator<Item = TypeRow<T>>) -> Self {
        variant_rows
            .into_iter()
            .map(Type::new_tuple)
            .collect_vec()
            .into()
    }

    /// Returns the smallest [TypeTag] that contains all elements of the row
    pub fn containing_tag(&self) -> TypeTag {
        self.iter()
            .map(Type::bounding_tag)
            .fold(TypeTag::Hashable, TypeTag::union)
    }
}

impl<T: TypeClass> Default for TypeRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, T: TypeClass> From<F> for TypeRow<T>
where
    F: Into<Cow<'static, [Type<T>]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl<T: TypeClass> Deref for TypeRow<T> {
    type Target = [Type<T>];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl<T: TypeClass> DerefMut for TypeRow<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

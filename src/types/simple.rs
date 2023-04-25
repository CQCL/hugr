//! Dataflow types

use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use smol_str::SmolStr;

use super::{custom::CustomType, Signature};
use crate::resource::ResourceSet;

/// A type that represents concrete data.
///
/// TODO: Derive pyclass
///
/// TODO: Compare performance vs flattening this into a single enum
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum SimpleType {
    Classic(ClassicType),
    Linear(LinearType),
}

/// Trait of primitive types (ClassicType or LinearType)
pub trait PrimType {
    // may be updated with functions in future for necessary shared functionality
    // across ClassicType and LinearType
    // currently used to constrain Container<T>
}

// For algebraic types Sum, Tuple if one element of type row is linear, the
// overall type is too
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Container<T: PrimType> {
    /// Variable sized list of T
    List(Box<T>),
    /// Hash map from hashable key type to value T
    Map(Box<(ClassicType, T)>),
    /// Product type, known-size tuple over elements of type row
    Tuple(Box<TypeRow>),
    /// Product type, variants are tagged by their position in the type row
    Sum(Box<TypeRow>),
    /// Known size array of T
    Array(Box<T>, usize),
    /// Named type defined by, but distinct from, T
    NewType(SmolStr, Box<T>),
}

impl From<Container<ClassicType>> for SimpleType {
    fn from(value: Container<ClassicType>) -> Self {
        Self::Classic(ClassicType::Container(value))
    }
}

impl From<Container<LinearType>> for SimpleType {
    fn from(value: Container<LinearType>) -> Self {
        Self::Linear(LinearType::Container(value))
    }
}

/// A type that represents concrete classical data.
///
/// Uses `Box`es on most variants to reduce the memory footprint.
///
/// TODO: Derive pyclass
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ClassicType {
    Variable(SmolStr),
    Int(usize),
    F64,
    String,
    Graph(Box<(ResourceSet, Signature)>),
    Container(Container<ClassicType>),
    /// An opaque operation that can be downcasted by the extensions that define it.
    Opaque(CustomType),
}

impl ClassicType {
    /// Create a graph type with the given signature, using default resources.
    /// TODO in the future we'll probably need versions of this that take resources.
    pub fn graph_from_sig(signature: Signature) -> Self {
        ClassicType::Graph(Box::new((Default::default(), signature)))
    }

    #[inline]
    pub const fn int<const N: usize>() -> Self {
        Self::Int(N)
    }

    #[inline]
    pub const fn i64() -> Self {
        Self::int::<64>()
    }

    #[inline]
    pub const fn bit() -> Self {
        Self::int::<1>()
    }
}

impl Default for ClassicType {
    fn default() -> Self {
        Self::int::<1>()
    }
}

impl PrimType for ClassicType {}

/// A type that represents concrete linear data.
///
/// TODO: Derive pyclass
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LinearType {
    #[default]
    Qubit,
    /// A linear opaque operation that can be downcasted by the extensions that define it.
    Qpaque(CustomType),
    Container(Container<LinearType>),
}

impl PrimType for LinearType {}

impl SimpleType {
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear(_))
    }

    pub fn is_classical(&self) -> bool {
        matches!(self, Self::Classic(_))
    }

    pub fn new_sum(row: TypeRow) -> Self {
        if row.purely_classical() {
            Container::<ClassicType>::Sum(Box::new(row)).into()
        } else {
            Container::<LinearType>::Sum(Box::new(row)).into()
        }
    }

    pub fn new_tuple(row: TypeRow) -> Self {
        if row.purely_classical() {
            Container::<ClassicType>::Tuple(Box::new(row)).into()
        } else {
            Container::<LinearType>::Tuple(Box::new(row)).into()
        }
    }
}

impl Default for SimpleType {
    fn default() -> Self {
        Self::Linear(Default::default())
    }
}

impl From<ClassicType> for SimpleType {
    fn from(typ: ClassicType) -> Self {
        Self::Classic(typ)
    }
}

impl From<LinearType> for SimpleType {
    fn from(typ: LinearType) -> Self {
        Self::Linear(typ)
    }
}

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[non_exhaustive]
pub struct TypeRow {
    /// The datatypes in the row.
    types: Cow<'static, [SimpleType]>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl TypeRow {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.types.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.types.len() == 0
    }

    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.types.iter().all(|typ| typ.is_linear())
    }

    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        self.types.iter().all(SimpleType::is_classical)
    }
}
impl TypeRow {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Create a new row from a Cow slice of types.
    ///
    /// See [`type_row!`] for a more ergonomic way to create a statically allocated rows.
    ///
    /// [`type_row!`]: crate::macros::type_row
    pub fn from(types: impl Into<Cow<'static, [SimpleType]>>) -> Self {
        Self {
            types: types.into(),
        }
    }

    /// Iterator over the types in the row.
    pub fn iter(&self) -> impl Iterator<Item = &SimpleType> {
        self.types.iter()
    }

    /// Mutable iterator over the types in the row.
    pub fn to_mut(&mut self) -> &mut Vec<SimpleType> {
        self.types.to_mut()
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: usize) -> Option<&SimpleType> {
        self.types.get(offset)
    }

    #[inline(always)]
    /// Returns the port type given an offset. Returns `None` if the offset is out of bounds.
    pub fn get_mut(&mut self, offset: usize) -> Option<&mut SimpleType> {
        self.types.to_mut().get_mut(offset)
    }
}

impl Default for TypeRow {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<T> for TypeRow
where
    T: Into<Cow<'static, [SimpleType]>>,
{
    fn from(types: T) -> Self {
        Self::from(types.into())
    }
}

impl Deref for TypeRow {
    type Target = [SimpleType];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl DerefMut for TypeRow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

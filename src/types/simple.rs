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
// TODO: Derive pyclass
//
// TODO: Compare performance vs flattening this into a single enum
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum SimpleType {
    /// A type containing classical data. Elements of this type can be copied.
    Classic(ClassicType),
    /// A type containing linear data. Elements of this type must be used exactly once.
    Linear(LinearType),
}

/// Trait of primitive types (ClassicType or LinearType).
pub trait PrimType {
    // may be updated with functions in future for necessary shared functionality
    // across ClassicType and LinearType
    // currently used to constrain Container<T>
}

/// A type that represents a container of other types.
///
/// For algebraic types Sum, Tuple if one element of type row is linear, the
/// overall type is too.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Container<T: PrimType> {
    /// Variable sized list of T.
    List(Box<T>),
    /// Hash map from hashable key type to value T.
    Map(Box<(ClassicType, T)>),
    /// Product type, known-size tuple over elements of type row.
    Tuple(Box<TypeRow>),
    /// Product type, variants are tagged by their position in the type row.
    Sum(Box<TypeRow>),
    /// Known size array of T.
    Array(Box<T>, usize),
    /// Named type defined by, but distinct from, T.
    NewType(SmolStr, Box<T>),
}

impl From<Container<ClassicType>> for SimpleType {
    #[inline]
    fn from(value: Container<ClassicType>) -> Self {
        Self::Classic(ClassicType::Container(value))
    }
}

impl From<Container<LinearType>> for SimpleType {
    #[inline]
    fn from(value: Container<LinearType>) -> Self {
        Self::Linear(LinearType::Container(value))
    }
}

/// A type that represents concrete classical data.
///
/// Uses `Box`es on most variants to reduce the memory footprint.
///
/// TODO: Derive pyclass.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ClassicType {
    /// A type variable identified by a name.
    Variable(SmolStr),
    /// An arbitrary size integer.
    Int(usize),
    /// A 64-bit floating point number.
    F64,
    /// An arbitrary length string.
    String,
    /// A graph encoded as a value. It contains a concrete signature and a set of required resources.
    Graph(Box<(ResourceSet, Signature)>),
    /// A nested definition containing other classic types.
    Container(Container<ClassicType>),
    /// An opaque operation that can be downcasted by the extensions that define it.
    Opaque(CustomType),
}

impl ClassicType {
    /// Create a graph type with the given signature, using default resources.
    /// TODO in the future we'll probably need versions of this that take resources.
    #[inline]
    pub fn graph_from_sig(signature: Signature) -> Self {
        ClassicType::Graph(Box::new((Default::default(), signature)))
    }

    /// Returns a new integer type with the given number of bits.
    #[inline]
    pub const fn int<const N: usize>() -> Self {
        Self::Int(N)
    }

    /// Returns a new 64-bit integer type.
    #[inline]
    pub const fn i64() -> Self {
        Self::int::<64>()
    }

    /// Returns a new 1-bit integer type.
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
/// TODO: Derive pyclass.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LinearType {
    /// A qubit.
    #[default]
    Qubit,
    /// A linear opaque operation that can be downcasted by the extensions that define it.
    Qpaque(CustomType),
    /// A nested definition containing other linear types.
    Container(Container<LinearType>),
}

impl PrimType for LinearType {}

impl SimpleType {
    /// Returns whether the type contains only linear data.
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear(_))
    }

    /// Returns whether the type contains only classic data.
    pub fn is_classical(&self) -> bool {
        matches!(self, Self::Classic(_))
    }

    /// New Sum type, variants defined by TypeRow.
    pub fn new_sum(row: impl Into<TypeRow>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            Container::<ClassicType>::Sum(Box::new(row)).into()
        } else {
            Container::<LinearType>::Sum(Box::new(row)).into()
        }
    }

    /// New Tuple type, elements defined by TypeRow.
    pub fn new_tuple(row: impl Into<TypeRow>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            Container::<ClassicType>::Tuple(Box::new(row)).into()
        } else {
            Container::<LinearType>::Tuple(Box::new(row)).into()
        }
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::Classic(ClassicType::Container(Container::Tuple(Box::new(
            TypeRow::new(),
        ))))
    }

    /// New Sum of Unit types, used as predicates in branching.
    pub fn new_predicate(size: usize) -> Self {
        let rowvec = vec![Self::new_unit(); size];
        Self::Classic(ClassicType::Container(Container::Sum(Box::new(
            rowvec.into(),
        ))))
    }

    /// Convert to a named NewType.
    pub fn into_new_type(self, name: impl Into<SmolStr>) -> SimpleType {
        match self {
            // annoying that the arms have the same code
            SimpleType::Classic(typ) => {
                Container::<ClassicType>::NewType(name.into(), Box::new(typ)).into()
            }
            SimpleType::Linear(typ) => {
                Container::<LinearType>::NewType(name.into(), Box::new(typ)).into()
            }
        }
    }
}

impl Default for SimpleType {
    fn default() -> Self {
        Self::Linear(Default::default())
    }
}

/// Implementations of Into and TryFrom for SimpleType and &'a SimpleType.
macro_rules! impl_from_into_simple_type {
    ($target:ident, $matcher:pat, $unpack:expr, $new:expr) => {
        impl From<$target> for SimpleType {
            fn from(typ: $target) -> Self {
                $new(typ)
            }
        }

        impl TryFrom<SimpleType> for $target {
            type Error = ();

            fn try_from(op: SimpleType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }

        impl<'a> TryFrom<&'a SimpleType> for &'a $target {
            type Error = ();

            fn try_from(op: &'a SimpleType) -> Result<Self, Self::Error> {
                match op {
                    $matcher => Ok($unpack),
                    _ => Err(()),
                }
            }
        }
    };
}
impl_from_into_simple_type!(
    ClassicType,
    SimpleType::Classic(typ),
    typ,
    SimpleType::Classic
);
impl_from_into_simple_type!(LinearType, SimpleType::Linear(typ), typ, SimpleType::Linear);

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

    /// Returns whether the row contains only linear data.
    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.types.iter().all(|typ| typ.is_linear())
    }

    /// Returns whether the row contains only classic data.
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
    /// [`type_row!`]: crate::macros::type_row.
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

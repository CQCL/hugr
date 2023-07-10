//! Dataflow types

use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter, Write},
    ops::{Deref, DerefMut},
};

use itertools::Itertools;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use smol_str::SmolStr;

use super::{custom::CustomType, Signature};
use crate::{ops::constant::HugrIntWidthStore, utils::display_list};
use crate::{resource::ResourceSet, type_row};

/// A type that represents concrete data. Can include both linear and classical parts.
///
// TODO: Derive pyclass
//
// TODO: Compare performance vs flattening this into a single enum
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "serialize::SerSimpleType", into = "serialize::SerSimpleType")]
#[non_exhaustive]
pub enum SimpleType {
    /// A type containing only classical data. Elements of this type can be copied.
    Classic(ClassicType),
    /// A qubit.
    Qubit,
    /// A linear opaque type that can be downcasted by the extensions that define it.
    Qpaque(CustomType),
    /// A nested definition containing other linear types (possibly as well as classical ones)
    Qontainer(Container<SimpleType>),
}

mod serialize;

impl Display for SimpleType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimpleType::Classic(ty) => ty.fmt(f),
            SimpleType::Qubit => f.write_str("Qubit"),
            SimpleType::Qpaque(custom) => custom.fmt(f),
            SimpleType::Qontainer(c) => c.fmt(f),
        }
    }
}

/// Trait of primitive types (SimpleType or ClassicType).
pub trait PrimType: std::fmt::Debug + Clone + 'static {
    // may be updated with functions in future for necessary shared functionality
    // across ClassicType and SimpleType
    // currently used to constrain Container<T>

    /// Is this type linear? (I.e. does it have any linear components?)
    const LINEAR: bool;
}

/// A type that represents a container of other types.
///
/// For algebraic types Sum, Tuple if one element of type row is linear, the
/// overall type is too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Container<T: PrimType> {
    /// Variable sized list of T.
    List(Box<T>),
    /// Hash map from hashable key type to value T.
    Map(Box<(ClassicType, T)>),
    /// Product type, known-size tuple over elements of type row.
    Tuple(Box<TypeRow<T>>),
    /// Product type, variants are tagged by their position in the type row.
    Sum(Box<TypeRow<T>>),
    /// Known size array of T.
    Array(Box<T>, usize),
    /// Alias defined in AliasDefn or AliasDecl nodes.
    Alias(SmolStr),
}

impl<T: Display + PrimType> Display for Container<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Container::List(ty) => write!(f, "List({})", ty.as_ref()),
            Container::Map(tys) => write!(f, "Map({}, {})", tys.as_ref().0, tys.as_ref().1),
            Container::Tuple(row) => write!(f, "Tuple({})", row.as_ref()),
            Container::Sum(row) => write!(f, "Sum({})", row.as_ref()),
            Container::Array(t, size) => write!(f, "Array({}, {})", t, size),
            Container::Alias(str) => f.write_str(str),
        }
    }
}

impl From<Container<ClassicType>> for SimpleType {
    #[inline]
    fn from(value: Container<ClassicType>) -> Self {
        Self::Classic(ClassicType::Container(value))
    }
}

impl From<Container<SimpleType>> for SimpleType {
    #[inline]
    fn from(value: Container<SimpleType>) -> Self {
        Self::Qontainer(value)
    }
}

/// A type that represents concrete classical data.
///
/// Uses `Box`es on most variants to reduce the memory footprint.
///
/// TODO: Derive pyclass.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SimpleType", into = "SimpleType")]
#[non_exhaustive]
pub enum ClassicType {
    /// A type variable identified by a name.
    Variable(SmolStr),
    /// An arbitrary size integer.
    Int(HugrIntWidthStore),
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
    pub const fn int<const N: HugrIntWidthStore>() -> Self {
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

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::Container(Container::Tuple(Box::new(TypeRow::new())))
    }

    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = TypeRow<ClassicType>>) -> Self {
        Self::Container(Container::Sum(Box::new(TypeRow::predicate_variants_row(
            variant_rows,
        ))))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(type_row![]).take(size))
    }
}

impl Default for ClassicType {
    fn default() -> Self {
        Self::int::<1>()
    }
}

impl Display for ClassicType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ClassicType::Variable(x) => f.write_str(x),
            ClassicType::Int(i) => {
                f.write_char('I')?;
                f.write_str(&i.to_string())
            }
            ClassicType::F64 => f.write_str("F64"),
            ClassicType::String => f.write_str("String"),
            ClassicType::Graph(data) => {
                let (rs, sig) = data.as_ref();
                write!(f, "[{:?}]", rs)?;
                sig.fmt(f)
            }
            ClassicType::Container(c) => c.fmt(f),
            ClassicType::Opaque(custom) => custom.fmt(f),
        }
    }
}

impl PrimType for ClassicType {
    const LINEAR: bool = false;
}

impl PrimType for SimpleType {
    const LINEAR: bool = true;
}

impl SimpleType {
    /// Returns whether the type contains linear data (perhaps as well as classical)
    pub fn is_linear(&self) -> bool {
        !self.is_classical()
    }

    /// Returns whether the type contains only classic data.
    pub fn is_classical(&self) -> bool {
        matches!(self, Self::Classic(_))
    }

    /// New Sum type, variants defined by TypeRow.
    pub fn new_sum(row: impl Into<TypeRow<SimpleType>>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            let row: TypeRow<ClassicType> = row.try_convert_elems().unwrap();
            Container::<ClassicType>::Sum(Box::new(row)).into()
        } else {
            Container::<SimpleType>::Sum(Box::new(row)).into()
        }
    }

    /// New Tuple type, elements defined by TypeRow.
    pub fn new_tuple(row: impl Into<TypeRow<SimpleType>>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            let row: TypeRow<ClassicType> = row.try_convert_elems().unwrap();
            Container::<ClassicType>::Tuple(Box::new(row)).into()
        } else {
            Container::<SimpleType>::Tuple(Box::new(row)).into()
        }
    }

    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = TypeRow<ClassicType>>) -> Self {
        Self::Classic(ClassicType::new_predicate(variant_rows))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::Classic(ClassicType::new_simple_predicate(size))
    }
}

impl From<ClassicType> for SimpleType {
    fn from(typ: ClassicType) -> Self {
        SimpleType::Classic(typ)
    }
}

impl TryFrom<SimpleType> for ClassicType {
    type Error = &'static str;

    fn try_from(op: SimpleType) -> Result<Self, Self::Error> {
        match op {
            SimpleType::Classic(typ) => Ok(typ),
            _ => Err("Invalid type conversion"),
        }
    }
}

impl<'a> TryFrom<&'a SimpleType> for &'a ClassicType {
    type Error = &'static str;

    fn try_from(op: &'a SimpleType) -> Result<Self, Self::Error> {
        match op {
            SimpleType::Classic(typ) => Ok(typ),
            _ => Err("Invalid type conversion"),
        }
    }
}

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow<T: PrimType> {
    /// The datatypes in the row.
    types: Cow<'static, [T]>,
}

impl<T: Display + PrimType> Display for TypeRow<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl<T: PrimType> TypeRow<T> {
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
}

impl TypeRow<SimpleType> {
    /// Returns whether the row contains only linear data.
    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.types.iter().all(SimpleType::is_linear)
    }

    /// Returns whether the row contains only classic data.
    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        self.types.iter().all(SimpleType::is_classical)
    }
}

impl TypeRow<ClassicType> {
    #[inline]
    /// Return the type row of variants required to define a Sum of Tuples type
    /// given the rows of each tuple
    pub fn predicate_variants_row(
        variant_rows: impl IntoIterator<Item = TypeRow<ClassicType>>,
    ) -> Self {
        variant_rows
            .into_iter()
            .map(|row| ClassicType::Container(Container::Tuple(Box::new(row))))
            .collect_vec()
            .into()
    }
}

impl<T: PrimType> TypeRow<T> {
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

    fn try_convert_elems<D: PrimType + TryFrom<T>>(self) -> Result<TypeRow<D>, D::Error> {
        let elems: Vec<D> = self
            .into_owned()
            .into_iter()
            .map(D::try_from)
            .collect::<Result<_, _>>()?;
        Ok(TypeRow::from(elems))
    }

    /// Converts the elements of this TypeRow into some other type that they can `.into()`
    pub fn map_into<T2: PrimType + From<T>>(self) -> TypeRow<T2> {
        TypeRow::from(
            self.into_owned()
                .into_iter()
                .map(T2::from)
                .collect::<Vec<T2>>(),
        )
    }
}

impl<T: PrimType> Default for TypeRow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, T: PrimType> From<F> for TypeRow<T>
where
    F: Into<Cow<'static, [T]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl<T: PrimType> Deref for TypeRow<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl<T: PrimType> DerefMut for TypeRow<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

//! Dataflow types

use std::{
    borrow::Cow,
    fmt::{self, Display, Formatter, Write},
    ops::{Deref, DerefMut},
};

use itertools::Itertools;
use serde_repr::{Deserialize_repr, Serialize_repr};
use smol_str::SmolStr;

use super::{custom::CustomType, AbstractSignature};
use crate::{classic_row, ops::constant::HugrIntWidthStore, utils::display_list};

/// A type that represents concrete data. Can include both linear and classical parts.
///
// TODO: Derive pyclass
//
// TODO: Compare performance vs flattening this into a single enum
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(
    try_from = "super::serialize::SerSimpleType",
    into = "super::serialize::SerSimpleType"
)]
#[non_exhaustive]
pub enum SimpleType {
    /// A type containing only classical data. Elements of this type can be copied.
    Classic(ClassicType),
    /// A qubit.
    Qubit,
    /// A nested definition containing other linear types (possibly as well as classical ones)
    Qontainer(Container<SimpleType>),
}

impl Display for SimpleType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimpleType::Classic(ty) => ty.fmt(f),
            SimpleType::Qubit => f.write_str("Qubit"),
            SimpleType::Qontainer(c) => c.fmt(f),
        }
    }
}

/// Categorizes types into three classes according to basic operations supported.
#[derive(
    Copy, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize_repr, Deserialize_repr,
)]
#[repr(u8)]
pub enum TypeTag {
    /// Any [SimpleType], including linear and quantum types;
    /// cannot necessarily be copied or discarded.
    Simple = 0,
    /// Subset of [TypeTag::Simple]; types that can be copied and discarded. See [ClassicType]
    Classic = 1,
    /// Subset of [TypeTag::Classic]: types that can also be hashed and support
    /// a strong notion of equality. See [HashableType]
    Hashable = 2,
}

impl TypeTag {
    /// Returns the smallest TypeTag containing both the receiver and argument.
    /// (This will be one of the receiver or the argument.)
    pub fn union(self, other: Self) -> Self {
        if self == Self::Simple || other == Self::Simple {
            Self::Simple
        } else if self == Self::Classic || other == Self::Classic {
            Self::Classic
        } else {
            Self::Hashable
        }
    }

    /// Do types in this tag contain only classic data
    /// (which can be copied and discarded, i.e. [ClassicType]s)
    pub fn is_classical(self) -> bool {
        self != Self::Simple
    }

    /// Do types in this tag contain only hashable classic data
    /// (with a strong notion of equality, i.e. [HashableType]s)
    pub fn is_hashable(self) -> bool {
        self == Self::Hashable
    }
}

/// Base trait for anything that can be put in a [TypeRow]
pub trait TypeRowElem: std::fmt::Debug + Clone + 'static {}

impl TypeRowElem for SimpleType {}
impl TypeRowElem for ClassicType {}
impl TypeRowElem for HashableType {}

/// Trait of primitive types, i.e. that are uniquely identified by a [TypeTag]
pub trait PrimType: TypeRowElem + sealed::Sealed {
    // may be updated with functions in future for necessary shared functionality
    // across ClassicType, SimpleType and HashableType.
    // Currently used to constrain Container<T>
    /// Tells us the [TypeTag] of the type represented by the receiver.
    fn tag(&self) -> TypeTag;
}

// sealed trait pattern to prevent users extending PrimType
mod sealed {
    use super::{ClassicType, HashableType, SimpleType};
    pub trait Sealed {}
    impl Sealed for SimpleType {}
    impl Sealed for ClassicType {}
    impl Sealed for HashableType {}
}
/// A type that represents a container of other types.
///
/// For algebraic types Sum, Tuple if one element of type row is linear, the
/// overall type is too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Container<T: TypeRowElem> {
    /// Variable sized list of T.
    List(Box<T>),
    /// Hash map from hashable key type to value T.
    Map(Box<(HashableType, T)>),
    /// Product type, known-size tuple over elements of type row.
    Tuple(Box<TypeRow<T>>),
    /// Product type, variants are tagged by their position in the type row.
    Sum(Box<TypeRow<T>>),
    /// Known size array of T.
    Array(Box<T>, usize),
    /// Alias defined in AliasDefn or AliasDecl nodes.
    Alias(SmolStr),
    /// An opaque type that can be downcasted by the extensions that define it.
    /// This will always have a [`TypeTag`] contained within that of the Container
    Opaque(CustomType),
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
            Container::Opaque(c) => write!(f, "Opaque({})", c),
        }
    }
}

impl From<Container<HashableType>> for ClassicType {
    fn from(value: Container<HashableType>) -> Self {
        ClassicType::Hashable(HashableType::Container(value))
    }
}

impl From<Container<HashableType>> for SimpleType {
    fn from(value: Container<HashableType>) -> Self {
        let ty: ClassicType = value.into();
        ty.into()
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
    /// A 64-bit floating point number.
    F64,
    /// A graph encoded as a value. It contains a concrete signature and a set of required resources.
    /// TODO this can be moved out into an extension/resource
    Graph(Box<AbstractSignature>),
    /// A nested definition containing other classic types.
    Container(Container<ClassicType>),
    /// A type which can be hashed
    Hashable(HashableType),
}

/// A type that represents concrete classical data that supports hashing
/// and a strong notion of equality. (So, e.g., no floating-point.)
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ClassicType", into = "ClassicType")]
#[non_exhaustive]
pub enum HashableType {
    /// A type variable identified by a name.
    /// TODO of course this is not necessarily hashable, or even classic,
    /// depending on how it is instantiated...
    Variable(SmolStr),
    /// An arbitrary size integer.
    Int(HugrIntWidthStore),
    /// An arbitrary length string.
    String,
    /// A container (all of whose elements can be hashed)
    Container(Container<HashableType>),
}

impl ClassicType {
    /// Returns whether the type contains only hashable data.
    pub fn is_hashable(&self) -> bool {
        matches!(self, Self::Hashable(_))
    }

    /// Create a graph type with the given signature, using default resources.
    #[inline]
    pub fn graph_from_sig(signature: AbstractSignature) -> Self {
        ClassicType::Graph(Box::new(signature))
    }

    /// Returns a new integer type with the given number of bits.
    #[inline]
    pub const fn int<const N: HugrIntWidthStore>() -> Self {
        Self::Hashable(HashableType::Int(N))
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
        Self::Container(Container::Tuple(Box::new(classic_row![])))
    }

    /// New Tuple type, elements defined by TypeRow
    pub fn new_tuple(row: impl Into<TypeRow<ClassicType>>) -> Self {
        let row = row.into();
        if row.purely_hashable() {
            // This should succeed given purely_hashable returned True
            let row = row.try_convert_elems().unwrap();
            Container::<HashableType>::Tuple(Box::new(row)).into()
        } else {
            ClassicType::Container(Container::Tuple(Box::new(row)))
        }
    }

    /// New Sum type, variants defined by TypeRow
    pub fn new_sum(row: impl Into<TypeRow<ClassicType>>) -> Self {
        let row = row.into();
        if row.purely_hashable() {
            // This should succeed given purely_hashable returned True
            let row = row.try_convert_elems().unwrap();
            Container::<HashableType>::Sum(Box::new(row)).into()
        } else {
            ClassicType::Container(Container::Sum(Box::new(row)))
        }
    }

    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = ClassicRow>) -> Self {
        Self::new_sum(TypeRow::predicate_variants_row(variant_rows))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(classic_row![]).take(size))
    }
}

impl Display for ClassicType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ClassicType::F64 => f.write_str("F64"),
            ClassicType::Graph(data) => {
                let sig = data.as_ref();
                write!(f, "[{:?}]", sig.resource_reqs)?;
                sig.fmt(f)
            }
            ClassicType::Container(c) => c.fmt(f),
            ClassicType::Hashable(h) => h.fmt(f),
        }
    }
}

impl Display for HashableType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            HashableType::Variable(x) => f.write_str(x),
            HashableType::Int(i) => {
                f.write_char('I')?;
                f.write_str(&i.to_string())
            }
            HashableType::String => f.write_str("String"),
            HashableType::Container(c) => c.fmt(f),
        }
    }
}

impl PrimType for ClassicType {
    fn tag(&self) -> TypeTag {
        if self.is_hashable() {
            TypeTag::Hashable
        } else {
            TypeTag::Classic
        }
    }
}

impl PrimType for SimpleType {
    fn tag(&self) -> TypeTag {
        match self {
            Self::Classic(c) => c.tag(),
            _ => TypeTag::Simple,
        }
    }
}

impl PrimType for HashableType {
    fn tag(&self) -> TypeTag {
        TypeTag::Hashable
    }
}

impl SimpleType {
    /// New Sum type, variants defined by TypeRow.
    pub fn new_sum(row: impl Into<TypeRow<SimpleType>>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            // This should succeed given purely_classical has returned True
            let row: TypeRow<ClassicType> = row.try_convert_elems().unwrap();
            ClassicType::new_sum(row).into()
        } else {
            Container::<SimpleType>::Sum(Box::new(row)).into()
        }
    }

    /// New Tuple type, elements defined by TypeRow.
    pub fn new_tuple(row: impl Into<TypeRow<SimpleType>>) -> Self {
        let row = row.into();
        if row.purely_classical() {
            // This should succeed given purely_classical has returned True
            let row: TypeRow<ClassicType> = row.try_convert_elems().unwrap();
            ClassicType::new_tuple(row).into()
        } else {
            Container::<SimpleType>::Tuple(Box::new(row)).into()
        }
    }

    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = ClassicRow>) -> Self {
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

impl From<HashableType> for ClassicType {
    fn from(typ: HashableType) -> Self {
        ClassicType::Hashable(typ)
    }
}

impl From<HashableType> for SimpleType {
    fn from(value: HashableType) -> Self {
        let c: ClassicType = value.into();
        c.into()
    }
}

impl TryFrom<SimpleType> for ClassicType {
    type Error = String;

    fn try_from(op: SimpleType) -> Result<Self, Self::Error> {
        match op {
            SimpleType::Classic(typ) => Ok(typ),
            _ => Err(format!("Invalid type conversion, {:?} is not classic", op)),
        }
    }
}

impl TryFrom<ClassicType> for HashableType {
    type Error = String;

    fn try_from(value: ClassicType) -> Result<Self, Self::Error> {
        match value {
            ClassicType::Hashable(typ) => Ok(typ),
            _ => Err(format!(
                "Invalid type conversion, {:?} is not hashable",
                value
            )),
        }
    }
}

impl TryFrom<SimpleType> for HashableType {
    type Error = String;

    fn try_from(op: SimpleType) -> Result<Self, Self::Error> {
        let typ: ClassicType = op.try_into()?;
        typ.try_into()
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
//#[cfg_attr(feature = "pyo3", pyclass)] // TODO: expose unparameterized versions
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow<T: TypeRowElem> {
    /// The datatypes in the row.
    types: Cow<'static, [T]>,
}

/// A row of [SimpleType]s
pub type SimpleRow = TypeRow<SimpleType>;

/// A row of [ClassicType]s
pub type ClassicRow = TypeRow<ClassicType>;

impl<T: Display + TypeRowElem> Display for TypeRow<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl TypeRow<SimpleType> {
    /// Returns whether the row contains only classic data.
    /// (Note: this is defined only on [`TypeRow<SimpleType>`] because
    /// it is guaranteed true for any other TypeRow)
    #[inline]
    pub fn purely_classical(&self) -> bool {
        self.types
            .iter()
            .map(PrimType::tag)
            .all(TypeTag::is_classical)
    }
}

impl TypeRow<ClassicType> {
    /// Return the type row of variants required to define a Sum of Tuples type
    /// given the rows of each tuple
    pub fn predicate_variants_row(variant_rows: impl IntoIterator<Item = ClassicRow>) -> Self {
        variant_rows
            .into_iter()
            .map(ClassicType::new_tuple)
            .collect_vec()
            .into()
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

impl<T: PrimType> TypeRow<T> {
    /// Returns whether the row contains only hashable classic data.
    #[inline(always)]
    pub fn purely_hashable(&self) -> bool {
        self.types
            .iter()
            .map(PrimType::tag)
            .all(TypeTag::is_hashable)
    }

    /// Returns the smallest [TypeTag] that contains all elements of the row
    pub fn containing_tag(&self) -> TypeTag {
        self.types
            .iter()
            .map(PrimType::tag)
            .fold(TypeTag::Hashable, TypeTag::union)
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

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;

    #[test]
    fn new_tuple() {
        let simp = vec![SimpleType::Qubit, SimpleType::Classic(ClassicType::F64)];
        let ty = SimpleType::new_tuple(simp);
        assert_matches!(ty, SimpleType::Qontainer(Container::Tuple(_)));

        let clas: ClassicRow = vec![
            ClassicType::F64,
            ClassicType::Container(Container::List(Box::new(ClassicType::F64))),
        ]
        .into();
        let ty = SimpleType::new_tuple(clas.map_into());
        assert_matches!(
            ty,
            SimpleType::Classic(ClassicType::Container(Container::Tuple(_)))
        );

        let hash = vec![
            SimpleType::Classic(ClassicType::Hashable(HashableType::Int(8))),
            SimpleType::Classic(ClassicType::Hashable(HashableType::String)),
        ];
        let ty = SimpleType::new_tuple(hash);
        assert_matches!(
            ty,
            SimpleType::Classic(ClassicType::Hashable(HashableType::Container(
                Container::Tuple(_)
            )))
        );
    }

    #[test]
    fn new_sum() {
        let clas = vec![
            SimpleType::Classic(ClassicType::F64),
            SimpleType::Classic(ClassicType::Container(Container::List(Box::new(
                ClassicType::F64,
            )))),
        ];
        let ty = SimpleType::new_sum(clas);
        assert_matches!(
            ty,
            SimpleType::Classic(ClassicType::Container(Container::Sum(_)))
        );

        let hash: TypeRow<HashableType> = vec![HashableType::Int(4), HashableType::String].into();
        let ty = SimpleType::new_sum(hash.map_into());
        assert_matches!(
            ty,
            SimpleType::Classic(ClassicType::Hashable(HashableType::Container(
                Container::Sum(_)
            )))
        );
    }
}

//! Dataflow types

use std::fmt::{self, Display, Formatter, Write};

use super::type_row::{TypeRow, TypeRowElem};
use super::{custom::CustomType, AbstractSignature};
use crate::{classic_row, ops::constant::HugrIntWidthStore};
use itertools::Itertools;
use serde_repr::{Deserialize_repr, Serialize_repr};
use smol_str::SmolStr;

/// A type that represents concrete data. Can include both linear and classical parts.
///
// TODO: Derive pyclass
//
// TODO: Compare performance vs flattening this into a single enum
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "serialize::SerSimpleType", into = "serialize::SerSimpleType")]
#[non_exhaustive]
pub enum SimpleElem {
    /// A type containing only classical data. Elements of this type can be copied.
    Classic(ClassicElem),
    /// A qubit.
    Qubit,
}

pub type SimpleType = Container<SimpleElem>;

mod serialize;

impl Display for SimpleElem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimpleElem::Classic(ty) => ty.fmt(f),
            SimpleElem::Qubit => f.write_str("Qubit"),
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

/// Trait of primitive types, i.e. that are uniquely identified by a [TypeTag]
pub trait PrimType: std::fmt::Debug + Clone + 'static + sealed::Sealed {
    // may be updated with functions in future for necessary shared functionality
    // across ClassicType, SimpleType and HashableType.
    // Currently used to constrain Container<T>
    /// Tells us the [TypeTag] of the type represented by the receiver.
    fn tag(&self) -> TypeTag;
}

// sealed trait pattern to prevent users extending PrimType
mod sealed {
    use super::{ClassicElem, HashableElem, SimpleElem};
    pub trait Sealed {}
    impl Sealed for SimpleElem {}
    impl Sealed for ClassicElem {}
    impl Sealed for HashableElem {}
}

/// A type that represents a container of other types.
///
/// For algebraic types Sum, Tuple if one element of type row is linear, the
/// overall type is too.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Container<T: PrimType> {
    /// A single element
    Single(T),
    /// Variable sized list of T.
    List(Box<Container<T>>),
    /// Hash map from hashable key type to value T.
    Map(Box<(HashableType, Container<T>)>),
    /// Product type, known-size tuple over elements of type row.
    Tuple(Box<TypeRow<Container<T>>>),
    /// Product type, variants are tagged by their position in the type row.
    Sum(Box<TypeRow<Container<T>>>),
    /// Known size array of T.
    Array(Box<Container<T>>, usize),
    /// Alias defined in AliasDefn or AliasDecl nodes.
    Alias(SmolStr),
    /// An opaque type that can be downcasted by the extensions that define it.
    /// This will always have a [`TypeTag`] contained within that of the Container
    Opaque(CustomType),
}

impl<T: PrimType> TypeRowElem for Container<T> {}

impl<T: PrimType> Container<T> {
    fn all(&self, f: impl Fn(&T) -> bool) -> bool {
        match self {
            Container::Single(e) => f(&*e),
            Container::List(e) => e.all(f),
            Container::Map(kv) => kv.1.all(f),
            Container::Tuple(r) => (*r).iter().all(|c| c.all(f)),
            Container::Sum(r) => (*r).iter().all(|c| c.all(f)),
            Container::Array(e, _) => e.all(f),
            Container::Alias(_) => todo!(),
            Container::Opaque(_) => todo!(),
        }
    }

    fn map_into<T2: PrimType>(self) -> Container<T2>
    where
        T2: From<T>,
    {
        match self {
            Container::Single(e) => Container::Single(e.into()),
            Container::List(e) => Container::List(Box::new(e.map_into())),
            Container::Map(kv) => Container::Map(Box::new((kv.0, kv.1.map_into()))),
            Container::Tuple(r) => Container::Tuple(Box::new(r.map_into())),
            Container::Sum(r) => Container::Tuple(Box::new(r.map_into())),
            Container::Array(e, sz) => Container::Array(Box::new(e.map_into()), sz),
            Container::Alias(n) => Container::Alias(n),
            Container::Opaque(t) => Container::Opaque(t),
        }
    }
}

impl<T: Display + PrimType> Display for Container<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Container::Single(elem) => Display::fmt(elem, f),
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

/// A type that represents concrete classical data.
///
/// Uses `Box`es on most variants to reduce the memory footprint.
///
/// TODO: Derive pyclass.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SimpleElem", into = "SimpleElem")]
#[non_exhaustive]
pub enum ClassicElem {
    /// A 64-bit floating point number.
    F64,
    /// A graph encoded as a value. It contains a concrete signature and a set of required resources.
    /// TODO this can be moved out into an extension/resource
    Graph(Box<AbstractSignature>),
    /// A type which can be hashed
    Hashable(HashableElem),
}

impl From<HashableElem> for ClassicElem {
    fn from(value: HashableElem) -> Self {
        Self::Hashable(value)
    }
}

/// A type that represents concrete classical data that supports hashing
/// and a strong notion of equality. (So, e.g., no floating-point.)
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ClassicType", into = "ClassicType")]
#[non_exhaustive]
pub enum HashableElem {
    /// A type variable identified by a name.
    /// TODO of course this is not necessarily hashable, or even classic,
    /// depending on how it is instantiated...
    Variable(SmolStr),
    /// An arbitrary size integer.
    Int(HugrIntWidthStore),
    /// An arbitrary length string.
    String,
}

pub type HashableType = Container<HashableElem>;

pub type ClassicType = Container<ClassicElem>;

impl ClassicElem {
    /// Returns whether the type contains only hashable data.
    pub fn is_hashable(&self) -> bool {
        matches!(self, Self::Hashable(_))
    }

    /// Create a graph type with the given signature, using default resources.
    #[inline]
    pub fn graph_from_sig(signature: AbstractSignature) -> Self {
        ClassicElem::Graph(Box::new(signature))
    }
}

impl ClassicType {
    /// Returns a new integer type with the given number of bits.
    #[inline]
    pub const fn int<const N: HugrIntWidthStore>() -> Self {
        Self::Single(ClassicElem::Hashable(HashableElem::Int(N)))
    }

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
        Container::Tuple(Box::new(classic_row![]))
    }

    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = ClassicRow>) -> Self {
        Container::Sum(Box::new(TypeRow::predicate_variants_row(variant_rows)))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(classic_row![]).take(size))
    }
}

impl Display for ClassicElem {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ClassicType::F64 => f.write_str("F64"),
            ClassicType::Variable(x) => f.write_str(x),
            ClassicType::Int(i) => {
                f.write_char('I')?;
                f.write_str(&i.to_string())
            }
            ClassicType::String => f.write_str("String"),
            ClassicElem::Opaque(custom) => custom.fmt(f),
        }
    }
}

impl PrimType for ClassicElem {
    fn tag(&self) -> TypeTag {
        if self.is_hashable() {
            TypeTag::Hashable
        } else {
            TypeTag::Classic
        }
    }
}

impl PrimType for SimpleElem {
    fn tag(&self) -> TypeTag {
        match self {
            Self::Classic(c) => c.tag(),
            _ => TypeTag::Simple,
        }
    }
}

impl PrimType for HashableElem {
    fn tag(&self) -> TypeTag {
        TypeTag::Hashable
    }
}

impl SimpleElem {
    pub fn is_classical(&self) -> bool {
        matches!(self, SimpleElem::Classic(_))
    }
}

impl SimpleType {
    /// Returns whether the type contains only classic data.
    pub fn is_classical(&self) -> bool {
        self.all(|e| e.is_classical())
    }
}

impl From<ClassicElem> for SimpleElem {
    fn from(typ: ClassicElem) -> Self {
        SimpleElem::Classic(typ)
    }
}

// for deserialization
impl TryFrom<SimpleElem> for ClassicElem {
    type Error = String;

    fn try_from(value: SimpleElem) -> Result<Self, Self::Error> {
        match value {
            SimpleElem::Classic(e) => Ok(e),
            _ => Err(format!("Not classic: {:?}", value)),
        }
    }
}

/* Do we need these?
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
}*/

/// A row of [SimpleType]s
pub type SimpleRow = TypeRow<SimpleType>;

/// A row of [ClassicType]s
pub type ClassicRow = TypeRow<ClassicType>;

impl TypeRow<SimpleType> {
    /// Returns whether the row contains only classic data.
    /// (Note: this is defined only on [`TypeRow<SimpleType>`] because
    /// it is guaranteed true for any other TypeRow)
    #[inline]
    pub fn purely_classical(&self) -> bool {
        self.iter().map(PrimType::tag).all(TypeTag::is_classical)
    }
}

impl TypeRow<ClassicType> {
    #[inline]
    /// Return the type row of variants required to define a Sum of Tuples type
    /// given the rows of each tuple
    pub fn predicate_variants_row(variant_rows: impl IntoIterator<Item = ClassicRow>) -> Self {
        variant_rows
            .into_iter()
            .map(|row| Container::Tuple(Box::new(row)))
            .collect_vec()
            .into()
    }

    #[inline(always)]
    pub fn purely_hashable(&self) -> bool {
        self.iter().map(PrimType::tag).all(TypeTag::is_hashable)
    }
}

impl<T: PrimType> TypeRow<Container<T>> {
    /// Returns the smallest [TypeTag] that contains all elements of the row
    pub fn containing_tag(&self) -> TypeTag {
        self.iter()
            .map(PrimType::tag)
            .fold(TypeTag::Hashable, TypeTag::union)
    }
}

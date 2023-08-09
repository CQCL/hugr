#![allow(missing_docs)]

use std::{fmt::Write, marker::PhantomData};

use crate::{ops::AliasDecl, utils::display_list};
use std::fmt::{self, Debug, Display};

use super::{
    leaf::{AnyLeaf, CopyableLeaf, EqLeaf, InvalidBound, Tagged, TypeClass},
    AbstractSignature, CustomType, TypeTag,
};

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
#[display(bound = "T: Display")]
#[display(fmt = "{}")]
pub enum Type<T: TypeClass> {
    Prim(T),
    Extension(Tagged<CustomType, T>),
    #[display(fmt = "Alias({})", "_0.inner().name()")]
    Alias(Tagged<AliasDecl, T>),
    #[display(fmt = "Array[{};{}]", "_0", "_1")]
    Array(Box<Type<T>>, usize),
    #[display(fmt = "Tuple({})", "DisplayRow(_0)")]
    Tuple(Vec<Type<T>>),
    #[display(fmt = "Sum({})", "DisplayRow(_0)")]
    Sum(Vec<Type<T>>),
}

struct DisplayRow<'a, T: TypeClass>(&'a Vec<Type<T>>);
impl<'a, T: Display + TypeClass> Display for DisplayRow<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.0, f)?;
        f.write_char(']')
    }
}

impl<T: TypeClass> Type<T> {
    pub const BOUND_TAG: TypeTag = T::BOUND_TAG;
    pub const fn bounding_tag(&self) -> TypeTag {
        T::BOUND_TAG
    }

    pub fn new_tuple(types: impl IntoIterator<Item = Type<T>>) -> Self {
        Self::Tuple(types.into_iter().collect())
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    pub fn new_sum(types: impl IntoIterator<Item = Type<T>>) -> Self {
        Self::Sum(types.into_iter().collect())
    }

    pub fn new_extension(opaque: CustomType) -> Result<Self, InvalidBound> {
        Ok(Self::Extension(Tagged::new(opaque)?))
    }
    pub fn new_alias(alias: AliasDecl) -> Result<Self, InvalidBound> {
        Ok(Self::Alias(Tagged::new(alias)?))
    }
}

impl Type<AnyLeaf> {
    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate<V>(variant_rows: impl IntoIterator<Item = V>) -> Self
    where
        V: IntoIterator<Item = Type<AnyLeaf>>,
    {
        Self::new_sum(predicate_variants_row(variant_rows))
    }
    /// New simple predicate with empty Tuple variants

    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(vec![]).take(size))
    }
}

impl<T: From<EqLeaf> + TypeClass> Type<T> {
    pub fn usize() -> Self {
        Self::Prim(EqLeaf::USize.into())
    }
}

impl<T: From<CopyableLeaf> + TypeClass> Type<T> {
    pub fn graph(signature: AbstractSignature) -> Self {
        Self::Prim(CopyableLeaf::Graph(Box::new(signature)).into())
    }
}

impl<T: TypeClass> Type<T> {
    #[inline]
    fn upcast<T2: From<T> + TypeClass>(self) -> Type<T2> {
        match self {
            Type::Prim(t) => Type::Prim(t.into()),
            Type::Extension(Tagged(t, _)) => Type::Extension(Tagged(t, PhantomData)),
            Type::Alias(Tagged(t, _)) => Type::Alias(Tagged(t, PhantomData)),
            Type::Array(t, l) => Type::Array(Box::new(t.upcast()), l),
            Type::Tuple(row) => Type::Tuple(row.into_iter().map(Type::<T>::upcast).collect()),
            Type::Sum(row) => Type::Sum(row.into_iter().map(Type::<T>::upcast).collect()),
        }
    }
}

/// Return the type row of variants required to define a Sum of Tuples type
/// given the rows of each tuple
pub(crate) fn predicate_variants_row<V>(
    variant_rows: impl IntoIterator<Item = V>,
) -> Vec<Type<AnyLeaf>>
where
    V: IntoIterator<Item = Type<AnyLeaf>>,
{
    variant_rows.into_iter().map(Type::new_tuple).collect()
}

impl From<Type<EqLeaf>> for Type<CopyableLeaf> {
    fn from(value: Type<EqLeaf>) -> Self {
        value.upcast()
    }
}

impl From<Type<EqLeaf>> for Type<AnyLeaf> {
    fn from(value: Type<EqLeaf>) -> Self {
        value.upcast()
    }
}

impl From<Type<CopyableLeaf>> for Type<AnyLeaf> {
    fn from(value: Type<CopyableLeaf>) -> Self {
        value.upcast()
    }
}

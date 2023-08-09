#![allow(missing_docs)]

use std::marker::PhantomData;

use crate::ops::AliasDecl;
use itertools::Itertools;
use std::fmt::Display;

use super::{
    leaf::{AnyLeaf, CopyableLeaf, EqLeaf, InvalidBound, Tagged, TypeClass},
    AbstractSignature, CustomType, TypeTag,
};

use super::new_type_row::TypeRow;

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
    #[display(fmt = "Tuple({})", "_0")]
    Tuple(Box<TypeRow<T>>),
    #[display(fmt = "Sum({})", "_0")]
    Sum(Box<TypeRow<T>>),
}

impl<T: TypeClass> Type<T> {
    pub const BOUND_TAG: TypeTag = T::BOUND_TAG;
    pub const fn bounding_tag(&self) -> TypeTag {
        T::BOUND_TAG
    }

    pub fn new_tuple(types: impl Into<TypeRow<T>>) -> Self {
        Self::Tuple(Box::new(types.into()))
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    pub fn new_sum(types: impl Into<TypeRow<T>>) -> Self {
        Self::Sum(Box::new(types.into()))
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
    pub fn new_predicate(variant_rows: impl IntoIterator<Item = TypeRow<AnyLeaf>>) -> Self {
        Self::new_sum(TypeRow::predicate_variants_row(variant_rows))
    }
    /// New simple predicate with empty Tuple variants

    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(vec![]).map_into().take(size))
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
            Type::Tuple(row) => Type::Tuple(Box::new(upcast_row(*row))),
            Type::Sum(row) => Type::Sum(Box::new(upcast_row(*row))),
        }
    }
}

fn upcast_row<T: TypeClass, T2: From<T> + TypeClass>(row: TypeRow<T>) -> TypeRow<T2> {
    row.into_owned()
        .into_iter()
        .map(Type::<T>::upcast)
        .collect_vec()
        .into()
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

#![allow(missing_docs)]

use std::{fmt::Display, marker::PhantomData};

use crate::ops::AliasDecl;

use super::{
    leaf::{AnyLeaf, CopyableLeaf, EqLeaf, InvalidBound, Tagged, TypeClass},
    AbstractSignature, CustomType, TypeTag,
};

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Type<T> {
    Prim(T),
    Extension(Tagged<CustomType, T>),
    Alias(Tagged<AliasDecl, T>),
    Array(Box<Type<T>>, usize),
    Tuple(Vec<Type<T>>),
    Sum(Vec<Type<T>>),
}

impl<T: TypeClass> Type<T> {
    pub const BOUND_TAG: TypeTag = T::BOUND_TAG;
    pub const fn bounding_tag(&self) -> TypeTag {
        T::BOUND_TAG
    }

    pub fn new_tuple(types: impl IntoIterator<Item = Type<T>>) -> Self {
        Self::Tuple(types.into_iter().collect())
    }

    pub fn new_extension(opaque: CustomType) -> Result<Self, InvalidBound> {
        Ok(Self::Extension(Tagged::new(opaque)?))
    }
    pub fn new_alias(alias: AliasDecl) -> Result<Self, InvalidBound> {
        Ok(Self::Alias(Tagged::new(alias)?))
    }
}

impl<T: From<EqLeaf>> Type<T> {
    pub fn usize() -> Self {
        Self::Prim(EqLeaf::USize.into())
    }
}

impl<T: From<CopyableLeaf>> Type<T> {
    pub fn graph(signature: AbstractSignature) -> Self {
        Self::Prim(CopyableLeaf::Graph(Box::new(signature)).into())
    }
}

impl<T> Type<T> {
    #[inline]
    fn upcast<T2: From<T>>(self) -> Type<T2> {
        match self {
            Type::Prim(t) => Type::Prim(t.into()),
            Type::Extension(Tagged(t, _)) => Type::Extension(Tagged(t, PhantomData)),
            Type::Alias(Tagged(t, _)) => Type::Alias(Tagged(t, PhantomData)),
            Type::Array(t, l) => Type::Array(Box::new(t.upcast()), l),
            Type::Tuple(vec) => Type::Tuple(vec.into_iter().map(Type::<T>::upcast).collect()),
            Type::Sum(vec) => Type::Sum(vec.into_iter().map(Type::<T>::upcast).collect()),
        }
    }
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

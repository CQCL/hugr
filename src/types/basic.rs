#![allow(missing_docs)]

use std::marker::PhantomData;

use crate::ops::AliasDecl;

use super::{AbstractSignature, CustomType, TypeTag};
use thiserror::Error;

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum EqLeaf {
    USize,
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum ClassicLeaf {
    E(EqLeaf),
    Graph(Box<AbstractSignature>),
}
#[derive(Clone, PartialEq, Debug, Eq)]
pub enum AnyLeaf {
    C(ClassicLeaf),
}

impl From<EqLeaf> for ClassicLeaf {
    fn from(value: EqLeaf) -> Self {
        ClassicLeaf::E(value)
    }
}

impl<T: Into<ClassicLeaf>> From<T> for AnyLeaf {
    fn from(value: T) -> Self {
        AnyLeaf::C(value.into())
    }
}

mod sealed {
    use super::{AnyLeaf, ClassicLeaf, EqLeaf};
    pub trait Sealed {}
    impl Sealed for AnyLeaf {}
    impl Sealed for ClassicLeaf {}
    impl Sealed for EqLeaf {}
}
pub trait TypeClass: sealed::Sealed {
    const BOUND_TAG: TypeTag;
}

impl TypeClass for EqLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Hashable;
}

impl TypeClass for ClassicLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Classic;
}
impl TypeClass for AnyLeaf {
    const BOUND_TAG: TypeTag = TypeTag::Simple;
}

#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Tagged<I, T>(I, PhantomData<T>);

pub trait ActualTag {
    fn actual_tag(&self) -> TypeTag;
}

impl ActualTag for CustomType {
    fn actual_tag(&self) -> TypeTag {
        self.tag()
    }
}

impl ActualTag for AliasDecl {
    fn actual_tag(&self) -> TypeTag {
        self.tag
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
#[error("The tag reported by the object ({found:?} is not contained by the tag of the Type ({bound:?}).")]
pub struct InvalidBound {
    bound: TypeTag,
    found: TypeTag,
}

impl<T: ActualTag, C: TypeClass> Tagged<T, C> {
    pub fn new(inner: T) -> Result<Self, InvalidBound> {
        if C::BOUND_TAG.contains(inner.actual_tag()) {
            Ok(Self(inner, PhantomData))
        } else {
            Err(InvalidBound {
                bound: C::BOUND_TAG,
                found: inner.actual_tag(),
            })
        }
    }
    pub fn inner(&self) -> &T {
        &self.0
    }
}

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

impl<T: From<ClassicLeaf>> Type<T> {
    pub fn graph(signature: AbstractSignature) -> Self {
        Self::Prim(ClassicLeaf::Graph(Box::new(signature)).into())
    }
}

impl<T> Type<T> {
    #[inline]
    fn upcast<T2: From<T>>(self) -> Type<T2> {
        match self {
            Type::Prim(t) => Type::Prim(t.into()),
            Type::Extension(Tagged(t, _)) => Type::Extension(Tagged(t, PhantomData)),
            Type::Alias(Tagged(t, _)) => Type::Alias(Tagged(t, PhantomData)),
            Type::Array(_, _) => todo!(),
            Type::Tuple(vec) => Type::Tuple(vec.into_iter().map(Type::<T>::upcast).collect()),
            Type::Sum(_) => todo!(),
        }
    }
}

impl From<Type<EqLeaf>> for Type<ClassicLeaf> {
    fn from(value: Type<EqLeaf>) -> Self {
        value.upcast()
    }
}

impl From<Type<EqLeaf>> for Type<AnyLeaf> {
    fn from(value: Type<EqLeaf>) -> Self {
        value.upcast()
    }
}

impl From<Type<ClassicLeaf>> for Type<AnyLeaf> {
    fn from(value: Type<ClassicLeaf>) -> Self {
        value.upcast()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn construct() {
        let t: Type<ClassicLeaf> = Type::new_tuple([
            Type::usize(),
            Type::graph(AbstractSignature::new_linear(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_resource",
                TypeTag::Classic,
            ))
            .unwrap(),
            Type::new_alias(AliasDecl::new("my_alias", TypeTag::Hashable)).unwrap(),
        ]);
        assert_eq!(t.bounding_tag(), TypeTag::Classic);
        let t_any: Type<AnyLeaf> = t.into();

        assert_eq!(t_any.bounding_tag(), TypeTag::Simple);
    }

    #[test]
    fn test_bad_dynamic() {
        let res: Result<Type<ClassicLeaf>, _> =
            Type::new_alias(AliasDecl::new("my_alias", TypeTag::Simple));
        assert_eq!(
            res,
            Err(InvalidBound {
                bound: TypeTag::Classic,
                found: TypeTag::Simple
            })
        );
    }
    #[test]
    fn all_constructors() {
        Type::<EqLeaf>::usize();
        Type::<ClassicLeaf>::usize();
        Type::<AnyLeaf>::usize();
        Type::<ClassicLeaf>::graph(AbstractSignature::new_linear(vec![]));
        Type::<AnyLeaf>::graph(AbstractSignature::new_linear(vec![]));
    }
}

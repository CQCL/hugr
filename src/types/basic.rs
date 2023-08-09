#![allow(missing_docs)]

use std::marker::PhantomData;

use super::{AbstractSignature, CustomType, TypeTag};
use thiserror::Error;
pub enum EqLeaf {
    USize,
}
pub enum ClassicLeaf {
    E(EqLeaf),
    Graph(Box<AbstractSignature>),
}
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
    use super::{AnyLeaf, ClassicLeaf, EqLeaf, Type};
    pub trait SealedLeaf {}
    impl SealedLeaf for AnyLeaf {}
    impl SealedLeaf for ClassicLeaf {}
    impl SealedLeaf for EqLeaf {}

    pub trait SealedType {}
    impl<T: SealedLeaf> SealedType for Type<T> {}
}
pub trait TypeClass: sealed::SealedLeaf {
    const TAG: TypeTag;
}

impl TypeClass for EqLeaf {
    const TAG: TypeTag = TypeTag::Hashable;
}

impl TypeClass for ClassicLeaf {
    const TAG: TypeTag = TypeTag::Classic;
}
impl TypeClass for AnyLeaf {
    const TAG: TypeTag = TypeTag::Simple;
}
pub struct Tagged<I, T>(I, PhantomData<T>);

pub trait GetTag {
    fn tag(&self) -> TypeTag;
}

impl GetTag for CustomType {
    fn tag(&self) -> TypeTag {
        self.tag()
    }
}

#[derive(Debug, Clone, Error)]
#[error("The tag reported by the object is not contained by the tag of the Type.")]
pub struct InvalidBound;

impl<T: GetTag, C: TypeClass> Tagged<T, C> {
    pub fn new(inner: T) -> Result<Self, InvalidBound> {
        if C::TAG.contains(inner.tag()) {
            Ok(Self(inner, PhantomData))
        } else {
            Err(InvalidBound)
        }
    }
    pub fn inner(&self) -> &T {
        &self.0
    }
}

pub struct OpaqueType(CustomType);
pub enum Type<T> {
    Prim(T),
    Extension(Tagged<CustomType, T>),
    Array(Box<Type<T>>, usize),
    Tuple(Vec<Type<T>>),
    Sum(Vec<Type<T>>),
}

impl<T: TypeClass> Type<T> {
    pub const TAG: TypeTag = T::TAG;
    pub const fn tag(&self) -> TypeTag {
        T::TAG
    }

    pub fn new_tuple(types: impl IntoIterator<Item = Type<T>>) -> Self {
        Self::Tuple(types.into_iter().collect())
    }

    pub fn new_opaque(opaque: CustomType) -> Result<Self, InvalidBound> {
        Ok(Self::Extension(Tagged::new(opaque)?))
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
            Type::new_opaque(CustomType::new(
                "my_custom",
                [],
                "my_resource",
                TypeTag::Classic,
            ))
            .unwrap(),
        ]);
        assert_eq!(t.tag(), TypeTag::Classic);
        let t_any: Type<AnyLeaf> = t.into();

        assert_eq!(t_any.tag(), TypeTag::Simple);
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

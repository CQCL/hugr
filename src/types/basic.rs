#![allow(missing_docs)]

use super::{AbstractSignature, CustomType, TypeTag};

enum EqTypeImpl {
    USize,
}
enum ClassicTypeImpl {
    E(EqTypeImpl),
    Graph(Box<AbstractSignature>),
}
enum AnyTypeImpl {
    C(ClassicTypeImpl),
}

pub struct Eq(EqTypeImpl);
pub struct Classic(ClassicTypeImpl);
pub struct Any(AnyTypeImpl);

mod sealed {
    use super::{Any, Classic, Eq};
    pub trait Sealed {}
    impl Sealed for Any {}
    impl Sealed for Classic {}
    impl Sealed for Eq {}
}
pub trait TypeClass: sealed::Sealed {
    const TAG: TypeTag;
}

impl TypeClass for Eq {
    const TAG: TypeTag = TypeTag::Hashable;
}

impl TypeClass for Classic {
    const TAG: TypeTag = TypeTag::Classic;
}
impl TypeClass for Any {
    const TAG: TypeTag = TypeTag::Simple;
}
pub struct TaggedWrapper<T>(TypeTag, T);

impl<T> TaggedWrapper<T> {
    pub fn tag(&self) -> TypeTag {
        self.0
    }

    pub fn inner(&self) -> &T {
        &self.1
    }
}

pub struct OpaqueType(CustomType);
pub enum Type<T> {
    Prim(T),
    Extension(OpaqueType),
    Alias(TaggedWrapper<String>),
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

    pub fn new_opaque(opaque: CustomType) -> Self {
        assert!(T::TAG.contains(opaque.tag()), "tag ");
        Self::Extension(OpaqueType(opaque))
    }
}

/*
Public traits for construction
*/
pub trait NewEq {
    const USIZE: Self;
}

pub trait NewClassic: NewEq {
    fn graph(signature: AbstractSignature) -> Self;
}

impl NewEq for Type<Eq> {
    const USIZE: Self = Self::Prim(Eq(EqTypeImpl::USize));
}

impl NewEq for Type<Classic> {
    const USIZE: Self = Self::Prim(Classic(ClassicTypeImpl::E(EqTypeImpl::USize)));
}

impl NewClassic for Type<Classic> {
    fn graph(signature: AbstractSignature) -> Self {
        Self::Prim(Classic(ClassicTypeImpl::Graph(Box::new(signature))))
    }
}

impl NewEq for Type<Any> {
    const USIZE: Self = Self::Prim(Any(AnyTypeImpl::C(ClassicTypeImpl::E(EqTypeImpl::USize))));
}
impl NewClassic for Type<Any> {
    fn graph(signature: AbstractSignature) -> Self {
        Type::<Classic>::graph(signature).upcast()
    }
}

pub trait UpCastTo<T2>: Sized {
    fn upcast(self) -> T2;
}

impl UpCastTo<Type<Any>> for Type<Classic> {
    fn upcast(self: Type<Classic>) -> Type<Any> {
        match self {
            Type::Prim(t) => Type::Prim(Any(AnyTypeImpl::C(t.0))),
            Type::Extension(t) => Type::Extension(t),
            Type::Alias(_) => todo!(),
            Type::Array(_, _) => todo!(),
            Type::Tuple(vec) => Type::Tuple(vec.into_iter().map(UpCastTo::upcast).collect()),
            Type::Sum(_) => todo!(),
        }
    }
}

impl UpCastTo<Type<Classic>> for Type<Eq> {
    fn upcast(self) -> Type<Classic> {
        todo!()
    }
}

impl UpCastTo<Type<Any>> for Type<Eq> {
    fn upcast(self) -> Type<Any> {
        let cl: Type<Classic> = self.upcast();
        cl.upcast()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn construct() {
        let t: Type<Classic> = Type::new_tuple([
            Type::USIZE,
            Type::graph(AbstractSignature::new_linear(vec![])),
            Type::new_opaque(CustomType::new(
                "my_custom",
                [],
                "my_resource",
                TypeTag::Classic,
            )),
        ]);
        assert_eq!(t.tag(), TypeTag::Classic);
        let t_any: Type<Any> = t.upcast();

        assert_eq!(t_any.tag(), TypeTag::Simple);
    }
}

#![allow(missing_docs)]

use super::{AbstractSignature, CustomType, TypeTag};

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

mod sealed {
    use super::{AnyLeaf, ClassicLeaf, EqLeaf};
    pub trait Sealed {}
    impl Sealed for AnyLeaf {}
    impl Sealed for ClassicLeaf {}
    impl Sealed for EqLeaf {}
}
pub trait TypeClass: sealed::Sealed {
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

impl NewEq for Type<EqLeaf> {
    const USIZE: Self = Self::Prim(EqLeaf::USize);
}

impl NewEq for Type<ClassicLeaf> {
    const USIZE: Self = Self::Prim(ClassicLeaf::E(EqLeaf::USize));
}

impl NewClassic for Type<ClassicLeaf> {
    fn graph(signature: AbstractSignature) -> Self {
        Self::Prim(ClassicLeaf::Graph(Box::new(signature)))
    }
}

impl NewEq for Type<AnyLeaf> {
    const USIZE: Self = Self::Prim(AnyLeaf::C(ClassicLeaf::E(EqLeaf::USize)));
}
impl NewClassic for Type<AnyLeaf> {
    fn graph(signature: AbstractSignature) -> Self {
        Type::<ClassicLeaf>::graph(signature).upcast()
    }
}

pub trait UpCastTo<T2>: Sized {
    fn upcast(self) -> T2;
}

impl<T: UpCastTo<T2>, T2> UpCastTo<Type<T2>> for Type<T> {
    fn upcast(self) -> Type<T2> {
        match self {
            Type::Prim(t) => Type::Prim(t.upcast()),
            Type::Extension(t) => Type::Extension(t),
            Type::Alias(_) => todo!(),
            Type::Array(_, _) => todo!(),
            Type::Tuple(vec) => Type::Tuple(vec.into_iter().map(UpCastTo::upcast).collect()),
            Type::Sum(_) => todo!(),
        }
    }
}

impl UpCastTo<ClassicLeaf> for EqLeaf {
    fn upcast(self) -> ClassicLeaf {
        ClassicLeaf::E(self)
    }
}

impl From<EqLeaf> for ClassicLeaf {
    fn from(value: EqLeaf) -> Self {
        ClassicLeaf::E(value)
    }
}

impl<T: Into<ClassicLeaf>> UpCastTo<AnyLeaf> for T {
    fn upcast(self) -> AnyLeaf {
        AnyLeaf::C(self.into())
    }
}

impl<T: Into<ClassicLeaf>> From<T> for AnyLeaf {
    fn from(value: T) -> Self {
        value.upcast()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn construct() {
        let t: Type<ClassicLeaf> = Type::new_tuple([
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
        let t_any: Type<AnyLeaf> = t.upcast();

        assert_eq!(t_any.tag(), TypeTag::Simple);
    }
}

use super::{AnyLeaf, CopyableLeaf, EqLeaf, InvalidBound, Type, TypeClass, TypeTag};

use itertools::Itertools;
use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::super::AbstractSignature;

use crate::ops::AliasDecl;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    I,
    G(AbstractSignature),
    Tuple(Vec<SerSimpleType>),
    Sum(Vec<SerSimpleType>),
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
    },
    Opaque(CustomType),
    Alias {
        name: SmolStr,
        c: TypeTag,
    },
}

impl<T: SerLeaf> From<Type<T>> for SerSimpleType {
    fn from(value: Type<T>) -> Self {
        match value {
            Type::Prim(t) => t.ser(),
            Type::Sum(inner) => SerSimpleType::Sum(inner.into_iter().map_into().collect()),
            Type::Tuple(inner) => SerSimpleType::Tuple(inner.into_iter().map_into().collect()),
            Type::Array(inner, len) => SerSimpleType::Array {
                inner: Box::new((*inner).into()),
                len,
            },
            Type::Alias(decl) => SerSimpleType::Alias {
                name: decl.inner().name.clone(),
                c: decl.inner().tag,
            },
            Type::Extension(custom) => SerSimpleType::Opaque(custom.inner().clone()),
        }
    }
}

pub(super) trait SerLeaf: TypeClass {
    fn usize() -> Type<Self>;
    fn graph(sig: AbstractSignature) -> Result<Type<Self>, InvalidBound>;
    fn ser(&self) -> SerSimpleType;
}

impl SerLeaf for EqLeaf {
    fn usize() -> Type<EqLeaf> {
        Type::usize()
    }
    fn graph(_sig: AbstractSignature) -> Result<Type<EqLeaf>, InvalidBound> {
        Err(InvalidBound {
            bound: TypeTag::Hashable,
            found: TypeTag::Classic,
        })
    }
    fn ser(&self) -> SerSimpleType {
        match self {
            EqLeaf::USize => SerSimpleType::I,
        }
    }
}

impl SerLeaf for CopyableLeaf {
    fn usize() -> Type<CopyableLeaf> {
        Type::usize()
    }
    fn graph(sig: AbstractSignature) -> Result<Type<CopyableLeaf>, InvalidBound> {
        Ok(Type::graph(sig))
    }
    fn ser(&self) -> SerSimpleType {
        match self {
            CopyableLeaf::E(e) => e.ser(),
            CopyableLeaf::Graph(sig) => SerSimpleType::G((**sig).clone()),
        }
    }
}

impl SerLeaf for AnyLeaf {
    fn usize() -> Type<AnyLeaf> {
        Type::usize()
    }
    fn graph(sig: AbstractSignature) -> Result<Type<AnyLeaf>, InvalidBound> {
        Ok(Type::graph(sig))
    }
    fn ser(&self) -> SerSimpleType {
        match self {
            AnyLeaf::C(c) => c.ser(),
        }
    }
}

impl<T: TypeClass + SerLeaf> TryFrom<SerSimpleType> for Type<T> {
    type Error = InvalidBound;

    fn try_from(value: SerSimpleType) -> Result<Self, InvalidBound> {
        match value {
            SerSimpleType::I => Ok(T::usize()),
            SerSimpleType::G(sig) => T::graph(sig),
            SerSimpleType::Tuple(elems) => Ok(Type::new_tuple(
                elems
                    .into_iter()
                    .map(Type::<T>::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            SerSimpleType::Sum(elems) => Ok(Type::new_sum(
                elems
                    .into_iter()
                    .map(Type::<T>::try_from)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            SerSimpleType::Array { inner, len } => {
                Ok(Type::Array(Box::new((*inner).try_into()?), len))
            }
            SerSimpleType::Opaque(custom) => Type::new_extension(custom),
            SerSimpleType::Alias { name, c } => Type::new_alias(AliasDecl::new(name, c)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::custom::test::CLASSIC_CUST;
    use crate::types::type_enum::{AnyLeaf, CopyableLeaf, EqLeaf, Type};
    use crate::types::AbstractSignature;

    #[test]
    fn serialize_types_roundtrip() {
        let g: Type<CopyableLeaf> = Type::graph(AbstractSignature::new_linear(vec![
            crate::types::SimpleType::Qubit,
        ]));

        // A Simple tuple
        let t = Type::<AnyLeaf>::new_tuple([Type::usize(), g.into()]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::<CopyableLeaf>::new_sum([
            Type::usize(),
            Type::new_extension(CLASSIC_CUST).unwrap(),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable array
        let t: Type<EqLeaf> = Type::Array(Box::new(Type::usize()), 3);
        assert_eq!(ser_roundtrip(&t), t);
    }
}

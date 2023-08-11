use super::{Type, TypeEnum};

use itertools::Itertools;

use super::custom::CustomType;

use super::AbstractSignature;

use crate::ops::AliasDecl;
use crate::types::primitive::PrimType;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    I,
    G(Box<AbstractSignature>),
    Tuple { inner: Vec<SerSimpleType> },
    Sum { inner: Vec<SerSimpleType> },
    Array { inner: Box<SerSimpleType>, len: u64 },
    Opaque(Box<CustomType>),
    Alias(AliasDecl),
}

impl From<Type> for SerSimpleType {
    fn from(value: Type) -> Self {
        let Type(value, _) = value;
        match value {
            TypeEnum::Prim(t) => match t {
                PrimType::E(c) => SerSimpleType::Opaque(Box::new(*c)),
                PrimType::A(a) => SerSimpleType::Alias(a),
                PrimType::Graph(sig) => SerSimpleType::G(Box::new(*sig)),
            },
            TypeEnum::Sum(inner) => SerSimpleType::Sum {
                inner: inner.into_iter().map_into().collect(),
            },
            TypeEnum::Tuple(inner) => SerSimpleType::Tuple {
                inner: inner.into_iter().map_into().collect(),
            },
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Type {
        match value {
            SerSimpleType::I => Type::usize(),
            SerSimpleType::G(sig) => Type::graph(*sig),
            SerSimpleType::Tuple { inner } => Type::new_tuple(inner.into_iter().map_into()),
            SerSimpleType::Sum { inner } => Type::new_sum(inner.into_iter().map_into()),
            SerSimpleType::Array { inner, len } => Type::new_array((*inner).into(), len),
            SerSimpleType::Opaque(custom) => Type::new_extension(*custom),
            SerSimpleType::Alias(a) => Type::new_alias(a),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::custom::test::COPYABLE_CUST;
    use crate::types::AbstractSignature;
    use crate::types::Type;

    #[test]
    fn serialize_types_roundtrip() {
        let g: Type = Type::graph(AbstractSignature::new_linear(vec![]));

        assert_eq!(ser_roundtrip(&g), g);

        // A Simple tuple
        let t = Type::new_tuple([Type::usize(), g]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::new_sum([Type::usize(), Type::new_extension(COPYABLE_CUST)]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable array
        // TODO uncomment once refactor complete
        // let t: Type = Type::new_array(Type::usize(), 3);
        // assert_eq!(ser_roundtrip(&t), t);
    }
}

use super::{Type, TypeEnum};

use itertools::Itertools;

use super::super::custom::CustomType;

use super::super::AbstractSignature;

use crate::ops::AliasDecl;
use crate::types::leaf::PrimType;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    I,
    G(AbstractSignature),
    Tuple {
        inner: Vec<SerSimpleType>,
    },
    Sum {
        inner: Vec<SerSimpleType>,
    },
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
    },
    Opaque(CustomType),
    Alias(AliasDecl),
}

impl From<Type> for SerSimpleType {
    fn from(value: Type) -> Self {
        let Type(value, _) = value;
        match value {
            TypeEnum::Prim(t) => match t {
                PrimType::E(c) => SerSimpleType::Opaque(c),
                PrimType::A(a) => SerSimpleType::Alias(a),
            },
            TypeEnum::Sum(inner) => SerSimpleType::Sum {
                inner: inner.into_iter().map_into().collect(),
            },
            TypeEnum::Tuple(inner) => SerSimpleType::Tuple {
                inner: inner.into_iter().map_into().collect(),
            },
            TypeEnum::Array(inner, len) => SerSimpleType::Array {
                inner: Box::new((*inner).into()),
                len,
            },
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Type {
        match value {
            SerSimpleType::I => Type::usize(),
            SerSimpleType::G(sig) => Type::graph(sig),
            SerSimpleType::Tuple { inner } => Type::new_tuple(inner.into_iter().map_into()),
            SerSimpleType::Sum { inner } => Type::new_sum(inner.into_iter().map_into()),
            SerSimpleType::Array { inner, len } => Type::new_array((*inner).into(), len),
            SerSimpleType::Opaque(custom) => Type::new_extension(custom),
            SerSimpleType::Alias(a) => Type::new_alias(a),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::custom::test::CLASSIC_CUST;
    use crate::types::type_enum::Type;
    use crate::types::AbstractSignature;

    #[test]
    fn serialize_types_roundtrip() {
        let g: Type = Type::graph(AbstractSignature::new_linear(vec![
            crate::types::SimpleType::Qubit,
        ]));

        assert_eq!(ser_roundtrip(&g), g);

        // A Simple tuple
        let t = Type::new_tuple([Type::usize(), g]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::new_sum([Type::usize(), Type::new_extension(CLASSIC_CUST)]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable array
        let t: Type = Type::new_array(Type::usize(), 3);
        assert_eq!(ser_roundtrip(&t), t);
    }
}

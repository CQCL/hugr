use super::{SumType, Type, TypeBound, TypeEnum, TypeRow};

use super::custom::CustomType;

use super::FunctionType;

use crate::extension::prelude::{new_array, QB_T, USIZE_T};
use crate::ops::AliasDecl;
use crate::types::primitive::PrimType;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(super) enum SerSimpleType {
    Q,
    I,
    G(Box<FunctionType>),
    Tuple { inner: TypeRow },
    Sum(SumType),
    Array { inner: Box<SerSimpleType>, len: u64 },
    Opaque(CustomType),
    Alias(AliasDecl),
    V { i: usize, b: TypeBound },
}

impl From<Type> for SerSimpleType {
    fn from(value: Type) -> Self {
        if value == QB_T {
            return SerSimpleType::Q;
        }
        if value == USIZE_T {
            return SerSimpleType::I;
        }
        // TODO short circuiting for array.
        let Type(value, _) = value;
        match value {
            TypeEnum::Prim(t) => match t {
                PrimType::Extension(c) => SerSimpleType::Opaque(c),
                PrimType::Alias(a) => SerSimpleType::Alias(a),
                PrimType::Function(sig) => SerSimpleType::G(Box::new(*sig)),
                PrimType::Variable(i, b) => SerSimpleType::V { i, b },
            },
            TypeEnum::Sum(sum) => SerSimpleType::Sum(sum),
            TypeEnum::Tuple(inner) => SerSimpleType::Tuple { inner },
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Type {
        match value {
            SerSimpleType::Q => QB_T,
            SerSimpleType::I => USIZE_T,
            SerSimpleType::G(sig) => Type::new_function(*sig),
            SerSimpleType::Tuple { inner } => Type::new_tuple(inner),
            SerSimpleType::Sum(sum) => sum.into(),
            SerSimpleType::Array { inner, len } => new_array((*inner).into(), len),
            SerSimpleType::Opaque(custom) => Type::new_extension(custom),
            SerSimpleType::Alias(a) => Type::new_alias(a),
            SerSimpleType::V { i, b } => Type::new_var_use(i, b),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::extension::prelude::USIZE_T;
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
    use crate::types::FunctionType;
    use crate::types::Type;

    #[test]
    fn serialize_types_roundtrip() {
        let g: Type = Type::new_function(FunctionType::new_linear(vec![]));

        assert_eq!(ser_roundtrip(&g), g);

        // A Simple tuple
        let t = Type::new_tuple(vec![USIZE_T, g]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::new_sum(vec![USIZE_T, FLOAT64_TYPE]);
        assert_eq!(ser_roundtrip(&t), t);

        // A simple predicate
        let t = Type::new_simple_predicate(4);
        assert_eq!(ser_roundtrip(&t), t);
    }
}

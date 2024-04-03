use super::{PolyFuncType, SumType, Type, TypeArg, TypeBound, TypeEnum};

use super::custom::CustomType;

use crate::extension::prelude::{array_type, QB_T, USIZE_T};
use crate::ops::AliasDecl;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(super) enum SerSimpleType {
    Q,
    I,
    G(Box<PolyFuncType>),
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
            TypeEnum::Extension(c) => SerSimpleType::Opaque(c),
            TypeEnum::Alias(a) => SerSimpleType::Alias(a),
            TypeEnum::Function(sig) => SerSimpleType::G(sig),
            TypeEnum::Variable(i, b) => SerSimpleType::V { i, b },
            TypeEnum::Sum(sum) => SerSimpleType::Sum(sum),
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Type {
        match value {
            SerSimpleType::Q => QB_T,
            SerSimpleType::I => USIZE_T,
            SerSimpleType::G(sig) => Type::new_function(*sig),
            SerSimpleType::Sum(sum) => sum.into(),
            SerSimpleType::Array { inner, len } => {
                array_type(TypeArg::BoundedNat { n: len }, (*inner).into())
            }
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
    use crate::type_row;
    use crate::types::FunctionType;
    use crate::types::Type;

    #[test]
    fn serialize_types_roundtrip() {
        let g: Type = Type::new_function(FunctionType::new_endo(vec![]));

        assert_eq!(ser_roundtrip(&g), g);

        // A Simple tuple
        let t = Type::new_tuple(vec![USIZE_T, g]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::new_sum([type_row![USIZE_T], type_row![FLOAT64_TYPE]]);
        assert_eq!(ser_roundtrip(&t), t);

        let t = Type::new_unit_sum(4);
        assert_eq!(ser_roundtrip(&t), t);
    }
}

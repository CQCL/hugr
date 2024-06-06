use super::{FunctionType, SumType, Type, TypeArg, TypeBound, TypeEnum};

use super::custom::CustomType;

use crate::extension::prelude::{array_type, QB_T, USIZE_T};
use crate::ops::AliasDecl;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(super) enum SerSimpleType {
    Q,
    I,
    G(Box<FunctionType>),
    Sum(SumType),
    Array { inner: Box<SerSimpleType>, len: u64 },
    Opaque(CustomType),
    Alias(AliasDecl),
    V { i: usize, b: TypeBound },
    R { i: usize, b: TypeBound },
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
            TypeEnum::Extension(o) => SerSimpleType::Opaque(o),
            TypeEnum::Alias(a) => SerSimpleType::Alias(a),
            TypeEnum::Function(sig) => SerSimpleType::G(sig),
            TypeEnum::Variable(i, b) => SerSimpleType::V { i, b },
            TypeEnum::RowVariable(i, b) => SerSimpleType::R { i, b },
            TypeEnum::Sum(st) => SerSimpleType::Sum(st),
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Type {
        match value {
            SerSimpleType::Q => QB_T,
            SerSimpleType::I => USIZE_T,
            SerSimpleType::G(sig) => Type::new_function(*sig),
            SerSimpleType::Sum(st) => st.into(),
            SerSimpleType::Array { inner, len } => {
                array_type(TypeArg::BoundedNat { n: len }, (*inner).into())
            }
            SerSimpleType::Opaque(o) => Type::new_extension(o),
            SerSimpleType::Alias(a) => Type::new_alias(a),
            SerSimpleType::V { i, b } => Type::new_var_use(i, b),
            SerSimpleType::R { i, b } => Type::new_row_var_use(i, b),
        }
    }
}

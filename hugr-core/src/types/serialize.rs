use super::{FunctionTypeRV, SumType, TypeArg, TypeBase, TypeBound, TypeEnum};

use super::custom::CustomType;

use crate::extension::prelude::{array_type, QB_T, USIZE_T};
use crate::extension::SignatureError;
use crate::ops::AliasDecl;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(super) enum SerSimpleType {
    Q,
    I,
    G(Box<FunctionTypeRV>),
    Sum(SumType),
    Array { inner: Box<SerSimpleType>, len: u64 },
    Opaque(CustomType),
    Alias(AliasDecl),
    V { i: usize, b: TypeBound },
    R { i: usize, b: TypeBound },
}

impl<const RV: bool> From<TypeBase<RV>> for SerSimpleType {
    fn from(value: TypeBase<RV>) -> Self {
        if value == QB_T {
            return SerSimpleType::Q;
        };
        if value == USIZE_T {
            return SerSimpleType::I;
        };
        match value.0 {
            TypeEnum::Extension(o) => SerSimpleType::Opaque(o),
            TypeEnum::Alias(a) => SerSimpleType::Alias(a),
            TypeEnum::Function(sig) => SerSimpleType::G(sig),
            TypeEnum::Variable(i, b) => SerSimpleType::V { i, b },
            TypeEnum::RowVariable(i, b) => SerSimpleType::R { i, b },
            TypeEnum::Sum(st) => SerSimpleType::Sum(st),
        }
    }
}

impl<const RV: bool> TryFrom<SerSimpleType> for TypeBase<RV> {
    type Error = SignatureError;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        Ok(match value {
            SerSimpleType::Q => QB_T.into_(),
            SerSimpleType::I => USIZE_T.into_(),
            SerSimpleType::G(sig) => TypeBase::new_function(*sig),
            SerSimpleType::Sum(st) => st.into(),
            SerSimpleType::Array { inner, len } => {
                array_type(TypeArg::BoundedNat { n: len }, (*inner).try_into().unwrap()).into_()
            }
            SerSimpleType::Opaque(o) => TypeBase::new_extension(o),
            SerSimpleType::Alias(a) => TypeBase::new_alias(a),
            SerSimpleType::V { i, b } => TypeBase::new_var_use(i, b),
            // We can't use new_row_var because that returns Type<true> not Type<RV>.
            SerSimpleType::R { i, b } => TypeBase::new(TypeEnum::RowVariable(i, b)).try_into_()?,
        })
    }
}

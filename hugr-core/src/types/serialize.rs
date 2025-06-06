use super::{FuncValueType, MaybeRV, RowVariable, SumType, TypeBase, TypeBound, TypeEnum};

use super::custom::CustomType;

use crate::extension::SignatureError;
use crate::extension::prelude::{qb_t, usize_t};
use crate::ops::AliasDecl;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(super) enum SerSimpleType {
    Q,
    I,
    G(Box<FuncValueType>),
    Sum(SumType),
    Opaque(CustomType),
    Alias(AliasDecl),
    V { i: usize, b: TypeBound },
    R { i: usize, b: TypeBound },
}

impl<RV: MaybeRV> From<TypeBase<RV>> for SerSimpleType {
    fn from(value: TypeBase<RV>) -> Self {
        if value == qb_t() {
            return SerSimpleType::Q;
        }
        if value == usize_t() {
            return SerSimpleType::I;
        }
        match value.0 {
            TypeEnum::Extension(o) => SerSimpleType::Opaque(o),
            TypeEnum::Alias(a) => SerSimpleType::Alias(a),
            TypeEnum::Function(sig) => SerSimpleType::G(sig),
            TypeEnum::Variable(i, b) => SerSimpleType::V { i, b },
            TypeEnum::RowVar(rv) => {
                let RowVariable(idx, bound) = rv.as_rv();
                SerSimpleType::R { i: *idx, b: *bound }
            }
            TypeEnum::Sum(st) => SerSimpleType::Sum(st),
        }
    }
}

impl<RV: MaybeRV> TryFrom<SerSimpleType> for TypeBase<RV> {
    type Error = SignatureError;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        Ok(match value {
            SerSimpleType::Q => qb_t().into_(),
            SerSimpleType::I => usize_t().into_(),
            SerSimpleType::G(sig) => TypeBase::new_function(*sig),
            SerSimpleType::Sum(st) => st.into(),
            SerSimpleType::Opaque(o) => TypeBase::new_extension(o),
            SerSimpleType::Alias(a) => TypeBase::new_alias(a),
            SerSimpleType::V { i, b } => TypeBase::new_var_use(i, b),
            // We can't use new_row_var because that returns TypeRV not TypeBase<RV>.
            SerSimpleType::R { i, b } => TypeBase::new(TypeEnum::RowVar(
                RV::try_from_rv(RowVariable(i, b))
                    .map_err(|var| SignatureError::RowVarWhereTypeExpected { var })?,
            )),
        })
    }
}

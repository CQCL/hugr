//! Logic for checking values against types.
use thiserror::Error;

use crate::values::{PrimValue, Value};

use super::{primitive::PrimType, CustomType, Type, TypeEnum};

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CustomCheckFail {
    /// The value had a specific type that was not what was expected
    #[error("Expected type: {0} but value was of type: {1}")]
    TypeMismatch(CustomType, CustomType),
    /// Any other message
    #[error("{0}")]
    Message(String),
}

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
pub enum ConstTypeError {
    /// Found a Var type constructor when we're checking a const val
    #[error("Type of a const value can't be Var")]
    ConstCantBeVar,
    /// Type we were checking against was an Alias.
    /// This should have been resolved to an actual type.
    #[error("Type of a const value can't be an Alias {0}")]
    NoAliases(String),
    /// The length of the tuple value doesn't match the length of the tuple type
    #[error("Tuple of wrong length")]
    TupleWrongLength,
    /// Tag for a sum value exceeded the number of variants
    #[error("Tag of Sum value is invalid")]
    InvalidSumTag,
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ValueCheckFail(Type, Value),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFail),
}

impl PrimType {
    pub fn check_type(&self, val: &PrimValue) -> Result<(), ConstTypeError> {
        if let PrimType::A(alias) = self {
            return Err(ConstTypeError::NoAliases(alias.name().to_string()));
        }

        match (self, val) {
            (PrimType::E(e), PrimValue::Extension(e_val)) => {
                e_val.0.check_custom_type(e)?;
                Ok(())
            }
            (PrimType::A(s), _) => Err(ConstTypeError::NoAliases(s.name().to_string())),
            (PrimType::Graph(_), PrimValue::Graph) => todo!(),
            _ => Err(ConstTypeError::ValueCheckFail(
                Type::new(TypeEnum::Prim(self.clone())),
                Value::Prim(val.clone()),
            )),
        }
    }
}

impl Type {
    pub fn check_type(&self, val: &Value) -> Result<(), ConstTypeError> {
        match (self.0, val) {
            (TypeEnum::Prim(p), Value::Prim(p_v)) => p.check_type(p_v),
            (TypeEnum::Tuple(t), Value::Tuple(t_v)) => {
                if t.len() != t_v.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                for (elem, ty) in t_v.iter().zip(t.iter()) {
                    ty.check_type(elem)?
                }
                Ok(())
            }
            (TypeEnum::Sum(variants), Value::Sum(tag, value)) => variants
                .get(*tag)
                .ok_or(ConstTypeError::InvalidSumTag)?
                .check_type(value),
            _ => Err(ConstTypeError::ValueCheckFail(self.clone(), val.clone())),
        }
    }
}

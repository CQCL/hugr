//! Logic for checking values against types.
use thiserror::Error;

use crate::{
    ops::{FuncDecl, FuncDefn, OpType},
    values::Value,
    Hugr, HugrView,
};

use super::{CustomType, PolyFuncType, Type, TypeEnum};

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CustomCheckFailure {
    /// The value had a specific type that was not what was expected
    #[error("Expected type: {expected} but value was of type: {found}")]
    TypeMismatch {
        /// The expected custom type.
        expected: CustomType,
        /// The custom type found when checking.
        found: CustomType,
    },
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
    CustomCheckFail(#[from] CustomCheckFailure),
}

fn check_ts(v: &Hugr, t: &PolyFuncType) -> bool {
    // exact signature equality, in future this may need to be
    // relaxed to be compatibility checks between the signatures.
    let root_op = v.get_optype(v.root());
    if let OpType::FuncDecl(FuncDecl { signature, .. })
    | OpType::FuncDefn(FuncDefn { signature, .. }) = root_op
    {
        signature == t
    } else {
        v.get_function_type()
            .is_some_and(|ft| &PolyFuncType::from(ft) == t)
    }
}

impl Type {
    /// Check that a [`Value`] is a valid instance of this [`Type`].
    ///
    /// # Errors
    ///
    /// This function will return an error if there is a type check error.
    pub fn check_type(&self, val: &Value) -> Result<(), ConstTypeError> {
        match (&self.0, val) {
            (TypeEnum::Extension(e), Value::Extension { c: e_val }) => {
                e_val.0.check_custom_type(e)?;
                Ok(())
            }
            (TypeEnum::Function(t), Value::Function { hugr: v }) if check_ts(v, t) => Ok(()),
            (TypeEnum::Tuple(t), Value::Tuple { vs: t_v }) => {
                if t.len() != t_v.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                t_v.iter()
                    .zip(t.iter())
                    .try_for_each(|(elem, ty)| ty.check_type(elem))
                    .map_err(|_| ConstTypeError::ValueCheckFail(self.clone(), val.clone()))
            }
            (TypeEnum::Sum(sum), Value::Sum { tag, value }) => sum
                .get_variant(*tag)
                .ok_or(ConstTypeError::InvalidSumTag)?
                .check_type(value),
            _ => Err(ConstTypeError::ValueCheckFail(self.clone(), val.clone())),
        }
    }
}

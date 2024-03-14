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
        found: Type,
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
    /// The length of the sum value doesn't match the length of the variant of
    /// the sum type
    #[error("Sum variant of wrong length")]
    SumWrongLength,
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

fn type_sig_equal(v: &Hugr, t: &PolyFuncType) -> bool {
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

impl super::SumType {
    /// Check that a [`Value`] is a valid instance of this [`SumType`].
    ///
    ///   [`SumType`]: crate::types::SumType
    ///
    /// # Errors
    ///
    /// This function will return an error if there is a type check error.
    pub fn check_type(&self, tag: usize, val: &[Box<Value>]) -> Result<(), ConstTypeError> {
        if self
            .get_variant(tag)
            .ok_or(ConstTypeError::InvalidSumTag)?
            .len()
            != val.len()
        {
            Err(ConstTypeError::SumWrongLength)?
        }

        for (t, v) in itertools::zip_eq(
            self.get_variant(tag)
                .ok_or(ConstTypeError::InvalidSumTag)?
                .iter(),
            val.iter(),
        ) {
            t.check_type(v)?;
        }
        Ok(())
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
            (TypeEnum::Extension(expected), Value::Extension { c: (e_val,) }) => {
                let found = e_val.get_type();
                if found == expected.clone().into() {
                    Ok(e_val.validate()?)
                } else {
                    Err(CustomCheckFailure::TypeMismatch {
                        expected: expected.clone(),
                        found,
                    }
                    .into())
                }
            }
            (TypeEnum::Function(t), Value::Function { hugr: v }) if type_sig_equal(v, t) => Ok(()),
            (TypeEnum::Tuple(t), Value::Tuple { vs: t_v }) => {
                if t.len() != t_v.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                t_v.iter()
                    .zip(t.iter())
                    .try_for_each(|(elem, ty)| ty.check_type(elem))
                    .map_err(|_| ConstTypeError::ValueCheckFail(self.clone(), val.clone()))
            }
            (TypeEnum::Sum(sum), Value::Sum { tag, values }) => sum.check_type(*tag, values),
            _ => Err(ConstTypeError::ValueCheckFail(self.clone(), val.clone())),
        }
    }
}

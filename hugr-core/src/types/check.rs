//! Logic for checking values against types.

use thiserror::Error;

use super::{Type, TypeRow};
use crate::{extension::SignatureError, ops::Value};

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum SumTypeError {
    /// The type of the variant doesn't match the type of the value.
    #[error("Expected type {expected} for element {index} of variant #{tag}, but found {}", .found.get_type())]
    InvalidValueType {
        /// Tag of the variant.
        tag: usize,
        /// The element in the tuple that was wrong.
        index: usize,
        /// The expected type.
        expected: Box<Type>,
        /// The value that was found.
        found: Box<Value>,
    },
    /// The type of the variant we were trying to convert into contained type variables
    #[error("Sum variant #{tag} contained a variable #{varidx}")]
    VariantNotConcrete {
        /// The variant index
        tag: usize,
        /// The index (identifier) of the type-variable
        varidx: usize,
    },
    /// The length of the sum value doesn't match the length of the variant of
    /// the sum type.
    #[error("Sum variant #{tag} should have length {expected}, but has length {found}")]
    WrongVariantLength {
        /// The variant index.
        tag: usize,
        /// The expected length of the sum variant.
        expected: usize,
        /// The length of the sum variant found.
        found: usize,
    },
    /// Tag for a sum value exceeded the number of variants.
    #[error("Invalid tag {tag} for sum type with {num_variants} variants")]
    InvalidTag {
        /// The tag of the sum value.
        tag: usize,
        /// The number of variants in the sum type.
        num_variants: usize,
    },
}

impl super::SumType {
    /// Check if a sum variant is a valid instance of this [`SumType`].
    ///
    /// Since [`Value::Sum`] variants always contain a tuple of values,
    /// `val` must be a slice of [`Value`]s.
    ///
    ///   [`SumType`]: crate::types::SumType
    ///
    /// # Errors
    ///
    /// This function will return an error if there is a type check error.
    pub fn check_type(&self, tag: usize, val: &[Value]) -> Result<(), SumTypeError> {
        let variant = self
            .get_variant(tag)
            .ok_or_else(|| SumTypeError::InvalidTag {
                tag,
                num_variants: self.num_variants(),
            })?;
        let variant: TypeRow = variant.clone().try_into().map_err(|e| {
            let SignatureError::RowVarWhereTypeExpected { var } = e else {
                panic!("Unexpected error")
            };
            SumTypeError::VariantNotConcrete { tag, varidx: var.0 }
        })?;

        if variant.len() != val.len() {
            Err(SumTypeError::WrongVariantLength {
                tag,
                expected: variant.len(),
                found: val.len(),
            })?;
        }

        for (index, (t, v)) in itertools::zip_eq(variant.iter(), val.iter()).enumerate() {
            if v.get_type() != *t {
                Err(SumTypeError::InvalidValueType {
                    tag,
                    index,
                    expected: Box::new(t.clone()),
                    found: Box::new(v.clone()),
                })?;
            }
        }
        Ok(())
    }
}

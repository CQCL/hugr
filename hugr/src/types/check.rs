//! Logic for checking values against types.

use thiserror::Error;

use super::{Type, TypeBound, TypeRow};
use crate::ops::Const;

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
pub enum SumTypeError {
    /// The type of the variant doesn't match the type of the value.
    #[error("Expected type {expected} for element {index} of variant #{tag}, but found {}", .found.const_type())]
    InvalidValueType {
        /// Tag of the variant.
        tag: usize,
        /// The element in the tuple that was wrong.
        index: usize,
        /// The expected type.
        expected: Type,
        /// The value that was found.
        found: Const,
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
    /// The value claims to be of a variant which contains a row variable, i.e. is not a concrete type
    #[error("Sum variant #{tag} contained row variable with index {idx} standing for a row of {bound} types")]
    ContainsRowVariables {
        /// The tag of the sum (and value)
        tag: usize,
        /// (DeBruijn) index of the row variable
        idx: usize,
        /// Bound on types which can be substituted for the variable
        bound: TypeBound,
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
    /// Since [`Const::Sum`] variants always contain a tuple of values,
    /// `val` must be a slice of [`Const`]s.
    ///
    ///   [`SumType`]: crate::types::SumType
    ///
    /// # Errors
    ///
    /// This function will return an error if there is a type check error.
    pub fn check_type(&self, tag: usize, val: &[Const]) -> Result<(), SumTypeError> {
        let variant = self
            .get_variant(tag)
            .ok_or_else(|| SumTypeError::InvalidTag {
                tag,
                num_variants: self.num_variants(),
            })?;

        // TODO how is it possible to call len() here?
        if variant.len() != val.len() {
            Err(SumTypeError::WrongVariantLength {
                tag,
                expected: variant.len(),
                found: val.len(),
            })?;
        }

        let variant: TypeRow = variant
            .clone()
            .try_into()
            .map_err(|(idx, bound)| SumTypeError::ContainsRowVariables { tag, idx, bound })?;

        for (index, (t, v)) in itertools::zip_eq(variant.iter(), val.iter()).enumerate() {
            if v.const_type() != *t {
                Err(SumTypeError::InvalidValueType {
                    tag,
                    index,
                    expected: t.clone(),
                    found: v.clone(),
                })?;
            }
        }
        Ok(())
    }
}

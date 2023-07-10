//! Simple type checking - takes a hugr and some extra info and checks whether
//! the types at the sources of each wire match those of the targets

use lazy_static::lazy_static;

use std::collections::HashSet;

use crate::hugr::*;
use crate::types::TypeRow;

// For static typechecking
use crate::ops::ConstValue;
use crate::types::{ClassicType, Container};

use crate::ops::constant::{HugrIntValueStore, HugrIntWidthStore, HUGR_MAX_INT_WIDTH};

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ConstTypeError {
    /// This case hasn't been implemented. Possibly because we don't have value
    /// constructors to check against it
    #[error("Const type checking unimplemented for {0}")]
    Unimplemented(ClassicType),
    /// The given type and term are incompatible
    #[error("Invalid const value for type {0}")]
    Failed(ClassicType),
    /// The value exceeds the max value of its `I<n>` type
    /// E.g. checking 300 against I8
    #[error("Const int {1} too large for type I{0}")]
    IntTooLarge(HugrIntWidthStore, HugrIntValueStore),
    /// Width (n) of an `I<n>` type doesn't fit into a HugrIntWidthStore
    #[error("Int type too large: I{0}")]
    IntWidthTooLarge(HugrIntWidthStore),
    /// The width of an integer type wasn't a power of 2
    #[error("The int type I{0} is invalid, because {0} is not a power of 2")]
    IntWidthInvalid(HugrIntWidthStore),
    /// Expected width (packed with const int) doesn't match type
    #[error("Type mismatch for int: expected I{0}, but found I{1}")]
    IntWidthMismatch(HugrIntWidthStore, HugrIntWidthStore),
    /// Found a Var type constructor when we're checking a const val
    #[error("Type of a const value can't be Var")]
    ConstCantBeVar,
    /// The length of the tuple value doesn't match the length of the tuple type
    #[error("Tuple of wrong length")]
    TupleWrongLength,
    /// Tag for a sum value exceeded the number of variants
    #[error("Tag of Sum value is invalid")]
    InvalidSumTag,
    /// For a value which embeds its type (e.g. sum or opaque) - a mismatch
    /// between the embedded type and the type we're checking against
    #[error("Type mismatch for const - expected {0}, found {1}")]
    TypeMismatch(ClassicType, ClassicType),
    /// A mismatch between the embedded type and the type we're checking
    /// against, as above, but for rows instead of simple types
    #[error("Type mismatch for const - expected {0}, found {1}")]
    TypeRowMismatch(TypeRow<ClassicType>, TypeRow<ClassicType>),
}

lazy_static! {
    static ref VALID_WIDTHS: HashSet<HugrIntWidthStore> =
        HashSet::from_iter((0..8).map(|a| HugrIntWidthStore::pow(2, a)));
}

/// Per the spec, valid widths for integers are 2^n for all n in [0,7]
fn check_valid_width(width: HugrIntWidthStore) -> Result<(), ConstTypeError> {
    if width > HUGR_MAX_INT_WIDTH {
        return Err(ConstTypeError::IntWidthTooLarge(width));
    }

    if VALID_WIDTHS.contains(&width) {
        Ok(())
    } else {
        Err(ConstTypeError::IntWidthInvalid(width))
    }
}

/// Typecheck a constant value
pub fn typecheck_const(typ: &ClassicType, val: &ConstValue) -> Result<(), ConstTypeError> {
    match (typ, val) {
        (ClassicType::Int(exp_width), ConstValue::Int { value, width }) => {
            // Check that the types make sense
            check_valid_width(*exp_width)?;
            check_valid_width(*width)?;
            // Check that the terms make sense against the types
            if exp_width == width {
                let max_value = if *width == HUGR_MAX_INT_WIDTH {
                    HugrIntValueStore::MAX
                } else {
                    HugrIntValueStore::pow(2, *width as u32) - 1
                };
                if value <= &max_value {
                    Ok(())
                } else {
                    Err(ConstTypeError::IntTooLarge(*width, *value))
                }
            } else {
                Err(ConstTypeError::IntWidthMismatch(*exp_width, *width))
            }
        }
        (ty @ ClassicType::F64, _) => Err(ConstTypeError::Unimplemented(ty.clone())),
        (ty @ ClassicType::Container(c), tm) => match (c, tm) {
            (Container::Tuple(row), ConstValue::Tuple(xs)) => {
                if row.len() != xs.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                for (ty, tm) in row.iter().zip(xs.iter()) {
                    typecheck_const(ty, tm)?
                }
                Ok(())
            }
            (Container::Sum(row), ConstValue::Sum { tag, variants, val }) => {
                if tag > &row.len() {
                    return Err(ConstTypeError::InvalidSumTag);
                }
                if **row != *variants {
                    return Err(ConstTypeError::TypeRowMismatch(
                        *row.clone(),
                        variants.clone(),
                    ));
                }
                let ty = variants.get(*tag).unwrap();
                typecheck_const(ty, val.as_ref())
            }
            _ => Err(ConstTypeError::Unimplemented(ty.clone())),
        },
        (ty @ ClassicType::Graph(_), _) => Err(ConstTypeError::Unimplemented(ty.clone())),
        (ty @ ClassicType::String, _) => Err(ConstTypeError::Unimplemented(ty.clone())),
        (ClassicType::Variable(_), _) => Err(ConstTypeError::ConstCantBeVar),
        (ClassicType::Opaque(ty), ConstValue::Opaque(_tm, ty2)) => {
            // The type we're checking against
            let ty_exp = ty.clone().classic_type();
            let ty_act = ty2.const_type();
            if ty_exp != ty_act {
                return Err(ConstTypeError::TypeMismatch(ty_exp, ty_act));
            }
            Ok(())
        }
        (ty, _) => Err(ConstTypeError::Failed(ty.clone())),
    }
}

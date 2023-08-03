//! Simple type checking for int constants - currently this is just the bits that are
//! shared between the old [TypeArg] and the new [ConstValue]/[HashableValue].
//!
//! [TypeArg]: crate::types::type_param::TypeArg
//! [ConstValue]: crate::ops::constant::ConstValue
//! [HashableValue]: crate::values::HashableValue
use lazy_static::lazy_static;

use std::collections::HashSet;

use thiserror::Error;

// For static typechecking
use crate::ops::constant::{HugrIntValueStore, HugrIntWidthStore, HUGR_MAX_INT_WIDTH};

/// An error in fitting an integer constant into its size
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ConstIntError {
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
}

lazy_static! {
    static ref VALID_WIDTHS: HashSet<HugrIntWidthStore> =
        HashSet::from_iter((0..8).map(|a| HugrIntWidthStore::pow(2, a)));
}

/// Per the spec, valid widths for integers are 2^n for all n in [0,7]
pub(crate) fn check_int_fits_in_width(
    value: HugrIntValueStore,
    width: HugrIntWidthStore,
) -> Result<(), ConstIntError> {
    if width > HUGR_MAX_INT_WIDTH {
        return Err(ConstIntError::IntWidthTooLarge(width));
    }

    if VALID_WIDTHS.contains(&width) {
        let max_value = if width == HUGR_MAX_INT_WIDTH {
            HugrIntValueStore::MAX
        } else {
            HugrIntValueStore::pow(2, width as u32) - 1
        };
        if value <= max_value {
            Ok(())
        } else {
            Err(ConstIntError::IntTooLarge(width, value))
        }
    } else {
        Err(ConstIntError::IntWidthInvalid(width))
    }
}

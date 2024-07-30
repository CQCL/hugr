//! Experiments for `Extension` definitions.
//!
//! These may be moved to other crates in the future, or dropped altogether.

use crate::extension::ExtensionRegistry;

pub mod arithmetic;
pub mod collections;
pub mod logic;
pub mod ptr;

/// Extension registry with all standard extensions and prelude.
pub fn std_reg() -> ExtensionRegistry {
    ExtensionRegistry::try_new([
        crate::extension::prelude::PRELUDE.to_owned(),
        arithmetic::int_ops::EXTENSION.to_owned(),
        arithmetic::int_types::EXTENSION.to_owned(),
        arithmetic::conversions::EXTENSION.to_owned(),
        arithmetic::float_ops::EXTENSION.to_owned(),
        arithmetic::float_types::EXTENSION.to_owned(),
        logic::EXTENSION.to_owned(),
        ptr::EXTENSION.to_owned(),
    ])
    .unwrap()
}

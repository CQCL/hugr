//! Experiments for `Extension` definitions.
//!
//! These may be moved to other crates in the future, or dropped altogether.

use crate::extension::ExtensionRegistry;
use std::sync::LazyLock;

pub mod arithmetic;
pub mod collections;
pub mod logic;
pub mod ptr;

/// Extension registry with all standard extensions and prelude.
#[must_use]
pub fn std_reg() -> ExtensionRegistry {
    let reg = ExtensionRegistry::new([
        crate::extension::prelude::PRELUDE.clone(),
        arithmetic::int_ops::EXTENSION.to_owned(),
        arithmetic::int_types::EXTENSION.to_owned(),
        arithmetic::conversions::EXTENSION.to_owned(),
        arithmetic::float_ops::EXTENSION.to_owned(),
        arithmetic::float_types::EXTENSION.to_owned(),
        collections::array::EXTENSION.to_owned(),
        collections::list::EXTENSION.to_owned(),
        collections::borrow_array::EXTENSION.to_owned(),
        collections::static_array::EXTENSION.to_owned(),
        collections::value_array::EXTENSION.to_owned(),
        logic::EXTENSION.to_owned(),
        ptr::EXTENSION.to_owned(),
    ]);
    reg.validate()
        .expect("Standard extension registry is valid");
    reg
}

/// Standard extension registry.
pub static STD_REG: LazyLock<ExtensionRegistry> = LazyLock::new(std_reg);

//! Compilation passes acting on the HUGR program representation.

pub mod const_fold;
pub mod force_order;
mod half_node;
pub mod lower;
pub mod merge_bbs;
pub mod nest_cfgs;
pub mod non_local;
pub mod validation;
pub use force_order::{force_order, force_order_by_key};
pub use lower::{lower_ops, replace_many_ops};
pub use non_local::{ensure_no_nonlocal_edges, nonlocal_edges};

#[cfg(test)]
pub(crate) mod test {

    use lazy_static::lazy_static;

    use hugr_core::extension::{ExtensionRegistry, PRELUDE};
    use hugr_core::std_extensions::arithmetic;
    use hugr_core::std_extensions::collections;
    use hugr_core::std_extensions::logic;

    lazy_static! {
        /// A registry containing various extensions for testing.
        pub(crate) static ref TEST_REG: ExtensionRegistry = ExtensionRegistry::try_new([
            PRELUDE.clone(),
            arithmetic::int_ops::EXTENSION.clone(),
            arithmetic::int_types::EXTENSION.clone(),
            arithmetic::float_types::EXTENSION.clone(),
            arithmetic::float_ops::EXTENSION.clone(),
            logic::EXTENSION.clone(),
            arithmetic::conversions::EXTENSION.to_owned(),
            collections::EXTENSION.to_owned(),
        ])
        .unwrap();
    }
}

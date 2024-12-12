//! Compilation passes acting on the HUGR program representation.

pub mod const_fold;
pub mod dataflow;
pub mod force_order;
mod half_node;
pub mod lower;
pub mod merge_bbs;
mod monomorphize;
pub use monomorphize::{monomorphize, remove_polyfuncs};
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
        pub(crate) static ref TEST_REG: ExtensionRegistry = ExtensionRegistry::new([
            PRELUDE.to_owned(),
            arithmetic::int_ops::EXTENSION.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
            arithmetic::float_types::EXTENSION.to_owned(),
            arithmetic::float_ops::EXTENSION.to_owned(),
            logic::EXTENSION.to_owned(),
            arithmetic::conversions::EXTENSION.to_owned(),
            collections::list::EXTENSION.to_owned(),
        ]);
    }
}

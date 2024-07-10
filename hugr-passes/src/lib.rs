//! Compilation passes acting on the HUGR program representation.

pub mod const_fold;
pub mod force_order;
mod half_node;
pub mod merge_bbs;
pub mod nest_cfgs;
pub mod validation;

pub use force_order::{force_order, force_order_by_key};

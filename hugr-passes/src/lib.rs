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

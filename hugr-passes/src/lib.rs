//! Compilation passes acting on the HUGR program representation.

pub mod call_graph;
pub mod composable;
pub use composable::ComposablePass;
pub mod const_fold;
pub mod dataflow;
pub mod dead_code;
pub use dead_code::DeadCodeElimPass;
mod dead_funcs;
pub use dead_funcs::{remove_dead_funcs, RemoveDeadFuncsError, RemoveDeadFuncsPass};
pub mod force_order;
mod half_node;
pub mod lower;
pub mod merge_bbs;
mod monomorphize;

// TODO: Deprecated re-export. Remove on a breaking release.
#[deprecated(
    since = "0.14.1",
    note = "Use `hugr_passes::RemoveDeadFuncsPass` instead."
)]
#[allow(deprecated)]
pub use monomorphize::remove_polyfuncs;
pub use monomorphize::{monomorphize, MonomorphizePass};
pub mod nest_cfgs;
pub mod non_local;
pub use force_order::{force_order, force_order_by_key};
pub use lower::{lower_ops, replace_many_ops};
pub use non_local::{ensure_no_nonlocal_edges, nonlocal_edges};

//! Compilation passes acting on the HUGR program representation.

pub mod call_graph;
pub mod composable;
pub use composable::ComposablePass;
pub mod const_fold;
pub mod dataflow;
pub mod dead_code;
pub use dead_code::DeadCodeElimPass;
mod dead_funcs;
pub use dead_funcs::{RemoveDeadFuncsError, RemoveDeadFuncsPass, remove_dead_funcs};
pub mod force_order;
mod half_node;
pub mod linearize_array;
pub use linearize_array::LinearizeArrayPass;
pub mod lower;
pub mod merge_bbs;
mod monomorphize;
pub mod untuple;

pub use monomorphize::{MonomorphizePass, mangle_name, monomorphize};
pub mod replace_types;
pub use replace_types::ReplaceTypes;
pub mod nest_cfgs;
pub mod non_local;
pub use force_order::{force_order, force_order_by_key};
pub use lower::{lower_ops, replace_many_ops};
pub use non_local::{ensure_no_nonlocal_edges, nonlocal_edges};
pub use untuple::UntuplePass;

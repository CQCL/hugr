//! Compilation passes acting on the HUGR program representation.

pub mod call_graph;
pub mod composable;
pub use composable::ComposablePass;
pub mod const_fold;
pub mod dataflow;
pub mod dead_code;
pub use dead_code::DeadCodeElimPass;
mod dead_funcs;
#[deprecated(
    note = "Does not account for visibility; use remove_dead_funcs2 or manually configure RemoveDeadFuncsPass"
)]
#[allow(deprecated)] // When original removed, rename remove_dead_funcs2=>remove_dead_funcs
pub use dead_funcs::remove_dead_funcs;
pub use dead_funcs::{RemoveDeadFuncsError, RemoveDeadFuncsPass, remove_dead_funcs2};
pub mod force_order;
mod half_node;
pub mod linearize_array;
use hugr_core::HugrView;
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
/// A policy for selecting [FuncDefn] and [FuncDecl]s using their [Visibility],
/// e.g. (typically) to use as starting points for analysis
///
/// [FuncDefn]: hugr_core::ops::FuncDefn
/// [FuncDecl]: hugr_core::ops::FuncDecl
/// [Visibility]: hugr_core::Visibility
pub enum VisPolicy {
    /// All [Public] functions should be used
    ///
    /// [Public]: hugr_core::Visibility::Public
    AllPublic,
    /// Do not select any functions
    None,
    /// Use the [Public] functions if the Hugr's [entrypoint] is the [module_root],
    /// otherwise do not use any.
    ///
    /// [Public]: hugr_core::Visibility::Public
    /// [entrypoint]: hugr_core::HugrView::entrypoint
    /// [module_root]: hugr_core::HugrView::module_root
    #[default]
    PublicIfModuleEntrypoint,
}

impl VisPolicy {
    /// Returns whether to include the public functions of a particular Hugr
    fn for_hugr(&self, h: &impl HugrView) -> bool {
        matches!(
            (self, h.entrypoint() == h.module_root()),
            (Self::AllPublic, _) | (Self::PublicIfModuleEntrypoint, true)
        )
    }
}

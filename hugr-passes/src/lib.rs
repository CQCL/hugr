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
pub mod inline_dfgs;
pub mod inline_funcs;
pub use inline_funcs::inline_acyclic;
pub mod linearize_array;
pub use linearize_array::LinearizeArrayPass;
pub mod lower;
mod monomorphize;
pub mod normalize_cfgs;
pub mod untuple;

/// Merge basic blocks. Subset of [normalize_cfgs], use the latter.
#[deprecated(note = "Use normalize_cfgs", since = "0.23.0")]
pub mod merge_bbs {
    use hugr_core::hugr::{hugrmut::HugrMut, views::RootCheckable};
    use hugr_core::ops::handle::CfgID;

    /// Merge any basic blocks that are direct children of the specified CFG
    /// i.e. where a basic block B has a single successor B' whose only predecessor
    /// is B, B and B' can be combined.
    ///
    /// # Panics
    ///
    /// If the `entrypoint` of `cfg` is not an [OpType::CFG]
    ///
    /// [OpType::CFG]: hugr_core::ops::OpType::CFG
    #[deprecated(note = "Use version in normalize_cfgs", since = "0.23.0")]
    pub fn merge_basic_blocks<'h, H: 'h + HugrMut>(
        cfg: impl RootCheckable<&'h mut H, CfgID<H::Node>>,
    ) {
        let checked = cfg.try_into_checked().expect("Hugr must be a CFG region");
        super::normalize_cfgs::merge_basic_blocks(checked.into_hugr()).unwrap();
    }
}

pub use monomorphize::{MonomorphizePass, mangle_name, monomorphize};
pub mod replace_types;
pub use replace_types::ReplaceTypes;
pub mod nest_cfgs;
pub mod non_local;
pub use force_order::{force_order, force_order_by_key};
pub use lower::{lower_ops, replace_many_ops};
pub use non_local::{ensure_no_nonlocal_edges, nonlocal_edges};
pub use untuple::UntuplePass;

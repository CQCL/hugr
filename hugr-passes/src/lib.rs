//! Compilation passes acting on the HUGR program representation.

pub mod call_graph;
pub mod const_fold;
pub mod dataflow;
mod dead_code;
mod dead_funcs;
pub use dead_funcs::{remove_dead_funcs, RemoveDeadFuncsError, RemoveDeadFuncsPass};
pub mod force_order;
mod half_node;
pub mod lower;
pub mod merge_bbs;
mod monomorphize;
use hugr_core::{HugrView, Node};
// TODO: Deprecated re-export. Remove on a breaking release.
#[deprecated(
    since = "0.14.1",
    note = "Use `hugr_passes::RemoveDeadFuncsPass` instead."
)]
#[allow(deprecated)]
pub use monomorphize::remove_polyfuncs;
// TODO: Deprecated re-export. Remove on a breaking release.
#[deprecated(
    since = "0.14.1",
    note = "Use `hugr_passes::MonomorphizePass` instead."
)]
#[allow(deprecated)]
pub use monomorphize::monomorphize;
pub use monomorphize::{MonomorphizeError, MonomorphizePass};
pub mod nest_cfgs;
pub mod non_local;
pub mod validation;
pub use force_order::{force_order, force_order_by_key};
pub use lower::{lower_ops, replace_many_ops};
pub use non_local::{ensure_no_nonlocal_edges, nonlocal_edges};

fn find_main(h: &impl HugrView) -> Option<Node> {
    let root = h.root();
    if !h.get_optype(root).is_module() {
        return None;
    }
    h.children(root).find(|n| {
        h.get_optype(*n)
            .as_func_defn()
            .is_some_and(|f| f.name == "main")
    })
}

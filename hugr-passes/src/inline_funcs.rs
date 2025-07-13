/// Pass to inline calls to selected functions in a Hugr.
use std::collections::{HashSet, VecDeque};

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::patch::inline_call::{InlineCall, InlineCallError};
use petgraph::algo::tarjan_scc;

use crate::call_graph::{CallGraph, CallGraphNode};

/// Error raised by [inline_acyclic]
#[derive(Clone, Debug, thiserror::Error)]
pub enum InlineAllError<N> {
    /// Raised to indicate a request to inline calls to a function that is part of an SCC
    /// in the call graph.
    #[error("Cannot inline calls to {0} as in a cycle {1:?}")]
    FunctionOnCycle(N, Vec<N>),
    /// An error inlining a call.
    #[error(transparent)]
    InlineCallError(#[from] InlineCallError),
}

/// Inline all [Call]s to the specified `target_funcs` subject to a filter function
/// that is given the [Call] node and the target [FuncDefn] node. (Note the [Call]
/// may be created as a result of inlining and so may not have existed in the input
/// Hugr).
///
/// # Errors
///
/// [InlineAllError::FunctionOnCycle] if any element of `target_funcs` is in a cycle
/// in the call graph
///
/// [Call]: hugr_core::ops::Call
/// [FuncDefn]: hugr_core::ops::FuncDefn
pub fn inline_acyclic<H: HugrMut>(
    h: &mut H,
    cg: CallGraph<H::Node>,
    target_funcs: HashSet<H::Node>, // If empty, inline ALL
    filt_func: impl Fn(H::Node, H::Node) -> bool,
) -> Result<(), InlineAllError<H::Node>> {
    let g = cg.graph();
    let sccs = tarjan_scc(g)
        .into_iter()
        .map(|ns| {
            ns.into_iter()
                .map(|n| {
                    let CallGraphNode::FuncDefn(fd) = g.node_weight(n).unwrap() else {
                        panic!("Expected only FuncDefns in sccs")
                    };
                    *fd
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let all_sccs: HashSet<_> = sccs.iter().cloned().flatten().collect();
    let target_funcs = if target_funcs.is_empty() {
        h.children(h.module_root())
            .filter(|n| !all_sccs.contains(n))
            .collect()
    } else if let Some(n) = target_funcs.intersection(&all_sccs).next() {
        let scc = sccs.iter().find(|ns| ns.as_slice().contains(n)).unwrap();
        return Err(InlineAllError::FunctionOnCycle(*n, scc.clone()));
    } else {
        target_funcs
    };
    let mut q = VecDeque::from([h.entrypoint()]);
    while let Some(n) = q.pop_front() {
        if h.get_optype(n).is_call() {
            if let Some(t) = h.static_source(n) {
                if target_funcs.contains(&t) && filt_func(n, t) {
                    h.apply_patch(InlineCall::new(n)).unwrap();
                }
            }
        }
        // If that was a call, `n` is now a DFG containing the function body, so explore inside
        q.extend(h.children(n));
    }
    Ok(())
}

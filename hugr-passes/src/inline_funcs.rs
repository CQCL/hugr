use std::collections::HashSet;

use hugr_core::hugr::{hugrmut::HugrMut, patch::inline_call::InlineCall};
use petgraph::{Direction, algo::tarjan_scc, visit::EdgeRef};

use crate::call_graph::{CallGraph, CallGraphEdge, CallGraphNode};

/// Inline all [Call]s except those to functions in cycles on the call graph,
/// and subject to a filter function that is given the [Call] node and the target
/// [FuncDefn] node.
///
/// [Call]: hugr_core::ops::Call
/// [FuncDefn]: hugr_core::ops::FuncDefn
pub fn inline_acyclic<H: HugrMut>(
    h: &mut H,
    cg: CallGraph<H::Node>,
    filt_func: impl Fn(H::Node, H::Node) -> bool,
) {
    let g = cg.graph();
    let sccs = tarjan_scc(cg.graph());
    let all_sccs: HashSet<_> = sccs.iter().flatten().collect::<HashSet<_>>();
    if h.entrypoint() == h.module_root() {
        // Can modify everywhere. Traverse post-order (callee before caller).
        let mut seen = HashSet::new();
        let mut stack: Vec<_> = h
            .children(h.module_root())
            .map(|n| (cg.node_index(n).unwrap(), false))
            .collect();
        while let Some((func, children_done)) = stack.pop() {
            let edges = cg
                .graph()
                .edges_directed(func, Direction::Outgoing)
                .filter_map(|e| {
                    assert_eq!(e.source(), func);
                    let d = e.target();
                    (!all_sccs.contains(&d)).then_some((g.edge_weight(e.id()).unwrap(), d))
                });
            if !children_done {
                stack.push((func, true));
                stack.extend(edges.map(|(_, tgt)| (tgt, false)));
                // We know no tgt can push `func` because we have already filtered out cycles
                continue
            }
            // Callees have all been processed.
            if !seen.insert(func) { continue } // Caller has too!
            for (edge, callee) in edges {
                let tgt_func: H::Node = match g.node_weight(callee).unwrap() {
                    CallGraphNode::FuncDecl(_) => continue,
                    CallGraphNode::FuncDefn(n) => *n,
                    CallGraphNode::NonFuncRoot => unreachable!("Call to non-func"),
                };
                if let CallGraphEdge::Call(n) = edge {
                    // If we've processed `func` already, the calls will have become DFGs
                    assert_eq!(h.static_source(*n), Some(tgt_func));
                    if filt_func(*n, tgt_func) {
                        h.apply_patch(InlineCall::new(tgt_func)).unwrap();
                    }
                }
            }
            
        }
    } else {
        todo!() // Can only modify inside entrypoint. But, can keep inlining.
    }
}

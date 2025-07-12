use std::collections::{HashMap, HashSet};

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
        enum Status {
            PushChildren,
            Todo,
            Done,
        }
        let mut status: HashMap<_, _> = g
            .node_indices()
            .map(|n| (n, Status::PushChildren))
            .collect();
        let mut stack: Vec<_> = h
            .children(h.module_root())
            .map(|n| cg.node_index(n).unwrap())
            .collect();
        while let Some(func) = stack.pop() {
            let edges = cg
                .graph()
                .edges_directed(func, Direction::Outgoing)
                .filter_map(|e| {
                    assert_eq!(e.source(), func);
                    let d = e.target();
                    (!all_sccs.contains(&d)).then_some((g.edge_weight(e.id()).unwrap(), d))
                });
            match status.get(&func).unwrap() {
                Status::Done => (),
                Status::PushChildren => {
                    stack.push(func);
                    status.insert(func, Status::Todo);
                    stack.extend(edges.map(|(_, tgt)| tgt));
                }
                Status::Todo => {
                    // Callees have all been inlined.
                    for (edge, callee) in edges {
                        let tgt_func: H::Node = match g.node_weight(callee).unwrap() {
                            CallGraphNode::FuncDecl(_) => continue,
                            CallGraphNode::FuncDefn(n) => *n,
                            CallGraphNode::NonFuncRoot => unreachable!("Call to non-func"),
                        };
                        if let CallGraphEdge::Call(n) = edge {
                            if filt_func(*n, tgt_func) {
                                h.apply_patch(InlineCall::new(tgt_func)).unwrap();
                            }
                        }
                    }
                    status.insert(func, Status::Done);
                }
            }
        }
    } else {
        todo!() // Can only modify inside entrypoint. But, can keep inlining.
    }
}

//! Contains a pass to inline calls to selected functions in a Hugr.
use std::collections::{HashSet, VecDeque};

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::patch::inline_call::InlineCall;
use itertools::Itertools;
use petgraph::algo::tarjan_scc;

use crate::call_graph::{CallGraph, CallGraphNode};

/// Error raised by [inline_acyclic]
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum InlineFuncsError {}

/// Inline (a subset of) [Call]s whose target [FuncDefn]s are not in cycles of the call
/// graph.
///
/// The function `call_predicate` is passed each such [Call] node and can return
/// `false` to prevent that Call from being inlined. (Note the [Call] may be created as
/// a result of previous inlinings so may not have existed in the original Hugr).
///
/// [Call]: hugr_core::ops::Call
/// [FuncDefn]: hugr_core::ops::FuncDefn
pub fn inline_acyclic<H: HugrMut>(
    h: &mut H,
    call_predicate: impl Fn(&H, H::Node) -> bool,
) -> Result<(), InlineFuncsError> {
    let cg = CallGraph::new(&*h);
    let g = cg.graph();
    let all_funcs_in_cycles = tarjan_scc(g)
        .into_iter()
        .flat_map(|mut ns| {
            if let Ok(n) = ns.iter().exactly_one() {
                if g.edges_connecting(*n, *n).next().is_none() {
                    ns.clear(); // Single-node SCC has no self edge, so discard
                }
            }
            ns.into_iter().map(|n| {
                let CallGraphNode::FuncDefn(fd) = g.node_weight(n).unwrap() else {
                    panic!("Expected only FuncDefns in sccs")
                };
                *fd
            })
        })
        .collect::<HashSet<_>>();
    let target_funcs: HashSet<H::Node> = h
        .children(h.module_root())
        .filter(|n| h.get_optype(*n).is_func_defn() && !all_funcs_in_cycles.contains(n))
        .collect();
    let mut q = VecDeque::from([h.entrypoint()]);
    while let Some(n) = q.pop_front() {
        if h.get_optype(n).is_call() {
            if let Some(t) = h.static_source(n) {
                if target_funcs.contains(&t) && call_predicate(h, n) {
                    // We've already checked all error conditions
                    h.apply_patch(InlineCall::new(n)).unwrap();
                }
            }
        }
        // Traverse children - including any resulting from turning Call into DFG
        q.extend(h.children(n));
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use hugr_core::core::HugrNode;
    use hugr_core::ops::OpType;
    use itertools::Itertools;
    use petgraph::visit::EdgeRef;

    use hugr_core::HugrView;
    use hugr_core::builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use hugr_core::{Hugr, extension::prelude::qb_t, types::Signature};
    use rstest::rstest;

    use crate::call_graph::{CallGraph, CallGraphNode};
    use crate::inline_funcs::inline_acyclic;

    ///          /->-\
    /// main -> f     g -> b -> c
    ///        / \-<-/
    ///       /
    ///       \-> a -> x
    fn make_test_hugr() -> Hugr {
        let sig = || Signature::new_endo(qb_t());
        let mut mb = ModuleBuilder::new();
        let x = mb.declare("x", sig().into()).unwrap();
        let a = {
            let mut fb = mb.define_function("a", sig()).unwrap();
            let ins = fb.input_wires();
            let res = fb.call(&x, &[], ins).unwrap();
            fb.finish_with_outputs(res.outputs()).unwrap()
        };
        let c = {
            let fb = mb.define_function("c", sig()).unwrap();
            let ins = fb.input_wires();
            fb.finish_with_outputs(ins).unwrap()
        };
        let b = {
            let mut fb = mb.define_function("b", sig()).unwrap();
            let ins = fb.input_wires();
            let res = fb.call(c.handle(), &[], ins).unwrap().outputs();
            fb.finish_with_outputs(res).unwrap()
        };
        let f = mb.declare("f", sig().into()).unwrap();
        let g = {
            let mut fb = mb.define_function("g", sig()).unwrap();
            let ins = fb.input_wires();
            let c1 = fb.call(&f, &[], ins).unwrap();
            let c2 = fb.call(b.handle(), &[], c1.outputs()).unwrap();
            fb.finish_with_outputs(c2.outputs()).unwrap()
        };
        let _f = {
            let mut fb = mb.define_declaration(&f).unwrap();
            let ins = fb.input_wires();
            let c1 = fb.call(g.handle(), &[], ins).unwrap();
            let c2 = fb.call(a.handle(), &[], c1.outputs()).unwrap();
            fb.finish_with_outputs(c2.outputs()).unwrap()
        };
        mb.finish_hugr().unwrap()
    }

    fn find_func<H: HugrView>(h: &H, name: &str) -> H::Node {
        h.children(h.module_root())
            .find(|n| {
                h.get_optype(*n)
                    .as_func_defn()
                    .is_some_and(|fd| fd.func_name() == name)
            })
            .unwrap()
    }

    #[rstest]
    #[case(["a", "b", "c"], ["a", "b", "c"], [vec!["g", "x"], vec!["f"], vec!["x"], vec![], vec![]])]
    #[case(["a", "b"], ["a", "b"], [vec!["g", "x"], vec!["f", "c"], vec!["x"], vec!["c"], vec![]])]
    #[case(["c"], ["c"], [vec!["g", "a"], vec!("f", "b"), vec!["x"], vec![], vec![]])]
    fn test_inline(
        #[case] req: impl IntoIterator<Item = &'static str>,
        #[case] check_not_called: impl IntoIterator<Item = &'static str>,
        #[case] calls_fgabc: [Vec<&'static str>; 5],
    ) {
        let mut h = make_test_hugr();
        let target_funcs = req
            .into_iter()
            .map(|name| find_func(&h, name))
            .collect::<HashSet<_>>();
        inline_acyclic(&mut h, |h, call| {
            let tgt = h.static_source(call).unwrap();
            // Check the callback is never asked about an impossible inlining
            assert!(["a", "b", "c"].contains(&func_name(h, tgt).as_str()));
            target_funcs.contains(&tgt)
        })
        .unwrap();
        let cg = CallGraph::new(&h);
        for fname in check_not_called {
            let fnode = find_func(&h, fname);
            let fnode = cg.node_index(fnode).unwrap();
            assert_eq!(
                None,
                cg.graph()
                    .edges_directed(fnode, petgraph::Direction::Incoming)
                    .next()
            );
        }
        for (fname, tgts) in ["f", "g", "a", "b", "c"].into_iter().zip_eq(calls_fgabc) {
            let fnode = find_func(&h, fname);
            assert_eq!(
                outgoing_calls(&cg, fnode)
                    .into_iter()
                    .map(|n| func_name(&h, n).as_str())
                    .collect::<HashSet<_>>(),
                HashSet::from_iter(tgts),
                "Calls from {fname}"
            );
        }
    }

    fn outgoing_calls<N: HugrNode>(cg: &CallGraph<N>, src: N) -> Vec<N> {
        let src = cg.node_index(src).unwrap();
        cg.graph()
            .edges_directed(src, petgraph::Direction::Outgoing)
            .map(|e| func_node(cg.graph().node_weight(e.target()).unwrap()))
            .collect()
    }

    #[test]
    fn test_filter_caller() {
        let mut h = make_test_hugr();
        let [g, b, c] = ["g", "b", "c"].map(|n| find_func(&h, n));
        // Inline calls contained within `g`
        inline_acyclic(&mut h, |h, mut call| {
            loop {
                if call == g {
                    return true;
                };
                let Some(parent) = h.get_parent(call) else {
                    return false;
                };
                call = parent;
            }
        })
        .unwrap();
        let cg = CallGraph::new(&h);
        // b and then c should have been inlined into g, leaving only cyclic call to f
        assert_eq!(outgoing_calls(&cg, g), [find_func(&h, "f")]);
        // But c should not have been inlined into b:
        assert_eq!(outgoing_calls(&cg, b), [c]);
    }

    fn func_node<N: Copy>(cgn: &CallGraphNode<N>) -> N {
        match cgn {
            CallGraphNode::FuncDecl(n) | CallGraphNode::FuncDefn(n) => *n,
            CallGraphNode::NonFuncRoot => panic!(),
        }
    }

    fn func_name<H: HugrView>(h: &H, n: H::Node) -> &String {
        match h.get_optype(n) {
            OpType::FuncDecl(fd) => fd.func_name(),
            OpType::FuncDefn(fd) => fd.func_name(),
            _ => panic!(),
        }
    }
}

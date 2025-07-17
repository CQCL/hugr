//! Contains a pass to inline calls to selected functions in a Hugr.
use std::collections::{HashSet, VecDeque};

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::patch::inline_call::{InlineCall, InlineCallError};
use hugr_core::ops::OpType;
use itertools::Itertools;
use petgraph::algo::tarjan_scc;

use crate::call_graph::{CallGraph, CallGraphNode};

/// Error raised by [inline_acyclic]
#[derive(Clone, Debug, thiserror::Error)]
pub enum InlineAllError<N> {
    /// Raised to indicate a request to inline calls to a node that is not a FuncDefn
    #[error("Can only inline calls to FuncDefns; {0} is a {1:?}")]
    NotAFuncDefn(N, OpType),
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
/// If `target_funcs` is empty, rather than inlining no functions, inline all
/// possible functions (i.e. all that are part of a cycle).
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
    target_funcs: HashSet<H::Node>, // If empty, inline ALL
    filt_func: impl Fn(&H, H::Node, H::Node) -> bool,
) -> Result<(), InlineAllError<H::Node>> {
    let cg = CallGraph::new(&*h);
    let g = cg.graph();
    let sccs = tarjan_scc(g)
        .into_iter()
        .filter_map(|ns| {
            if let Ok(n) = ns.iter().exactly_one() {
                if g.edges_connecting(*n, *n).next().is_none() {
                    return None; // A 1-node SCC might be a cycle, but this is just a node.
                }
            }
            Some(
                ns.into_iter()
                    .map(|n| {
                        let CallGraphNode::FuncDefn(fd) = g.node_weight(n).unwrap() else {
                            panic!("Expected only FuncDefns in sccs")
                        };
                        *fd
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let all_sccs: HashSet<_> = sccs.iter().cloned().flatten().collect();
    let target_funcs = if target_funcs.is_empty() {
        h.children(h.module_root())
            .filter(|n| h.get_optype(*n).is_func_defn() && !all_sccs.contains(n))
            .collect()
    } else {
        for &tgt in &target_funcs {
            let op = h.get_optype(tgt);
            if !op.is_func_defn() {
                return Err(InlineAllError::NotAFuncDefn(tgt, op.clone()));
            }
            if all_sccs.contains(&tgt) {
                let scc = sccs.iter().find(|ns| ns.as_slice().contains(&tgt)).unwrap();
                return Err(InlineAllError::FunctionOnCycle(tgt, scc.clone()));
            }
        }
        target_funcs
    };
    let mut q = VecDeque::from([h.entrypoint()]);
    while let Some(n) = q.pop_front() {
        if h.get_optype(n).is_call() {
            if let Some(t) = h.static_source(n) {
                if target_funcs.contains(&t) && filt_func(&h, n, t) {
                    h.apply_patch(InlineCall::new(n)).unwrap();
                }
            }
        }
        // If that was a call, `n` is now a DFG containing the function body, so explore inside
        q.extend(h.children(n));
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use hugr_core::ops::OpType;
    use petgraph::visit::EdgeRef;

    use hugr_core::HugrView;
    use hugr_core::builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use hugr_core::{Hugr, extension::prelude::qb_t, types::Signature};
    use rstest::rstest;

    use crate::call_graph::{CallGraph, CallGraphNode};
    use crate::inline_funcs::{InlineAllError, inline_acyclic};

    ///          /->-\
    /// main -> f     g -> a -> b -> x
    ///        / \-<-/
    ///       /
    ///       \-> c -> d
    fn make_test_hugr() -> Hugr {
        let sig = || Signature::new_endo(qb_t());
        let mut mb = ModuleBuilder::new();
        let x = mb.declare("x", sig().into()).unwrap();
        let b = {
            let mut fb = mb.define_function("b", sig()).unwrap();
            let ins = fb.input_wires();
            let res = fb.call(&x, &[], ins).unwrap();
            fb.finish_with_outputs(res.outputs()).unwrap()
        };
        let a = {
            let mut fb = mb.define_function("a", sig()).unwrap();
            let ins = fb.input_wires();
            let res = fb.call(b.handle(), &[], ins).unwrap().outputs();
            fb.finish_with_outputs(res).unwrap()
        };
        let d = {
            let fb = mb.define_function("d", sig()).unwrap();
            let ins = fb.input_wires();
            fb.finish_with_outputs(ins).unwrap()
        };
        let c = {
            let mut fb = mb.define_function("c", sig()).unwrap();
            let ins = fb.input_wires();
            let res = fb.call(d.handle(), &[], ins).unwrap().outputs();
            fb.finish_with_outputs(res).unwrap()
        };
        let f = mb.declare("f", sig().into()).unwrap();
        let g = {
            let mut fb = mb.define_function("g", sig()).unwrap();
            let ins = fb.input_wires();
            let c1 = fb.call(&f, &[], ins).unwrap();
            let c2 = fb.call(a.handle(), &[], c1.outputs()).unwrap();
            fb.finish_with_outputs(c2.outputs()).unwrap()
        };
        let _f = {
            let mut fb = mb.define_declaration(&f).unwrap();
            let ins = fb.input_wires();
            let c1 = fb.call(g.handle(), &[], ins).unwrap();
            let c2 = fb.call(c.handle(), &[], c1.outputs()).unwrap();
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
    #[case(["f"], "f")]
    #[case(["g", "a", "b", "d"], "g")]
    fn test_cycles(
        #[case] funcs: impl IntoIterator<Item = &'static str>,
        #[case] exp_err: &'static str,
    ) {
        let h = make_test_hugr();
        let target_funcs = funcs.into_iter().map(|name| find_func(&h, name)).collect();
        let mut h2 = h.clone();
        let r = inline_acyclic(&mut h2, target_funcs, |_, _, _| panic!());
        assert_eq!(h, h2); // Did nothing
        let Err(InlineAllError::FunctionOnCycle(tgt, scc)) = r else {
            panic!()
        };
        assert_eq!(
            h.get_optype(tgt).as_func_defn().unwrap().func_name(),
            exp_err
        );
        let [f, g] = ["f", "g"].map(|n| find_func(&h, n));
        assert_eq!(HashSet::from([f, g]), HashSet::from_iter(scc));
    }

    #[rstest]
    #[case([], ["a", "b", "c", "d"], [("f", vec!["g"]), ("g", vec!["f", "x"]), ("c", vec![]), ("a", vec!["x"])])]
    #[case(["a", "c"], ["a", "c"], [("f", vec!["g", "d"]), ("g", vec!["f", "b"]), ("c", vec!["d"]), ("a", vec!["b"])])]
    fn test_inline(
        #[case] req: impl IntoIterator<Item = &'static str>,
        #[case] check_not_called: impl IntoIterator<Item = &'static str>,
        #[case] check_calls: impl IntoIterator<Item = (&'static str, Vec<&'static str>)>,
    ) {
        let mut h = make_test_hugr();
        let target_funcs = req.into_iter().map(|name| find_func(&h, name)).collect();
        inline_acyclic(&mut h, target_funcs, |_, _, _| true).unwrap();
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
        for (fname, tgts) in check_calls {
            let fnode = find_func(&h, fname);
            let fnode = cg.node_index(fnode).unwrap();
            assert_eq!(
                cg.graph()
                    .edges_directed(fnode, petgraph::Direction::Outgoing)
                    .map(|e| { get_name(&h, cg.graph().node_weight(e.target()).unwrap()).as_str() })
                    .collect::<HashSet<_>>(),
                HashSet::from_iter(tgts),
                "Calls from {fname}"
            );
        }
    }

    fn get_name<'a, H: HugrView>(h: &'a H, cgn: &'a CallGraphNode<H::Node>) -> &'a String {
        let n = match cgn {
            CallGraphNode::FuncDecl(n) | CallGraphNode::FuncDefn(n) => n,
            CallGraphNode::NonFuncRoot => panic!(),
        };
        match h.get_optype(*n) {
            OpType::FuncDecl(fd) => fd.func_name(),
            OpType::FuncDefn(fd) => fd.func_name(),
            _ => panic!(),
        }
    }
}

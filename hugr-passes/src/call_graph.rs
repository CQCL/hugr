//! Data structure for call graphs of a Hugr
use std::collections::HashMap;

use hugr_core::{HugrView, Node, core::HugrNode, ops::OpType};
use petgraph::Graph;

/// Weight for an edge in a [`CallGraph`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallGraphEdge<N = Node> {
    /// Edge corresponds to a [Call](OpType::Call) node (specified) in the Hugr
    Call(N),
    /// Edge corresponds to a [`LoadFunction`](OpType::LoadFunction) node (specified) in the Hugr
    LoadFunction(N),
}

/// Weight for a petgraph-node in a [`CallGraph`]
pub enum CallGraphNode<N = Node> {
    /// petgraph-node corresponds to a [`FuncDecl`](OpType::FuncDecl) node (specified) in the Hugr
    FuncDecl(N),
    /// petgraph-node corresponds to a [`FuncDefn`](OpType::FuncDefn) node (specified) in the Hugr
    FuncDefn(N),
    /// petgraph-node corresponds to the root node of the hugr, that is not
    /// a [`FuncDefn`](OpType::FuncDefn). Note that it will not be a [Module](OpType::Module)
    /// either, as such a node could not have outgoing edges, so is not represented in the petgraph.
    NonFuncRoot,
}

/// Details the [`Call`]s and [`LoadFunction`]s in a Hugr.
///
/// Each node in the `CallGraph` corresponds to a [`FuncDefn`] or [`FuncDecl`] in the Hugr;
/// each edge corresponds to a [`Call`]/[`LoadFunction`] of the edge's target, contained in
/// the edge's source.
///
/// For Hugrs whose entrypoint is neither a [Module](OpType::Module) nor a [`FuncDefn`], the
/// call graph will have an additional [`CallGraphNode::NonFuncRoot`] corresponding to the Hugr's
/// entrypoint, with no incoming edges.
///
/// [`Call`]: OpType::Call
/// [`FuncDecl`]: OpType::FuncDecl
/// [`FuncDefn`]: OpType::FuncDefn
/// [`LoadFunction`]: OpType::LoadFunction
pub struct CallGraph<N = Node> {
    g: Graph<CallGraphNode<N>, CallGraphEdge<N>>,
    node_to_g: HashMap<N, petgraph::graph::NodeIndex<u32>>,
}

impl<N: HugrNode> CallGraph<N> {
    /// Makes a new `CallGraph` for a Hugr.
    pub fn new(hugr: &impl HugrView<Node = N>) -> Self {
        let mut g = Graph::default();
        let mut node_to_g = hugr
            .children(hugr.module_root())
            .filter_map(|n| {
                let weight = match hugr.get_optype(n) {
                    OpType::FuncDecl(_) => CallGraphNode::FuncDecl(n),
                    OpType::FuncDefn(_) => CallGraphNode::FuncDefn(n),
                    _ => return None,
                };
                Some((n, g.add_node(weight)))
            })
            .collect::<HashMap<_, _>>();
        if !hugr.entrypoint_optype().is_module() && !node_to_g.contains_key(&hugr.entrypoint()) {
            node_to_g.insert(hugr.entrypoint(), g.add_node(CallGraphNode::NonFuncRoot));
        }
        for (func, cg_node) in &node_to_g {
            traverse(hugr, *cg_node, *func, &mut g, &node_to_g);
        }
        fn traverse<N: HugrNode>(
            h: &impl HugrView<Node = N>,
            enclosing_func: petgraph::graph::NodeIndex<u32>,
            node: N, // Nonstrict-descendant of `enclosing_func``
            g: &mut Graph<CallGraphNode<N>, CallGraphEdge<N>>,
            node_to_g: &HashMap<N, petgraph::graph::NodeIndex<u32>>,
        ) {
            for ch in h.children(node) {
                if h.get_optype(ch).is_func_defn() {
                    continue;
                }
                traverse(h, enclosing_func, ch, g, node_to_g);
                let weight = match h.get_optype(ch) {
                    OpType::Call(_) => CallGraphEdge::Call(ch),
                    OpType::LoadFunction(_) => CallGraphEdge::LoadFunction(ch),
                    _ => continue,
                };
                if let Some(target) = h.static_source(ch) {
                    g.add_edge(enclosing_func, *node_to_g.get(&target).unwrap(), weight);
                }
            }
        }
        CallGraph { g, node_to_g }
    }

    /// Allows access to the petgraph
    #[must_use]
    pub fn graph(&self) -> &Graph<CallGraphNode<N>, CallGraphEdge<N>> {
        &self.g
    }

    /// Convert a Hugr [Node] into a petgraph node index.
    /// Result will be `None` if `n` is not a [`FuncDefn`](OpType::FuncDefn),
    /// [`FuncDecl`](OpType::FuncDecl) or the [HugrView::entrypoint].
    pub fn node_index(&self, n: N) -> Option<petgraph::graph::NodeIndex<u32>> {
        self.node_to_g.get(&n).copied()
    }
}

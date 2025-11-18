//! Data structure summarizing static nodes of a Hugr and their uses
use std::collections::HashMap;

use crate::{HugrView, Node, core::HugrNode, ops::OpType};
use petgraph::{Graph, visit::EdgeRef};

/// Weight for an edge in a [`StaticGraph`]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StaticEdge<N = Node> {
    /// Edge corresponds to a [Call](OpType::Call) node (specified) in the Hugr
    Call(N),
    /// Edge corresponds to a [`LoadFunction`](OpType::LoadFunction) node (specified) in the Hugr
    LoadFunction(N),
    /// Edge corresponds to a [LoadConstant](OpType::LoadConstant) node (specified) in the Hugr
    LoadConstant(N),
}

/// Weight for a petgraph-node in a [`StaticGraph`]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StaticNode<N = Node> {
    /// petgraph-node corresponds to a [`FuncDecl`](OpType::FuncDecl) node (specified) in the Hugr
    FuncDecl(N),
    /// petgraph-node corresponds to a [`FuncDefn`](OpType::FuncDefn) node (specified) in the Hugr
    FuncDefn(N),
    /// petgraph-node corresponds to the [HugrView::entrypoint], that is not
    /// a [`FuncDefn`](OpType::FuncDefn). Note that it will not be a [Module](OpType::Module)
    /// either, as such a node could not have edges, so is not represented in the petgraph.
    NonFuncEntrypoint,
    /// petgraph-node corresponds to a constant; will have no outgoing edges, and incoming
    /// edges will be [StaticEdge::LoadConstant]
    Const(N),
}

/// Details the [`FuncDefn`]s, [`FuncDecl`]s and module-level [`Const`]s in a Hugr,
/// in a Hugr, along with the [`Call`]s, [`LoadFunction`]s, and [`LoadConstant`]s connecting them.
///
/// Each node in the `StaticGraph` corresponds to a module-level function or const;
/// each edge corresponds to a use of the target contained in the edge's source.
///
/// For Hugrs whose entrypoint is neither a [Module](OpType::Module) nor a [`FuncDefn`],
/// the static graph will have an additional [`StaticNode::NonFuncEntrypoint`]
/// corresponding to the Hugr's entrypoint, with no incoming edges.
///
/// [`Call`]: OpType::Call
/// [`Const`]: OpType::Const
/// [`FuncDecl`]: OpType::FuncDecl
/// [`FuncDefn`]: OpType::FuncDefn
/// [`LoadConstant`]: OpType::LoadConstant
/// [`LoadFunction`]: OpType::LoadFunction
pub struct StaticGraph<N = Node> {
    g: Graph<StaticNode<N>, StaticEdge<N>>,
    node_to_g: HashMap<N, petgraph::graph::NodeIndex<u32>>,
}

impl<N: HugrNode> StaticGraph<N> {
    /// Makes a new `CallGraph` for a Hugr.
    pub fn new(hugr: &impl HugrView<Node = N>) -> Self {
        let mut g = Graph::default();
        let mut node_to_g = hugr
            .children(hugr.module_root())
            .filter_map(|n| {
                let weight = match hugr.get_optype(n) {
                    OpType::FuncDecl(_) => StaticNode::FuncDecl(n),
                    OpType::FuncDefn(_) => StaticNode::FuncDefn(n),
                    OpType::Const(_) => StaticNode::Const(n),
                    _ => return None,
                };
                Some((n, g.add_node(weight)))
            })
            .collect::<HashMap<_, _>>();
        if !hugr.entrypoint_optype().is_module() && !node_to_g.contains_key(&hugr.entrypoint()) {
            node_to_g.insert(hugr.entrypoint(), g.add_node(StaticNode::NonFuncEntrypoint));
        }
        for (func, cg_node) in &node_to_g {
            traverse(hugr, *cg_node, *func, &mut g, &node_to_g);
        }
        fn traverse<N: HugrNode>(
            h: &impl HugrView<Node = N>,
            enclosing_func: petgraph::graph::NodeIndex<u32>,
            node: N, // Nonstrict-descendant of `enclosing_func``
            g: &mut Graph<StaticNode<N>, StaticEdge<N>>,
            node_to_g: &HashMap<N, petgraph::graph::NodeIndex<u32>>,
        ) {
            for ch in h.children(node) {
                traverse(h, enclosing_func, ch, g, node_to_g);
                let weight = match h.get_optype(ch) {
                    OpType::Call(_) => StaticEdge::Call(ch),
                    OpType::LoadFunction(_) => StaticEdge::LoadFunction(ch),
                    OpType::LoadConstant(_) => StaticEdge::LoadConstant(ch),
                    _ => continue,
                };
                if let Some(target) = h.static_source(ch) {
                    if h.get_parent(target) == Some(h.module_root()) {
                        g.add_edge(enclosing_func, node_to_g[&target], weight);
                    } else {
                        assert!(!node_to_g.contains_key(&target));
                        assert!(h.get_optype(ch).is_load_constant());
                        assert!(h.get_optype(target).is_const());
                    }
                }
            }
        }
        StaticGraph { g, node_to_g }
    }

    /// Allows access to the petgraph
    #[must_use]
    pub fn graph(&self) -> &Graph<StaticNode<N>, StaticEdge<N>> {
        &self.g
    }

    /// Convert a Hugr [Node] into a petgraph node index.
    /// Result will be `None` if `n` is not a [`FuncDefn`](OpType::FuncDefn),
    /// [`FuncDecl`](OpType::FuncDecl) or the [HugrView::entrypoint].
    pub fn node_index(&self, n: N) -> Option<petgraph::graph::NodeIndex<u32>> {
        self.node_to_g.get(&n).copied()
    }

    /// Returns an iterator over the out-edges from the given Node, i.e.
    /// edges to the functions/constants called/loaded by it.
    ///
    /// If the node is not recognised as a function or the entrypoint,
    /// for example if it is a [`Const`](OpType::Const), the iterator will be empty.
    pub fn out_edges(&self, n: N) -> impl Iterator<Item = (&StaticEdge<N>, &StaticNode<N>)> {
        let g = self.graph();
        self.node_index(n).into_iter().flat_map(move |n| {
            self.graph().edges(n).map(|e| {
                (
                    g.edge_weight(e.id()).unwrap(),
                    g.node_weight(e.target()).unwrap(),
                )
            })
        })
    }
}

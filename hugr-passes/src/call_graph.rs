//! Data structure for call graphs of a Hugr, and some transformations using them.
use std::collections::{HashMap, HashSet};

use hugr_core::{
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait, OpType},
    HugrView, Node,
};
use petgraph::{graph::NodeIndex, visit::Bfs, Graph};

enum CallGraphEdge {
    Call(Node),
    LoadFunction(Node),
}

pub struct CallGraph {
    g: Graph<Node, CallGraphEdge>,
    node_to_g: HashMap<Node, NodeIndex<u32>>,
}

impl CallGraph {
    /// Makes a new CallGraph for a specified (subview) of a Hugr.
    /// Calls to functions outside the view will be dropped.
    pub fn new(hugr: &impl HugrView) -> Self {
        let mut g = Graph::default();
        let root = (!hugr.get_optype(hugr.root()).is_module()).then_some(hugr.root());
        let node_to_g = hugr
            .nodes()
            // For non-Module-rooted Hugrs, make sure we include the root
            .filter(|&n| Some(n) == root || OpTag::Function.is_superset(hugr.get_optype(n).tag()))
            .map(|n| (n, g.add_node(n)))
            .collect::<HashMap<_, _>>();
        for (func, cg_node) in node_to_g.iter() {
            traverse(hugr, *func, *cg_node, &mut g, &node_to_g)
        }
        fn traverse(
            h: &impl HugrView,
            node: Node,
            enclosing: NodeIndex<u32>,
            g: &mut Graph<Node, CallGraphEdge>,
            node_to_g: &HashMap<Node, NodeIndex<u32>>,
        ) {
            for ch in h.children(node) {
                if h.get_optype(ch).is_func_defn() {
                    continue;
                };
                traverse(h, ch, enclosing, g, node_to_g);
                let weight = match h.get_optype(ch) {
                    OpType::Call(_) => CallGraphEdge::Call(ch),
                    OpType::LoadFunction(_) => CallGraphEdge::LoadFunction(ch),
                    _ => continue,
                };
                if let Some(target) = h.static_source(ch) {
                    g.add_edge(enclosing, *node_to_g.get(&target).unwrap(), weight);
                }
            }
        }
        CallGraph { g, node_to_g }
    }

    fn reachable_funcs(
        &self,
        h: &impl HugrView,
        roots: Option<impl IntoIterator<Item = Node>>,
    ) -> impl Iterator<Item = Node> + '_ {
        let mut b = if h.get_optype(h.root()).is_module() {
            let mut entry_points = match roots {
                Some(i) => i.into_iter().collect(),
                None => h
                    .children(h.root())
                    .filter(|n| {
                        h.get_optype(*n)
                            .as_func_defn()
                            .is_some_and(|fd| fd.name == "main")
                    })
                    .collect::<Vec<_>>(),
            }
            .into_iter()
            .map(|i| self.node_to_g.get(&i).unwrap());
            let mut b = Bfs::new(&self.g, *entry_points.next().unwrap());
            while let Some(e) = entry_points.next() {
                b.stack.push_back(*e);
            }
            b
        } else {
            Bfs::new(&self.g, *self.node_to_g.get(&h.root()).unwrap())
        };
        std::iter::from_fn(move || b.next(&self.g)).map(|i| *self.g.node_weight(i).unwrap())
    }
}

/// Delete from the Hugr any functions that are not used by either [Call](OpType::Call) or
/// [LoadFunction](OpType::LoadFunction) nodes in reachable parts.
///
/// For [Module](OpType::Module)-rooted Hugrs, `roots` may provide a list of entry points;
/// these are expected to be children of the root although this is not enforced. If `roots`
/// is absent, then the function called `main` (must be present and) is used as sole entry point.
///
///
pub fn remove_dead_funcs(h: &mut impl HugrMut, roots: Option<impl IntoIterator<Item = Node>>) {
    let cg = CallGraph::new(h);
    let reachable = cg.reachable_funcs(h, roots).collect::<HashSet<_>>();
    let unreachable = h
        .nodes()
        .filter(|n| h.get_optype(*n).is_func_defn() && !reachable.contains(n))
        .collect::<Vec<_>>();
    for n in unreachable {
        h.remove_subtree(n);
    }
}

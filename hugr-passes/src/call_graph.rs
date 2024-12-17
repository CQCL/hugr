//! Data structure for call graphs of a Hugr, and some transformations using them.
use std::collections::{HashMap, HashSet};

use hugr_core::{
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait, OpType},
    HugrView, Node,
};
use itertools::Itertools;
use petgraph::{graph::NodeIndex, visit::Bfs, Graph};

/// Weight for an edge in a [CallGraph]
pub enum CallGraphEdge {
    /// Edge corresponds to a [Call](OpType::Call) node (specified) in the Hugr
    Call(Node),
    /// Edge corresponds to a [LoadFunction](OpType::LoadFunction) node (specified) in the Hugr
    LoadFunction(Node),
}

/// Details the [Call]s and [LoadFunction]s in a Hugr.
/// Each node in the `CallGraph` corresponds to a [FuncDefn] in the Hugr; each edge corresponds
/// to a [Call]/[LoadFunction] of the edge's target, contained in the edge's source.
///
/// For Hugrs whose root is neither a [Module](OpType::Module) nor a [FuncDefn], the call graph
/// will have an additional node corresponding to the Hugr's root, with no incoming edges.
///
/// [Call]: OpType::Call
/// [FuncDefn]: OpType::FuncDefn
/// [LoadFunction]: OpType::LoadFunction
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
        roots: impl IntoIterator<Item = Node>,
    ) -> impl Iterator<Item = Node> + '_ {
        let mut entry_points = roots.into_iter().collect_vec();
        let mut b = if h.get_optype(h.root()).is_module() {
            if entry_points.is_empty() {
                entry_points.push(
                    h.children(h.root())
                        .filter(|n| {
                            h.get_optype(*n)
                                .as_func_defn()
                                .is_some_and(|fd| fd.name == "main")
                        })
                        .exactly_one()
                        .ok()
                        .expect("No entry_points provided, Module must contain `main`"),
                );
            }
            let mut entry_points = entry_points
                .into_iter()
                .map(|i| self.node_to_g.get(&i).unwrap());
            let mut b = Bfs::new(&self.g, *entry_points.next().unwrap());
            b.stack.extend(entry_points);
            b
        } else {
            assert!(entry_points.is_empty());
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
/// is empty, then the root must have exactly one child being a function called `main`,
/// which is used as sole entry point.
///
/// For non-Module-rooted Hugrs, `entry_points` must be empty; the root node is used.
///
/// # Panics
/// * If the Hugr is non-Module-rooted and `entry_points` is non-empty
/// * If the Hugr is Module-rooted, but does not declare `main`, and `entry_points` is empty
/// * If the Hugr is Module-rooted, and `entry_points` is non-empty but contains nodes that
///      are not [FuncDefn](OpType::FuncDefn)s
pub fn remove_dead_funcs(h: &mut impl HugrMut, entry_points: impl IntoIterator<Item = Node>) {
    let cg = CallGraph::new(h);
    let reachable = cg.reachable_funcs(h, entry_points).collect::<HashSet<_>>();
    let unreachable = h
        .nodes()
        .filter(|n| h.get_optype(*n).is_func_defn() && !reachable.contains(n))
        .collect::<Vec<_>>();
    for n in unreachable {
        h.remove_subtree(n);
    }
}

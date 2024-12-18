#![warn(missing_docs)]
//! Data structure for call graphs of a Hugr, and some transformations using them.
use std::collections::{HashMap, HashSet};

use hugr_core::{
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait, OpType},
    HugrView, Node,
};
use itertools::Itertools;
use petgraph::{graph::NodeIndex, visit::Bfs, Graph};

use crate::validation::{ValidatePassError, ValidationLevel};

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
        // For non-Module-rooted Hugrs, make sure we include the root
        let root = (!hugr.get_optype(hugr.root()).is_module()).then_some(hugr.root());
        let node_to_g = hugr
            .nodes()
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
}

fn reachable_funcs<'a>(
    cg: &'a CallGraph,
    h: &impl HugrView,
    entry_points: impl IntoIterator<Item = Node>,
) -> impl Iterator<Item = Node> + 'a {
    let mut roots = entry_points.into_iter().collect_vec();
    let mut b = if h.get_optype(h.root()).is_module() {
        if roots.is_empty() {
            roots.extend(h.children(h.root()).filter(|n| {
                h.get_optype(*n)
                    .as_func_defn()
                    .is_some_and(|fd| fd.name == "main")
            }));
            assert_eq!(roots.len(), 1, "No entry_points for Module and no `main`");
        }
        let mut roots = roots.into_iter().map(|i| cg.node_to_g.get(&i).unwrap());
        let mut b = Bfs::new(&cg.g, *roots.next().unwrap());
        b.stack.extend(roots);
        b
    } else {
        assert!(roots.is_empty());
        Bfs::new(&cg.g, *cg.node_to_g.get(&h.root()).unwrap())
    };
    std::iter::from_fn(move || b.next(&cg.g)).map(|i| *cg.g.node_weight(i).unwrap())
}

#[derive(Debug, Clone, Default)]
/// A configuration for the Dead Function Removal pass.
pub struct RemoveDeadFuncsPass {
    validation: ValidationLevel,
    entry_points: Vec<Node>,
}

impl RemoveDeadFuncsPass {
    /// Sets the validation level used before and after the pass is run
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Adds new entry points for a Module-rooted Hugr,
    /// i.e., FuncDefns that are children of the root.
    pub fn with_module_entry_points(
        mut self,
        entry_points: impl IntoIterator<Item = Node>,
    ) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    /// Runs the pass (see [remove_dead_funcs]) with this configuration
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), ValidatePassError> {
        self.validation.run_validated_pass(hugr, |hugr: &mut H, _| {
            remove_dead_funcs(hugr, self.entry_points.iter().cloned());
            Ok(())
        })
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
    let reachable = reachable_funcs(&CallGraph::new(h), h, entry_points).collect::<HashSet<_>>();
    let unreachable = h
        .nodes()
        .filter(|n| h.get_optype(*n).is_func_defn() && !reachable.contains(n))
        .collect::<Vec<_>>();
    for n in unreachable {
        h.remove_subtree(n);
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use itertools::Itertools;
    use rstest::rstest;

    use hugr_core::builder::{
        Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use hugr_core::{extension::prelude::usize_t, types::Signature, HugrView};

    use super::remove_dead_funcs;

    #[rstest]
    #[case([], vec!["from_main", "main"])]
    #[case(["main"], vec!["from_main", "main"])]
    #[case(["from_main"], vec!["from_main"])]
    #[case(["other1"], vec!["other1", "other2"])]
    #[case(["other2"], vec!["other2"])]
    #[case(["other1", "other2"], vec!["other1", "other2"])]
    fn remove_dead_funcs_entry_points(
        #[case] entry_points: impl IntoIterator<Item = &'static str>,
        #[case] retained_funcs: Vec<&'static str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut hb = ModuleBuilder::new();
        let o2 = hb.define_function("other2", Signature::new_endo(usize_t()))?;
        let o2inp = o2.input_wires();
        let o2 = o2.finish_with_outputs(o2inp)?;
        let mut o1 = hb.define_function("other1", Signature::new_endo(usize_t()))?;

        let o1c = o1.call(o2.handle(), &[], o1.input_wires())?;
        o1.finish_with_outputs(o1c.outputs())?;

        let fm = hb.define_function("from_main", Signature::new_endo(usize_t()))?;
        let f_inp = fm.input_wires();
        let fm = fm.finish_with_outputs(f_inp)?;
        let mut m = hb.define_function("main", Signature::new_endo(usize_t()))?;
        let mc = m.call(fm.handle(), &[], m.input_wires())?;
        m.finish_with_outputs(mc.outputs())?;

        let mut hugr = hb.finish_hugr()?;

        let avail_funcs = hugr
            .nodes()
            .filter_map(|n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .map(|fd| (fd.name.clone(), n))
            })
            .collect::<HashMap<_, _>>();

        remove_dead_funcs(
            &mut hugr,
            entry_points
                .into_iter()
                .map(|name| *avail_funcs.get(name).unwrap())
                .collect::<Vec<_>>(),
        );
        let remaining_funcs = hugr
            .nodes()
            .filter_map(|n| hugr.get_optype(n).as_func_defn().map(|fd| fd.name.as_str()))
            .sorted()
            .collect_vec();
        assert_eq!(remaining_funcs, retained_funcs);
        Ok(())
    }
}

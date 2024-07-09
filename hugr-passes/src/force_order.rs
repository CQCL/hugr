//! Provides [force_order], a tool for fixing the order of nodes in a Hugr.
use std::iter::zip;

use hugr_core::{
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, HierarchyView, SiblingGraph},
        HugrError,
    },
    ops::{OpTag, OpTrait},
    Direction, HugrView as _, Node,
};
use itertools::Itertools as _;
use petgraph::{
    algo::dominators::simple_fast,
    visit::{
        GraphBase, GraphRef, IntoNeighbors as _, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeFiltered, Reversed, VisitMap, Visitable, Walker,
    },
    Direction::Incoming,
};

/// Insert order edges into a Hugr according to a rank function.
///
/// All dataflow parents which are transitive children of `root`, including
/// `root` itself, will be ordered.
///
/// Dataflow parents are ordered by inserting order edges between their
/// immediate children. A dataflow parent with `C` children will have at most
/// `C-1` edges added. Any node than can be ordered will be.
///
/// Nodes are ordered according to the rank function. Nodes of lower rank will
/// be ordered earlier in their parent. Note that if  `rank(n1) > rank(n2)` it
/// is not guaranteed that `n1` will be ordered after `n2`. If `n2` dominates
/// `n1` it cannot be ordered after `n1`. Nodes of equal rank will be ordered
/// arbitrarily.
pub fn force_order(
    hugr: &mut impl HugrMut,
    root: Node,
    rank: impl Fn(Node) -> i64,
) -> Result<(), HugrError> {
    let dataflow_parents = DescendantsGraph::<Node>::try_new(hugr, root)?
        .nodes()
        .filter(|n| hugr.get_optype(*n).tag() <= OpTag::DataflowParent)
        .collect_vec();
    for dp in dataflow_parents {
        let sg = SiblingGraph::<Node>::try_new(hugr, dp)?;
        let petgraph = NodeFiltered::from_fn(sg.as_petgraph(), |x| x != dp);
        let ordered_nodes = ForceOrder::new(&petgraph, &rank)
            .iter(&petgraph)
            .collect_vec();

        let [i, _] = hugr.get_io(dp).unwrap();
        let dominators = simple_fast(&petgraph, i);
        for (&n1, &n2) in zip(&ordered_nodes[..], &ordered_nodes[1..]) {
            // there is already an edge here, order edge unnecessary
            if dominators.immediately_dominated_by(n1).contains(&n2) {
                continue;
            }

            // we can only add an order edge if the two ops support it
            let (n1_ot, n2_ot) = (hugr.get_optype(n1), hugr.get_optype(n2));
            let expected_edge_kind = Some(hugr_core::types::EdgeKind::StateOrder);
            if n1_ot.other_port_kind(Direction::Outgoing) != expected_edge_kind
                || n2_ot.other_port_kind(Direction::Incoming) != expected_edge_kind
            {
                continue;
            }

            hugr.connect(
                n1,
                n1_ot.other_output_port().unwrap(),
                n2,
                n2_ot.other_input_port().unwrap(),
            );
        }
    }

    Ok(())
}

/// An adaption of [petgraph::visit::Topo]. We differ only in that we sort nodes
/// by the rank function before adding them to the internal work stack. This
/// ensures we visit lower ranked nodes before higher ranked nodes whenever the
/// topology of the graph allows.
#[derive(Clone)]
struct ForceOrder<N, VM, F> {
    tovisit: Vec<N>,
    ordered: VM,
    rank: F,
}

impl<N, VM, F: Fn(N) -> i64> ForceOrder<N, VM, F>
where
    N: Copy + PartialEq,
    VM: VisitMap<N>,
{
    pub fn new<G>(graph: G, rank: F) -> Self
    where
        G: IntoNodeIdentifiers + IntoNeighborsDirected + Visitable<NodeId = N, Map = VM>,
    {
        let mut topo = Self::empty(graph, rank);
        topo.extend_with_initials(graph);
        topo
    }

    fn empty<G>(graph: G, rank: F) -> Self
    where
        G: GraphRef + Visitable<NodeId = N, Map = VM>,
    {
        Self {
            ordered: graph.visit_map(),
            tovisit: Default::default(),
            rank,
        }
    }

    fn extend_with_initials<G>(&mut self, g: G)
    where
        G: IntoNodeIdentifiers + IntoNeighborsDirected<NodeId = N>,
    {
        // find all initial nodes (nodes without incoming edges)
        self.extend(
            g.node_identifiers()
                .filter(move |&a| g.neighbors_directed(a, Incoming).next().is_none()),
        );
    }

    fn extend(&mut self, new_nodes: impl IntoIterator<Item = N>) {
        let mut new_nodes = new_nodes.into_iter().collect_vec();
        new_nodes.sort_by_cached_key(|&k| !(self.rank)(k));
        // Lower rank nodes must be ordered earlier in the graph.
        // This means we should visit them earlier, so they should be at the
        // end of the list passed to extend.
        self.tovisit.extend(new_nodes.into_iter());
    }

    /// Return the next node in the current topological order traversal, or
    /// `None` if the traversal is at the end.
    ///
    /// *Note:* The graph may not have a complete topological order, and the only
    /// way to know is to run the whole traversal and make sure it visits every node.
    pub fn next<G>(&mut self, g: G) -> Option<N>
    where
        G: IntoNeighborsDirected + Visitable<NodeId = N, Map = VM>,
    {
        // Take an unvisited element and find which of its neighbors are next
        while let Some(nix) = self.tovisit.pop() {
            if self.ordered.is_visited(&nix) {
                continue;
            }
            self.ordered.visit(nix);
            // Look at each neighbor, and those that only have incoming edges
            // from the already ordered list, they are the next to visit.
            let new_nodes = g
                .neighbors(nix)
                .filter(|&n| {
                    Reversed(g)
                        .neighbors(n)
                        .all(|b| self.ordered.is_visited(&b))
                })
                .collect_vec();

            self.extend(new_nodes);
            return Some(nix);
        }
        None
    }
}

impl<G: Visitable + IntoNeighborsDirected, F: Fn(G::NodeId) -> i64> Walker<G>
    for ForceOrder<G::NodeId, G::Map, F>
{
    type Item = <G as GraphBase>::NodeId;

    fn walk_next(&mut self, g: G) -> Option<Self::Item> {
        self.next(g)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::*;
    use hugr_core::builder::{BuildHandle, Dataflow, DataflowHugr};
    use hugr_core::ops::handle::{DataflowOpID, NodeHandle};
    use hugr_core::std_extensions::arithmetic::int_ops::{self, IntOpDef};
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::types::FunctionType;
    use hugr_core::{builder::DFGBuilder, hugr::Hugr};
    use hugr_core::{HugrView, Wire};

    use petgraph::visit::Topo;

    const I: u8 = 3;

    fn build_neg(builder: &mut impl Dataflow, wire: Wire) -> BuildHandle<DataflowOpID> {
        builder
            .add_dataflow_op(IntOpDef::ineg.with_log_width(I), [wire])
            .unwrap()
    }

    fn build_add(builder: &mut impl Dataflow, w1: Wire, w2: Wire) -> BuildHandle<DataflowOpID> {
        builder
            .add_dataflow_op(IntOpDef::iadd.with_log_width(I), [w1, w2])
            .unwrap()
    }

    /// Our tests use the following hugr:
    ///
    /// a DFG with sig: [i8,i8] -> [i8,i8]
    ///
    ///         Input
    ///    |        |
    ///    | iw1    | iw2
    ///   v0(neg)  v1(neg)
    ///    |   \    |
    ///    |    \   |
    ///    |     \  |
    ///   v2(neg)  v3(add)
    ///    |        |
    ///      Output
    fn test_hugr() -> (Hugr, [Node; 4]) {
        let t = INT_TYPES[I as usize].clone();
        let mut builder =
            DFGBuilder::new(FunctionType::new_endo(vec![t.clone(), t.clone()])).unwrap();
        let [iw1, iw2] = builder.input_wires_arr();
        let v0 = build_neg(&mut builder, iw1);
        let v1 = build_neg(&mut builder, iw2);
        let v2 = build_neg(&mut builder, v0.out_wire(0));
        let v3 = build_add(&mut builder, v0.out_wire(0), v1.out_wire(0));
        let nodes = [v0, v1, v2, v3]
            .into_iter()
            .map(|x| x.handle().node())
            .collect_vec()
            .try_into()
            .unwrap();
        (
            builder
                .finish_hugr_with_outputs(
                    [v2.out_wire(0), v3.out_wire(0)],
                    &int_ops::INT_OPS_REGISTRY,
                )
                .unwrap(),
            nodes,
        )
    }

    type RankMap = HashMap<Node, i64>;

    fn force_order_test_impl(hugr: &mut Hugr, rank_map: RankMap) -> Vec<Node> {
        force_order(hugr, hugr.root(), |n| *rank_map.get(&n).unwrap_or(&0)).unwrap();

        let topo_sorted = Topo::new(&hugr.as_petgraph())
            .iter(&hugr.as_petgraph())
            .filter(|n| rank_map.contains_key(n))
            .collect_vec();
        hugr.validate_no_extensions(&int_ops::INT_OPS_REGISTRY)
            .unwrap();

        topo_sorted
    }

    #[test]
    fn test_force_order_1() {
        let (mut hugr, [v0, v1, v2, v3]) = test_hugr();

        // v0 has a higher rank than v2, but v0 dominates v2, so cannot be
        // ordered before it.
        //
        // v1 and v3 are pushed to the bottom of the graph with high weights.
        let rank_map = [(v0, 2), (v2, 1), (v1, 10), (v3, 9)].into_iter().collect();

        let topo_sort = force_order_test_impl(&mut hugr, rank_map);
        assert_eq!(vec![v0, v2, v1, v3], topo_sort);
    }

    #[test]
    fn test_force_order_2() {
        let (mut hugr, [v0, v1, v2, v3]) = test_hugr();

        // v1 and v3 are pulled to the top of the graph with low weights.
        // v3 cannot ascend past v0 because it is dominated by v0
        let rank_map = [(v0, 2), (v2, 1), (v1, -10), (v3, -9)]
            .into_iter()
            .collect();
        let topo_sort = force_order_test_impl(&mut hugr, rank_map);
        assert_eq!(vec![v1, v0, v3, v2], topo_sort);
    }
}

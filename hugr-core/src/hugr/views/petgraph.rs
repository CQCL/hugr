//! Implementations of petgraph's traits for Hugr Region views.

use crate::hugr::HugrView;
use crate::ops::OpType;
use crate::types::EdgeKind;
use crate::NodeIndex;
use crate::{Node, Port};

use petgraph::visit as pv;

/// Wrapper for a HugrView that implements petgraph's traits.
///
/// It can be used to apply petgraph's algorithms to a Hugr.
#[derive(Debug)]
pub struct PetgraphWrapper<'a, T> {
    pub(crate) hugr: &'a T,
}

impl<'a, T> Clone for PetgraphWrapper<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for PetgraphWrapper<'a, T> {}

impl<'a, T> From<&'a T> for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    fn from(hugr: &'a T) -> Self {
        Self { hugr }
    }
}

impl<'a, T> pv::GraphBase for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type NodeId = Node;
    type EdgeId = ((Node, Port), (Node, Port));
}

impl<'a, T> pv::GraphProp for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type EdgeType = petgraph::Directed;
}

impl<'a, T> pv::GraphRef for PetgraphWrapper<'a, T> where T: HugrView {}

impl<'a, T> pv::NodeCount for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    fn node_count(&self) -> usize {
        HugrView::node_count(self.hugr)
    }
}

impl<'a, T> pv::NodeIndexable for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    fn node_bound(&self) -> usize {
        HugrView::node_count(self.hugr)
    }

    fn to_index(&self, ix: Self::NodeId) -> usize {
        ix.index()
    }

    fn from_index(&self, ix: usize) -> Self::NodeId {
        portgraph::NodeIndex::new(ix).into()
    }
}

impl<'a, T> pv::EdgeCount for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    fn edge_count(&self) -> usize {
        HugrView::edge_count(self.hugr)
    }
}

impl<'a, T> pv::Data for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type NodeWeight = OpType;
    type EdgeWeight = EdgeKind;
}

impl<'a, T> pv::IntoNodeReferences for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type NodeRef = HugrNodeRef<'a>;
    type NodeReferences = Box<dyn Iterator<Item = HugrNodeRef<'a>> + 'a>;

    fn node_references(self) -> Self::NodeReferences {
        Box::new(
            self.hugr
                .nodes()
                .map(|n| HugrNodeRef::from_node(n, self.hugr)),
        )
    }
}

impl<'a, T> pv::IntoNodeIdentifiers for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type NodeIdentifiers = Box<dyn Iterator<Item = Node> + 'a>;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        Box::new(self.hugr.nodes())
    }
}

impl<'a, T> pv::IntoNeighbors for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type Neighbors = Box<dyn Iterator<Item = Node> + 'a>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        Box::new(self.hugr.output_neighbours(n))
    }
}

impl<'a, T> pv::IntoNeighborsDirected for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type NeighborsDirected = Box<dyn Iterator<Item = Node> + 'a>;

    fn neighbors_directed(
        self,
        n: Self::NodeId,
        d: petgraph::Direction,
    ) -> Self::NeighborsDirected {
        Box::new(self.hugr.neighbours(n, d.into()))
    }
}

impl<'a, T> pv::Visitable for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type Map = std::collections::HashSet<Self::NodeId>;

    fn visit_map(&self) -> Self::Map {
        std::collections::HashSet::new()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear();
    }
}

impl<'a, T> pv::GetAdjacencyMatrix for PetgraphWrapper<'a, T>
where
    T: HugrView,
{
    type AdjMatrix = std::collections::HashSet<(Self::NodeId, Self::NodeId)>;

    fn adjacency_matrix(&self) -> Self::AdjMatrix {
        let mut matrix = std::collections::HashSet::new();
        for node in self.hugr.nodes() {
            for neighbour in self.hugr.output_neighbours(node) {
                matrix.insert((node, neighbour));
            }
        }
        matrix
    }

    fn is_adjacent(&self, matrix: &Self::AdjMatrix, a: Self::NodeId, b: Self::NodeId) -> bool {
        matrix.contains(&(a, b))
    }
}

/// Reference to a Hugr node and its associated OpType.
#[derive(Debug, Clone, Copy)]
pub struct HugrNodeRef<'a> {
    node: Node,
    op: &'a OpType,
}

impl<'a> HugrNodeRef<'a> {
    pub(self) fn from_node(node: Node, hugr: &'a impl HugrView) -> Self {
        Self {
            node,
            op: hugr.get_optype(node),
        }
    }
}

impl pv::NodeRef for HugrNodeRef<'_> {
    type NodeId = Node;

    type Weight = OpType;

    fn id(&self) -> Self::NodeId {
        self.node
    }

    fn weight(&self) -> &Self::Weight {
        self.op
    }
}

#[cfg(test)]
mod test {
    use petgraph::visit::{
        EdgeCount, GetAdjacencyMatrix, IntoNodeReferences, NodeCount, NodeIndexable, NodeRef,
    };

    use crate::hugr::views::tests::sample_hugr;
    use crate::ops::handle::NodeHandle;
    use crate::HugrView;

    use super::PetgraphWrapper;

    #[test]
    fn test_petgraph_wrapper() {
        let (hugr, cx1, cx2) = sample_hugr();
        let wrapper = PetgraphWrapper::from(&hugr);

        assert_eq!(wrapper.node_count(), 5);
        assert_eq!(wrapper.node_bound(), 5);
        assert_eq!(wrapper.edge_count(), 7);

        let cx1_index = cx1.node().pg_index().index();
        assert_eq!(wrapper.to_index(cx1.node()), cx1_index);
        assert_eq!(wrapper.from_index(cx1_index), cx1.node());

        let cx1_ref = wrapper
            .node_references()
            .find(|n| n.id() == cx1.node())
            .unwrap();
        assert_eq!(cx1_ref.weight(), hugr.get_optype(cx1.node()));

        let adj = wrapper.adjacency_matrix();
        assert!(wrapper.is_adjacent(&adj, cx1.node(), cx2.node()));
    }
}

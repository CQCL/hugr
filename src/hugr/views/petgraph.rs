//! Implementations of petgraph's traits for Hugr Region views.

use crate::hugr::HugrView;
use crate::ops::OpType;
use crate::types::EdgeKind;
use crate::{Node, Port};

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use delegate::delegate;
use petgraph::visit as pv;
use portgraph::NodeIndex;

use super::sealed::HugrInternals;

/// Wrapper for a HugrView that implements petgraph's traits.
///
/// It can be used to apply petgraph's algorithms to a Hugr.
#[derive(Debug, Clone, Copy)]
pub struct PetgraphWrapper<T> {
    hugr: T,
}

impl<T> PetgraphWrapper<T>
where
    T: HugrView,
{
    /// Wrap a HugrView in a PetgraphWrapper.
    pub fn new(hugr: T) -> Self {
        Self { hugr }
    }
}

impl<T> pv::GraphBase for PetgraphWrapper<T>
where
    T: HugrView,
{
    type NodeId = Node;
    type EdgeId = ((Node, Port), (Node, Port));
}

impl<T> pv::GraphProp for PetgraphWrapper<T>
where
    T: HugrView,
{
    type EdgeType = petgraph::Directed;
}

impl<T> pv::NodeCount for PetgraphWrapper<T>
where
    T: HugrView,
{
    fn node_count(&self) -> usize {
        HugrView::node_count(self)
    }
}

impl<T> pv::NodeIndexable for PetgraphWrapper<T>
where
    T: HugrView,
{
    fn node_bound(&self) -> usize {
        HugrView::node_count(self)
    }

    fn to_index(&self, ix: Self::NodeId) -> usize {
        ix.index.into()
    }

    fn from_index(&self, ix: usize) -> Self::NodeId {
        NodeIndex::new(ix).into()
    }
}

impl<T> pv::EdgeCount for PetgraphWrapper<T>
where
    T: HugrView,
{
    fn edge_count(&self) -> usize {
        HugrView::edge_count(self)
    }
}

impl<T> pv::Data for PetgraphWrapper<T>
where
    T: HugrView,
{
    type NodeWeight = OpType;
    type EdgeWeight = EdgeKind;
}

impl<'g, T> pv::IntoNodeReferences for &'g PetgraphWrapper<T>
where
    T: HugrView,
{
    type NodeRef = HugrNodeRef<'g>;
    type NodeReferences = MapWithCtx<<T as HugrView>::Nodes<'g>, Self, HugrNodeRef<'g>>;

    fn node_references(self) -> Self::NodeReferences {
        self.nodes()
            .with_context(self)
            .map_with_context(|n, &hugr| HugrNodeRef::from_node(n, hugr))
    }
}

impl<'g, T> pv::IntoNodeIdentifiers for &'g PetgraphWrapper<T>
where
    T: HugrView,
{
    type NodeIdentifiers = <T as HugrView>::Nodes<'g>;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.nodes()
    }
}

impl<'g, T> pv::IntoNeighbors for &'g PetgraphWrapper<T>
where
    T: HugrView,
{
    type Neighbors = <T as HugrView>::Neighbours<'g>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        self.output_neighbours(n)
    }
}

impl<'g, T> pv::IntoNeighborsDirected for &'g PetgraphWrapper<T>
where
    T: HugrView,
{
    type NeighborsDirected = <T as HugrView>::Neighbours<'g>;

    fn neighbors_directed(
        self,
        n: Self::NodeId,
        d: petgraph::Direction,
    ) -> Self::NeighborsDirected {
        self.neighbours(n, d.into())
    }
}

impl<T> pv::Visitable for PetgraphWrapper<T>
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

impl<T> pv::GetAdjacencyMatrix for PetgraphWrapper<T>
where
    T: HugrView,
{
    type AdjMatrix = std::collections::HashSet<(Self::NodeId, Self::NodeId)>;

    fn adjacency_matrix(&self) -> Self::AdjMatrix {
        let mut matrix = std::collections::HashSet::new();
        for node in self.nodes() {
            for neighbour in self.output_neighbours(node) {
                matrix.insert((node, neighbour));
            }
        }
        matrix
    }

    fn is_adjacent(&self, matrix: &Self::AdjMatrix, a: Self::NodeId, b: Self::NodeId) -> bool {
        matrix.contains(&(a, b))
    }
}

impl<T> HugrInternals for PetgraphWrapper<T>
where
    T: HugrView,
{
    type Portgraph<'p> = <T as HugrInternals>::Portgraph<'p>
    where
        Self: 'p;

    delegate! {
        to self.hugr {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &crate::Hugr;
            fn root_node(&self) -> Node;
        }
    }
}

impl<T> HugrView for PetgraphWrapper<T>
where
    T: HugrView,
{
    type RootHandle = T::RootHandle;

    type Nodes<'a> = T::Nodes<'a>
    where
        Self: 'a;

    type NodePorts<'a> = T::NodePorts<'a>
    where
        Self: 'a;

    type Children<'a> = T::Children<'a>
    where
        Self: 'a;

    type Neighbours<'a> = T::Neighbours<'a>
    where
        Self: 'a;

    type PortLinks<'a> = T::PortLinks<'a>
    where
        Self: 'a;

    type NodeConnections<'a> = T::NodeConnections<'a>
    where
        Self: 'a;

    delegate! {
        to self.hugr {
            fn contains_node(&self, node: Node) -> bool;
            fn get_parent(&self, node: Node) -> Option<Node>;
            fn get_optype(&self, node: Node) -> &OpType;
            fn get_nodetype(&self, node: Node) -> &crate::hugr::NodeType;
            fn get_metadata(&self, node: Node) -> &crate::hugr::NodeMetadata;
            fn node_count(&self) -> usize;
            fn edge_count(&self) -> usize;
            fn nodes(&self) -> Self::Nodes<'_>;
            fn node_ports(&self, node: Node, dir: crate::Direction) -> Self::NodePorts<'_>;
            fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_>;
            fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_>;
            fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_>;
            fn num_ports(&self, node: Node, dir: crate::Direction) -> usize;
            fn children(&self, node: Node) -> Self::Children<'_>;
            fn neighbours(&self, node: Node, dir: crate::Direction) -> Self::Neighbours<'_>;
            fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_>;
            fn get_io(&self, node: Node) -> Option<[Node; 2]>;
            fn get_function_type(&self) -> Option<&crate::types::FunctionType>;
        }
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

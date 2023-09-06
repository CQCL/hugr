//! Implementations of petgraph's traits for Hugr Region views.

use super::{DescendantsGraph, SiblingGraph};
use crate::hugr::views::sealed::HugrInternals;
use crate::hugr::HugrView;
use crate::ops::handle::NodeHandle;
use crate::ops::OpType;
use crate::types::EdgeKind;
use crate::{Hugr, Node, Port};

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use petgraph::visit as pv;
use portgraph::NodeIndex;

/// A trait grouping the multiple petgraph traits implemented on Hugrs
//
// TODO: We could just require these in `HugrView`.
pub trait PetgraphHugr:
    HugrView
    + pv::GraphBase<NodeId = Node>
    + pv::GraphProp
    + pv::NodeCount
    + pv::NodeIndexable
    + pv::EdgeCount
    + pv::Visitable
    + pv::GetAdjacencyMatrix
    + pv::Visitable
where
    for<'g> &'g Self: pv::IntoNeighborsDirected + pv::IntoNodeIdentifiers,
{
}

impl pv::GraphBase for Hugr {
    type NodeId = Node;
    type EdgeId = ((Node, Port), (Node, Port));
}

impl pv::GraphProp for Hugr {
    type EdgeType = petgraph::Directed;
}

impl pv::NodeCount for Hugr {
    fn node_count(&self) -> usize {
        HugrView::node_count(self)
    }
}

impl pv::NodeIndexable for Hugr {
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

impl pv::EdgeCount for Hugr {
    fn edge_count(&self) -> usize {
        HugrView::edge_count(self)
    }
}

impl pv::Data for Hugr {
    type NodeWeight = OpType;
    type EdgeWeight = EdgeKind;
}

impl<'g> pv::IntoNodeIdentifiers for &'g Hugr {
    type NodeIdentifiers = <Hugr as HugrView>::Nodes<'g>;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.nodes()
    }
}

impl<'g> pv::IntoNodeReferences for &'g Hugr {
    type NodeRef = HugrNodeRef<'g>;
    type NodeReferences = MapWithCtx<<Hugr as HugrView>::Nodes<'g>, Self, HugrNodeRef<'g>>;

    fn node_references(self) -> Self::NodeReferences {
        self.nodes()
            .with_context(self)
            .map_with_context(|n, &hugr| HugrNodeRef::from_node(n, hugr))
    }
}

impl<'g> pv::IntoNeighbors for &'g Hugr {
    type Neighbors = <Hugr as HugrView>::Neighbours<'g>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        self.output_neighbours(n)
    }
}

impl<'g> pv::IntoNeighborsDirected for &'g Hugr {
    type NeighborsDirected = <Hugr as HugrView>::Neighbours<'g>;

    fn neighbors_directed(
        self,
        n: Self::NodeId,
        d: petgraph::Direction,
    ) -> Self::NeighborsDirected {
        self.neighbours(n, d.into())
    }
}

impl pv::Visitable for Hugr {
    type Map = std::collections::HashSet<Self::NodeId>;

    fn visit_map(&self) -> Self::Map {
        std::collections::HashSet::new()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        map.clear();
    }
}

impl pv::GetAdjacencyMatrix for Hugr {
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

impl PetgraphHugr for Hugr {}

macro_rules! impl_petgraph_into_noderefs {
    ($hugr:ident) => {
        impl<'g, 'a, Root, Base> pv::IntoNodeReferences for &'g $hugr<'a, Root, Base>
        where 
        Root:NodeHandle, 'g: 'a, Base: HugrInternals + HugrView
        {
            type NodeRef = HugrNodeRef<'a>;
            type NodeReferences =
                MapWithCtx<<$hugr<'a, Root, Base> as HugrView>::Nodes<'a>, Self, HugrNodeRef<'a>>;

            fn node_references(self) -> Self::NodeReferences {
                self.nodes()
                    .with_context(self)
                    .map_with_context(|n, &hugr| HugrNodeRef::from_node(n, hugr))
            }
        }
    }
}

macro_rules! impl_region_petgraph_traits {
    ($hugr:ident $(< $($l:lifetime,)? $($v:ident: $b:tt $(+ $b2:tt)*),* >)? ) => {
        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::GraphBase for $hugr$(<$($l,)? $($v),*>)?
        {
            type NodeId = Node;
            type EdgeId = ((Node, Port), (Node, Port));
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::GraphProp for $hugr$(<$($l,)? $($v),*>)?
        {
            type EdgeType = petgraph::Directed;
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::NodeCount for $hugr$(<$($l,)? $($v),*>)?
        {
            fn node_count(&self) -> usize {
                HugrView::node_count(self)
            }
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::NodeIndexable for $hugr$(<$($l,)? $($v),*>)?
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

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::EdgeCount for $hugr$(<$($l,)? $($v),*>)?
        {
            fn edge_count(&self) -> usize {
                HugrView::edge_count(self)
            }
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::Data for $hugr$(<$($l,)? $($v),*>)?
        {
            type NodeWeight = OpType;
            type EdgeWeight = EdgeKind;
        }

        impl <'g $(, $($l,)? $($v: $b $(+ $b2)*),*)?> pv::IntoNodeIdentifiers for &'g $hugr$(<$($l,)? $($v),*>)?
        {
            type NodeIdentifiers = <$hugr$(<$($l,)? $($v),*>)? as HugrView>::Nodes<'g>;

            fn node_identifiers(self) -> Self::NodeIdentifiers {
                self.nodes()
            }
        }

        impl <'g $(, $($l,)? $($v: $b $(+ $b2)*),*)?> pv::IntoNeighbors for &'g $hugr$(<$($l,)? $($v),*>)?
        {
            type Neighbors = <$hugr$(<$($l,)? $($v),*>)? as HugrView>::Neighbours<'g>;

            fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
                self.output_neighbours(n)
            }
        }

        impl <'g $(, $($l,)? $($v: $b $(+ $b2)*),*)?> pv::IntoNeighborsDirected for &'g $hugr$(<$($l,)? $($v),*>)?
        {
            type NeighborsDirected = <$hugr$(<$($l,)? $($v),*>)? as HugrView>::Neighbours<'g>;

            fn neighbors_directed(
                self,
                n: Self::NodeId,
                d: petgraph::Direction,
            ) -> Self::NeighborsDirected {
                self.neighbours(n, d.into())
            }
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::Visitable for $hugr$(<$($l,)? $($v),*>)?
        {
            type Map = std::collections::HashSet<Self::NodeId>;

            fn visit_map(&self) -> Self::Map {
                std::collections::HashSet::new()
            }

            fn reset_map(&self, map: &mut Self::Map) {
                map.clear();
            }
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? pv::GetAdjacencyMatrix for $hugr$(<$($l,)? $($v),*>)?
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

            fn is_adjacent(
                &self,
                matrix: &Self::AdjMatrix,
                a: Self::NodeId,
                b: Self::NodeId,
            ) -> bool {
                matrix.contains(&(a, b))
            }
        }

        impl $(<$($l,)? $($v: $b $(+ $b2)*),*>)? PetgraphHugr for $hugr$(<$($l,)? $($v),*>)?
        {
        }
    };
}

impl_petgraph_into_noderefs!(SiblingGraph);
impl_petgraph_into_noderefs!(DescendantsGraph);
impl_region_petgraph_traits!(SiblingGraph<'a, Root:NodeHandle, Base: HugrInternals + HugrView>);
impl_region_petgraph_traits!(DescendantsGraph<'a, Root: NodeHandle, Base: HugrInternals + HugrView>);

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

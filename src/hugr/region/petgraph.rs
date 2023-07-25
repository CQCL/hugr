//! Implementations of petgraph's traits for Hugr Region views.

use super::{FlatRegionView, RegionView};
use crate::hugr::view::sealed::HugrInternals;
use crate::hugr::HugrView;
use crate::ops::OpType;
use crate::types::EdgeKind;
use crate::{Node, Port};

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use petgraph::visit as pv;
use portgraph::NodeIndex;

macro_rules! impl_region_petgraph_traits {
    ($hugr:ident) => {
        impl<'a, Base> pv::GraphBase for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type NodeId = Node;
            type EdgeId = ((Node, Port), (Node, Port));
        }

        impl<'a, Base> pv::GraphProp for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type EdgeType = petgraph::Directed;
        }

        impl<'a, Base> pv::NodeCount for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            fn node_count(&self) -> usize {
                HugrView::node_count(self)
            }
        }

        impl<'a, Base> pv::NodeIndexable for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
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

        impl<'a, Base> pv::EdgeCount for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            fn edge_count(&self) -> usize {
                HugrView::edge_count(self)
            }
        }

        impl<'a, Base> pv::Data for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type NodeWeight = OpType;
            type EdgeWeight = EdgeKind;
        }

        impl<'g, 'a, Base> pv::IntoNodeIdentifiers for &'g $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type NodeIdentifiers = <$hugr<'a, Base> as HugrView>::Nodes<'g>;

            fn node_identifiers(self) -> Self::NodeIdentifiers {
                self.nodes()
            }
        }

        impl<'g, 'a, Base> pv::IntoNodeReferences for &'g $hugr<'a, Base>
        where
            'g: 'a,
            Base: HugrInternals + HugrView,
        {
            type NodeRef = HugrNodeRef<'a>;
            type NodeReferences =
                MapWithCtx<<$hugr<'a, Base> as HugrView>::Nodes<'a>, Self, HugrNodeRef<'a>>;

            fn node_references(self) -> Self::NodeReferences {
                self.nodes()
                    .with_context(self)
                    .map_with_context(|n, &hugr| HugrNodeRef::from_node(n, hugr))
            }
        }

        impl<'g, 'a, Base> pv::IntoNeighbors for &'g $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type Neighbors = <$hugr<'a, Base> as HugrView>::Neighbours<'g>;

            fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
                self.output_neighbours(n)
            }
        }

        impl<'g, 'a, Base> pv::IntoNeighborsDirected for &'g $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type NeighborsDirected = <$hugr<'a, Base> as HugrView>::Neighbours<'g>;

            fn neighbors_directed(
                self,
                n: Self::NodeId,
                d: petgraph::Direction,
            ) -> Self::NeighborsDirected {
                self.neighbours(n, d.into())
            }
        }

        impl<'a, Base> pv::Visitable for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
        {
            type Map = std::collections::HashSet<Self::NodeId>;

            fn visit_map(&self) -> Self::Map {
                std::collections::HashSet::new()
            }

            fn reset_map(&self, map: &mut Self::Map) {
                map.clear();
            }
        }

        impl<'a, Base> pv::GetAdjacencyMatrix for $hugr<'a, Base>
        where
            Base: HugrInternals + HugrView,
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
    };
}

impl_region_petgraph_traits!(FlatRegionView);
impl_region_petgraph_traits!(RegionView);

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

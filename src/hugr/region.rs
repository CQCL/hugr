//! Region-level views of a HUGR.

pub mod petgraph;

use context_iterators::{ContextIterator, IntoContextIterator, MapCtx, WithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{
    multiportgraph::SubportIndex, Hierarchy, LinkView, MultiPortGraph, PortView, UnmanagedDenseMap,
};

use crate::{ops::OpType, Direction, Hugr, Node, Port};

use super::HugrView;

type FlatRegionGraph<'g> = portgraph::view::FlatRegion<'g, MultiPortGraph>;

/// Single region view of a HUGR. Ignores any nodes that are not direct children
/// of the root.
///
/// For a view that includes all the descendants of the root, see [`RegionView`].
#[derive(Clone, Debug)]
pub struct FlatRegionView<'g> {
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: FlatRegionGraph<'g>,

    /// The node hierarchy.
    hierarchy: &'g Hierarchy,

    /// Operation types for each node.
    op_types: &'g UnmanagedDenseMap<portgraph::NodeIndex, OpType>,
}

impl<'g> FlatRegionView<'g> {
    /// Create a new region view of a HUGR containing only the direct children
    /// of the root.
    pub fn new(hugr: &'g Hugr, root: Node) -> Self {
        let Hugr {
            graph,
            hierarchy,
            op_types,
            ..
        } = hugr;
        Self {
            root,
            graph: FlatRegionGraph::new_flat_region(graph, hierarchy, root.index),
            hierarchy,
            op_types,
        }
    }
}

impl<'g> HugrView for FlatRegionView<'g> {
    type Nodes<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<FlatRegionGraph<'g> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<FlatRegionGraph<'g> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapCtx<
        WithCtx<<FlatRegionGraph<'g> as LinkView>::PortLinks<'a>, &'a Self>,
        fn((SubportIndex, SubportIndex), &&'a Self) -> (Node, Port),
    > where
        Self: 'a;

    fn root(&self) -> Node {
        self.root
    }

    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hierarchy
            .parent(node.index)
            .map(Into::into)
            .filter(|&n| n == self.root)
    }

    fn get_optype(&self, node: Node) -> &OpType {
        self.op_types.get(node.index)
    }

    fn node_count(&self) -> usize {
        self.hierarchy.child_count(self.root.index) + 1
    }

    fn edge_count(&self) -> usize {
        // Faster implementation than filtering all the nodes in the internal graph.
        self.nodes()
            .map(|n| self.output_neighbours(n).count())
            .sum()
    }

    fn nodes(&self) -> Self::Nodes<'_> {
        // Faster implementation than filtering all the nodes in the internal graph.
        self.hierarchy.children(self.root.index).map_into()
    }

    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.graph.port_offsets(node.index, dir).map_into()
    }

    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.graph.all_port_offsets(node.index).map_into()
    }

    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        let port = self.graph.port_index(node.index, port.offset).unwrap();
        self.graph
            .port_links(port)
            .with_context(self)
            .map_with_context(|(_, link), region| {
                let port = link.port();
                let node = region.graph.port_node(port).unwrap();
                let offset = region.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.index, dir)
    }

    fn children(&self, node: Node) -> Self::Children<'_> {
        let mut iter = self.hierarchy.children(node.index).map_into();
        if node != self.root {
            // Eagerly empty the iterator.
            // Ideally we would construct an empty iterator directly, but
            // `Children` is not `Default`.
            while iter.next().is_some() {}
        }
        iter
    }

    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.graph.neighbours(node.index, dir).map_into()
    }

    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.graph.all_neighbours(node.index).map_into()
    }
}

type RegionGraph<'g> = portgraph::view::Region<'g, MultiPortGraph>;

/// Single region view of a HUGR. Ignores any nodes that are not descendants of
/// the root.
///
/// For a view that includes only the direct children of the root, see
/// [`FlatRegionView`]. Prefer using [`FlatRegionView`] over this type when
/// possible, as it is more efficient.
#[derive(Clone, Debug)]
pub struct RegionView<'g> {
    /// The chosen root node.
    root: Node,

    /// The graph encoding the adjacency structure of the HUGR.
    graph: RegionGraph<'g>,

    /// The node hierarchy.
    hierarchy: &'g Hierarchy,

    /// Operation types for each node.
    op_types: &'g UnmanagedDenseMap<portgraph::NodeIndex, OpType>,
}

impl<'g> RegionView<'g> {
    /// Create a new region view of a HUGR containing only the direct children
    /// of the root.
    pub fn new(hugr: &'g Hugr, root: Node) -> Self {
        let Hugr {
            graph,
            hierarchy,
            op_types,
            ..
        } = hugr;
        Self {
            root,
            graph: RegionGraph::new_region(graph, hierarchy, root.index),
            hierarchy,
            op_types,
        }
    }
}

impl<'g> HugrView for RegionView<'g> {
    type Nodes<'a> = MapInto<<RegionGraph<'g> as PortView>::Nodes<'a>, Node>
    where
        Self: 'a;

    type NodePorts<'a> = MapInto<<RegionGraph<'g> as PortView>::NodePortOffsets<'a>, Port>
    where
        Self: 'a;

    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>
    where
        Self: 'a;

    type Neighbours<'a> = MapInto<<RegionGraph<'g> as LinkView>::Neighbours<'a>, Node>
    where
        Self: 'a;

    type PortLinks<'a> = MapCtx<
        WithCtx<<RegionGraph<'g> as LinkView>::PortLinks<'a>, &'a Self>,
        fn((SubportIndex, SubportIndex), &&'a Self) -> (Node, Port),
    > where
        Self: 'a;

    fn root(&self) -> Node {
        self.root
    }

    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hierarchy
            .parent(node.index)
            .filter(|&parent| self.graph.contains_node(parent))
            .map(Into::into)
    }

    fn get_optype(&self, node: Node) -> &OpType {
        self.op_types.get(node.index)
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn edge_count(&self) -> usize {
        self.graph.link_count()
    }

    fn nodes(&self) -> Self::Nodes<'_> {
        self.graph.nodes_iter().map_into()
    }

    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.graph.port_offsets(node.index, dir).map_into()
    }

    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.graph.all_port_offsets(node.index).map_into()
    }

    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        let port = self.graph.port_index(node.index, port.offset).unwrap();
        self.graph
            .port_links(port)
            .with_context(self)
            .map_with_context(|(_, link), region| {
                let port = link.port();
                let node = region.graph.port_node(port).unwrap();
                let offset = region.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.index, dir)
    }

    fn children(&self, node: Node) -> Self::Children<'_> {
        let mut iter = self.hierarchy.children(node.index).map_into();
        if !self.graph.contains_node(node.index) {
            // Eagerly empty the iterator.
            // Ideally we would construct an empty iterator directly, but
            // `Children` is not `Default`.
            while iter.next().is_some() {}
        }
        iter
    }

    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.graph.neighbours(node.index, dir).map_into()
    }

    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.graph.all_neighbours(node.index).map_into()
    }
}

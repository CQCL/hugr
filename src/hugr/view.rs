#![allow(unused)]
//! A Trait for "read-only" HUGRs.

use std::iter::FusedIterator;
use std::ops::Deref;

use context_iterators::{ContextIterator, IntoContextIterator, MapCtx, MapWithCtx, WithCtx};
use itertools::{Itertools, MapInto};
use portgraph::{multiportgraph, LinkView, MultiPortGraph, PortView};

use super::Hugr;
use super::{Node, Port};
use crate::ops::OpType;
use crate::Direction;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView {
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a>: Iterator<Item = Port>
    where
        Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a>: Iterator<Item = (Node, Port)>
    where
        Self: 'a;

    /// Return index of HUGR root node.
    fn root(&self) -> Node;

    /// Return the type of the HUGR root node.
    fn root_type(&self) -> &OpType {
        self.get_optype(self.root())
    }

    /// Returns the parent of a node.
    fn get_parent(&self, node: Node) -> Option<Node>;

    /// Returns the operation type of a node.
    fn get_optype(&self, node: Node) -> &OpType;

    /// Returns the number of nodes in the hugr.
    fn node_count(&self) -> usize;

    /// Returns the number of edges in the hugr.
    fn edge_count(&self) -> usize;

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> Self::Nodes<'_>;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_>;

    /// Iterator over output ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn node_outputs(&self, node: Node) -> Self::NodePorts<'_> {
        self.node_ports(node, Direction::Outgoing)
    }

    /// Iterator over inputs ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn node_inputs(&self, node: Node) -> Self::NodePorts<'_> {
        self.node_ports(node, Direction::Incoming)
    }

    /// Iterator over both the input and output ports of node.
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_>;

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_>;

    /// Returns whether a port is connected.
    fn is_linked(&self, node: Node, port: Port) -> bool {
        self.linked_ports(node, port).next().is_some()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Node, dir: Direction) -> usize;

    /// Number of inputs to a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Incoming)
    }

    /// Number of outputs from a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Outgoing)
    }

    /// Return iterator over children of node.
    fn children(&self, node: Node) -> Self::Children<'_>;

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_>;

    /// Iterates over the input neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Incoming)`.
    #[inline]
    fn input_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Incoming)
    }

    /// Iterates over the output neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Outgoing)`.
    #[inline]
    fn output_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Outgoing)
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_>;
}

impl<T> HugrView for T
where
    T: AsRef<Hugr>,
{
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a> = MapInto<multiportgraph::Nodes<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a> = MapInto<portgraph::portgraph::NodePortOffsets, Port> where Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a> = MapInto<multiportgraph::Neighbours<'a>, Node> where Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a> = MapWithCtx<multiportgraph::PortLinks<'a>, &'a Hugr, (Node, Port)>
    where
        Self: 'a;

    #[inline]
    fn root(&self) -> Node {
        self.as_ref().root.into()
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.as_ref().hierarchy.parent(node.index).map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.as_ref().op_types.get(node.index)
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.as_ref().graph.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.as_ref().graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> Self::Nodes<'_> {
        self.as_ref().graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.as_ref().graph.port_offsets(node.index, dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.as_ref().graph.all_port_offsets(node.index).map_into()
    }

    #[inline]
    fn linked_ports(&self, node: Node, port: Port) -> Self::PortLinks<'_> {
        let hugr = self.as_ref();
        let port = hugr.graph.port_index(node.index, port.offset).unwrap();
        hugr.graph
            .port_links(port)
            .with_context(hugr)
            .map_with_context(|(_, link), hugr| {
                let port = link.port();
                let node = hugr.graph.port_node(port).unwrap();
                let offset = hugr.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.as_ref().graph.num_ports(node.index, dir)
    }

    #[inline]
    fn children(&self, node: Node) -> Self::Children<'_> {
        self.as_ref().hierarchy.children(node.index).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.as_ref().graph.neighbours(node.index, dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.as_ref().graph.all_neighbours(node.index).map_into()
    }
}

/// Internal trait for accessing the underlying portgraph of a hugr view.
pub trait AsPortgraph: HugrView + sealed::AsPortgraph {}

impl<T> AsPortgraph for T where T: AsRef<Hugr> {}

pub(crate) mod sealed {
    use super::*;

    pub trait AsPortgraph {
        /// The underlying portgraph view type.
        type Portgraph: LinkView;

        /// Returns a reference to the underlying portgraph.
        fn as_portgraph(&self) -> &Self::Portgraph;
    }

    impl<T> AsPortgraph for T
    where
        T: AsRef<super::Hugr>,
    {
        type Portgraph = MultiPortGraph;

        fn as_portgraph(&self) -> &Self::Portgraph {
            &self.as_ref().graph
        }
    }
}

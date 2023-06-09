#![allow(unused)]
//! A Trait for "read-only" HUGRs.

use std::iter::FusedIterator;
use std::ops::Deref;

use itertools::{Itertools, MapInto};

use super::Hugr;
use super::{Node, Port};
use crate::ops::OpType;
use crate::Direction;

/// An Iterator over the nodes in a Hugr(View)
pub type Nodes<'a> = MapInto<crate::hugr::multiportgraph::Nodes<'a>, Node>;

/// An Iterator over (some or all) ports of a node
pub type NodePorts = MapInto<portgraph::portgraph::NodePortOffsets, Port>;

/// An Iterator over the children of a node
pub type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>;

/// An Iterator over (some or all) the nodes neighbouring a node
pub type Neighbours<'a> = MapInto<crate::hugr::multiportgraph::Neighbours<'a>, Node>;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView {
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
    fn nodes(&self) -> Nodes<'_>;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Node, dir: Direction) -> NodePorts;

    /// Iterator over output ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`.
    fn node_outputs(&self, node: Node) -> NodePorts;

    /// Iterator over inputs ports of node.
    /// Shorthand for [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`.
    fn node_inputs(&self, node: Node) -> NodePorts;

    /// Iterator over both the input and output ports of node.
    fn all_node_ports(&self, node: Node) -> NodePorts;

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(&self, node: Node, port: Port) -> PortLinks<'_>;

    /// Returns whether a port is connected.
    fn is_linked(&self, node: Node, port: Port) -> bool {
        self.linked_ports(node, port).next().is_some()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Node, dir: Direction) -> usize;

    /// Number of inputs to a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Incoming)`.
    fn num_inputs(&self, node: Node) -> usize;

    /// Number of outputs from a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Outgoing)`.
    fn num_outputs(&self, node: Node) -> usize;

    /// Return iterator over children of node.
    fn children(&self, node: Node) -> Children<'_>;

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(&self, node: Node, dir: Direction) -> Neighbours<'_>;

    /// Iterates over the input neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Incoming)`.
    fn input_neighbours(&self, node: Node) -> Neighbours<'_>;

    /// Iterates over the output neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Outgoing)`.
    fn output_neighbours(&self, node: Node) -> Neighbours<'_>;

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Node) -> Neighbours<'_>;
}

impl<T> HugrView for T
where
    T: DerefHugr,
{
    #[inline]
    fn root(&self) -> Node {
        self.hugr().root.into()
    }

    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hugr().hierarchy.parent(node.index).map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.hugr().op_types.get(node.index)
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.hugr().graph.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.hugr().graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> Nodes<'_> {
        self.hugr().graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> NodePorts {
        self.hugr().graph.port_offsets(node.index, dir).map_into()
    }

    #[inline]
    fn node_outputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.output_offsets(node.index).map_into()
    }

    #[inline]
    fn node_inputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.input_offsets(node.index).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> NodePorts {
        self.hugr().graph.all_port_offsets(node.index).map_into()
    }

    #[inline]
    fn linked_ports(&self, node: Node, port: Port) -> PortLinks<'_> {
        let hugr = self.hugr();
        let port = hugr.graph.port_index(node.index, port.offset).unwrap();
        let links = hugr.graph.port_links(port);
        PortLinks { hugr, links }
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.hugr().graph.num_ports(node.index, dir)
    }

    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.hugr().graph.num_inputs(node.index)
    }

    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.hugr().graph.num_outputs(node.index)
    }

    #[inline]
    fn children(&self, node: Node) -> Children<'_> {
        self.hugr().hierarchy.children(node.index).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Neighbours<'_> {
        self.hugr().graph.neighbours(node.index, dir).map_into()
    }

    #[inline]
    fn input_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.input_neighbours(node.index).map_into()
    }

    #[inline]
    fn output_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.output_neighbours(node.index).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.all_neighbours(node.index).map_into()
    }
}

/// Trait for things that can be converted into a reference to a Hugr.
///
/// This is equivalent to `Deref<Target=Hugr>`, but we use a local definition to
/// be able to write blanket implementations.
pub(crate) trait DerefHugr {
    fn hugr(&self) -> &Hugr;
}

impl DerefHugr for Hugr {
    fn hugr(&self) -> &Hugr {
        self
    }
}

impl<T> DerefHugr for T
where
    T: Deref<Target = Hugr>,
{
    fn hugr(&self) -> &Hugr {
        self.deref()
    }
}

/// Iterator over the links of a port
#[derive(Clone)]
pub struct PortLinks<'a> {
    hugr: &'a Hugr,
    links: crate::hugr::multiportgraph::PortLinks<'a>,
}

impl<'a> Iterator for PortLinks<'a> {
    type Item = (Node, Port);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.links.next().map(|(_, link)| {
            let port = link.port();
            let node = self.hugr.graph.port_node(port).unwrap();
            let offset = self.hugr.graph.port_offset(port).unwrap();
            (node.into(), offset.into())
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.links.size_hint()
    }
}

impl<'a> DoubleEndedIterator for PortLinks<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.links.next_back().map(|(_, link)| {
            let port = link.port();
            let node = self.hugr.graph.port_node(port).unwrap();
            let offset = self.hugr.graph.port_offset(port).unwrap();
            (node.into(), offset.into())
        })
    }
}

impl<'a> FusedIterator for PortLinks<'a> {}

#![allow(unused)]
//! Internal API for HUGRs, not intended for use by users.

use std::ops::Deref;

use itertools::{Itertools, MapInto};

use super::Hugr;
use super::{Node, Port};
use crate::ops::OpType;
use crate::Direction;

type Nodes<'a> = MapInto<portgraph::portgraph::Nodes<'a>, Node>;
type NodePorts = MapInto<portgraph::portgraph::NodePortOffsets, Port>;
type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>;
type Neighbours<'a> = MapInto<portgraph::portgraph::Neighbours<'a>, Node>;

/// Internal API for HUGRs, not intended for use by users.
///
/// TODO: Wraps the underlying graph and hierarchy, producing a view where
/// non-linear ports can be connected to multiple nodes via implicit copies
/// (which correspond to copy nodes in the internal graph).
pub(crate) trait HugrView: DerefHugr {
    /// Return index of HUGR root node.
    #[inline]
    fn root(&self) -> Node {
        self.hugr().root.into()
    }

    /// Returns the parent of a node.
    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.hugr().hierarchy.parent(node.index).map(Into::into)
    }

    /// Returns the operation type of a node.
    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        self.hugr().op_types.get(node.index)
    }

    /// Returns the number of nodes in the hugr.
    #[inline]
    fn node_count(&self) -> usize {
        self.hugr().graph.node_count()
    }

    /// Returns the number of edges in the hugr.
    #[inline]
    fn edge_count(&self) -> usize {
        self.hugr().graph.link_count()
    }

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> Nodes<'_> {
        self.hugr().graph.nodes_iter().map_into()
    }

    /// Iterator over ports of node in a given direction.
    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> NodePorts {
        self.hugr().graph.port_offsets(node.index, dir).map_into()
    }

    /// Iterator over output ports of node. Shorthand for [`Hugr::node_ports`].
    #[inline]
    fn node_outputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.output_offsets(node.index).map_into()
    }

    /// Iterator over inputs ports of node. Shorthand for [`Hugr::node_ports`].
    #[inline]
    fn node_inputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.input_offsets(node.index).map_into()
    }

    /// Iterator over both the input and output ports of node.
    #[inline]
    fn all_node_ports(&self, node: Node) -> NodePorts {
        self.hugr().graph.all_port_offsets(node.index).map_into()
    }

    /// Return node and port connected to provided port, if not connected return None.
    #[inline]
    fn linked_port(&self, node: Node, port: Port) -> Option<(Node, Port)> {
        let raw = self.hugr();
        let port = raw.graph.port_index(node.index, port.offset)?;
        let link = raw.graph.port_link(port)?;
        Some((
            raw.graph.port_node(link).map(Into::into)?,
            raw.graph.port_offset(port).map(Into::into)?,
        ))
    }

    /// Number of ports in node for a given direction.
    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.hugr().graph.num_ports(node.index, dir)
    }

    /// Number of inputs to a node. Shorthand for [`Hugr::num_ports`].
    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.hugr().graph.num_inputs(node.index)
    }

    /// Number of outputs from a node. Shorthand for [`Hugr::num_ports`].
    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.hugr().graph.num_outputs(node.index)
    }

    /// Return iterator over children of node.
    #[inline]
    fn children(&self, node: Node) -> Children<'_> {
        self.hugr().hierarchy.children(node.index).map_into()
    }

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Neighbours<'_> {
        self.hugr().graph.neighbours(node.index, dir).map_into()
    }

    /// Iterates over the input neighbours of the `node`. Shorthand for [`Hugr::neighbours`].
    #[inline]
    fn input_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.input_neighbours(node.index).map_into()
    }

    /// Iterates over the output neighbours of the `node`. Shorthand for [`Hugr::neighbours`].
    #[inline]
    fn output_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.output_neighbours(node.index).map_into()
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    #[inline]
    fn all_neighbours(&self, node: Node) -> Neighbours<'_> {
        self.hugr().graph.all_neighbours(node.index).map_into()
    }
}

impl<T> HugrView for T where T: DerefHugr {}

/// Trait for converting a reference to a Hugr into a RawHugr.
///
/// This is equivalent to `Deref<Target=Hugr>`, but we use a local definition to
/// be able to write blanket implementations.
pub(crate) trait DerefHugr: Sized {
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

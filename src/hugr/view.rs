//! Internal API for HUGRs, not intended for use by users.

use std::ops::Deref;

use itertools::{Itertools, MapInto};
use portgraph::portgraph::NodePorts;

use super::Hugr;
use super::{Node, Port};
use crate::ops::OpType;
use crate::Direction;

/// A type of Iterator over children of a node
pub type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node>;

/// A type of Iterator over nodes neighbouring another in a single direction
pub type Neighbours<'a> = MapInto<portgraph::portgraph::Neighbours<'a>, Node>;

/// Internal API for HUGRs, not intended for use by users.
///
/// TODO: Wraps the underlying graph and hierarchy, producing a view where
/// non-linear ports can be connected to multiple nodes via implicit copies
/// (which correspond to copy nodes in the internal graph).
pub trait HugrView: DerefHugr {
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

    /// Iterator over outputs of node.
    ///
    /// TODO: In the future this will return hugr ports, not `PortIndices`.
    #[inline]
    fn node_outputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.outputs(node.index)
    }

    /// Iterator over inputs of node.
    ///
    /// TODO: In the future this will return hugr ports, not `PortIndices`.
    #[inline]
    fn node_inputs(&self, node: Node) -> NodePorts {
        self.hugr().graph.inputs(node.index)
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

    /// Number of inputs to node.
    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.hugr().graph.num_inputs(node.index)
    }

    /// Number of outputs to node.
    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.hugr().graph.num_outputs(node.index)
    }

    /// Return iterator over children of node.
    #[inline]
    fn children(&self, node: Node) -> Children<'_> {
        self.hugr().hierarchy.children(node.index).map_into()
    }

    /// Return iterator over neighbouring nodes, in a particular direction
    fn neighbours(&self, node: Node, dir: Direction) -> Neighbours<'_> {
        self.hugr().graph.neighbours(node.index, dir).map_into()
    }
}

impl<T> HugrView for T where T: DerefHugr {}

/// Trait for things that can be converted into a reference to a Hugr.
///
/// This is equivalent to `Deref<Target=Hugr>`, but we use a local definition to
/// be able to write blanket implementations.
pub trait DerefHugr {
    /// Gets the reference. Same idea as `Deref::deref()`
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

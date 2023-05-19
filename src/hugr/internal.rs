//! Internal API for HUGRs, not intended for use by users.

use std::ops::Deref;

use portgraph::hierarchy::Children;
use portgraph::portgraph::NodePorts;
use portgraph::{NodeIndex, PortOffset};

use super::Hugr;
use crate::ops::OpType;

/// Internal API for HUGRs, not intended for use by users.
///
/// TODO: Wraps the underlying graph and hierarchy, producing a view where
/// non-linear ports can be connected to multiple nodes via implicit copies
/// (which correspond to copy nodes in the internal graph).
pub(crate) trait HugrView: DerefHugr {
    /// Returns the parent of a node.
    #[inline]
    fn get_parent(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.hugr().hierarchy.parent(node)
    }

    /// Returns the operation type of a node.
    #[inline]
    fn get_optype(&self, node: NodeIndex) -> &OpType {
        self.hugr().op_types.get(node)
    }

    /// Iterator over outputs of node.
    #[inline]
    fn node_outputs(&self, node: NodeIndex) -> NodePorts {
        self.hugr().graph.outputs(node)
    }

    /// Iterator over inputs of node.
    #[inline]
    fn node_inputs(&self, node: NodeIndex) -> NodePorts {
        self.hugr().graph.inputs(node)
    }

    /// Return node and port connected to provided port, if not connected return None.
    #[inline]
    fn linked_port(&self, node: NodeIndex, offset: PortOffset) -> Option<(NodeIndex, PortOffset)> {
        let raw = self.hugr();
        let port = raw.graph.port_index(node, offset)?;
        let link = raw.graph.port_link(port)?;
        Some((raw.graph.port_node(link)?, raw.graph.port_offset(port)?))
    }

    /// Number of inputs to node.
    #[inline]
    fn num_inputs(&self, node: NodeIndex) -> usize {
        self.hugr().graph.num_inputs(node)
    }

    /// Number of outputs to node.
    #[inline]
    fn num_outputs(&self, node: NodeIndex) -> usize {
        self.hugr().graph.num_outputs(node)
    }

    /// Return iterator over children of node.
    #[inline]
    fn children(&self, node: NodeIndex) -> Children<'_> {
        self.hugr().hierarchy.children(node)
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

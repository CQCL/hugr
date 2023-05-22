//! The Hugr data structure, and its basic component handles.

mod hugrmut;
pub(crate) mod internal;

pub mod serialize;
pub mod validate;

use derive_more::From;
pub use hugrmut::HugrMut;
pub use validate::ValidationError;

use portgraph::dot::{hier_graph_dot_string_with, DotEdgeStyle};
use portgraph::{Hierarchy, PortGraph, SecondaryMap};
use thiserror::Error;

use self::internal::HugrView;
use crate::ops::{ModuleOp, OpType};
use crate::rewrite::{Rewrite, RewriteError};
use crate::types::EdgeKind;

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    graph: PortGraph,

    /// The node hierarchy.
    hierarchy: Hierarchy,

    /// The single root node in the hierarchy.
    /// It must correspond to a [`ModuleOp::Root`] node.
    ///
    /// [`ModuleOp::Root`]: crate::ops::ModuleOp::Root.
    root: portgraph::NodeIndex,

    /// Operation types for each node.
    op_types: SecondaryMap<portgraph::NodeIndex, OpType>,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(ModuleOp::Root)
    }
}

/// A handle to a node in the HUGR.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, From)]
pub struct Node(portgraph::NodeIndex);

/// A handle to a port for a node in the HUGR.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug, From)]
pub struct Port(portgraph::PortOffset);

/// The direction of a port.
pub type Direction = portgraph::Direction;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

/// Public API for HUGRs.
impl Hugr {
    /// Returns an immutable view over the graph.
    pub fn view(&self) {
        unimplemented!()
    }

    /// Applies a rewrite to the graph.
    pub fn apply_rewrite(mut self, rewrite: Rewrite) -> Result<(), RewriteError> {
        // Get the open graph for the rewrites, and a HUGR with the additional components.
        let (rewrite, mut replacement, parents) = rewrite.into_parts();

        // TODO: Use `parents` to update the hierarchy, and keep the internal hierarchy from `replacement`.
        let _ = parents;

        let node_inserted = |old, new| {
            std::mem::swap(&mut self.op_types[new], &mut replacement.op_types[old]);
            // TODO: metadata (Fn parameter ?)
        };
        rewrite.apply_with_callbacks(
            &mut self.graph,
            |_| {},
            |_| {},
            node_inserted,
            |_, _| {},
            |_, _| {},
        )?;

        // TODO: Check types

        Ok(())
    }

    /// Return dot string showing underlying graph and hierarchy side by side.
    pub fn dot_string(&self) -> String {
        hier_graph_dot_string_with(
            &self.graph,
            &self.hierarchy,
            |n| {
                format!(
                    "({ni}) {name}",
                    name = self.op_types[n].name(),
                    ni = n.index()
                )
            },
            |p| {
                let src = self.graph.port_node(p).unwrap();
                let Some(tgt_port) = self.graph.port_link(p) else {
                        return ("".into(), DotEdgeStyle::None);
                    };
                let tgt = self.graph.port_node(tgt_port).unwrap();
                let style = if self.hierarchy.parent(src) != self.hierarchy.parent(tgt) {
                    DotEdgeStyle::Some("dashed".into())
                } else if self
                    .get_optype(src.into())
                    .port_kind(self.graph.port_offset(p).unwrap())
                    == Some(EdgeKind::StateOrder)
                {
                    DotEdgeStyle::Some("dotted".into())
                } else {
                    DotEdgeStyle::None
                };

                ("".into(), style)
            },
        )
    }
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new(root_op: impl Into<OpType>) -> Self {
        let mut graph = PortGraph::default();
        let hierarchy = Hierarchy::new();
        let mut op_types = SecondaryMap::new();
        let root = graph.add_node(0, 0);
        op_types[root] = root_op.into();

        Self {
            graph,
            hierarchy,
            root,
            op_types,
        }
    }
}

impl Port {
    /// Creates a new port offset.
    #[inline]
    pub fn new(direction: Direction, offset: usize) -> Self {
        Self(portgraph::PortOffset::new(direction, offset))
    }

    /// Creates a new incoming port offset.
    #[inline]
    pub fn new_incoming(offset: usize) -> Self {
        Self(portgraph::PortOffset::new_incoming(offset))
    }

    /// Creates a new outgoing port offset.
    #[inline]
    pub fn new_outgoing(offset: usize) -> Self {
        Self(portgraph::PortOffset::new_outgoing(offset))
    }

    /// Returns the direction of the port.
    #[inline]
    pub fn direction(self) -> Direction {
        self.0.direction()
    }

    /// Returns the offset of the port.
    #[inline(always)]
    pub fn index(self) -> usize {
        self.0.index()
    }
}

impl Wire {
    /// Create a new wire from a node and an offset.
    pub fn new(node: Node, offset: usize) -> Self {
        Self(node, offset)
    }

    /// The node that this wire is connected to.
    pub fn node(&self) -> Node {
        self.0
    }

    /// The offset of the output port that this wire is connected to.
    pub fn source(&self) -> Port {
        Port::new_outgoing(self.1)
    }
}

/// Errors that can occur while manipulating a Hugr.
///
/// TODO: Better descriptions, not just re-exporting portgraph errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HugrError {
    /// An error occurred while connecting nodes.
    #[error("An error occurred while connecting the nodes.")]
    ConnectionError(#[from] portgraph::LinkError),
    /// An error occurred while manipulating the hierarchy.
    #[error("An error occurred while manipulating the hierarchy.")]
    HierarchyError(#[from] portgraph::hierarchy::AttachError),
}

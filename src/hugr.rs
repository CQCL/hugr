//! The Hugr data structure, and its basic component handles.

mod hugrmut;

pub mod rewrite;
pub mod serialize;
pub mod typecheck;
pub mod validate;
pub mod view;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

use derive_more::From;
pub use rewrite::{Replace, ReplaceError, Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::dot::{DotFormat, EdgeStyle, NodeStyle, PortStyle};
use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, LinkView, PortMut, PortView, UnmanagedDenseMap};
use thiserror::Error;

pub use self::view::HugrView;
use crate::ops::{OpName, OpType};
use crate::types::EdgeKind;

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    graph: MultiPortGraph,

    /// The node hierarchy.
    hierarchy: Hierarchy,

    /// The single root node in the hierarchy.
    root: portgraph::NodeIndex,

    /// Operation types for each node.
    op_types: UnmanagedDenseMap<portgraph::NodeIndex, OpType>,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(crate::ops::Module)
    }
}

impl AsRef<Hugr> for Hugr {
    fn as_ref(&self) -> &Hugr {
        self
    }
}

impl AsMut<Hugr> for Hugr {
    fn as_mut(&mut self) -> &mut Hugr {
        self
    }
}

/// A handle to a node in the HUGR.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    From,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Node {
    index: portgraph::NodeIndex,
}

/// A handle to a port for a node in the HUGR.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug, From)]
pub struct Port {
    offset: portgraph::PortOffset,
}

/// The direction of a port.
pub type Direction = portgraph::Direction;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

/// Public API for HUGRs.
impl Hugr {
    /// Applies a rewrite to the graph.
    pub fn apply_rewrite<E>(&mut self, rw: impl Rewrite<Error = E>) -> Result<(), E> {
        rw.apply(self)
    }

    /// Return dot string showing underlying graph and hierarchy side by side.
    pub fn dot_string(&self) -> String {
        self.graph
            .dot_format()
            .with_hierarchy(&self.hierarchy)
            .with_node_style(|n| {
                NodeStyle::Box(format!(
                    "({ni}) {name}",
                    ni = n.index(),
                    name = self.op_types[n].name()
                ))
            })
            .with_port_style(|port| {
                let node = self.graph.port_node(port).unwrap();
                let optype = self.op_types.get(node);
                let offset = self.graph.port_offset(port).unwrap();
                match optype.port_kind(offset).unwrap() {
                    EdgeKind::Const(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::Value(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::StateOrder => match self.graph.port_links(port).count() > 0 {
                        true => PortStyle::text("", false),
                        false => PortStyle::Hidden,
                    },
                    _ => PortStyle::text("", true),
                }
            })
            .with_edge_style(|src, tgt| {
                let src_node = self.graph.port_node(src).unwrap();
                let src_optype = self.op_types.get(src_node);
                let src_offset = self.graph.port_offset(src).unwrap();
                let tgt_node = self.graph.port_node(tgt).unwrap();

                if self.hierarchy.parent(src_node) != self.hierarchy.parent(tgt_node) {
                    EdgeStyle::Dashed
                } else if src_optype.port_kind(src_offset) == Some(EdgeKind::StateOrder) {
                    EdgeStyle::Dotted
                } else {
                    EdgeStyle::Solid
                }
            })
            .finish()
    }
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new(root_op: impl Into<OpType>) -> Self {
        Self::with_capacity(root_op, 0, 0)
    }

    /// Create a new Hugr, with a single root node and preallocated capacity.
    pub(crate) fn with_capacity(root_op: impl Into<OpType>, nodes: usize, ports: usize) -> Self {
        let mut graph = MultiPortGraph::with_capacity(nodes, ports);
        let hierarchy = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
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
    /// Creates a new port.
    #[inline]
    pub fn new(direction: Direction, port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new(direction, port),
        }
    }

    /// Creates a new incoming port.
    #[inline]
    pub fn new_incoming(port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new_incoming(port),
        }
    }

    /// Creates a new outgoing port.
    #[inline]
    pub fn new_outgoing(port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new_outgoing(port),
        }
    }

    /// Returns the direction of the port.
    #[inline]
    pub fn direction(self) -> Direction {
        self.offset.direction()
    }

    /// Returns the offset of the port.
    #[inline(always)]
    pub fn index(self) -> usize {
        self.offset.index()
    }
}

impl Wire {
    /// Create a new wire from a node and a port.
    #[inline]
    pub fn new(node: Node, port: Port) -> Self {
        Self(node, port.index())
    }

    /// The node that this wire is connected to.
    #[inline]
    pub fn node(&self) -> Node {
        self.0
    }

    /// The output port that this wire is connected to.
    #[inline]
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

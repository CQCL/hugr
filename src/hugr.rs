//! The Hugr data structure, and its basic component handles.

mod hugrmut;

pub mod region;
pub mod rewrite;
pub mod serialize;
pub mod typecheck;
pub mod validate;
pub mod view;

use std::collections::VecDeque;
use std::iter;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

use delegate::delegate;
use derive_more::From;
pub use rewrite::{Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, PortMut, UnmanagedDenseMap};
use thiserror::Error;

pub use self::view::HugrView;
use crate::ops::{tag::OpTag, OpTrait, OpType, ValidateOp};
use crate::resource::ResourceSet;
use crate::types::{AbstractSignature, EdgeKind, Signature, SignatureDescription};

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
    op_types: UnmanagedDenseMap<portgraph::NodeIndex, NodeType>,

    /// Node metadata
    metadata: UnmanagedDenseMap<portgraph::NodeIndex, NodeMetadata>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// The type of a node on a graph
pub struct NodeType {
    /// The underlying OpType
    pub op: OpType,
    /// The resources that the signature has been specialised to
    pub input_resources: ResourceSet,
}

impl NodeType {
    /// Instantiate an OpType with no resources
    pub fn pure(op: impl Into<OpType>) -> Self {
        NodeType {
            op: op.into(),
            input_resources: ResourceSet::new(),
        }
    }

    /// Use the input resources to calculate the concrete signature of the node
    pub fn signature(&self) -> Signature {
        self.op
            .op_signature()
            .with_input_resources(self.input_resources.clone())
    }
}

impl OpType {
    /// Convert an OpType to a NodeType by giving it some input resources
    pub fn with_resources(self, rs: ResourceSet) -> NodeType {
        NodeType {
            op: self,
            input_resources: rs,
        }
    }
}

impl OpType {
    /// The edge kind for the non-dataflow or constant-input ports of the
    /// operation, not described by the signature.
    ///
    /// If not None, a single extra multiport of that kind will be present on
    /// the given direction.
    pub fn other_port(&self, dir: Direction) -> Option<EdgeKind> {
        match dir {
            Direction::Incoming => self.other_input(),
            Direction::Outgoing => self.other_output(),
        }
    }

    /// Returns the edge kind for the given port.
    pub fn port_kind(&self, port: impl Into<Port>) -> Option<EdgeKind> {
        let signature = self.op_signature();
        let port = port.into();
        let dir = port.direction();
        match port.index() < signature.port_count(dir) {
            true => signature.get(port),
            false => self.other_port(dir),
        }
    }

    /// The non-dataflow port for the operation, not described by the signature.
    /// See `[OpType::other_port]`.
    ///
    /// Returns None if there is no such port, or if the operation defines multiple non-dataflow ports.
    pub fn other_port_index(&self, dir: Direction) -> Option<Port> {
        let non_df_count = self.validity_flags().non_df_port_count(dir).unwrap_or(1);
        if self.other_port(dir).is_some() && non_df_count == 1 {
            Some(Port::new(dir, self.op_signature().port_count(dir)))
        } else {
            None
        }
    }

    /// Returns the number of ports for the given direction.
    pub fn port_count(&self, dir: Direction) -> usize {
        let signature = self.op_signature();
        let has_other_ports = self.other_port(dir).is_some();
        let non_df_count = self
            .validity_flags()
            .non_df_port_count(dir)
            .unwrap_or(has_other_ports as usize);
        signature.port_count(dir) + non_df_count
    }

    /// Returns the number of inputs ports for the operation.
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of outputs ports for the operation.
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }
}

impl OpTrait for NodeType {
    delegate! {
        to self.op {
            fn description(&self) -> &str;
            fn tag(&self) -> OpTag;
            fn op_signature(&self) -> AbstractSignature;
            fn signature_desc(&self) -> SignatureDescription;
            fn other_input(&self) -> Option<EdgeKind>;
            fn other_output(&self) -> Option<EdgeKind>;
        }
    }
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

/// Public API for HUGRs.
impl Hugr {
    /// Applies a rewrite to the graph.
    pub fn apply_rewrite<E>(&mut self, rw: impl Rewrite<Error = E>) -> Result<(), E> {
        rw.apply(self)
    }
}

/// Arbitrary metadata for a node.
pub type NodeMetadata = serde_json::Value;

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new(root_op: impl Into<OpType>) -> Self {
        Self::with_capacity(root_op, 0, 0)
    }

    /// Create a new Hugr, with a single root node and preallocated capacity.
    // TODO: Make this take a NodeType
    pub(crate) fn with_capacity(root_op: impl Into<OpType>, nodes: usize, ports: usize) -> Self {
        let mut graph = MultiPortGraph::with_capacity(nodes, ports);
        let hierarchy = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
        let root = graph.add_node(0, 0);
        op_types[root] = NodeType {
            op: root_op.into(),
            input_resources: ResourceSet::new(),
        };

        Self {
            graph,
            hierarchy,
            root,
            op_types,
            metadata: UnmanagedDenseMap::with_capacity(nodes),
        }
    }

    /// Produce a canonical ordering of the nodes.
    ///
    /// Used by [`HugrMut::canonicalize_nodes`] and the serialization code.
    fn canonical_order(&self) -> impl Iterator<Item = Node> + '_ {
        // Generate a BFS-ordered list of nodes based on the hierarchy
        let mut queue = VecDeque::from([self.root.into()]);
        iter::from_fn(move || {
            let node = queue.pop_front()?;
            for child in self.children(node) {
                queue.push_back(child);
            }
            Some(node)
        })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

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

/// Enum for uniquely identifying the origin of linear wires in a circuit-like
/// dataflow region.
///
/// Falls back to [`Wire`] if the wire is not linear or if it's not possible to
/// track the origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CircuitUnit {
    /// Arbitrary input wire.
    Wire(Wire),
    /// Index to region input.
    Linear(usize),
}

impl From<usize> for CircuitUnit {
    fn from(value: usize) -> Self {
        CircuitUnit::Linear(value)
    }
}

impl From<Wire> for CircuitUnit {
    fn from(value: Wire) -> Self {
        CircuitUnit::Wire(value)
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

#[cfg(test)]
mod test {
    use super::Hugr;

    #[test]
    fn impls_send_and_sync() {
        // Send and Sync are automatically impl'd by the compiler, if possible.
        // This test will fail to compile if that wasn't possible.
        trait Test: Send + Sync {}
        impl Test for Hugr {}
    }
}

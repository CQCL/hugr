//! The Hugr data structure, and its basic component handles.

mod hugrmut;

pub mod rewrite;
pub mod serialize;
pub mod validate;
pub mod views;

use std::collections::VecDeque;
use std::iter;

pub(crate) use self::hugrmut::HugrInternalsMut;
pub use self::validate::ValidationError;

use derive_more::From;
pub use rewrite::{Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, PortMut, UnmanagedDenseMap};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use self::views::HugrView;
use crate::extension::{infer_extensions, ExtensionSet, ExtensionSolution, InferExtensionError};
use crate::ops::{OpTag, OpTrait, OpType};
use crate::types::{FunctionType, Signature};

use delegate::delegate;

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
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

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
/// The type of a node on a graph. In addition to the [`OpType`], it also
/// describes the extensions inferred to be used by the node.
pub struct NodeType {
    /// The underlying OpType
    op: OpType,
    /// The extensions that the signature has been specialised to
    input_extensions: Option<ExtensionSet>,
}

impl NodeType {
    /// Create a new optype with some ExtensionSet
    pub fn new(op: impl Into<OpType>, input_extensions: ExtensionSet) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: Some(input_extensions),
        }
    }

    /// Instantiate an OpType with no input extensions
    pub fn pure(op: impl Into<OpType>) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: Some(ExtensionSet::new()),
        }
    }

    /// Instantiate an OpType with an unknown set of input extensions
    /// (to be inferred later)
    pub fn open_extensions(op: impl Into<OpType>) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: None,
        }
    }

    /// Use the input extensions to calculate the concrete signature of the node
    pub fn signature(&self) -> Option<Signature> {
        self.input_extensions
            .as_ref()
            .map(|rs| self.op.signature().with_input_extensions(rs.clone()))
    }

    /// Get the function type from the embedded op
    pub fn op_signature(&self) -> FunctionType {
        self.op.signature()
    }

    /// The input extensions defined for this node.
    ///
    /// The output extensions will correspond to the input extensions plus any
    /// extension delta defined by the operation type.
    ///
    /// If the input extensions are not known, this will return None.
    pub fn input_extensions(&self) -> Option<&ExtensionSet> {
        self.input_extensions.as_ref()
    }
}

impl NodeType {
    delegate! {
        to self.op {
            /// Tag identifying the operation.
            pub fn tag(&self) -> OpTag;
            /// Returns the number of inputs ports for the operation.
            pub fn input_count(&self) -> usize;
            /// Returns the number of outputs ports for the operation.
            pub fn output_count(&self) -> usize;
        }
    }
}

impl<'a> From<&'a NodeType> for &'a OpType {
    fn from(nt: &'a NodeType) -> Self {
        &nt.op
    }
}

impl OpType {
    /// Convert an OpType to a NodeType by giving it some input extensions
    pub fn with_extensions(self, rs: ExtensionSet) -> NodeType {
        NodeType {
            op: self,
            input_extensions: Some(rs),
        }
    }
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(NodeType::pure(crate::ops::Module))
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
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Node {
    index: portgraph::NodeIndex,
}

/// A handle to a port for a node in the HUGR.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug, From)]
#[cfg_attr(feature = "pyo3", pyclass)]
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

    /// Infer extension requirements
    pub fn infer_extensions(&mut self) -> Result<(), InferExtensionError> {
        let solution = infer_extensions(self)?;
        self.instantiate_extensions(solution);
        Ok(())
    }

    /// TODO: Write this
    fn instantiate_extensions(&mut self, _solution: ExtensionSolution) {
        //todo!()
    }
}

/// Arbitrary metadata for a node.
pub type NodeMetadata = serde_json::Value;

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new(root_node: NodeType) -> Self {
        Self::with_capacity(root_node, 0, 0)
    }

    /// Create a new Hugr, with a single root node and preallocated capacity.
    // TODO: Make this take a NodeType
    pub(crate) fn with_capacity(root_node: NodeType, nodes: usize, ports: usize) -> Self {
        let mut graph = MultiPortGraph::with_capacity(nodes, ports);
        let hierarchy = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
        let root = graph.add_node(0, 0);
        // TODO: These extensions should be open in principle, but lets wait
        // until extensions can be inferred for open sets until changing this
        op_types[root] = root_node;

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

#[cfg(feature = "pyo3")]
impl From<HugrError> for PyErr {
    fn from(err: HugrError) -> Self {
        // We may want to define more specific python-level errors at some point.
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
    }
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

    #[test]
    fn io_node() {
        use crate::builder::test::simple_dfg_hugr;
        use crate::hugr::views::HugrView;
        use cool_asserts::assert_matches;

        let hugr = simple_dfg_hugr();
        assert_matches!(hugr.get_io(hugr.root()), Some(_));
    }
}

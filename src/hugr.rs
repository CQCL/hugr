//! The Hugr data structure, and its basic component handles.

pub mod hugrmut;

mod ident;
pub mod rewrite;
pub mod serialize;
pub mod validate;
pub mod views;

use std::collections::VecDeque;
use std::iter;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

use derive_more::From;
pub use ident::{IdentList, InvalidIdentifier};
pub use rewrite::{Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, NodeIndex, PortMut, UnmanagedDenseMap};
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::{create_exception, exceptions::PyException, pyclass, PyErr};

pub use self::views::HugrView;
use crate::extension::{
    infer_extensions, ExtensionRegistry, ExtensionSet, ExtensionSolution, InferExtensionError,
};
use crate::ops::{OpTag, OpTrait, OpType, DEFAULT_OPTYPE};
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

/// The default NodeType, with open extensions
pub const DEFAULT_NODETYPE: NodeType = NodeType {
    op: DEFAULT_OPTYPE,
    input_extensions: None, // Default for any Option
};

impl NodeType {
    /// Create a new optype with some ExtensionSet
    pub fn new(op: impl Into<OpType>, input_extensions: impl Into<Option<ExtensionSet>>) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: input_extensions.into(),
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
#[derive(
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Hash,
    Default,
    Debug,
    From,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Port {
    offset: portgraph::PortOffset,
}

/// The direction of a port.
pub type Direction = portgraph::Direction;

/// Public API for HUGRs.
impl Hugr {
    /// Run resource inference and pass the closure into validation
    pub fn infer_and_validate(
        &mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), ValidationError> {
        let closure = self.infer_extensions()?;
        self.validate_with_extension_closure(closure, extension_registry)?;
        Ok(())
    }

    /// Infer extension requirements and add new information to `op_types` field
    ///
    /// See [`infer_extensions`] for details on the "closure" value
    pub fn infer_extensions(&mut self) -> Result<ExtensionSolution, InferExtensionError> {
        let (solution, extension_closure) = infer_extensions(self)?;
        self.instantiate_extensions(solution);
        Ok(extension_closure)
    }

    /// Add extension requirement information to the hugr in place.
    fn instantiate_extensions(&mut self, solution: ExtensionSolution) {
        // We only care about inferred _input_ extensions, because `NodeType`
        // uses those to infer the output extensions
        for (node, input_extensions) in solution.iter() {
            let nodetype = self.op_types.try_get_mut(node.index).unwrap();
            match &nodetype.input_extensions {
                None => nodetype.input_extensions = Some(input_extensions.clone()),
                Some(existing_ext_reqs) => {
                    debug_assert_eq!(existing_ext_reqs, input_extensions)
                }
            }
        }
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

    /// Add a node to the graph, with the default conversion from OpType to NodeType
    pub(crate) fn add_op(&mut self, op: impl Into<OpType>) -> Node {
        // TODO: Default to `NodeType::open_extensions` once we can infer extensions
        self.add_node(NodeType::pure(op))
    }

    /// Add a node to the graph.
    pub(crate) fn add_node(&mut self, nodetype: NodeType) -> Node {
        let node = self
            .graph
            .add_node(nodetype.input_count(), nodetype.output_count());
        self.op_types[node] = nodetype;
        node.into()
    }

    /// Produce a canonical ordering of the descendant nodes of a root,
    /// following the graph hierarchy.
    ///
    /// This starts with the root, and then proceeds in BFS order through the
    /// contained regions.
    ///
    /// Used by [`HugrMut::canonicalize_nodes`] and the serialization code.
    fn canonical_order(&self, root: Node) -> impl Iterator<Item = Node> + '_ {
        // Generate a BFS-ordered list of nodes based on the hierarchy
        let mut queue = VecDeque::from([root]);
        iter::from_fn(move || {
            let node = queue.pop_front()?;
            for child in self.children(node) {
                queue.push_back(child);
            }
            Some(node)
        })
    }

    /// Compact the nodes indices of the hugr to be contiguous, and order them as a breadth-first
    /// traversal of the hierarchy.
    ///
    /// The rekey function is called for each moved node with the old and new indices.
    ///
    /// After this operation, a serialization and deserialization of the Hugr is guaranteed to
    /// preserve the indices.
    pub fn canonicalize_nodes(&mut self, mut rekey: impl FnMut(Node, Node)) {
        // Generate the ordered list of nodes
        let mut ordered = Vec::with_capacity(self.node_count());
        let root = self.root();
        ordered.extend(self.as_mut().canonical_order(root));

        // Permute the nodes in the graph to match the order.
        //
        // Invariant: All the elements before `position` are in the correct place.
        for position in 0..ordered.len() {
            // Find the element's location. If it originally came from a previous position
            // then it has been swapped somewhere else, so we follow the permutation chain.
            let mut source: Node = ordered[position];
            while position > source.index.index() {
                source = ordered[source.index.index()];
            }

            let target: Node = NodeIndex::new(position).into();
            if target != source {
                self.graph.swap_nodes(target.index, source.index);
                self.op_types.swap(target.index, source.index);
                self.hierarchy.swap_nodes(target.index, source.index);
                rekey(source, target);
            }
        }
        self.root = NodeIndex::new(0);

        // Finish by compacting the copy nodes.
        // The operation nodes will be left in place.
        // This step is not strictly necessary.
        self.graph.compact_nodes(|_, _| {});
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

impl CircuitUnit {
    /// Check if this is a wire.
    pub fn is_wire(&self) -> bool {
        matches!(self, CircuitUnit::Wire(_))
    }

    /// Check if this is a linear unit.
    pub fn is_linear(&self) -> bool {
        matches!(self, CircuitUnit::Linear(_))
    }
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
    /// The node doesn't exist.
    #[error("Invalid node {0:?}.")]
    InvalidNode(Node),
}

#[cfg(feature = "pyo3")]
create_exception!(
    pyrs,
    PyHugrError,
    PyException,
    "Errors that can occur while manipulating a Hugr"
);

#[cfg(feature = "pyo3")]
impl From<HugrError> for PyErr {
    fn from(err: HugrError) -> Self {
        PyHugrError::new_err(err.to_string())
    }
}

#[cfg(test)]
mod test {
    use super::{Hugr, HugrView, NodeType};
    use crate::builder::test::closed_dfg_root_hugr;
    use crate::extension::ExtensionSet;
    use crate::hugr::HugrMut;
    use crate::ops;
    use crate::type_row;
    use crate::types::{FunctionType, Type};

    use std::error::Error;

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

    #[test]
    fn extension_instantiation() -> Result<(), Box<dyn Error>> {
        const BIT: Type = crate::extension::prelude::USIZE_T;
        let r = ExtensionSet::singleton(&"R".try_into().unwrap());

        let mut hugr = closed_dfg_root_hugr(
            FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(&r),
        );
        let [input, output] = hugr.get_io(hugr.root()).unwrap();
        let lift = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::LeafOp::Lift {
                type_row: type_row![BIT],
                new_extension: "R".try_into().unwrap(),
            }),
        )?;
        hugr.connect(input, 0, lift, 0)?;
        hugr.connect(lift, 0, output, 0)?;
        hugr.infer_extensions()?;

        assert_eq!(
            hugr.get_nodetype(lift)
                .signature()
                .unwrap()
                .input_extensions,
            ExtensionSet::new()
        );
        assert_eq!(
            hugr.get_nodetype(output)
                .signature()
                .unwrap()
                .input_extensions,
            r
        );
        Ok(())
    }
}

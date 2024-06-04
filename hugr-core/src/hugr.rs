//! The Hugr data structure, and its basic component handles.

pub mod hugrmut;

pub mod attributes;
pub(crate) mod ident;
pub mod internal;
pub mod rewrite;
pub mod serialize;
pub mod validate;
pub mod views;

#[cfg(feature = "extension_inference")]
use std::collections::HashMap;
use std::collections::VecDeque;
use std::iter;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

pub use ident::{IdentList, InvalidIdentifier};
pub use rewrite::{Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, PortMut, UnmanagedDenseMap};
use thiserror::Error;

pub use self::views::{HugrView, RootTagged};
use crate::core::NodeIndex;
#[cfg(feature = "extension_inference")]
use crate::extension::infer_extensions;
use crate::extension::{ExtensionRegistry, ExtensionSet, ExtensionSolution, InferExtensionError};
use crate::ops::custom::resolve_extension_ops;
use crate::ops::{OpTag, OpTrait, OpType, DEFAULT_OPTYPE};
use crate::types::FunctionType;
use crate::{Direction, Node};

use delegate::delegate;

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
    metadata: UnmanagedDenseMap<portgraph::NodeIndex, Option<NodeMetadataMap>>,
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
    pub fn new_pure(op: impl Into<OpType>) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: Some(ExtensionSet::new()),
        }
    }

    /// Instantiate an OpType with an unknown set of input extensions
    /// (to be inferred later)
    pub fn new_open(op: impl Into<OpType>) -> Self {
        NodeType {
            op: op.into(),
            input_extensions: None,
        }
    }

    /// Instantiate an [OpType] with the default set of input extensions
    /// for that OpType.
    pub fn new_auto(op: impl Into<OpType>) -> Self {
        let op = op.into();
        if OpTag::ModuleOp.is_superset(op.tag()) {
            Self::new_pure(op)
        } else {
            Self::new_open(op)
        }
    }

    /// Get the function type from the embedded op
    pub fn op_signature(&self) -> Option<FunctionType> {
        self.op.dataflow_signature()
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

    /// The input and output extensions for this node, if set.
    ///
    /// `None`` if the [Self::input_extensions] is `None`.
    /// Otherwise, will return Some, with the output extensions computed from the node's delta
    pub fn io_extensions(&self) -> Option<(ExtensionSet, ExtensionSet)> {
        self.input_extensions
            .clone()
            .map(|e| (e.clone(), self.op.extension_delta().union(e)))
    }

    /// Gets the underlying [OpType] i.e. without any [input_extensions]
    ///
    /// [input_extensions]: NodeType::input_extensions
    pub fn op(&self) -> &OpType {
        &self.op
    }

    /// Returns the underlying [OpType] i.e. without any [input_extensions]
    ///
    /// [input_extensions]: NodeType::input_extensions
    pub fn into_op(self) -> OpType {
        self.op
    }

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

impl<T: Into<OpType>> From<T> for NodeType {
    fn from(value: T) -> Self {
        NodeType::new_auto(value.into())
    }
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(NodeType::new_pure(crate::ops::Module))
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

/// Arbitrary metadata entry for a node.
///
/// Each entry is associated to a string key.
pub type NodeMetadata = serde_json::Value;

/// The container of all the metadata entries for a node.
pub type NodeMetadataMap = serde_json::Map<String, NodeMetadata>;

/// Public API for HUGRs.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub fn new(root_node: NodeType) -> Self {
        Self::with_capacity(root_node, 0, 0)
    }

    /// Resolve extension ops, infer extensions used, and pass the closure into validation
    pub fn update_validate(
        &mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), ValidationError> {
        resolve_extension_ops(self, extension_registry)?;
        self.validate_no_extensions(extension_registry)?;
        #[cfg(feature = "extension_inference")]
        {
            self.infer_extensions()?;
            self.validate_extensions(HashMap::new())?;
        }
        Ok(())
    }

    /// Infer extension requirements and add new information to `op_types` field
    /// (if the "extension_inference" feature is on; otherwise, do nothing)
    pub fn infer_extensions(&mut self) -> Result<(), InferExtensionError> {
        #[cfg(feature = "extension_inference")]
        {
            let solution = infer_extensions(self)?;
            self.instantiate_extensions(&solution);
        }
        Ok(())
    }

    #[allow(dead_code)]
    /// Add extension requirement information to the hugr in place.
    fn instantiate_extensions(&mut self, solution: &ExtensionSolution) {
        // We only care about inferred _input_ extensions, because `NodeType`
        // uses those to infer the output extensions
        for (node, input_extensions) in solution.iter() {
            let nodetype = self.op_types.try_get_mut(node.pg_index()).unwrap();
            match &nodetype.input_extensions {
                None => nodetype.input_extensions = Some(input_extensions.clone()),
                Some(existing_ext_reqs) => {
                    debug_assert_eq!(existing_ext_reqs, input_extensions)
                }
            }
        }
    }
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
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
            while position > source.index() {
                source = ordered[source.index()];
            }

            let target: Node = portgraph::NodeIndex::new(position).into();
            if target != source {
                let pg_target = target.pg_index();
                let pg_source = source.pg_index();
                self.graph.swap_nodes(pg_target, pg_source);
                self.op_types.swap(pg_target, pg_source);
                self.hierarchy.swap_nodes(pg_target, pg_source);
                rekey(source, target);
            }
        }
        self.root = portgraph::NodeIndex::new(0);

        // Finish by compacting the copy nodes.
        // The operation nodes will be left in place.
        // This step is not strictly necessary.
        self.graph.compact_nodes(|_, _| {});
    }
}

/// Errors that can occur while manipulating a Hugr.
///
/// TODO: Better descriptions, not just re-exporting portgraph errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HugrError {
    /// The node was not of the required [OpTag]
    /// (e.g. to conform to the [RootTagged::RootHandle] of a [HugrView])
    #[error("Invalid tag: required a tag in {required} but found {actual}")]
    #[allow(missing_docs)]
    InvalidTag { required: OpTag, actual: OpTag },
    /// An invalid port was specified.
    #[error("Invalid port direction {0:?}.")]
    InvalidPortDirection(Direction),
}

#[cfg(test)]
mod test {
    use super::{Hugr, HugrView};
    #[cfg(feature = "extension_inference")]
    use std::error::Error;

    #[test]
    fn impls_send_and_sync() {
        // Send and Sync are automatically impl'd by the compiler, if possible.
        // This test will fail to compile if that wasn't possible.
        #[allow(dead_code)]
        trait Test: Send + Sync {}
        impl Test for Hugr {}
    }

    #[test]
    fn io_node() {
        use crate::builder::test::simple_dfg_hugr;
        use cool_asserts::assert_matches;

        let hugr = simple_dfg_hugr();
        assert_matches!(hugr.get_io(hugr.root()), Some(_));
    }

    #[cfg(feature = "extension_inference")]
    #[test]
    fn extension_instantiation() -> Result<(), Box<dyn Error>> {
        use crate::builder::test::closed_dfg_root_hugr;
        use crate::extension::ExtensionSet;
        use crate::hugr::HugrMut;
        use crate::ops::Lift;
        use crate::type_row;
        use crate::types::{FunctionType, Type};

        const BIT: Type = crate::extension::prelude::USIZE_T;
        let r = ExtensionSet::singleton(&"R".try_into().unwrap());

        let mut hugr = closed_dfg_root_hugr(
            FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(r.clone()),
        );
        let [input, output] = hugr.get_io(hugr.root()).unwrap();
        let lift = hugr.add_node_with_parent(
            hugr.root(),
            Lift {
                type_row: type_row![BIT],
                new_extension: "R".try_into().unwrap(),
            },
        );
        hugr.connect(input, 0, lift, 0);
        hugr.connect(lift, 0, output, 0);
        hugr.infer_extensions()?;

        assert_eq!(
            hugr.get_nodetype(lift).input_extensions().unwrap(),
            &ExtensionSet::new()
        );
        assert_eq!(hugr.get_nodetype(output).input_extensions().unwrap(), &r);
        Ok(())
    }
}

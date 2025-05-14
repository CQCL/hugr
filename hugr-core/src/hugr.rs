//! The Hugr data structure, and its basic component handles.

pub mod hugrmut;

pub(crate) mod ident;
pub mod internal;
pub mod patch;
pub mod serialize;
pub mod validate;
pub mod views;

use std::collections::VecDeque;
use std::io;
use std::iter;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

pub use ident::{IdentList, InvalidIdentifier};
use itertools::Itertools;
pub use patch::{Patch, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, PortMut, PortView, UnmanagedDenseMap};
use thiserror::Error;

pub use self::views::HugrView;
use crate::core::NodeIndex;
use crate::envelope::{self, EnvelopeConfig, EnvelopeError};
use crate::extension::resolution::{
    ExtensionResolutionError, WeakExtensionRegistry, resolve_op_extensions,
    resolve_op_types_extensions,
};
use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::ops::{self, Module, NamedOp, OpName, OpTag, OpTrait};
pub use crate::ops::{DEFAULT_OPTYPE, OpType};
use crate::package::Package;
use crate::{Direction, Node};

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    graph: MultiPortGraph,

    /// The node hierarchy.
    hierarchy: Hierarchy,

    /// The single root node in the portgraph hierarchy.
    ///
    /// This node is always a module node, containing all the other nodes.
    module_root: portgraph::NodeIndex,

    /// The distinguished entrypoint node of the HUGR.
    entrypoint: portgraph::NodeIndex,

    /// Operation types for each node.
    op_types: UnmanagedDenseMap<portgraph::NodeIndex, OpType>,

    /// Node metadata
    metadata: UnmanagedDenseMap<portgraph::NodeIndex, Option<NodeMetadataMap>>,

    /// Extensions used by the operations in the Hugr.
    extensions: ExtensionRegistry,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new()
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
    /// Create a new Hugr, with a single [`Module`] operation as the root node.
    #[must_use]
    pub fn new() -> Self {
        make_module_hugr(Module::new().into(), 0, 0).unwrap()
    }

    /// Create a new Hugr, with a given entrypoint operation.
    ///
    /// If the optype is [`OpType::Module`], the HUGR module root will match the
    /// entrypoint node. Otherwise, the entrypoint will be a descendent of the a
    /// module initialized at the node hierarchy root. The specific HUGR created
    /// depends on the operation type.
    ///
    /// # Error
    ///
    /// Returns [`HugrError::UnsupportedEntrypoint`] if the entrypoint operation
    /// requires additional context to be defined. This is the case for
    /// [`OpType::Case`], [`OpType::DataflowBlock`], and [`OpType::ExitBlock`]
    /// since they are context-specific definitions.
    pub fn new_with_entrypoint(entrypoint_op: impl Into<OpType>) -> Result<Self, HugrError> {
        Self::with_capacity(entrypoint_op, 0, 0)
    }

    /// Create a new Hugr, with a given entrypoint operation and preallocated capacity.
    ///
    /// If the optype is [`OpType::Module`], the HUGR module root will match the
    /// entrypoint node. Otherwise, the entrypoint will be a child of the a
    /// module initialized at the node hierarchy root. The specific HUGR created
    /// depends on the operation type.
    ///
    /// # Error
    ///
    /// Returns [`HugrError::UnsupportedEntrypoint`] if the entrypoint operation
    /// requires additional context to be defined. This is the case for
    /// [`OpType::Case`], [`OpType::DataflowBlock`], and [`OpType::ExitBlock`]
    /// since they are context-specific definitions.
    pub fn with_capacity(
        entrypoint_op: impl Into<OpType>,
        nodes: usize,
        ports: usize,
    ) -> Result<Self, HugrError> {
        let entrypoint_op: OpType = entrypoint_op.into();
        let op_name = entrypoint_op.name();
        make_module_hugr(entrypoint_op, nodes, ports)
            .ok_or(HugrError::UnsupportedEntrypoint { op: op_name })
    }

    /// Reserves enough capacity to insert at least the given number of
    /// additional nodes and links.
    ///
    /// This method does not take into account already allocated free space left
    /// after node removals, and may overallocate capacity.
    pub fn reserve(&mut self, nodes: usize, links: usize) {
        let ports = links * 2;
        self.graph.reserve(nodes, ports);
    }

    /// Read a Package from a HUGR envelope.
    pub fn load(
        reader: impl io::BufRead,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        let pkg = Package::load(reader, extensions)?;
        match pkg.modules.into_iter().exactly_one() {
            Ok(hugr) => Ok(hugr),
            Err(e) => Err(EnvelopeError::ExpectedSingleHugr { count: e.count() }),
        }
    }

    /// Read a Package from a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::load` with a bytearray instead.
    pub fn load_str(
        envelope: impl AsRef<str>,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        Self::load(envelope.as_ref().as_bytes(), extensions)
    }

    /// Store the Package in a HUGR envelope.
    pub fn store(
        &self,
        writer: impl io::Write,
        config: EnvelopeConfig,
    ) -> Result<(), EnvelopeError> {
        envelope::write_envelope_impl(writer, [self], &self.extensions, config)
    }

    /// Store the Package in a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::store` with a bytearray instead.
    /// See [`EnvelopeFormat::ascii_printable`][crate::envelope::EnvelopeFormat::ascii_printable].
    pub fn store_str(&self, config: EnvelopeConfig) -> Result<String, EnvelopeError> {
        if !config.format.ascii_printable() {
            return Err(EnvelopeError::NonASCIIFormat {
                format: config.format,
            });
        }

        let mut buf = Vec::new();
        self.store(&mut buf, config)?;
        Ok(String::from_utf8(buf).expect("Envelope is valid utf8"))
    }

    /// Given a Hugr that has been deserialized, collect all extensions used to
    /// define the HUGR while resolving all [`OpType::OpaqueOp`] operations into
    /// [`OpType::ExtensionOp`]s and updating the extension pointer in all
    /// internal [`crate::types::CustomType`]s to point to the extensions in the
    /// register.
    ///
    /// When listing "used extensions" we only care about _definitional_
    /// extension requirements, i.e., the operations and types that are required
    /// to define the HUGR nodes and wire types. This is computed from the union
    /// of all extension required across the HUGR.
    ///
    /// Updates the internal extension registry with the extensions used in the
    /// definition.
    ///
    /// # Parameters
    ///
    /// - `extensions`: The extension set considered when resolving opaque
    ///   operations and types. The original Hugr's internal extension
    ///   registry is ignored and replaced with the newly computed one.
    ///
    /// # Errors
    ///
    /// - If an opaque operation cannot be resolved to an extension operation.
    /// - If an extension operation references an extension that is missing from
    ///   the registry.
    /// - If a custom type references an extension that is missing from the
    ///   registry.
    pub fn resolve_extension_defs(
        &mut self,
        extensions: &ExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        let mut used_extensions = ExtensionRegistry::default();

        // Here we need to iterate the optypes in the hugr mutably, to avoid
        // having to clone and accumulate all replacements before finally
        // applying them.
        //
        // This is not something we want to expose it the API, so we manually
        // iterate instead of writing it as a method.
        //
        // Since we don't have a non-borrowing iterator over all the possible
        // NodeIds, we have to simulate it by iterating over all possible
        // indices and checking if the node exists.
        let weak_extensions: WeakExtensionRegistry = extensions.into();
        for n in 0..self.graph.node_capacity() {
            let pg_node = portgraph::NodeIndex::new(n);
            let node: Node = pg_node.into();
            if !self.contains_node(node) {
                continue;
            }

            let op = &mut self.op_types[pg_node];

            if let Some(extension) = resolve_op_extensions(node, op, extensions)? {
                used_extensions.register_updated_ref(extension);
            }
            used_extensions.extend(
                resolve_op_types_extensions(Some(node), op, &weak_extensions)?.map(|weak| {
                    weak.upgrade()
                        .expect("Extension comes from a valid registry")
                }),
            );
        }

        self.extensions = used_extensions;
        Ok(())
    }
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Add a node to the graph.
    pub(crate) fn add_node(&mut self, nodetype: OpType) -> Node {
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
        let mut ordered = Vec::with_capacity(self.num_nodes());
        let root = self.module_root();
        let mut new_entrypoint = self.entrypoint;
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
                let pg_target = target.into_portgraph();
                let pg_source = source.into_portgraph();
                self.graph.swap_nodes(pg_target, pg_source);
                self.op_types.swap(pg_target, pg_source);
                self.hierarchy.swap_nodes(pg_target, pg_source);
                rekey(source, target);

                if source.into_portgraph() == self.entrypoint {
                    new_entrypoint = target.into_portgraph();
                }
            }
        }
        self.module_root = portgraph::NodeIndex::new(0);
        self.entrypoint = new_entrypoint;

        // Finish by compacting the copy nodes.
        // The operation nodes will be left in place.
        // This step is not strictly necessary.
        self.graph.compact_nodes(|_, _| {});
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
#[error(
    "Parent node {parent} has extensions {parent_extensions} that are too restrictive for child node {child}, they must include child extensions {child_extensions}"
)]
/// An error in the extension deltas.
pub struct ExtensionError {
    parent: Node,
    parent_extensions: ExtensionSet,
    child: Node,
    child_extensions: ExtensionSet,
}

/// Errors that can occur while manipulating a Hugr.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HugrError {
    /// The node was not of the required [`OpTag`]
    #[error("Invalid tag: required a tag in {required} but found {actual}")]
    #[allow(missing_docs)]
    InvalidTag { required: OpTag, actual: OpTag },
    /// An invalid port was specified.
    #[error("Invalid port direction {0:?}.")]
    InvalidPortDirection(Direction),
    /// Cannot initialize a HUGR with the given entrypoint operation type.
    #[error("Cannot initialize a HUGR with entrypoint type {op}")]
    UnsupportedEntrypoint {
        /// The name of the unsupported operation.
        op: OpName,
    },
}

/// Create a new Hugr, with a given root node and preallocated capacity.
///
/// The root operation must be region root, i.e., define a node that can be
/// assigned as the parent of other nodes.
///
/// If the root optype is [`OpType::Module`], the HUGR module root will match
/// the root node.
///
/// Otherwise, the root node will be a child of the a module created at the node
/// hierarchy root. The specific HUGR created depends on the operation type, and
/// whether it can be contained in a module, function definition, etc.
///
/// Some operation types are not allowed and will result in a panic. This is the
/// case for [`OpType::Case`] and [`OpType::DataflowBlock`] since they are
/// context-specific operation.
fn make_module_hugr(root_op: OpType, nodes: usize, ports: usize) -> Option<Hugr> {
    let mut graph = MultiPortGraph::with_capacity(nodes, ports);
    let hierarchy = Hierarchy::new();
    let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
    let extensions = root_op.used_extensions().unwrap_or_default();

    // Filter out operations that are not region roots.
    let tag = root_op.tag();
    let container_tags = [
        OpTag::ModuleRoot,
        OpTag::DataflowParent,
        OpTag::Cfg,
        OpTag::Conditional,
    ];
    if !container_tags.iter().any(|t| t.is_superset(tag)) {
        return None;
    }

    let module = graph.add_node(0, 0);
    op_types[module] = OpType::Module(ops::Module::new());

    let mut hugr = Hugr {
        graph,
        hierarchy,
        module_root: module,
        entrypoint: module,
        op_types,
        metadata: UnmanagedDenseMap::with_capacity(nodes),
        extensions,
    };
    let module: Node = module.into();

    // Now the behaviour depends on the root node type.
    if root_op.is_module() {
        // The hugr is already a module, nothing to do.
    }
    // If possible, put the op directly in the module.
    else if OpTag::ModuleOp.is_superset(tag) {
        let node = hugr.add_node_with_parent(module, root_op);
        hugr.set_entrypoint(node);
    }
    // If it can exist inside a function definition, create a "main" function
    // and put the op inside it.
    else if OpTag::DataflowChild.is_superset(tag) && !root_op.is_input() && !root_op.is_output() {
        let signature = root_op
            .dataflow_signature()
            .unwrap_or_else(|| panic!("Dataflow child {} without signature", root_op.name()))
            .into_owned();
        let dataflow_inputs = signature.input_count();
        let dataflow_outputs = signature.output_count();

        let func = hugr.add_node_with_parent(module, ops::FuncDefn::new("main", signature.clone()));
        let inp = hugr.add_node_with_parent(
            func,
            ops::Input {
                types: signature.input.clone(),
            },
        );
        let out = hugr.add_node_with_parent(
            func,
            ops::Output {
                types: signature.output.clone(),
            },
        );
        let entrypoint = hugr.add_node_with_parent(func, root_op);

        // Wire the inputs and outputs of the entrypoint node to the function's
        // inputs and outputs.
        for port in 0..dataflow_inputs {
            hugr.connect(inp, port, entrypoint, port);
        }
        for port in 0..dataflow_outputs {
            hugr.connect(entrypoint, port, out, port);
        }

        hugr.set_entrypoint(entrypoint);
    }
    // Other more exotic ops are unsupported, and will cause a panic.
    else {
        debug_assert!(matches!(
            root_op,
            OpType::Input(_)
                | OpType::Output(_)
                | OpType::DataflowBlock(_)
                | OpType::ExitBlock(_)
                | OpType::Case(_)
        ));
        return None;
    }

    Some(hugr)
}

#[cfg(test)]
pub(crate) mod test {
    use std::{fs::File, io::BufReader};

    use super::*;

    use crate::envelope::{EnvelopeError, PackageEncodingError};
    use crate::ops::OpaqueOp;
    use crate::test_file;
    use cool_asserts::assert_matches;
    use portgraph::LinkView;

    /// Check that two HUGRs are equivalent, up to node renumbering.
    pub(crate) fn check_hugr_equality(lhs: &Hugr, rhs: &Hugr) {
        // Original HUGR, with canonicalized node indices
        //
        // The internal port indices may still be different.
        let mut lhs = lhs.clone();
        lhs.canonicalize_nodes(|_, _| {});
        let mut rhs = rhs.clone();
        rhs.canonicalize_nodes(|_, _| {});

        assert_eq!(rhs.module_root(), lhs.module_root());
        assert_eq!(rhs.entrypoint(), lhs.entrypoint());
        assert_eq!(rhs.hierarchy, lhs.hierarchy);
        assert_eq!(rhs.metadata, lhs.metadata);

        // Extension operations may have been downgraded to opaque operations.
        for node in rhs.nodes() {
            let new_op = rhs.get_optype(node);
            let old_op = lhs.get_optype(node);
            if !new_op.is_const() {
                match (new_op, old_op) {
                    (OpType::ExtensionOp(ext), OpType::OpaqueOp(opaque))
                    | (OpType::OpaqueOp(opaque), OpType::ExtensionOp(ext)) => {
                        let ext_opaque: OpaqueOp = ext.clone().into();
                        assert_eq!(ext_opaque, opaque.clone());
                    }
                    _ => assert_eq!(new_op, old_op),
                }
            }
        }

        // Check that the graphs are equivalent up to port renumbering.
        let new_graph = &rhs.graph;
        let old_graph = &lhs.graph;
        assert_eq!(new_graph.node_count(), old_graph.node_count());
        assert_eq!(new_graph.port_count(), old_graph.port_count());
        assert_eq!(new_graph.link_count(), old_graph.link_count());
        for n in old_graph.nodes_iter() {
            assert_eq!(new_graph.num_inputs(n), old_graph.num_inputs(n));
            assert_eq!(new_graph.num_outputs(n), old_graph.num_outputs(n));
            assert_eq!(
                new_graph.output_neighbours(n).collect_vec(),
                old_graph.output_neighbours(n).collect_vec()
            );
        }
    }

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
        assert_matches!(hugr.get_io(hugr.entrypoint()), Some(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_0() {
        // https://github.com/CQCL/hugr/issues/1091 bad case
        let hugr = Hugr::load(
            BufReader::new(File::open(test_file!("hugr-0.hugr")).unwrap()),
            None,
        );
        assert_matches!(
            hugr,
            Err(EnvelopeError::PackageEncoding {
                source: PackageEncodingError::JsonEncoding(_)
            })
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_1() {
        // https://github.com/CQCL/hugr/issues/1091 good case
        let hugr = Hugr::load(
            BufReader::new(File::open(test_file!("hugr-1.hugr")).unwrap()),
            None,
        );
        assert_matches!(&hugr, Ok(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_2() {
        // https://github.com/CQCL/hugr/issues/1185 bad case
        let hugr = Hugr::load(
            BufReader::new(File::open(test_file!("hugr-2.hugr")).unwrap()),
            None,
        )
        .unwrap();
        assert_matches!(hugr.validate(), Err(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_3() {
        // https://github.com/CQCL/hugr/issues/1185 good case
        let hugr = Hugr::load(
            BufReader::new(File::open(test_file!("hugr-3.hugr")).unwrap()),
            None,
        );
        assert_matches!(&hugr, Ok(_));
    }
}

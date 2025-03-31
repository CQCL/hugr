//! The Hugr data structure, and its basic component handles.

pub mod hugrmut;

pub(crate) mod ident;
pub mod internal;
pub mod rewrite;
pub mod serialize;
pub mod validate;
pub mod views;

use std::collections::VecDeque;
use std::io::Read;
use std::iter;

pub(crate) use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

pub use ident::{IdentList, InvalidIdentifier};
pub use rewrite::{Rewrite, SimpleReplacement, SimpleReplacementError};

use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{Hierarchy, PortMut, PortView, UnmanagedDenseMap};
use thiserror::Error;

pub use self::views::{HugrView, RootTagged};
use crate::core::NodeIndex;
use crate::extension::resolution::{
    resolve_op_extensions, resolve_op_types_extensions, ExtensionResolutionError,
    WeakExtensionRegistry,
};
use crate::extension::ExtensionRegistry;
use crate::ops::{OpTag, OpTrait};
pub use crate::ops::{OpType, DEFAULT_OPTYPE};
use crate::{Direction, Node};

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

    /// Node metadata
    metadata: UnmanagedDenseMap<portgraph::NodeIndex, Option<NodeMetadataMap>>,

    /// Extensions used by the operations in the Hugr.
    extensions: ExtensionRegistry,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(crate::ops::Module::new())
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
    pub fn new(root_node: impl Into<OpType>) -> Self {
        Self::with_capacity(root_node.into(), 0, 0)
    }

    /// Load a Hugr from a json reader.
    ///
    /// Validates the Hugr against the provided extension registry, ensuring all
    /// operations are resolved.
    ///
    /// If the feature `extension_inference` is enabled, we will ensure every function
    /// correctly specifies the extensions required by its contained ops.
    pub fn load_json(
        reader: impl Read,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, LoadHugrError> {
        let mut hugr: Hugr = serde_json::from_reader(reader)?;

        hugr.resolve_extension_defs(extension_registry)?;
        hugr.validate_no_extensions()?;

        if cfg!(feature = "extension_inference") {
            hugr.infer_extensions(false)?;
            hugr.validate_extensions()?;
        }

        Ok(hugr)
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
    /// This is distinct from _runtime_ extension requirements computed in
    /// [`Hugr::infer_extensions`], which are computed more granularly in each
    /// function signature by the `runtime_reqs` field and define the set
    /// of capabilities required by the runtime to execute each function.
    ///
    /// Updates the internal extension registry with the extensions used in the
    /// definition.
    ///
    /// # Parameters
    ///
    /// - `extensions`: The extension set considered when resolving opaque
    ///     operations and types. The original Hugr's internal extension
    ///     registry is ignored and replaced with the newly computed one.
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
    /// Create a new Hugr, with a single root node and preallocated capacity.
    pub(crate) fn with_capacity(root_node: OpType, nodes: usize, ports: usize) -> Self {
        let mut graph = MultiPortGraph::with_capacity(nodes, ports);
        let hierarchy = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
        let root = graph.add_node(root_node.input_count(), root_node.output_count());
        let extensions = root_node.used_extensions();
        op_types[root] = root_node;

        Self {
            graph,
            hierarchy,
            root,
            op_types,
            metadata: UnmanagedDenseMap::with_capacity(nodes),
            extensions: extensions.unwrap_or_default(),
        }
    }

    /// Set the root node of the hugr.
    pub(crate) fn set_root(&mut self, root: Node) {
        self.hierarchy.detach(self.root);
        self.root = root.pg_index();
    }

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

/// Errors that can occur while loading and validating a Hugr json.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LoadHugrError {
    /// Error while loading the Hugr from JSON.
    #[error("Error while loading the Hugr from JSON: {0}")]
    Load(#[from] serde_json::Error),
    /// Validation of the loaded Hugr failed.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// Error when resolving extension operations and types.
    #[error(transparent)]
    Extension(#[from] ExtensionResolutionError),
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::{fs::File, io::BufReader};

    use super::internal::HugrMutInternals;
    #[cfg(feature = "extension_inference")]
    use super::ValidationError;
    use super::{ExtensionError, Hugr, HugrMut, HugrView, Node};
    use crate::extension::{ExtensionId, ExtensionSet, PRELUDE_REGISTRY, TO_BE_INFERRED};
    use crate::ops::{ExtensionOp, OpName};
    use crate::types::type_param::TypeParam;
    use crate::types::{
        FuncValueType, PolyFuncTypeRV, Signature, Type, TypeArg, TypeBound, TypeRV, TypeRow,
    };

    use crate::{const_extension_ids, ops, test_file, type_row, Extension};
    use cool_asserts::assert_matches;
    use lazy_static::lazy_static;
    use rstest::rstest;

    const_extension_ids! {
        pub(crate) const LIFT_EXT_ID: ExtensionId = "LIFT_EXT_ID";
    }
    lazy_static! {
        /// Tests only extension holding an Op that can add arbitrary extensions to a row.
        pub(crate) static ref LIFT_EXT: Arc<Extension> = {
            Extension::new_arc(
                LIFT_EXT_ID,
                hugr::extension::Version::new(0, 0, 0),
                |ext, extension_ref| {
                    ext.add_op(
                        OpName::new_inline("Lift"),
                        "".into(),
                        PolyFuncTypeRV::new(
                            vec![TypeParam::Extensions, TypeParam::new_list(TypeBound::Any)],
                            FuncValueType::new_endo(TypeRV::new_row_var_use(1, TypeBound::Any))
                                .with_extension_delta(ExtensionSet::type_var(0)),
                        ),
                        extension_ref,
                    )
                    .unwrap();
                },
            )
        };
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
        assert_matches!(hugr.get_io(hugr.root()), Some(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_0() {
        // https://github.com/CQCL/hugr/issues/1091 bad case
        let hugr = Hugr::load_json(
            BufReader::new(File::open(test_file!("hugr-0.json")).unwrap()),
            &PRELUDE_REGISTRY,
        );
        assert_matches!(hugr, Err(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_1() {
        // https://github.com/CQCL/hugr/issues/1091 good case
        let hugr = Hugr::load_json(
            BufReader::new(File::open(test_file!("hugr-1.json")).unwrap()),
            &PRELUDE_REGISTRY,
        );
        assert_matches!(&hugr, Ok(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_2() {
        // https://github.com/CQCL/hugr/issues/1185 bad case
        let hugr = Hugr::load_json(
            BufReader::new(File::open(test_file!("hugr-2.json")).unwrap()),
            &PRELUDE_REGISTRY,
        );
        assert_matches!(hugr, Err(_));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_3() {
        // https://github.com/CQCL/hugr/issues/1185 good case
        let hugr = Hugr::load_json(
            BufReader::new(File::open(test_file!("hugr-3.json")).unwrap()),
            &PRELUDE_REGISTRY,
        );
        assert_matches!(&hugr, Ok(_));
    }

    const_extension_ids! {
        const XA: ExtensionId = "EXT_A";
        const XB: ExtensionId = "EXT_B";
    }

    fn build_ext_dfg(parent: ExtensionSet) -> (Hugr, Node) {
        let ty = Type::new_function(Signature::new_endo(type_row![]));
        let mut h = Hugr::new(ops::DFG {
            signature: Signature::new_endo(ty.clone()).with_extension_delta(parent.clone()),
        });
        let root = h.root();
        let mid = add_inliftout(&mut h, root, ty);
        (h, mid)
    }

    fn add_inliftout(h: &mut Hugr, p: Node, ty: Type) -> Node {
        let inp = h.add_node_with_parent(
            p,
            ops::Input {
                types: ty.clone().into(),
            },
        );
        let out = h.add_node_with_parent(
            p,
            ops::Output {
                types: ty.clone().into(),
            },
        );
        let mid = h.add_node_with_parent(p, lift_op(ty, XA));
        h.connect(inp, 0, mid, 0);
        h.connect(mid, 0, out, 0);
        mid
    }
}

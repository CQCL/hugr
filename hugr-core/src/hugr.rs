//! The Hugr data structure, and its basic component handles.

pub mod hugrmut;

pub(crate) mod ident;
pub mod internal;
pub mod rewrite;
pub mod serialize;
pub mod validate;
pub mod views;

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
use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::ops::custom::resolve_extension_ops;
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

    /// Resolve extension ops, infer extensions used, and pass the closure into validation
    pub fn update_validate(
        &mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), ValidationError> {
        resolve_extension_ops(self, extension_registry)?;
        self.validate_no_extensions(extension_registry)?;
        #[cfg(feature = "extension_inference")]
        {
            self.infer_extensions(false)?;
            self.validate_extensions()?;
        }
        Ok(())
    }

    /// Leaving this here as in the future we plan for it to infer deltas
    /// of container nodes e.g. [OpType::DFG]. For the moment it does nothing.
    pub fn infer_extensions(&mut self, remove: bool) -> Result<(), ExtensionError> {
        fn delta_mut(optype: &mut OpType) -> Option<&mut ExtensionSet> {
            match optype {
                OpType::DFG(dfg) => Some(&mut dfg.signature.extension_reqs),
                OpType::DataflowBlock(dfb) => Some(&mut dfb.extension_delta),
                OpType::TailLoop(tl) => Some(&mut tl.extension_delta),
                OpType::CFG(cfg) => Some(&mut cfg.signature.extension_reqs),
                OpType::Conditional(c) => Some(&mut c.extension_delta),
                OpType::Case(c) => Some(&mut c.signature.extension_reqs),
                //OpType::Lift(_) // Not ATM: only a single element, and we expect Lift to be removed
                //OpType::FuncDefn(_) // Not at present due to the possibility of recursion
                _ => None,
            }
        }
        fn infer(h: &mut Hugr, node: Node, remove: bool) -> Result<ExtensionSet, ExtensionError> {
            let mut child_sets = h
                .children(node)
                .collect::<Vec<_>>() // Avoid borrowing h over recursive call
                .into_iter()
                .map(|ch| Ok((ch, infer(h, ch, remove)?)))
                .collect::<Result<Vec<_>, _>>()?;

            let Some(es) = delta_mut(h.op_types.get_mut(node.pg_index())) else {
                return Ok(h.get_optype(node).extension_delta());
            };
            if es.contains(&ExtensionSet::TO_BE_INFERRED) {
                // Do not remove anything from current delta - any other elements are a lower bound
                child_sets.push((node, es.clone())); // "child_sets" now misnamed but we discard fst
            } else if remove {
                child_sets.iter().try_for_each(|(ch, ch_exts)| {
                    if !es.is_superset(ch_exts) {
                        return Err(ExtensionError {
                            parent: node,
                            parent_extensions: es.clone(),
                            child: *ch,
                            child_extensions: ch_exts.clone(),
                        });
                    }
                    Ok(())
                })?;
            } else {
                return Ok(es.clone()); // Can't neither add nor remove, so nothing to do
            }
            let merged = ExtensionSet::union_over(child_sets.into_iter().map(|(_, e)| e));
            *es = ExtensionSet::singleton(&ExtensionSet::TO_BE_INFERRED).missing_from(&merged);

            Ok(es.clone())
        }
        infer(self, self.root(), remove)?;
        Ok(())
    }

    // Note: tests
    // * all combinations of (remove or not, TO_BE_INFERRED present or absent, success(inferred-set) or failure (possible only if no TO_BE_INFERRED) )
    // * parent - child - grandchild tests:
    //   X - Y + INFER - X (ok with remove, but fails w/out remove)
    //   X - Y + INFER - Y or X - INFER - Y (mid fails against parent with just Y, regardless of remove)
    //   X - INFER - X (ok with-or-without remove)
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node and preallocated capacity.
    pub(crate) fn with_capacity(root_node: OpType, nodes: usize, ports: usize) -> Self {
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

#[derive(Debug, Clone, PartialEq, Error)]
#[error("Parent node {parent} has extensions {parent_extensions} that are too restrictive for child node {child}, they must include child extensions {child_extensions}")]
/// An error in the extension deltas.
pub struct ExtensionError {
    parent: Node,
    parent_extensions: ExtensionSet,
    child: Node,
    child_extensions: ExtensionSet,
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
}

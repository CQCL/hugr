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
use portgraph::{Hierarchy, PortMut, PortView, UnmanagedDenseMap};
use thiserror::Error;

pub use self::views::{HugrView, RootTagged};
use crate::core::NodeIndex;
use crate::extension::resolution::{
    update_op_extensions, update_op_types_extensions, ExtensionResolutionError,
};
use crate::extension::{ExtensionRegistry, ExtensionSet, TO_BE_INFERRED};
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

    /// Resolve extension ops, infer extensions used, and pass the closure into validation
    pub fn update_validate(
        &mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), ValidationError> {
        self.resolve_extension_defs(extension_registry)?;
        self.validate_no_extensions(extension_registry)?;
        #[cfg(feature = "extension_inference")]
        {
            self.infer_extensions(false)?;
            self.validate_extensions()?;
        }
        Ok(())
    }

    /// Infers an extension-delta for any non-function container node
    /// whose current [extension_delta] contains [TO_BE_INFERRED]. The inferred delta
    /// will be the smallest delta compatible with its children and that includes any
    /// other [ExtensionId]s in the current delta.
    ///
    /// If `remove` is true, for such container nodes *without* [TO_BE_INFERRED],
    /// ExtensionIds are removed from the delta if they are *not* used by any child node.
    ///
    /// The non-function container nodes are:
    /// [Case], [CFG], [Conditional], [DataflowBlock], [DFG], [TailLoop]
    ///
    /// [Case]: crate::ops::Case
    /// [CFG]: crate::ops::CFG
    /// [Conditional]: crate::ops::Conditional
    /// [DataflowBlock]: crate::ops::DataflowBlock
    /// [DFG]: crate::ops::DFG
    /// [TailLoop]: crate::ops::TailLoop
    /// [extension_delta]: crate::ops::OpType::extension_delta
    /// [ExtensionId]: crate::extension::ExtensionId
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
            if es.contains(&TO_BE_INFERRED) {
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
            *es = ExtensionSet::singleton(TO_BE_INFERRED).missing_from(&merged);

            Ok(es.clone())
        }
        infer(self, self.root(), remove)?;
        Ok(())
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
    /// function signature by the `required_extensions` field and define the set
    /// of capabilities required by the runtime to execute each function.
    ///
    /// Returns a new extension registry with the extensions used in the Hugr.
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
    ) -> Result<ExtensionRegistry, ExtensionResolutionError> {
        let mut used_extensions = ExtensionRegistry::default();

        // Here we need to iterate the optypes in the hugr mutably, to avoid
        // having to clone and accumulate all replacements before finally
        // applying them.
        //
        // This is not something we want to expose it the API, so we manually
        // iterate instead of writing it as a method.
        for n in 0..self.graph.node_capacity() {
            let pg_node = portgraph::NodeIndex::new(n);
            let node: Node = pg_node.into();
            if !self.contains_node(node) {
                continue;
            }

            let op = &mut self.op_types[pg_node];

            if let Some(extension) = update_op_extensions(node, op, extensions)? {
                used_extensions.register_updated_ref(extension);
            }
            update_op_types_extensions(node, op, extensions, &mut used_extensions)?;
        }

        Ok(used_extensions)
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
    use std::{fs::File, io::BufReader};

    use super::internal::HugrMutInternals;
    #[cfg(feature = "extension_inference")]
    use super::ValidationError;
    use super::{ExtensionError, Hugr, HugrMut, HugrView, Node};
    use crate::extension::prelude::Lift;
    use crate::extension::prelude::PRELUDE_ID;
    use crate::extension::{
        ExtensionId, ExtensionSet, EMPTY_REG, PRELUDE_REGISTRY, TO_BE_INFERRED,
    };
    use crate::types::{Signature, Type};
    use crate::{const_extension_ids, ops, test_file, type_row};
    use rstest::rstest;

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
    #[should_panic] // issue 1225: In serialization we do not distinguish between unknown CustomConst serialized value invalid but known CustomConst serialized values"
    fn hugr_validation_0() {
        // https://github.com/CQCL/hugr/issues/1091 bad case
        let mut hugr: Hugr = serde_json::from_reader(BufReader::new(
            File::open(test_file!("hugr-0.json")).unwrap(),
        ))
        .unwrap();
        assert!(
            hugr.update_validate(&PRELUDE_REGISTRY).is_err(),
            "HUGR should not validate."
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_1() {
        // https://github.com/CQCL/hugr/issues/1091 good case
        let mut hugr: Hugr = serde_json::from_reader(BufReader::new(
            File::open(test_file!("hugr-1.json")).unwrap(),
        ))
        .unwrap();
        assert!(hugr.update_validate(&PRELUDE_REGISTRY).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_2() {
        // https://github.com/CQCL/hugr/issues/1185 bad case
        let mut hugr: Hugr = serde_json::from_reader(BufReader::new(
            File::open(test_file!("hugr-2.json")).unwrap(),
        ))
        .unwrap();
        assert!(
            hugr.update_validate(&PRELUDE_REGISTRY).is_err(),
            "HUGR should not validate."
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_validation_3() {
        // https://github.com/CQCL/hugr/issues/1185 good case
        let mut hugr: Hugr = serde_json::from_reader(BufReader::new(
            File::open(test_file!("hugr-3.json")).unwrap(),
        ))
        .unwrap();
        assert!(hugr.update_validate(&PRELUDE_REGISTRY).is_ok());
    }

    const_extension_ids! {
        const XA: ExtensionId = "EXT_A";
        const XB: ExtensionId = "EXT_B";
    }

    #[rstest]
    #[case([], XA.into())]
    #[case([XA], XA.into())]
    #[case([XB], ExtensionSet::from_iter([XA, XB]))]

    fn infer_single_delta(
        #[case] parent: impl IntoIterator<Item = ExtensionId>,
        #[values(true, false)] remove: bool, // makes no difference when inferring
        #[case] result: ExtensionSet,
    ) {
        let parent = ExtensionSet::from_iter(parent).union(TO_BE_INFERRED.into());
        let (mut h, _) = build_ext_dfg(parent);
        h.infer_extensions(remove).unwrap();
        assert_eq!(h, build_ext_dfg(result.union(PRELUDE_ID.into())).0);
    }

    #[test]
    fn infer_removes_from_delta() {
        let parent = ExtensionSet::from_iter([XA, XB, PRELUDE_ID]);
        let mut h = build_ext_dfg(parent.clone()).0;
        let backup = h.clone();
        h.infer_extensions(false).unwrap();
        assert_eq!(h, backup); // did nothing
        h.infer_extensions(true).unwrap();
        assert_eq!(
            h,
            build_ext_dfg(ExtensionSet::from_iter([XA, PRELUDE_ID])).0
        );
    }

    #[test]
    fn infer_bad_remove() {
        let (mut h, mid) = build_ext_dfg(XB.into());
        let backup = h.clone();
        h.infer_extensions(false).unwrap();
        assert_eq!(h, backup); // did nothing
        let val_res = h.validate(&EMPTY_REG);
        let expected_err = ExtensionError {
            parent: h.root(),
            parent_extensions: XB.into(),
            child: mid,
            child_extensions: ExtensionSet::from_iter([XA, PRELUDE_ID]),
        };
        #[cfg(feature = "extension_inference")]
        assert_eq!(
            val_res,
            Err(ValidationError::ExtensionError(expected_err.clone()))
        );
        #[cfg(not(feature = "extension_inference"))]
        assert!(val_res.is_ok());

        let inf_res = h.infer_extensions(true);
        assert_eq!(inf_res, Err(expected_err));
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
        let mid = h.add_node_with_parent(p, Lift::new(ty.into(), XA));
        h.connect(inp, 0, mid, 0);
        h.connect(mid, 0, out, 0);
        mid
    }

    #[rstest]
    // Base case success: delta inferred for parent equals grandparent.
    #[case([XA], [TO_BE_INFERRED], true, [XA])]
    // Success: delta inferred for parent is subset of grandparent
    #[case([XA, XB], [TO_BE_INFERRED], true, [XA])]
    // Base case failure: infers [XA] for parent but grandparent has disjoint set
    #[case([XB], [TO_BE_INFERRED], false, [XA])]
    // Failure: as previous, but extra "lower bound" on parent that has no effect
    #[case([XB], [XA, TO_BE_INFERRED], false, [XA])]
    // Failure: grandparent ok wrt. child but parent specifies extra lower-bound XB
    #[case([XA], [XB, TO_BE_INFERRED], false, [XA, XB])]
    // Success: grandparent includes extra XB required for parent's "lower bound"
    #[case([XA, XB], [XB, TO_BE_INFERRED], true, [XA, XB])]
    // Success: grandparent is also inferred so can include 'extra' XB from parent
    #[case([TO_BE_INFERRED], [TO_BE_INFERRED, XB], true, [XA, XB])]
    // No inference: extraneous XB in parent is removed so all become [XA].
    #[case([XA], [XA, XB], true, [XA])]
    fn infer_three_generations(
        #[case] grandparent: impl IntoIterator<Item = ExtensionId>,
        #[case] parent: impl IntoIterator<Item = ExtensionId>,
        #[case] success: bool,
        #[case] result: impl IntoIterator<Item = ExtensionId>,
    ) {
        let ty = Type::new_function(Signature::new_endo(type_row![]));
        let grandparent = ExtensionSet::from_iter(grandparent).union(PRELUDE_ID.into());
        let parent = ExtensionSet::from_iter(parent).union(PRELUDE_ID.into());
        let result = ExtensionSet::from_iter(result).union(PRELUDE_ID.into());
        let root_ty = ops::Conditional {
            sum_rows: vec![type_row![]],
            other_inputs: ty.clone().into(),
            outputs: ty.clone().into(),
            extension_delta: grandparent.clone(),
        };
        let mut h = Hugr::new(root_ty.clone());
        let p = h.add_node_with_parent(
            h.root(),
            ops::Case {
                signature: Signature::new_endo(ty.clone()).with_extension_delta(parent),
            },
        );
        add_inliftout(&mut h, p, ty.clone());
        assert!(h.validate_extensions().is_err());
        let backup = h.clone();
        let inf_res = h.infer_extensions(true);
        if success {
            assert!(inf_res.is_ok());
            let expected_p = ops::Case {
                signature: Signature::new_endo(ty).with_extension_delta(result.clone()),
            };
            let mut expected = backup;
            expected.replace_op(p, expected_p).unwrap();
            let expected_gp = ops::Conditional {
                extension_delta: result,
                ..root_ty
            };
            expected.replace_op(h.root(), expected_gp).unwrap();

            assert_eq!(h, expected);
        } else {
            assert_eq!(
                inf_res,
                Err(ExtensionError {
                    parent: h.root(),
                    parent_extensions: grandparent,
                    child: p,
                    child_extensions: result
                })
            );
        }
    }
}

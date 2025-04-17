use std::collections::{BTreeSet, VecDeque};

use delegate::delegate;
use derive_more::From;
use itertools::Itertools;
use relrc::{HistoryGraph, RelRc};
use thiserror::Error;

use super::{Patch, PersistentHugr, PersistentReplacement, PointerEqResolver};
use crate::{Hugr, Node};

/// A copyable handle to a [`Patch`] vertex within a [`PatchStateSpace`]
pub(super) type PatchId = relrc::NodeId;

/// A Hugr node within a patch of the patch state space
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
pub struct PatchNode(pub PatchId, pub Node);

impl std::fmt::Display for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// The data stored in a [`Patch`], either the base [`Hugr`] (on which all other
/// patches are based), or a [`PersistentReplacement`]
#[derive(Debug, Clone, From)]
pub(super) enum PatchData {
    Base(Hugr),
    Replacement(PersistentReplacement),
}

/// A set of patches with directed (acyclic) dependencies between them.
///
/// Vertices in the [`PatchStateSpace`] are [`Patch`]es and there is an edge
/// from a patch `p1` to a patch `p2` if `p1` must be applied before `p2`: in
/// other words, if the `p2` deletes nodes that are introduced in `p1`. We say
/// `p2` depends on (or is a child of) `p1`.
///
/// A [`PatchStateSpace`] always has a unique base patch (the root of the
/// graph). All other patches are [`PersistentReplacement`]s that apply on top
/// of it. Patches are stored as [`RelRc`]s: they are reference-counted pointers
/// to the patch data that also maintain strong references to the patch's
/// parents. This means that patches can be cloned cheaply and dropped freely;
/// the memory of a patch will be released whenever no other patch in scope
/// depends on it.
///
/// Patches in a [`PatchStateSpace`] DO NOT represent a valid history in the
/// general case: pairs of patches may be mutually exclusive if they modify the
/// same subgraph. Use [`Self::try_extract_hugr`] to get a [`PersistentHugr`]
/// with a set of compatible patches.
#[derive(Clone, Debug)]
pub struct PatchStateSpace {
    /// A set of patches with directed (acyclic) dependencies between them.
    ///
    /// Each patch is stored as a [`RelRc`].
    graph: HistoryGraph<PatchData, (), PointerEqResolver>,
    /// The unique root of the patch graph.
    ///
    /// The only patch in the graph with variant `PatchData::Base`. All other
    /// patches are `PatchData::Replacement`s, and are descendants of this.
    base_patch: PatchId,
}

impl PatchStateSpace {
    /// Create a new patch state space with a single base patch.
    pub fn with_base(hugr: Hugr) -> Self {
        let patch = RelRc::new(hugr.into());
        let graph = HistoryGraph::new([patch.clone()], PointerEqResolver);
        let base_patch = graph
            .all_node_ids()
            .exactly_one()
            .ok()
            .expect("just added one patch");
        Self { graph, base_patch }
    }

    /// Create a new patch state space from a set of patches.
    ///
    /// Return a [`InvalidPatch::NonUniqueBase`] error if the patches do
    /// not share a unique common ancestor base patch.
    pub fn try_from_patches(
        patches: impl IntoIterator<Item = Patch>,
    ) -> Result<Self, InvalidPatch> {
        let graph = HistoryGraph::new(patches.into_iter().map_into(), PointerEqResolver);
        let base_patches = graph
            .all_node_ids()
            .filter(|&id| matches!(graph.get_node(id).value(), PatchData::Base(_)))
            .collect_vec();
        if base_patches.len() != 1 {
            Err(InvalidPatch::NonUniqueBase(base_patches.len()))
        } else {
            let base_patch = base_patches[0];
            Ok(Self { graph, base_patch })
        }
    }

    /// Add a replacement patch to the graph.
    ///
    /// Return an [`InvalidPatch::EmptyReplacement`] error if the replacement
    /// is empty.
    pub fn try_add_replacement(
        &mut self,
        replacement: PersistentReplacement,
    ) -> Result<PatchId, InvalidPatch> {
        Ok(self.add_patch(Patch::try_from_replacement(replacement, self)?))
    }

    /// Add a set of replacement patches (or a [`PersistentHugr`]) to the graph.
    pub fn extend(&mut self, patches: impl IntoIterator<Item = Patch>) {
        // TODO: make this more efficient
        for patch in patches {
            self.add_patch(patch);
        }
    }

    /// Check if `patch` is in the patch state space.
    pub fn contains(&self, patch: &Patch) -> bool {
        self.graph.contains(patch.as_relrc())
    }

    /// Extract a `PersistentHugr` from this state space, consisting of
    /// `patches` and their ancestors.
    ///
    /// All patches in the resulting `PersistentHugr` are guaranteed to be
    /// compatible. If the selected patches would include two patches which
    /// are incompatible, a [`InvalidPatch::IncompatibleHistory`] error is
    /// returned.
    pub fn try_extract_hugr(
        &self,
        patches: impl IntoIterator<Item = PatchId>,
    ) -> Result<PersistentHugr, InvalidPatch> {
        // Define patches as the set of all ancestors of the given patches
        let patches = get_all_ancestors(&self.graph, patches);

        // Check that all patches are compatible
        for &patch_id in &patches {
            if let Some(node) = find_conflicting_node(self, patch_id, &patches) {
                return Err(InvalidPatch::IncompatibleHistory(patch_id, node));
            }
        }

        let patches = patches
            .into_iter()
            .map(|id| self.get_patch(id).as_relrc().clone());
        let subgraph = HistoryGraph::new(patches, PointerEqResolver);

        Ok(PersistentHugr::from_patch_graph_unsafe(Self {
            graph: subgraph,
            base_patch: self.base_patch,
        }))
    }

    /// Get the base patch ID.
    pub fn base(&self) -> PatchId {
        self.base_patch
    }

    /// Get the base [`Hugr`].
    pub fn base_hugr(&self) -> &Hugr {
        let PatchData::Base(hugr) = self.graph.get_node(self.base_patch).value() else {
            panic!("base patch is not a base hugr");
        };
        hugr
    }

    /// Get the base patch.
    pub fn base_patch(&self) -> &Patch {
        self.get_patch(self.base_patch)
    }

    /// Get the patch with ID `patch_id`.
    pub fn get_patch(&self, patch_id: PatchId) -> &Patch {
        self.graph.get_node(patch_id).into()
    }

    /// Add a patch to the state space.
    pub fn add_patch(&mut self, patch: Patch) -> PatchId {
        self.graph.insert_node(patch.into())
    }

    /// Get an iterator over all patch IDs in the state space.
    pub fn all_patch_ids(&self) -> impl Iterator<Item = PatchId> + '_ {
        self.graph.all_node_ids()
    }

    /// Get the set of nodes removed by `patch_id` in `parent`.
    pub(super) fn deleted_nodes(
        &self,
        patch_id: PatchId,
        parent: PatchId,
    ) -> impl Iterator<Item = Node> + '_ {
        let patch = self.get_patch(patch_id);
        let ret = patch
            .deleted_nodes()
            .filter(move |n| n.0 == parent)
            .map(|n| n.1);
        Some(ret).into_iter().flatten()
    }

    delegate! {
        to self.graph {
            /// Get the parents of `patch_id`
            pub fn parents(&self, patch_id: PatchId) -> impl Iterator<Item = PatchId> + '_;
            /// Get the children of `patch_id`
            pub fn children(&self, patch_id: PatchId) -> impl Iterator<Item = PatchId> + '_;
        }
    }

    pub(super) fn as_history_graph(&self) -> &HistoryGraph<PatchData, (), PointerEqResolver> {
        &self.graph
    }
}

fn get_all_ancestors<N, E, R>(
    graph: &HistoryGraph<N, E, R>,
    patches: impl IntoIterator<Item = relrc::NodeId>,
) -> BTreeSet<relrc::NodeId> {
    let mut queue = VecDeque::from_iter(patches);
    let mut ancestors = BTreeSet::from_iter(queue.iter().copied());
    while let Some(patch) = queue.pop_front() {
        for parent in graph.parents(patch) {
            if ancestors.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    ancestors
}

/// An error that occurs when trying to add a patch to a patch history.
///
/// This can happen when:
/// - The patch requires a parent that is not present in the current history.
/// - The patch is incompatible with one or more patches already in the history.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum InvalidPatch {
    /// The patch conflicts with existing patches in the history.
    #[error("Incompatible history: children patches conflict in {0:?} conflict in {1:?}")]
    IncompatibleHistory(PatchId, Node),

    /// The patch has a parent not present in the history.
    #[error("Missing parent patch at position {0}")]
    UnknownParent(usize),

    /// The patch is not a replacement.
    #[error("Patch is not a replacement")]
    NotReplacement,

    /// The set of patches contains zero or more than one base patches.
    #[error("{0} base patches found (should be 1)")]
    NonUniqueBase(usize),

    /// The patch is an empty replacement.
    #[error("Not allowed: empty replacement")]
    EmptyReplacement,
}

/// Find a node that is invalidated by more than one child of `patch_id`.
fn find_conflicting_node(
    graph: &PatchStateSpace,
    patch_id: PatchId,
    patches: &BTreeSet<PatchId>,
) -> Option<Node> {
    let mut all_deleted = BTreeSet::new();
    let mut children = graph
        .children(patch_id)
        .filter(|&child_id| patches.contains(&child_id));

    children.find_map(|child_id| {
        let mut deleted_nodes = graph.deleted_nodes(child_id, patch_id);
        deleted_nodes.find(|&n| !all_deleted.insert(n))
    })
}

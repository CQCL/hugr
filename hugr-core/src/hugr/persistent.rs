//! Persistent data structure for Hugr mutations.
//!
//! This module provides a persistent data structure [`PersistentHugr`] that
//! implements [`crate::HugrView`]; mutations to the data are stored as a set of
//! [`Patch`]es along with dependencies between the patches.
//!
//! As a result of persistency, the entire mutation history of a hugr can be
//! traversed and references to previous versions of the data remain valid even
//! as new patches are applied.
//!
//! The data structure underlying [`PersistentHugr`], which stores the history
//! of all applied patches is [`PatchStateSpace`]. Multiple [`PersistentHugr`]
//! can be stored within a single [`PatchStateSpace`], which allows for the
//! efficient exploration of the space of all possible graph rewrites.
//!
//! ## Overlapping patches
//!
//! In general, [`PatchStateSpace`] may contain overlapping patches. Such
//! mutations are mutually exclusive as they modify the same nodes. It is
//! therefore not possible to apply all patches in a [`PatchStateSpace`]
//! simultaneously. A [`PersistentHugr`] on the other hand always corresponds to
//! a subgraph of a [`PatchStateSpace`] that is guaranteed to contain only
//! non-overlapping, compatible patches. By applying all patches in a
//! [`PersistentHugr`], we can materialize a [`Hugr`]. Traversing the
//! materialized Hugr is equivalent to using the [`crate::HugrView`]
//! implementation of the corresponding [`PersistentHugr`].
//!
//! ## Summary of data types
//!
//! - [`Patch`] A modification to a [`Hugr`] (currently a [`SimpleReplacement`])
//!   that forms the atomic unit of change for a [`PersistentHugr`] (like a
//!   commit in git). This is a reference-counted value that is cheap to clone
//!   and will be freed when the last reference is dropped.
//! - [`PersistentHugr`] A persistent data structure that implements
//!   [`crate::HugrView`] and can be used as a drop-in replacement for a
//!   [`crate::Hugr`] for read-only access and mutations through the
//!   [`VerifyPatch`] and [`ApplyPatch`] traits. It maintains the invariant that
//!   all contained patches are compatible.
//! - [`PatchStateSpace`] Stores patches, recording the dependencies between
//!   them. Includes the base Hugr and any number of possibly incompatible
//!   (overlapping) patches. Unlike a [`PersistentHugr`], a state space can
//!   contain mutually exclusive patches.
//!
//! ## Usage
//!
//! A [`PersistentHugr`] can be created from a base Hugr using
//! [`PersistentHugr::with_base`]. Replacements can then be applied to it
//! using [`PersistentHugr::add_replacement`]. Alternatively, if you already
//! have a populated state space, use [`PersistentHugr::try_new`] to create a
//! new Hugr with those patches.
//!
//! Commit a sequence of patches to a state space by mering a [`PersistentHugr`]
//! into it using [`PatchStateSpace::extend`] or directly using
//! [`PatchStateSpace::add_patch`].
//!
//! To obtain a [`PersistentHugr`] from your state space, use
//! [`PatchStateSpace::try_extract_hugr`]. A [`PersistentHugr`] can always be
//! materialised into a "real" Hugr using [`PersistentHugr::to_hugr`].

mod resolver;
mod state_space;
// mod trait_impls;

#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, BTreeSet},
    mem, vec,
};

use delegate::delegate;
use derive_more::derive::From;
use itertools::Itertools;
use relrc::RelRc;
pub use state_space::{InvalidPatch, PatchNode, PatchStateSpace};
use state_space::{PatchData, PatchId};

pub use resolver::PointerEqResolver;

use crate::{
    hugr::patch::{simple_replace, ApplyPatch, VerifyPatch},
    Hugr, HugrView, Node, SimpleReplacement,
};

/// A replacement operation that can be applied to a [`PersistentHugr`].
pub type PersistentReplacement = SimpleReplacement<PatchNode>;

/// A patch that can be applied to a [`PersistentHugr`] or added to a
/// [`PatchStateSpace`].
///
/// Patches are cheap to clone: they are reference-counted pointers to the
/// patch data. They also maintain strong references to the ancestor patches
/// that the patch may depend on (i.e. other replacements that must be applied
/// before `self` can be applied).
///
/// Currently, patches must be [`SimpleReplacement`]s.
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct Patch(RelRc<PatchData, ()>);

impl Patch {
    /// Create a patch from a replacement
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidPatch::EmptyReplacement`] error.
    ///
    /// Requires a reference to the patch graph instance that the nodes in
    /// `replacement` refer to.
    pub fn try_from_replacement(
        replacement: PersistentReplacement,
        graph: &PatchStateSpace,
    ) -> Result<Patch, InvalidPatch> {
        if replacement.subgraph().nodes().is_empty() {
            return Err(InvalidPatch::EmptyReplacement);
        }
        let parent_ids = replacement.invalidation_set().map(|n| n.0).unique();
        let parents = parent_ids
            .map(|id| graph.get_patch(id))
            .cloned()
            .collect_vec();
        let rc = RelRc::with_parents(
            replacement.into(),
            parents.into_iter().map(|p| (p.into(), ())),
        );
        Ok(Self(rc))
    }

    fn as_relrc(&self) -> &RelRc<PatchData, ()> {
        &self.0
    }

    fn all_parents(&self) -> impl Iterator<Item = &Patch> + '_ {
        self.0.all_parents().map_into()
    }

    /// Get the set of nodes removed by `self`.
    fn deleted_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        let replacement = match self.0.value() {
            PatchData::Base(_) => None,
            PatchData::Replacement(replacement) => Some(replacement),
        };
        replacement
            .into_iter()
            .flat_map(|r| r.subgraph().nodes())
            .copied()
    }

    delegate! {
        to self.0 {
            fn value(&self) -> &PatchData;
        }
    }
}

impl From<Patch> for RelRc<PatchData, ()> {
    fn from(patch: Patch) -> Self {
        patch.0
    }
}

impl<'a> From<&'a RelRc<PatchData, ()>> for &'a Patch {
    fn from(rc: &'a RelRc<PatchData, ()>) -> Self {
        // SAFETY: Patch is a transparent wrapper around RelRc
        unsafe { mem::transmute(rc) }
    }
}

/// A Hugr-like object that supports persistent mutation.
///
/// When mutations are applied to a [`PersistentHugr`], the object is mutated
/// as expected but all references to previous versions of the object remain
/// valid. Furthermore, older versions of the data can be recovered by
/// traversing the object's history with [`Self::as_state_space`].
///
/// Multiple references to various versions of a Hugr can be maintained in
/// parallel by extracting them from a shared [`PatchStateSpace`].
///
/// ## Supported access and mutation
///
/// [`PersistentHugr`] implements [`crate::HugrView`], so that it can used as
/// a drop-in substitute for a Hugr wherever read-only access is required. It
/// does not implement [`crate::hugr::HugrMut`], however. Mutations must be
/// performed by applying patches (see [`crate::hugr::patch::VerifyPatch`] and
/// [`crate::hugr::patch::ApplyPatch`]). Currently, only [`SimpleReplacement`]
/// patches are supported. You can use [`Self::add_replacement`] to add a patch
/// to `self`, or use the aforementioned patch traits.
///
/// ## Patches and history
///
/// A [`PersistentHugr`] is composed of a unique base Hugr, along with a set of
/// all mutations applied to it. All mutations are stored in the form of patches
/// that apply on top of the base Hugr. You may think of it as a "queue" of
/// patches: whenever a patch is "applied", it is in reality just added to the
/// queue. In practice, the total order of the queue is irrelevant, as patches
/// only depend on a subset of the previously applied patches. This creates a
/// partial order on the patches---a directed acyclic graph that we call the
/// "patch history". A patch history is in effect a subgraph of a patch state
/// space, with the additional invariant that all patches within the history
/// are compatible.
#[derive(Clone, Debug)]
pub struct PersistentHugr {
    /// The state space of all patches.
    ///
    /// Invariant: all patches are "compatible", meaning that no two patches
    /// invalidate the same node.
    patch_graph: PatchStateSpace,
}

impl PersistentHugr {
    /// Create a [`PersistentHugr`] with `hugr` as its base Hugr.
    ///
    /// All replacements added in the future will apply on top of `hugr`.
    pub fn with_base(hugr: Hugr) -> Self {
        let patch_graph = PatchStateSpace::with_base(hugr);
        Self { patch_graph }
    }

    /// Create a [`PersistentHugr`] from a list of patch ids.
    ///
    /// `Self` will correspond to the Hugr obtained by applying the history
    /// of all patches with the given IDs (and their ancestors).
    ///
    /// If the history of the patches would include two patches which are
    /// incompatible, an error is returned.
    pub fn try_new(patches: impl IntoIterator<Item = Patch>) -> Result<Self, InvalidPatch> {
        let graph = PatchStateSpace::try_from_patches(patches)?;
        graph.try_extract_hugr(graph.all_patch_ids())
    }

    /// Construct a [`PersistentHugr`] from a [`PatchStateSpace`].
    ///
    /// Does not check that the patches are compatible.
    fn from_patch_graph_unsafe(patch_graph: PatchStateSpace) -> Self {
        Self { patch_graph }
    }

    /// Add a replacement to `self`.
    ///
    /// The effect of this is equivalent to applying `replacement` to the
    /// equivalent Hugr, i.e. `self.to_hugr().apply(replacement)` is
    /// equivalent to `self.add_replacement(replacement).to_hugr()`.
    ///
    /// This may panic if the replacement is invalid. Use
    /// [`PersistentHugr::try_add_replacement`] instead for more graceful error
    /// handling.
    pub fn add_replacement(&mut self, replacement: PersistentReplacement) -> PatchId {
        self.try_add_replacement(replacement)
            .expect("invalid replacement")
    }

    /// Add a single patch to `self`.
    ///
    /// Return the ID of the patch if it was added successfully. This may
    /// return the following errors:
    /// - a [`InvalidPatch::IncompatibleHistory`] error if the patch is
    ///   incompatible with another patch already in `self`, or a
    /// - a [`InvalidPatch::UnknownParent`] error if one of the patches that
    ///   `replacement` applies to is not contained in `self`.
    pub fn try_add_replacement(
        &mut self,
        replacement: PersistentReplacement,
    ) -> Result<PatchId, InvalidPatch> {
        let patch = Patch::try_from_replacement(replacement, &self.patch_graph)?;
        // Recover ref to the replacement
        let PatchData::Replacement(replacement) = patch.value() else {
            unreachable!("created patch from replacement")
        };

        // Check that all parents are in the history
        if let Some(pos) = patch
            .all_parents()
            .position(|p| !self.patch_graph.contains(p))
        {
            return Err(InvalidPatch::UnknownParent(pos));
        }

        // Check that `patch` does not conflict with siblings at any of its parents
        let new_invalid_nodes = replacement
            .invalidation_set()
            .map(|PatchNode(id, node)| (id, node))
            .into_grouping_map()
            .collect::<BTreeSet<_>>();
        for (parent, new_invalid_nodes) in new_invalid_nodes {
            let invalidation_set = self.deleted_nodes(parent).collect();
            if let Some(&node) = new_invalid_nodes.intersection(&invalidation_set).next() {
                return Err(InvalidPatch::IncompatibleHistory(parent, node));
            }
        }

        Ok(self.patch_graph.add_patch(patch))
    }

    /// Convert this `PersistentHugr` to a materialized Hugr by applying all
    /// patches in `self`.
    ///
    /// This operation may be expensive and should be avoided in
    /// performance-critical paths. For read-only views into the data, rely
    /// instead on the [`crate::HugrView`] implementation when possible.
    pub fn to_hugr(&self) -> Hugr {
        let mut hugr = self.patch_graph.base_hugr().clone();
        let mut node_map = BTreeMap::from_iter(
            hugr.nodes()
                .map(|n| (PatchNode(self.patch_graph.base(), n), n)),
        );
        for patch_id in self.toposort_patches() {
            let PatchData::Replacement(repl) = self.patch_graph.get_patch(patch_id).value() else {
                continue;
            };
            let repl = repl.map_host_nodes(|n| node_map[&n]);
            let simple_replace::ApplyOutcome {
                node_map: new_node_map,
                ..
            } = repl.apply(&mut hugr).expect("invalid replacement");
            for (old_node, new_node) in new_node_map {
                let old_patch_node = PatchNode(patch_id, old_node);
                node_map.insert(old_patch_node, new_node);
            }
        }
        hugr
    }

    /// Get a reference to the underlying state space of `self`.
    pub fn as_state_space(&self) -> &PatchStateSpace {
        &self.patch_graph
    }

    /// Convert `self` into its underlying [`PatchStateSpace`].
    pub fn into_state_space(self) -> PatchStateSpace {
        self.patch_graph
    }

    delegate! {
        to self.patch_graph {
            /// Check if `patch` is in the PersistentHugr.
            pub fn contains(&self, patch: &Patch) -> bool;
            /// Get the base patch ID.
            pub fn base(&self) -> PatchId;
            /// Get the base [`Hugr`].
            pub fn base_hugr(&self) -> &Hugr;
            /// Get the base patch.
            pub fn base_patch(&self) -> &Patch;
        }
    }

    /// Get all patches in `self` in topological order.
    fn toposort_patches(&self) -> Vec<PatchId> {
        petgraph::algo::toposort(self.patch_graph.as_history_graph(), None)
            .expect("history is a DAG")
    }

    /// Iterator over the patch IDs in the history.
    ///
    /// The patches are not guaranteed to be in any particular order.
    fn patch_ids(&self) -> impl Iterator<Item = PatchId> + '_ {
        self.patch_graph.all_patch_ids()
    }

    /// Get the set of nodes of `patch_id` that are removed by children patches
    /// of `patch_id`.
    ///
    /// This is almost the invalidation set returned by
    /// [`SimpleReplacement::invalidation_set`]`, but does not include the
    /// output nodes. It is sufficient to consider the set of deleted nodes,
    /// rather than the full invalidation set because we disallow empty
    /// subgraphs in replacements.
    fn deleted_nodes<'a>(&'a self, patch_id: PatchId) -> impl Iterator<Item = Node> + 'a {
        let children = self
            .patch_graph
            .children(patch_id)
            .filter(|child_id| self.patch_ids().contains(child_id));
        children
            .flat_map(move |child_id| self.patch_graph.deleted_nodes(child_id, patch_id))
            .unique()
    }
}

impl IntoIterator for PersistentHugr {
    type Item = Patch;

    type IntoIter = vec::IntoIter<Patch>;

    fn into_iter(self) -> Self::IntoIter {
        self.patch_graph
            .all_patch_ids()
            .map(|id| self.patch_graph.get_patch(id).clone())
            .collect_vec()
            .into_iter()
    }
}

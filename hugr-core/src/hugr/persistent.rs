//! Persistent data structure for HUGR mutations.
//!
//! This module provides a persistent data structure [`PersistentHugr`] that
//! implements [`crate::HugrView`]; mutations to the data are stored
//! persistently as a set of [`Commit`]s along with the dependencies between the
//! commits.
//!
//! As a result of persistency, the entire mutation history of a HUGR can be
//! traversed and references to previous versions of the data remain valid even
//! as the HUGR graph is "mutated" by applying patches: the patches are in
//! effect added to the history as new commits.
//!
//! The data structure underlying [`PersistentHugr`], which stores the history
//! of all commits, is [`CommitStateSpace`]. Multiple [`PersistentHugr`] can be
//! stored within a single [`CommitStateSpace`], which allows for the efficient
//! exploration of the space of all possible graph rewrites.
//!
//! ## Overlapping commits
//!
//! In general, [`CommitStateSpace`] may contain overlapping commits. Such
//! mutations are mutually exclusive as they modify the same nodes. It is
//! therefore not possible to apply all commits in a [`CommitStateSpace`]
//! simultaneously. A [`PersistentHugr`] on the other hand always corresponds to
//! a subgraph of a [`CommitStateSpace`] that is guaranteed to contain only
//! non-overlapping, compatible commits. By applying all commits in a
//! [`PersistentHugr`], we can materialize a [`Hugr`]. Traversing the
//! materialized HUGR is equivalent to using the [`crate::HugrView`]
//! implementation of the corresponding [`PersistentHugr`].
//!
//! ## Summary of data types
//!
//! - [`Commit`] A modification to a [`Hugr`] (currently a
//!   [`SimpleReplacement`]) that forms the atomic unit of change for a
//!   [`PersistentHugr`] (like a commit in git). This is a reference-counted
//!   value that is cheap to clone and will be freed when the last reference is
//!   dropped.
//! - [`PersistentHugr`] A data structure that implements [`crate::HugrView`]
//!   and can be used as a drop-in replacement for a [`crate::Hugr`] for
//!   read-only access and mutations through the [`PatchVerification`] and
//!   [`Patch`] traits. Mutations are stored as a history of commits. Unlike
//!   [`CommitStateSpace`], it maintains the invariant that all contained
//!   commits are compatible with eachother.
//! - [`CommitStateSpace`] Stores commits, recording the dependencies between
//!   them. Includes the base HUGR and any number of possibly incompatible
//!   (overlapping) commits. Unlike a [`PersistentHugr`], a state space can
//!   contain mutually exclusive commits.
//!
//! ## Usage
//!
//! A [`PersistentHugr`] can be created from a base HUGR using
//! [`PersistentHugr::with_base`]. Replacements can then be applied to it
//! using [`PersistentHugr::add_replacement`]. Alternatively, if you already
//! have a populated state space, use [`PersistentHugr::try_new`] to create a
//! new HUGR with those commits.
//!
//! Add a sequence of commits to a state space by merging a [`PersistentHugr`]
//! into it using [`CommitStateSpace::extend`] or directly using
//! [`CommitStateSpace::try_add_commit`].
//!
//! To obtain a [`PersistentHugr`] from your state space, use
//! [`CommitStateSpace::try_extract_hugr`]. A [`PersistentHugr`] can always be
//! materialized into a [`Hugr`] type using [`PersistentHugr::to_hugr`].

mod resolver;
mod state_space;

use std::{
    collections::{BTreeMap, BTreeSet},
    mem, vec,
};

use delegate::delegate;
use derive_more::derive::From;
use itertools::Itertools;
use relrc::RelRc;
use state_space::{CommitData, CommitId};
pub use state_space::{CommitStateSpace, InvalidCommit, PatchNode};

pub use resolver::PointerEqResolver;

use crate::{
    Hugr, HugrView, Node, SimpleReplacement,
    hugr::patch::{Patch, PatchVerification, simple_replace},
};

/// A replacement operation that can be applied to a [`PersistentHugr`].
pub type PersistentReplacement = SimpleReplacement<PatchNode>;

/// A patch that can be applied to a [`PersistentHugr`] or a
/// [`CommitStateSpace`] as an atomic commit.
///
/// Commits are cheap to clone: they are reference-counted pointers to the
/// patch data. They also maintain strong references to the ancestor commits
/// that the patch may depend on (i.e. other patches that must be applied
/// before `self` can be applied).
///
/// Currently, patches must be [`SimpleReplacement`]s.
#[derive(Debug, Clone, From)]
#[repr(transparent)]
pub struct Commit(RelRc<CommitData, ()>);

impl Commit {
    /// Create a commit from a simple replacement.
    ///
    /// Requires a reference to the commit state space that the nodes in
    /// `replacement` refer to.
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidCommit::EmptyReplacement`] error.
    ///
    /// If any of the parents of the replacement are not in the commit state
    /// space, this function will return an [`InvalidCommit::UnknownParent`]
    /// error.
    pub fn try_from_replacement(
        replacement: PersistentReplacement,
        graph: &CommitStateSpace,
    ) -> Result<Commit, InvalidCommit> {
        if replacement.subgraph().nodes().is_empty() {
            return Err(InvalidCommit::EmptyReplacement);
        }
        let parent_ids = replacement.invalidation_set().map(|n| n.0).unique();
        let parents = parent_ids
            .map(|id| {
                if graph.contains_id(id) {
                    Ok(graph.get_commit(id).clone())
                } else {
                    Err(InvalidCommit::UnknownParent(id))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rc = RelRc::with_parents(
            replacement.into(),
            parents.into_iter().map(|p| (p.into(), ())),
        );
        Ok(Self(rc))
    }

    fn as_relrc(&self) -> &RelRc<CommitData, ()> {
        &self.0
    }

    fn replacement(&self) -> Option<&PersistentReplacement> {
        match self.0.value() {
            CommitData::Base(_) => None,
            CommitData::Replacement(replacement) => Some(replacement),
        }
    }

    /// Get the set of nodes invalidated by applying `self`.
    fn invalidation_set(&self) -> impl Iterator<Item = PatchNode> + '_ {
        self.replacement()
            .into_iter()
            .flat_map(|r| r.invalidation_set())
    }

    delegate! {
        to self.0 {
            fn value(&self) -> &CommitData;
        }
    }
}

impl From<Commit> for RelRc<CommitData, ()> {
    fn from(Commit(data): Commit) -> Self {
        data
    }
}

impl<'a> From<&'a RelRc<CommitData, ()>> for &'a Commit {
    fn from(rc: &'a RelRc<CommitData, ()>) -> Self {
        // SAFETY: Commit is a transparent wrapper around RelRc
        unsafe { mem::transmute(rc) }
    }
}

/// A HUGR-like object that tracks its mutation history.
///
/// When mutations are applied to a [`PersistentHugr`], the object is mutated
/// as expected but all references to previous versions of the object remain
/// valid. Furthermore, older versions of the data can be recovered by
/// traversing the object's history with [`Self::as_state_space`].
///
/// Multiple references to various versions of a Hugr can be maintained in
/// parallel by extracting them from a shared [`CommitStateSpace`].
///
/// ## Supported access and mutation
///
/// [`PersistentHugr`] implements [`crate::HugrView`], so that it can used as
/// a drop-in substitute for a Hugr wherever read-only access is required. It
/// does not implement [`HugrMut`](crate::hugr::HugrMut), however. Mutations
/// must be performed by applying patches (see [`PatchVerification`] and
/// [`Patch`]). Currently, only [`SimpleReplacement`] patches are supported. You
/// can use [`Self::add_replacement`] to add a patch to `self`, or use the
/// aforementioned patch traits.
///
/// ## Patches, commits and history
///
/// A [`PersistentHugr`] is composed of a unique base HUGR, along with a set of
/// mutations applied to it. All mutations are stored in the form of commits
/// that store the patches applied on top of a base HUGR. You may think of it
/// as a "queue" of patches: whenever the patch of a commit is "applied", it is
/// in reality just added to the queue. In practice, the total order of the
/// queue is irrelevant, as patches only depend on a subset of the previously
/// applied patches. This creates a partial order on the commits: a directed
/// acyclic graph that we call the "commit history". A commit history is in
/// effect a subgraph of a commit state space, with the additional invariant
/// that all commits within the history are compatible.
#[derive(Clone, Debug)]
pub struct PersistentHugr {
    /// The state space of all commits.
    ///
    /// Invariant: all commits are "compatible", meaning that no two patches
    /// invalidate the same node.
    state_space: CommitStateSpace,
}

impl PersistentHugr {
    /// Create a [`PersistentHugr`] with `hugr` as its base HUGR.
    ///
    /// All replacements added in the future will apply on top of `hugr`.
    pub fn with_base(hugr: Hugr) -> Self {
        let state_space = CommitStateSpace::with_base(hugr);
        Self { state_space }
    }

    /// Create a [`PersistentHugr`] from a list of commits.
    ///
    /// `Self` will correspond to the HUGR obtained by applying the patches of
    /// the given commits and of all their ancestors.
    ///
    /// If the state space of the commits would include two commits which are
    /// incompatible, or if the commits do not share a common base HUGR, then
    /// an error is returned.
    pub fn try_new(commits: impl IntoIterator<Item = Commit>) -> Result<Self, InvalidCommit> {
        let graph = CommitStateSpace::try_from_commits(commits)?;
        graph.try_extract_hugr(graph.all_commit_ids())
    }

    /// Construct a [`PersistentHugr`] from a [`CommitStateSpace`].
    ///
    /// Does not check that the commits are compatible.
    fn from_state_space_unsafe(state_space: CommitStateSpace) -> Self {
        Self { state_space }
    }

    /// Add a replacement to `self`.
    ///
    /// The effect of this is equivalent to applying `replacement` to the
    /// equivalent HUGR, i.e. `self.to_hugr().apply(replacement)` is
    /// equivalent to `self.add_replacement(replacement).to_hugr()`.
    ///
    /// This will panic if the replacement is invalid. Use
    /// [`PersistentHugr::try_add_replacement`] instead for more graceful error
    /// handling.
    pub fn add_replacement(&mut self, replacement: PersistentReplacement) -> CommitId {
        self.try_add_replacement(replacement)
            .expect("invalid replacement")
    }

    /// Add a replacement to `self`, with error handling.
    ///
    /// Return the ID of the commit if it was added successfully. This may
    /// return the following errors:
    /// - a [`InvalidCommit::IncompatibleHistory`] error if the replacement is
    ///   incompatible with another commit already in `self`, or
    /// - a [`InvalidCommit::UnknownParent`] error if one of the commits that
    ///   `replacement` applies on top of is not contained in `self`.
    pub fn try_add_replacement(
        &mut self,
        replacement: PersistentReplacement,
    ) -> Result<CommitId, InvalidCommit> {
        // Check that `replacement` does not conflict with siblings at any of its
        // parents
        let new_invalid_nodes = replacement
            .subgraph()
            .nodes()
            .iter()
            .map(|&PatchNode(id, node)| (id, node))
            .into_grouping_map()
            .collect::<BTreeSet<_>>();
        for (parent, new_invalid_nodes) in new_invalid_nodes {
            let invalidation_set = self.invalidation_set(parent).collect();
            if let Some(&node) = new_invalid_nodes.intersection(&invalidation_set).next() {
                return Err(InvalidCommit::IncompatibleHistory(parent, node));
            }
        }

        self.state_space.try_add_replacement(replacement)
    }

    /// Convert this `PersistentHugr` to a materialized HUGR by applying all
    /// commits in `self`.
    ///
    /// This operation may be expensive and should be avoided in
    /// performance-critical paths. For read-only views into the data, rely
    /// instead on the [`crate::HugrView`] implementation when possible.
    pub fn to_hugr(&self) -> Hugr {
        let mut hugr = self.state_space.base_hugr().clone();
        let mut node_map = BTreeMap::from_iter(
            hugr.nodes()
                .map(|n| (PatchNode(self.state_space.base(), n), n)),
        );
        for commit_id in self.toposort_commits() {
            let Some(repl) = self.state_space.get_commit(commit_id).replacement() else {
                continue;
            };
            let repl = repl.map_host_nodes(|n| node_map[&n]);

            let simple_replace::Outcome {
                node_map: new_node_map,
                ..
            } = repl.apply(&mut hugr).expect("invalid replacement");
            for (old_node, new_node) in new_node_map {
                let old_patch_node = PatchNode(commit_id, old_node);
                node_map.insert(old_patch_node, new_node);
            }
        }
        hugr
    }

    /// Get a reference to the underlying state space of `self`.
    pub fn as_state_space(&self) -> &CommitStateSpace {
        &self.state_space
    }

    /// Convert `self` into its underlying [`CommitStateSpace`].
    pub fn into_state_space(self) -> CommitStateSpace {
        self.state_space
    }

    delegate! {
        to self.state_space {
            /// Check if `commit` is in the PersistentHugr.
            pub fn contains(&self, commit: &Commit) -> bool;
            /// Get the base commit ID.
            pub fn base(&self) -> CommitId;
            /// Get the base [`Hugr`].
            pub fn base_hugr(&self) -> &Hugr;
            /// Get the base commit.
            pub fn base_commit(&self) -> &Commit;
        }
    }

    /// Get all commits in `self` in topological order.
    fn toposort_commits(&self) -> Vec<CommitId> {
        petgraph::algo::toposort(self.state_space.as_history_graph(), None)
            .expect("history is a DAG")
    }

    /// Iterator over the commit IDs in the history.
    ///
    /// The commits are not guaranteed to be in any particular order.
    fn commit_ids(&self) -> impl Iterator<Item = CommitId> + '_ {
        self.state_space.all_commit_ids()
    }

    /// Get the set of nodes of `commit_id` that are invalidated by applying
    /// children commits of `commit_id`.
    fn invalidation_set(&self, commit_id: CommitId) -> impl Iterator<Item = Node> + '_ {
        let children = self
            .state_space
            .children(commit_id)
            .filter(|child_id| self.commit_ids().contains(child_id));
        children
            .flat_map(move |child_id| self.state_space.invalidation_set(child_id, commit_id))
            .unique()
    }
}

impl IntoIterator for PersistentHugr {
    type Item = Commit;

    type IntoIter = vec::IntoIter<Commit>;

    fn into_iter(self) -> Self::IntoIter {
        self.state_space
            .all_commit_ids()
            .map(|id| self.state_space.get_commit(id).clone())
            .collect_vec()
            .into_iter()
    }
}

#[cfg(test)]
mod tests;

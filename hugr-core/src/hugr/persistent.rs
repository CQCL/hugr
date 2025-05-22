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

mod parents_view;
mod resolver;
mod state_space;
mod trait_impls;
pub mod walker;

pub use walker::{PinnedWire, Walker};

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    mem, vec,
};

use delegate::delegate;
use derive_more::derive::From;
use itertools::{Either, Itertools};
use relrc::RelRc;
use state_space::{CommitData, CommitId};
pub use state_space::{CommitStateSpace, InvalidCommit, PatchNode};

pub use resolver::PointerEqResolver;

use crate::{
    Direction, Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, SimpleReplacement,
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
    /// Use [`Self::try_new`] instead if the parents of the commit cannot be
    /// inferred from the invalidation set of `replacement` alone.
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidCommit::EmptyReplacement`] error.
    /// If any of the parents of the replacement are not in the commit state
    /// space, this function will return an [`InvalidCommit::UnknownParent`]
    /// error.
    pub fn try_from_replacement(
        replacement: PersistentReplacement,
        graph: &CommitStateSpace,
    ) -> Result<Commit, InvalidCommit> {
        Self::try_new(replacement, [], graph)
    }

    /// Create a new commit
    ///
    /// Requires a reference to the commit state space that the nodes in
    /// `replacement` refer to.
    ///
    /// The returned commit will correspond to the application of `replacement`
    /// and will be the child of the commits in `parents` as well as of all
    /// the commits in the invalidation set of `replacement`.
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidCommit::EmptyReplacement`] error.
    /// If any of the parents of the replacement are not in the commit state
    /// space, this function will return an [`InvalidCommit::UnknownParent`]
    /// error.
    pub fn try_new(
        replacement: PersistentReplacement,
        parents: impl IntoIterator<Item = Commit>,
        graph: &CommitStateSpace,
    ) -> Result<Commit, InvalidCommit> {
        if replacement.subgraph().nodes().is_empty() {
            return Err(InvalidCommit::EmptyReplacement);
        }
        let repl_parents = get_parent_commits(&replacement, graph)?;
        let parents = parents.into_iter().chain(repl_parents);
        let rc = RelRc::with_parents(
            replacement.into(),
            parents.into_iter().map(|p| (p.into(), ())),
        );
        Ok(Self(rc))
    }

    fn as_relrc(&self) -> &RelRc<CommitData, ()> {
        &self.0
    }

    /// Get the set of nodes inserted by the patch in `self`.
    pub fn inserted_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        match self.0.value() {
            CommitData::Base(base) => Either::Left(base.nodes()),
            CommitData::Replacement(repl) => {
                // Skip the entrypoint and the IO nodes
                Either::Right(repl.replacement().entry_descendants().skip(3))
            }
        }
        .into_iter()
    }

    fn all_parents(&self) -> impl Iterator<Item = &Commit> + '_ {
        self.0.all_parents().map_into()
    }

    /// Get the patch that `self` represents.
    pub fn replacement(&self) -> Option<&PersistentReplacement> {
        match self.0.value() {
            CommitData::Base(_) => None,
            CommitData::Replacement(replacement) => Some(replacement),
        }
    }

    /// Get the set of nodes invalidated by the patch in `self`.
    pub fn invalidation_set(&self) -> impl Iterator<Item = PatchNode> + '_ {
        self.replacement()
            .into_iter()
            .flat_map(|r| r.invalidation_set())
    }

    /// Get the set of nodes deleted by applying `self`.
    pub fn deleted_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        self.replacement()
            .into_iter()
            .flat_map(|r| r.subgraph().nodes())
            .copied()
    }

    delegate! {
        to self.0 {
            fn value(&self) -> &CommitData;
            fn as_ptr(&self) -> *const relrc::node::InnerData<CommitData, ()>;
        }
    }

    /// Get all ancestors of `self` in reverse topological order, up until and
    /// including the first commit for which `continue_fn` returns false.
    fn get_ancestors_while<'a>(
        &'a self,
        continue_fn: impl Fn(&'a Commit) -> bool,
    ) -> Vec<&'a Commit> {
        let mut next_ancestor = 0;
        let mut ancestors = vec![self];
        let mut seen = BTreeSet::from_iter([self.as_ptr()]);
        while next_ancestor < ancestors.len() {
            let commit = &ancestors[next_ancestor];
            next_ancestor += 1;
            if !continue_fn(commit) {
                continue;
            }
            for parent in commit.all_parents() {
                if seen.insert(parent.as_ptr()) {
                    ancestors.push(parent);
                }
            }
        }
        ancestors
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
///
/// ## Supported graph types
///
/// Currently, only patches that apply to subgraphs within dataflow regions
/// are supported.
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

    /// Create a [`PersistentHugr`] from a single commit and its ancestors.
    // This always defines a valid `PersistentHugr` as the ancestors of a commit
    // are guaranteed to be compatible with each other.
    pub fn from_commit(commit: Commit) -> Self {
        let state_space = CommitStateSpace::try_from_commits([commit]).expect("commit is valid");
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
    /// All parent commits must already be in `self`.
    ///
    /// Return the ID of) the commit if it was added successfully. This may
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

    /// Add a commit to `self` and all its ancestors.
    ///
    /// The commit and all its ancestors must be compatible with all existing
    /// commits in `self`. If this is not satisfied, an
    /// [`InvalidCommit::IncompatibleHistory`] error is returned. In this case,
    /// as many compatible commits as possible are added to `self`.
    pub fn try_add_commit(&mut self, commit: Commit) -> Result<CommitId, InvalidCommit> {
        let new_commits = commit.get_ancestors_while(|c| !self.contains(c));
        let mut commit_id = None;
        for &commit in new_commits.iter().rev() {
            commit_id = Some(self.state_space.try_add_commit(commit.clone())?);
            let commit_id = commit_id.unwrap();

            // Check that the new commit is compatible with all its (current and
            // future) children
            let curr_children = self
                .state_space
                .children(commit_id)
                .map(|id| self.get_commit(id));
            let new_children = new_commits
                .iter()
                .copied()
                .filter(|c| c.all_parents().any(|p| p.as_ptr() == commit.as_ptr()));
            if let Some(node) = find_conflicting_node(
                commit_id,
                curr_children.chain(new_children).unique_by(|c| c.as_ptr()),
            ) {
                return Err(InvalidCommit::IncompatibleHistory(commit_id, node));
            }
        }
        Ok(commit_id.expect("new_commits cannot be empty"))
    }

    /// Convert this `PersistentHugr` to a materialized Hugr by applying all
    /// commits in `self`.
    ///
    /// This operation may be expensive and should be avoided in
    /// performance-critical paths. For read-only views into the data, rely
    /// instead on the [`crate::HugrView`] implementation when possible.
    pub fn to_hugr(&self) -> Hugr {
        self.apply_all().0
    }

    /// Apply all commits in `self` to the base HUGR.
    ///
    /// Also returns a map from the nodes of the base HUGR to the nodes of the
    /// materialized HUGR.
    pub fn apply_all(&self) -> (Hugr, HashMap<PatchNode, Node>) {
        let mut hugr = self.state_space.base_hugr().clone();
        let mut node_map = HashMap::from_iter(
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
                removed_nodes,
            } = repl.apply(&mut hugr).expect("invalid replacement");

            debug_assert!(
                hugr.validate().is_ok(),
                "malformed patch in persistent hugr:\n{}",
                hugr.mermaid_string()
            );

            for (old_node, new_node) in new_node_map {
                let old_patch_node = PatchNode(commit_id, old_node);
                node_map.insert(old_patch_node, new_node);
            }
            for remove_node in removed_nodes.into_keys() {
                let &remove_patch_node = node_map
                    .iter()
                    .find_map(|(patch_node, &hugr_node)| {
                        (hugr_node == remove_node).then_some(patch_node)
                    })
                    .expect("node not found in node_map");
                node_map.remove(&remove_patch_node);
            }
        }
        (hugr, node_map)
    }

    /// Get a reference to the underlying state space of `self`.
    pub fn as_state_space(&self) -> &CommitStateSpace {
        &self.state_space
    }

    /// Convert `self` into its underlying [`CommitStateSpace`].
    pub fn into_state_space(self) -> CommitStateSpace {
        self.state_space
    }

    /// The unique outgoing port in `self` that `port` is attached to.
    ///
    /// Panics if `node` is not in `self` (in particular if it is deleted) or if
    /// `port` is not a value port in `node`.
    fn get_single_outgoing_port(
        &self,
        node: PatchNode,
        port: impl Into<IncomingPort>,
    ) -> (PatchNode, OutgoingPort) {
        let mut in_port = port.into();
        let PatchNode(commit_id, mut in_node) = node;

        assert!(self.is_value_port(node, in_port), "not a dataflow wire");
        assert!(self.contains_node(node), "node not in self");

        let hugr = self.commit_hugr(commit_id);
        let (mut out_node, mut out_port) = hugr
            .single_linked_output(in_node, in_port)
            .map(|(n, p)| (PatchNode(commit_id, n), p))
            .expect("invalid HUGR");

        // invariant: (out_node, out_port) -> (in_node, in_port) is a boundary
        // edge, i.e. it never is the case that both are deleted by the same
        // child commit
        loop {
            let commit_id = out_node.0;

            let is_input = || {
                let Some(repl) = self.replacement(commit_id) else {
                    return false;
                };
                repl.get_replacement_io()[0] == out_node.1
            };
            if let Some(deleted_by) = self.find_deleting_commit(out_node) {
                (out_node, out_port) = self
                    .state_space
                    .linked_child_output(PatchNode(commit_id, in_node), in_port, deleted_by)
                    .expect("valid boundary edge");
                // update (in_node, in_port)
                (in_node, in_port) = {
                    let new_commit_id = out_node.0;
                    let hugr = self.commit_hugr(new_commit_id);
                    hugr.linked_inputs(out_node.1, out_port)
                        // the out_node is either:
                        //  - connected to output node of replacement
                        //   => deleting commit is None
                        //  - a boundary edge in a parent commit of `deleted_by`
                        //   => there is in_node with deleting commit `deleted_by`)
                        .find(|&(n, _)| {
                            let del_commit = self.find_deleting_commit(PatchNode(new_commit_id, n));
                            del_commit.is_none() || del_commit == Some(deleted_by)
                        })
                        .expect("out_node is connected to output node or is boundary of deleted_by")
                };
            } else if is_input() {
                // out_node is an input node
                (out_node, out_port) = self
                    .as_state_space()
                    .linked_parent_input(PatchNode(commit_id, in_node), in_port);
                // update (in_node, in_port)
                (in_node, in_port) = {
                    let new_commit_id = out_node.0;
                    let hugr = self.commit_hugr(new_commit_id);
                    hugr.linked_inputs(out_node.1, out_port)
                        .find(|&(n, _)| {
                            self.find_deleting_commit(PatchNode(new_commit_id, n))
                                == Some(commit_id)
                        })
                        .expect("boundary edge must connect out_node to deleted node")
                };
            } else {
                // valid outgoing node!
                return (out_node, out_port);
            }
        }
    }

    /// All incoming ports that the given outgoing port is attached to.
    ///
    /// Panics if `out_node` is not in `self` (in particular if it is deleted)
    /// or if `out_port` is not a value port in `out_node`.
    fn get_all_incoming_ports(
        &self,
        out_node: PatchNode,
        out_port: OutgoingPort,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        self.get_all_incoming_ports_with_callback(out_node, out_port, ())
    }

    /// All incoming ports that the given outgoing port is attached to.
    ///
    /// Panics if `out_node` is not in `self` (in particular if it is deleted)
    /// or if `out_port` is not a value port in `out_node`.
    ///
    /// Takes an additional callback to register visits to commits that may not
    /// be contained in the commits that the returned ports belong to.
    fn get_all_incoming_ports_with_callback(
        &self,
        out_node: PatchNode,
        out_port: OutgoingPort,
        mut mark_visited: impl CommitVisitCallback,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        assert!(
            self.is_value_port(out_node, out_port),
            "not a dataflow wire"
        );
        assert!(self.contains_node(out_node), "node not in self");

        let mut nodes_visited = BTreeSet::new();
        // enqueue the outport and initialise the set of valid incoming ports
        // to the valid incoming ports in this commit
        let mut queue = VecDeque::from([(out_node, out_port, BTreeSet::new())]);
        let mut valid_incoming_ports = BTreeSet::from_iter(
            self.commit_hugr(out_node.0)
                .linked_inputs(out_node.1, out_port)
                .map(|(in_node, in_port)| (PatchNode(out_node.0, in_node), in_port))
                .filter(|(in_node, _)| self.contains_node(*in_node)),
        );

        // A simple BFS across the commit history to find all equivalent incoming ports.
        while let Some((out_node, out_port, mut empty_commits)) = queue.pop_front() {
            let commit_id = out_node.0;

            if !nodes_visited.insert((out_node, out_port)) {
                continue;
            }

            let hugr = self.commit_hugr(commit_id);
            let out_deleted_by = self.find_deleting_commit(out_node);
            let curr_repl_out = {
                let repl = self.replacement(commit_id);
                repl.map(|r| r.get_replacement_io()[1])
            };
            // incoming ports are of interest to us if
            //  (i) they are connected to the output of a replacement (then there will be a
            //      linked port in a parent commit), or
            //  (ii) they are deleted by a child commit and is not the same as the out_node
            //      (then there will be a linked port in a child commit)
            let (is_linked_to_output, deleted_by_child): (IteratorNonEmpty, BTreeSet<_>) = hugr
                .linked_inputs(out_node.1, out_port)
                .filter_map(|(in_node, _)| {
                    if Some(in_node) == curr_repl_out {
                        // Flag that we have found a link to output
                        Some(Either::Left(()))
                    } else {
                        let other_deleted_by =
                            self.find_deleting_commit(PatchNode(commit_id, in_node))?;
                        // (out_node, out_port) -> (in_node, in_port) is a boundary edge
                        // into the child commit `other_deleted_by`
                        (Some(other_deleted_by) != out_deleted_by)
                            .then_some(Either::Right(other_deleted_by))
                    }
                })
                .partition_map(|x| x);

            // Convert an incoming port to the unique outgoing port that it is linked to
            let to_outgoing_port = |PatchNode(commit_id, in_node), in_port, commits_visited| {
                let hugr = self.commit_hugr(commit_id);
                let (out_node, out_port) = hugr
                    .single_linked_output(in_node, in_port)
                    .expect("valid dfg wire");
                (PatchNode(commit_id, out_node), out_port, commits_visited)
            };

            if is_linked_to_output.0 {
                // Traverse boundary to parent(s)
                let new_ins = self
                    .as_state_space()
                    .linked_parent_outputs(out_node, out_port);
                for (in_node, in_port) in new_ins {
                    if self.contains_node(in_node) {
                        valid_incoming_ports.insert((in_node, in_port));
                        for commit_id in mem::take(&mut empty_commits) {
                            // mark all traversed commits as visited
                            mark_visited.visit(commit_id);
                        }
                    }
                    queue.push_back(to_outgoing_port(in_node, in_port, empty_commits.clone()));
                }
            }

            for child in deleted_by_child {
                // Traverse boundary to `child`
                let new_ins = self
                    .as_state_space()
                    .linked_child_inputs(out_node, out_port, child);
                for (in_node, in_port) in new_ins {
                    if in_node.0 != child {
                        empty_commits.insert(child);
                    }
                    if self.contains_node(in_node) {
                        valid_incoming_ports.insert((in_node, in_port));
                        for commit_id in mem::take(&mut empty_commits) {
                            // mark all traversed commits as visited
                            mark_visited.visit(commit_id);
                        }
                    }
                    queue.push_back(to_outgoing_port(in_node, in_port, empty_commits.clone()));
                }
            }
        }

        valid_incoming_ports.into_iter()
    }

    delegate! {
        to self.state_space {
            /// Check if `commit` is in the PersistentHugr.
            pub fn contains(&self, commit: &Commit) -> bool;
            /// Check if `commit_id` is in the PersistentHugr.
            pub fn contains_id(&self, commit_id: CommitId) -> bool;
            /// Get the base commit ID.
            pub fn base(&self) -> CommitId;
            /// Get the base [`Hugr`].
            pub fn base_hugr(&self) -> &Hugr;
            /// Get the base commit.
            pub fn base_commit(&self) -> &Commit;
            /// Get the commit with ID `commit_id`.
            pub fn get_commit(&self, commit_id: CommitId) -> &Commit;
            /// Get an iterator over all nodes inserted by `commit_id`.
            ///
            /// All nodes will be PatchNodes with commit ID `commit_id`.
            pub fn inserted_nodes(&self, commit_id: CommitId) -> impl Iterator<Item = PatchNode> + '_;
            /// Get the replacement for `commit_id`.
            fn replacement(&self, commit_id: CommitId) -> Option<&SimpleReplacement<PatchNode>>;
            /// Get the Hugr inserted by `commit_id`.
            ///
            /// This is either the replacement Hugr of a [`CommitData::Replacement`] or
            /// the base Hugr of a [`CommitData::Base`].
            pub(super) fn commit_hugr(&self, commit_id: CommitId) -> &Hugr;
            /// Get an iterator over all commit IDs in the persistent HUGR.
            pub fn all_commit_ids(&self) -> impl Iterator<Item = CommitId> + Clone + '_;
        }
    }

    /// Get all commits in `self` in topological order.
    fn toposort_commits(&self) -> Vec<CommitId> {
        petgraph::algo::toposort(self.state_space.as_history_graph(), None)
            .expect("history is a DAG")
    }

    /// Get the set of nodes of `commit_id` that are invalidated by the patches
    /// in the children commits of `commit_id`.
    ///
    /// The invalidation set must include all nodes that are deleted by the
    /// children commits (as returned by [`Self::deleted_nodes`]), but may
    /// also include further nodes to enforce stricter exclusivity constraints
    /// between patches.
    pub fn invalidation_set(&self, commit_id: CommitId) -> impl Iterator<Item = Node> + '_ {
        let children = self.state_space.children(commit_id);
        children
            .flat_map(move |child_id| self.state_space.invalidation_set(child_id, commit_id))
            .unique()
    }

    /// Get the set of nodes of `commit_id` that are deleted by applying
    /// the children commits of `commit_id`.
    ///
    /// This is a subset of [`Self::invalidation_set`]. Whilst the latter is
    /// used to establish exclusivity constraints between patches, this method
    /// is used when we are computing the set of nodes currently present in
    /// `self`.
    pub fn deleted_nodes(&self, commit_id: CommitId) -> impl Iterator<Item = Node> + '_ {
        let children = self.state_space.children(commit_id);
        children
            .flat_map(move |child_id| {
                let child = self.get_commit(child_id);
                child
                    .deleted_nodes()
                    .filter_map(move |PatchNode(id, node)| (commit_id == id).then_some(node))
            })
            .unique()
    }

    fn find_deleting_commit(&self, node @ PatchNode(commit_id, _): PatchNode) -> Option<CommitId> {
        let mut children = self.state_space.children(commit_id);
        children.find(move |&child_id| {
            let child = self.get_commit(child_id);
            child.deleted_nodes().contains(&node)
        })
    }

    /// Check if a patch node is in the PersistentHugr, that is, it belongs to
    /// a commit in the state space and is not deleted by any child commit.
    pub fn contains_node(&self, PatchNode(commit_id, node): PatchNode) -> bool {
        let is_replacement_io = || {
            self.replacement(commit_id)
                .is_some_and(|repl| repl.get_replacement_io().contains(&node))
        };
        let is_deleted = || self.deleted_nodes(commit_id).contains(&node);
        self.contains_id(commit_id) && !is_replacement_io() && !is_deleted()
    }

    fn is_value_port(&self, PatchNode(commit_id, node): PatchNode, port: impl Into<Port>) -> bool {
        self.commit_hugr(commit_id)
            .get_optype(node)
            .port_kind(port)
            .expect("invalid port")
            .is_value()
    }

    fn value_ports(
        &self,
        patch_node @ PatchNode(commit_id, node): PatchNode,
        dir: Direction,
    ) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        let hugr = self.commit_hugr(commit_id);
        let ports = hugr.node_ports(node, dir);
        ports.filter_map(move |p| self.is_value_port(patch_node, p).then_some((patch_node, p)))
    }

    fn output_value_ports(
        &self,
        patch_node: PatchNode,
    ) -> impl Iterator<Item = (PatchNode, OutgoingPort)> + '_ {
        self.value_ports(patch_node, Direction::Outgoing)
            .map(|(n, p)| (n, p.as_outgoing().expect("unexpected port direction")))
    }

    fn input_value_ports(
        &self,
        patch_node: PatchNode,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        self.value_ports(patch_node, Direction::Incoming)
            .map(|(n, p)| (n, p.as_incoming().expect("unexpected port direction")))
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

/// Find a node in `commit` that is invalidated by more than one child commit
/// among `children`.
fn find_conflicting_node<'a>(
    commit_id: CommitId,
    mut children: impl Iterator<Item = &'a Commit>,
) -> Option<Node> {
    let mut all_invalidated = BTreeSet::new();

    children.find_map(|child| {
        let mut new_invalidated =
            child
                .invalidation_set()
                .filter_map(|PatchNode(del_commit_id, node)| {
                    if del_commit_id == commit_id {
                        Some(node)
                    } else {
                        None
                    }
                });
        new_invalidated.find(|&n| !all_invalidated.insert(n))
    })
}

fn get_parent_commits(
    repl: &PersistentReplacement,
    graph: &CommitStateSpace,
) -> Result<Vec<Commit>, InvalidCommit> {
    let ids = repl.invalidation_set().map(|n| n.0).unique();
    ids.map(|id| {
        if graph.contains_id(id) {
            Ok(graph.get_commit(id).clone())
        } else {
            Err(InvalidCommit::UnknownParent(id))
        }
    })
    .collect()
}

/// A wrapper around a boolean that implements `Extend<V>`. The boolean
/// is true if a non-empty iterator was appended to `self`.
#[derive(Debug, Copy, Clone, Default)]
struct IteratorNonEmpty(bool);

impl<V> Extend<V> for IteratorNonEmpty {
    fn extend<T: IntoIterator<Item = V>>(&mut self, iter: T) {
        self.0 |= iter.into_iter().next().is_some();
    }
}

trait CommitVisitCallback {
    fn visit(&mut self, commit_id: CommitId);
}

impl CommitVisitCallback for &mut BTreeSet<CommitId> {
    fn visit(&mut self, commit_id: CommitId) {
        self.insert(commit_id);
    }
}

impl CommitVisitCallback for () {
    fn visit(&mut self, _commit_id: CommitId) {}
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod test_iterator_non_empty {
    use super::IteratorNonEmpty;

    use rstest::rstest;

    #[rstest]
    #[case(vec![])]
    #[case(vec![1])]
    #[case(vec![1, 2, 3])]
    fn test_extend(#[case] input: Vec<i32>) {
        let expected = !input.is_empty();
        let mut res = IteratorNonEmpty::default();
        res.extend(input);
        assert_eq!(res.0, expected);
    }
}

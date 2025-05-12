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
mod trait_impls;

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
    hugr::patch::{simple_replace, Patch, PatchVerification},
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, SimpleReplacement,
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
    fn get_single_outgoing_port(
        &self,
        node: PatchNode,
        port: impl Into<Port>,
    ) -> (PatchNode, OutgoingPort) {
        let port = port.into();

        assert!(self.is_value_port(node, port), "not a dataflow wire");

        use itertools::Either::{Left, Right};
        let (mut node, mut port) = match port.as_directed() {
            Left(incoming) => (node, incoming),
            Right(outgoing) => {
                // pick any incoming port, the result is the same by uniqueness
                // of the outgoing port
                let commit_id = node.0;
                let (node, port) = self
                    .commit_hugr(commit_id)
                    .linked_inputs(node.1, outgoing)
                    .next()
                    .expect("invalid dfg graph");
                (PatchNode(commit_id, node), port)
            }
        };

        loop {
            let commit_id = node.0;
            let hugr = self.commit_hugr(commit_id);
            let (outgoing_node, outgoing_port) = hugr
                .single_linked_output(node.1, port)
                .expect("invalid HUGR");

            let is_input = {
                if let Some(repl) = self.replacement(commit_id) {
                    repl.get_replacement_io().expect("invalid replacement")[0] == outgoing_node
                } else {
                    false
                }
            };
            let is_deleted = || self.deleted_nodes(commit_id).contains(&outgoing_node);
            if is_input || is_deleted() {
                if let Some((child_node, child_port)) = self.child_output_port(node, port) {
                    node = child_node;
                    port = child_port;
                } else {
                    debug_assert!(is_input, "found deleted node but no replacement for it");
                    (node, port) = self
                        .parent_input_port(node, port)
                        .expect("found input replacement port with no equivalent port in parents");
                }
            } else {
                // outgoing_node is a valid node in the current hugr!
                let outgoing_node = PatchNode(commit_id, outgoing_node);
                return (outgoing_node, outgoing_port);
            }
        }
    }

    /// All incoming ports that the given outgoing port is attached to.
    fn get_all_incoming_ports(
        &self,
        node: PatchNode,
        port: OutgoingPort,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        let mut valid_incoming_ports = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut queue = {
            let start_hugr = self.commit_hugr(node.0);
            let to_patch_node = |(n, p)| (PatchNode(node.0, n), p);
            VecDeque::from_iter(start_hugr.linked_inputs(node.1, port).map(to_patch_node))
        };

        // A simple BFS to find all equivalent incoming ports.
        while let Some((node, port)) = queue.pop_front() {
            if !visited.insert((node, port)) {
                continue;
            }
            let commit_id = node.0;
            let is_output = {
                if let Some(repl) = self.replacement(commit_id) {
                    repl.get_replacement_io().expect("invalid replacement")[1] == node.1
                } else {
                    false
                }
            };
            let is_deleted = || self.deleted_nodes(commit_id).contains(&node.1);
            if !is_output && !is_deleted() {
                valid_incoming_ports.insert((node, port));
            }
            queue.extend(self.children_input_ports(node, port));
            queue.extend(self.parents_output_ports(node, port));
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
            /// Get the input boundary ports that are equivalent to `(node, port)` in
            /// the children of the commit of `node`.
            fn children_input_ports(
                &self,
                node: PatchNode,
                port: IncomingPort,
            ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_;
            /// Get the input boundary port in a parent of the commit of `node`
            /// that is equivalent to `(node, port)`.
            fn parent_input_port(
                &self,
                patch_node: PatchNode,
                port: IncomingPort,
            ) -> Option<(PatchNode, IncomingPort)>;
            /// Get the output boundary ports that are equivalent to `(node, port)` in
            /// the parents of the commit of `node`.
            pub(super) fn parents_output_ports(
                &self,
                node: PatchNode,
                port: IncomingPort,
            ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_;
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

    /// Get the output boundary ports that are equivalent to `(node, port)` in
    /// the children of the commit of `node`.
    ///
    /// By compatibility of all commits in `self`, there can be at most one
    /// child replacement that defines at most one port equivalent to `(node,
    /// port)`.
    fn child_output_port(
        &self,
        node: PatchNode,
        port: IncomingPort,
    ) -> Option<(PatchNode, IncomingPort)> {
        self.as_state_space()
            .children_output_ports(node, port)
            .at_most_one()
            .ok()
            .expect("at most one definition of output port in children replacements")
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

    /// Check if a patch node is in the PersistentHugr, that is, it belongs to
    /// a commit in the state space and is not deleted by any child commit.
    pub fn contains_node(&self, PatchNode(commit_id, node): PatchNode) -> bool {
        self.contains_id(commit_id) && !self.deleted_nodes(commit_id).contains(&node)
    }

    fn is_value_port(&self, PatchNode(commit_id, node): PatchNode, port: impl Into<Port>) -> bool {
        self.commit_hugr(commit_id)
            .get_optype(node)
            .port_kind(port)
            .expect("invalid port")
            .is_value()
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

#[cfg(test)]
mod tests;

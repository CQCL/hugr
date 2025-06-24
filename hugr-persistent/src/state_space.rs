//! Store of commit histories for a [`PersistentHugr`].

use std::collections::{BTreeSet, VecDeque};

use delegate::delegate;
use derive_more::From;
use hugr_core::{
    Direction, Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, SimpleReplacement,
    hugr::{
        self,
        internal::HugrInternals,
        patch::{
            BoundaryPort,
            simple_replace::{IncludeReplacementNodes, InvalidReplacement},
        },
        views::{InvalidSignature, sibling_subgraph::InvalidSubgraph},
    },
    ops::OpType,
};
use itertools::{Either, Itertools};
use relrc::{HistoryGraph, RelRc};
use thiserror::Error;

use crate::{
    Commit, PersistentHugr, PersistentReplacement, PointerEqResolver, Resolver,
    find_conflicting_node, parents_view::ParentsView,
};

pub mod serial;

/// A copyable handle to a [`Commit`] vertex within a [`CommitStateSpace`]
pub type CommitId = relrc::NodeId;

/// A HUGR node within a commit of the commit state space
#[derive(
    Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct PatchNode(pub CommitId, pub Node);

impl PatchNode {
    /// Get the commit ID of the commit that owns this node.
    pub fn owner(&self) -> CommitId {
        self.0
    }
}

// Print out PatchNodes as `Node(x)@commit_hex`
impl std::fmt::Debug for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}@{}", self.1, self.0)
    }
}

impl std::fmt::Display for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

mod hidden {
    use super::*;

    /// The data stored in a [`Commit`], either the base [`Hugr`] (on which all
    /// other commits apply), or a [`PersistentReplacement`]
    ///
    /// This is a "unnamable" type: we do not expose this struct publicly in our
    /// API, but we can still use it in public trait bounds (see
    /// [`Resolver`](crate::resolver::Resolver)).
    #[derive(Debug, Clone, From)]
    pub enum CommitData {
        Base(Hugr),
        Replacement(PersistentReplacement),
    }
}
pub(crate) use hidden::CommitData;

/// A set of commits with directed (acyclic) dependencies between them.
///
/// Vertices in the [`CommitStateSpace`] are [`Commit`]s and there is an edge
/// from a commit `c1` to a commit `c2` if `c1` must be applied before `c2`:
/// in other words, if `c2` deletes nodes that are introduced in `c1`. We say
/// `c2` depends on (or is a child of) `c1`.
///
/// A [`CommitStateSpace`] always has a unique base commit (the root of the
/// graph). All other commits are [`PersistentReplacement`]s that apply on top
/// of it. Commits are stored as [`RelRc`]s: they are reference-counted pointers
/// to the patch data that also maintain strong references to the commit's
/// parents. This means that commits can be cloned cheaply and dropped freely;
/// the memory of a commit will be released whenever no other commit in scope
/// depends on it.
///
/// Commits in a [`CommitStateSpace`] DO NOT represent a valid history in the
/// general case: pairs of commits may be mutually exclusive if they modify the
/// same subgraph. Use [`Self::try_extract_hugr`] to get a [`PersistentHugr`]
/// with a set of compatible commits.
#[derive(Clone, Debug)]
pub struct CommitStateSpace<R = PointerEqResolver> {
    /// A set of commits with directed (acyclic) dependencies between them.
    ///
    /// Each commit is stored as a [`RelRc`].
    pub(super) graph: HistoryGraph<CommitData, (), R>,
    /// The unique root of the commit graph.
    ///
    /// The only commit in the graph with variant [`CommitData::Base`]. All
    /// other commits are [`CommitData::Replacement`]s, and are descendants
    /// of this.
    pub(super) base_commit: CommitId,
}

impl<R: Resolver> CommitStateSpace<R> {
    /// Create a new commit state space with a single base commit.
    pub fn with_base(hugr: Hugr) -> Self {
        let commit = RelRc::new(CommitData::Base(hugr));
        let graph = HistoryGraph::new([commit.clone()], R::default());
        let base_commit = graph
            .all_node_ids()
            .exactly_one()
            .ok()
            .expect("graph has exactly one commit (just added)");
        Self { graph, base_commit }
    }

    /// Create a new commit state space from a set of commits.
    ///
    /// Return a [`InvalidCommit::NonUniqueBase`] error if the commits do
    /// not share a unique common ancestor base commit.
    pub fn try_from_commits(
        commits: impl IntoIterator<Item = Commit>,
    ) -> Result<Self, InvalidCommit> {
        let graph = HistoryGraph::new(commits.into_iter().map_into(), R::default());
        let base_commits = graph
            .all_node_ids()
            .filter(|&id| matches!(graph.get_node(id).value(), CommitData::Base(_)))
            .collect_vec();
        let base_commit = base_commits
            .into_iter()
            .exactly_one()
            .map_err(|err| InvalidCommit::NonUniqueBase(err.len()))?;
        Ok(Self { graph, base_commit })
    }

    /// Add a replacement commit to the graph.
    ///
    /// Return an [`InvalidCommit::EmptyReplacement`] error if the replacement
    /// is empty.
    pub fn try_add_replacement(
        &mut self,
        replacement: PersistentReplacement,
    ) -> Result<CommitId, InvalidCommit> {
        let commit = Commit::try_from_replacement(replacement, self)?;
        self.try_add_commit(commit)
    }

    /// Add a commit (and all its ancestors) to the state space.
    ///
    /// Returns an [`InvalidCommit::NonUniqueBase`] error if the commit is a
    /// base commit and does not coincide with the existing base commit.
    pub fn try_add_commit(&mut self, commit: Commit) -> Result<CommitId, InvalidCommit> {
        let is_base = commit.as_relrc().ptr_eq(self.base_commit().as_relrc());
        if !is_base && matches!(commit.value(), CommitData::Base(_)) {
            return Err(InvalidCommit::NonUniqueBase(2));
        }
        let commit = commit.into();
        Ok(self.graph.insert_node(commit))
    }

    /// Add a set of commits to the state space.
    ///
    /// Commits must be valid replacement commits or coincide with the existing
    /// base commit.
    pub fn extend(&mut self, commits: impl IntoIterator<Item = Commit>) {
        // TODO: make this more efficient
        for commit in commits {
            self.try_add_commit(commit)
                .expect("invalid commit in extend");
        }
    }

    /// Extract a `PersistentHugr` from this state space, consisting of
    /// `commits` and their ancestors.
    ///
    /// All commits in the resulting `PersistentHugr` are guaranteed to be
    /// compatible. If the selected commits would include two commits which
    /// are incompatible, a [`InvalidCommit::IncompatibleHistory`] error is
    /// returned. If `commits` is empty, a [`InvalidCommit::NonUniqueBase`]
    /// error is returned.
    pub fn try_extract_hugr(
        &self,
        commits: impl IntoIterator<Item = CommitId>,
    ) -> Result<PersistentHugr<R>, InvalidCommit> {
        // Define commits as the set of all ancestors of the given commits
        let all_commit_ids = get_all_ancestors(&self.graph, commits);

        if all_commit_ids.is_empty() {
            return Err(InvalidCommit::NonUniqueBase(0));
        }
        debug_assert!(all_commit_ids.contains(&self.base()));

        // Check that all commits are compatible
        for &commit_id in &all_commit_ids {
            let selected_children = self
                .children(commit_id)
                .filter(|id| all_commit_ids.contains(id))
                .map(|id| self.get_commit(id));
            if let Some(node) = find_conflicting_node(commit_id, selected_children) {
                return Err(InvalidCommit::IncompatibleHistory(commit_id, node));
            }
        }

        let commits = all_commit_ids
            .into_iter()
            .map(|id| self.get_commit(id).as_relrc().clone());
        let subgraph = HistoryGraph::new(commits, R::default());

        Ok(PersistentHugr::from_state_space_unsafe(Self {
            graph: subgraph,
            base_commit: self.base_commit,
        }))
    }
}

impl<R> CommitStateSpace<R> {
    /// Check if `commit` is in the commit state space.
    pub fn contains(&self, commit: &Commit) -> bool {
        self.graph.contains(commit.as_relrc())
    }

    /// Check if `commit_id` is in the commit state space.
    pub fn contains_id(&self, commit_id: CommitId) -> bool {
        self.graph.contains_id(commit_id)
    }

    /// Get the base commit ID.
    pub fn base(&self) -> CommitId {
        self.base_commit
    }

    /// Get the base [`Hugr`].
    pub fn base_hugr(&self) -> &Hugr {
        let CommitData::Base(hugr) = self.graph.get_node(self.base_commit).value() else {
            panic!("base commit is not a base hugr");
        };
        hugr
    }

    /// Get the base commit.
    pub fn base_commit(&self) -> &Commit {
        self.get_commit(self.base_commit)
    }

    /// Get the commit with ID `commit_id`.
    pub fn get_commit(&self, commit_id: CommitId) -> &Commit {
        self.graph.get_node(commit_id).into()
    }

    /// Get an iterator over all commit IDs in the state space.
    pub fn all_commit_ids(&self) -> impl Iterator<Item = CommitId> + Clone + '_ {
        let vec = self.graph.all_node_ids().collect_vec();
        vec.into_iter()
    }

    /// Get an iterator over all nodes inserted by `commit_id`.
    ///
    /// All nodes will be PatchNodes with commit ID `commit_id`.
    pub fn inserted_nodes(&self, commit_id: CommitId) -> impl Iterator<Item = PatchNode> + '_ {
        let commit = self.get_commit(commit_id);
        let to_patch_node = move |node| PatchNode(commit_id, node);
        commit.inserted_nodes().map(to_patch_node)
    }

    /// Get the set of nodes invalidated by `commit_id` in `parent`.
    pub fn invalidation_set(
        &self,
        commit_id: CommitId,
        parent: CommitId,
    ) -> impl Iterator<Item = Node> + '_ {
        let commit = self.get_commit(commit_id);
        let ret = commit
            .invalidation_set()
            .filter(move |n| n.0 == parent)
            .map(|n| n.1);
        Some(ret).into_iter().flatten()
    }

    delegate! {
        to self.graph {
            /// Get the parents of `commit_id`
            pub fn parents(&self, commit_id: CommitId) -> impl Iterator<Item = CommitId> + '_;
            /// Get the children of `commit_id`
            pub fn children(&self, commit_id: CommitId) -> impl Iterator<Item = CommitId> + '_;
        }
    }

    pub(crate) fn as_history_graph(&self) -> &HistoryGraph<CommitData, (), R> {
        &self.graph
    }

    /// Get the Hugr inserted by `commit_id`.
    ///
    /// This is either the replacement Hugr of a [`CommitData::Replacement`] or
    /// the base Hugr of a [`CommitData::Base`].
    pub(crate) fn commit_hugr(&self, commit_id: CommitId) -> &Hugr {
        let commit = self.get_commit(commit_id);
        match commit.value() {
            CommitData::Base(base) => base,
            CommitData::Replacement(repl) => repl.replacement(),
        }
    }

    /// Whether the edge at `(node, port)` is a boundary edge of `child`.
    ///
    /// Check if `(node, port)` is outside of the subgraph of the patch of child
    /// and at least one opposite node is inside the subgraph.
    pub fn is_boundary_edge(
        &self,
        node: PatchNode,
        port: impl Into<Port>,
        child: CommitId,
    ) -> bool {
        let deleted_nodes: BTreeSet<_> = self.get_commit(child).deleted_nodes().collect();
        if deleted_nodes.contains(&node) {
            return false;
        }
        let mut opp_nodes = self
            .commit_hugr(node.0)
            .linked_ports(node.1, port)
            .map(|(n, _)| PatchNode(node.0, n));
        opp_nodes.any(|n| deleted_nodes.contains(&n))
    }

    /// Get the boundary inputs linked to `(node, port)` in `child`.
    ///
    /// The returned ports will be in the `child` commit unless the child commit
    /// is empty, in which case they will be in one of the parents of `child`.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not a boundary edge, or if `child` is not
    /// a valid commit ID.
    pub(crate) fn linked_child_inputs(
        &self,
        node: PatchNode,
        port: OutgoingPort,
        child: CommitId,
        return_invalid: IncludeReplacementNodes,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        assert!(
            self.is_boundary_edge(node, port, child),
            "not a boundary edge"
        );

        let parent_hugrs = ParentsView::from_commit(child, self);
        let repl = self.replacement(child).expect("valid child commit");
        repl.linked_replacement_inputs((node, port), &parent_hugrs, return_invalid)
            .collect_vec()
            .into_iter()
            .map(move |np| match np {
                BoundaryPort::Host(patch_node, port) => (patch_node, port),
                BoundaryPort::Replacement(node, port) => (PatchNode(child, node), port),
            })
    }

    /// Get the single boundary output linked to `(node, port)` in `child`.
    ///
    /// The returned port will be in the `child` commit unless the child commit
    /// is empty, in which case it will be in one of the parents of `child`.
    ///
    /// ## Panics
    ///
    /// Panics if `child` is not a valid commit ID.
    pub(crate) fn linked_child_output(
        &self,
        node: PatchNode,
        port: IncomingPort,
        child: CommitId,
        return_invalid: IncludeReplacementNodes,
    ) -> Option<(PatchNode, OutgoingPort)> {
        let parent_hugrs = ParentsView::from_commit(child, self);
        let repl = self.replacement(child).expect("valid child commit");
        match repl.linked_replacement_output((node, port), &parent_hugrs, return_invalid)? {
            BoundaryPort::Host(patch_node, port) => (patch_node, port),
            BoundaryPort::Replacement(node, port) => (PatchNode(child, node), port),
        }
        .into()
    }

    /// Get the boundary ports linked to `(node, port)` in `child`.
    ///
    /// See [`Self::linked_child_inputs`] and [`Self::linked_child_output`] for
    /// more details.
    pub(crate) fn linked_child_ports(
        &self,
        node: PatchNode,
        port: impl Into<Port>,
        child: CommitId,
        return_invalid: IncludeReplacementNodes,
    ) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        match port.into().as_directed() {
            Either::Left(incoming) => Either::Left(
                self.linked_child_output(node, incoming, child, return_invalid)
                    .into_iter()
                    .map(|(node, port)| (node, port.into())),
            ),
            Either::Right(outgoing) => Either::Right(
                self.linked_child_inputs(node, outgoing, child, return_invalid)
                    .map(|(node, port)| (node, port.into())),
            ),
        }
    }

    /// Get the single output boundary port linked to `(node, port)` in a
    /// parent of the commit of `node`.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not connected to the input node in the
    /// commit of `node`, or if the node is not valid.
    pub fn linked_parent_input(
        &self,
        PatchNode(commit_id, node): PatchNode,
        port: IncomingPort,
    ) -> (PatchNode, OutgoingPort) {
        let repl = self.replacement(commit_id).expect("valid commit");

        assert!(
            repl.replacement()
                .input_neighbours(node)
                .contains(&repl.get_replacement_io()[0]),
            "not connected to input"
        );

        let parent_hugrs = ParentsView::from_commit(commit_id, self);
        repl.linked_host_input((node, port), &parent_hugrs).into()
    }

    /// Get the input boundary ports linked to `(node, port)` in a
    /// parent of the commit of `node`.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not connected to the output node in the
    /// commit of `node`, or if the node is not valid.
    pub fn linked_parent_outputs(
        &self,
        PatchNode(commit_id, node): PatchNode,
        port: OutgoingPort,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        let repl = self.replacement(commit_id).expect("valid commit");

        assert!(
            repl.replacement()
                .output_neighbours(node)
                .contains(&repl.get_replacement_io()[1]),
            "not connected to output"
        );

        let parent_hugrs = ParentsView::from_commit(commit_id, self);
        repl.linked_host_outputs((node, port), &parent_hugrs)
            .map_into()
            .collect_vec()
            .into_iter()
    }

    /// Get the ports linked to `(node, port)` in a parent of the commit of
    /// `node`.
    ///
    /// See [`Self::linked_parent_input`] and [`Self::linked_parent_outputs`]
    /// for more details.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not connected to an IO node in the commit
    /// of `node`, or if the node is not valid.
    pub fn linked_parent_ports(
        &self,
        node: PatchNode,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        match port.into().as_directed() {
            Either::Left(incoming) => {
                let (node, port) = self.linked_parent_input(node, incoming);
                Either::Left(std::iter::once((node, port.into())))
            }
            Either::Right(outgoing) => Either::Right(
                self.linked_parent_outputs(node, outgoing)
                    .map(|(node, port)| (node, port.into())),
            ),
        }
    }

    /// Get the replacement for `commit_id`.
    pub(crate) fn replacement(&self, commit_id: CommitId) -> Option<&SimpleReplacement<PatchNode>> {
        let commit = self.get_commit(commit_id);
        commit.replacement()
    }
}

// The subset of HugrView methods that can be implemented on CommitStateSpace
// by simplify delegating to the patches' respective HUGRs
impl<R> CommitStateSpace<R> {
    /// Get the type of the operation at `node`.
    pub fn get_optype(&self, PatchNode(commit_id, node): PatchNode) -> &OpType {
        let hugr = self.commit_hugr(commit_id);
        hugr.get_optype(node)
    }

    /// Get the number of ports of `node` in `dir`.
    pub fn num_ports(&self, PatchNode(commit_id, node): PatchNode, dir: Direction) -> usize {
        self.commit_hugr(commit_id).num_ports(node, dir)
    }

    /// Iterator over output ports of node.
    /// Like [`CommitStateSpace::node_ports`](node, Direction::Outgoing)`
    /// but preserves knowledge that the ports are [OutgoingPort]s.
    #[inline]
    pub fn node_outputs(&self, node: PatchNode) -> impl Iterator<Item = OutgoingPort> + Clone + '_ {
        self.node_ports(node, Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Iterator over inputs ports of node.
    /// Like [`CommitStateSpace::node_ports`](node, Direction::Incoming)`
    /// but preserves knowledge that the ports are [IncomingPort]s.
    #[inline]
    pub fn node_inputs(&self, node: PatchNode) -> impl Iterator<Item = IncomingPort> + Clone + '_ {
        self.node_ports(node, Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Get an iterator over the ports of `node` in `dir`.
    pub fn node_ports(
        &self,
        PatchNode(commit_id, node): PatchNode,
        dir: Direction,
    ) -> impl Iterator<Item = Port> + Clone + '_ {
        self.commit_hugr(commit_id).node_ports(node, dir)
    }

    /// Get an iterator over all ports of `node`.
    pub fn all_node_ports(
        &self,
        PatchNode(commit_id, node): PatchNode,
    ) -> impl Iterator<Item = Port> + Clone + '_ {
        self.commit_hugr(commit_id).all_node_ports(node)
    }

    /// Get the metadata map of `node`.
    pub fn node_metadata_map(
        &self,
        PatchNode(commit_id, node): PatchNode,
    ) -> &hugr::NodeMetadataMap {
        self.commit_hugr(commit_id).node_metadata_map(node)
    }
}

fn get_all_ancestors<N, E, R>(
    graph: &HistoryGraph<N, E, R>,
    commits: impl IntoIterator<Item = CommitId>,
) -> BTreeSet<CommitId> {
    let mut queue = VecDeque::from_iter(commits);
    let mut ancestors = BTreeSet::from_iter(queue.iter().copied());
    while let Some(commit_id) = queue.pop_front() {
        for parent in graph.parents(commit_id) {
            if ancestors.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    ancestors
}

/// An error that occurs when trying to add a commit to a commit state space.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum InvalidCommit {
    /// The commit conflicts with existing commits in the state space.
    #[error("Incompatible history: children of commit {0:?} conflict in {1:?}")]
    IncompatibleHistory(CommitId, Node),

    /// The commit has a parent not present in the state space.
    #[error("Missing parent commit: {0:?}")]
    UnknownParent(CommitId),

    /// The commit is not a replacement.
    #[error("Commit is not a replacement")]
    NotReplacement,

    /// The set of commits contains zero or more than one base commit.
    #[error("{0} base commits found (should be 1)")]
    NonUniqueBase(usize),

    /// The commit is an empty replacement.
    #[error("Not allowed: empty replacement")]
    EmptyReplacement,

    #[error("Invalid subgraph: {0}")]
    /// The subgraph of the replacement is not convex.
    InvalidSubgraph(#[from] InvalidSubgraph<PatchNode>),

    /// The replacement of the commit is invalid.
    #[error("Invalid replacement: {0}")]
    InvalidReplacement(#[from] InvalidReplacement),

    /// The signature of the replacement is invalid.
    #[error("Invalid signature: {0}")]
    InvalidSignature(#[from] InvalidSignature),

    /// A wire has an unpinned port.
    #[error("Incomplete wire: {0} is unpinned")]
    IncompleteWire(PatchNode, Port),
}

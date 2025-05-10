use std::collections::{BTreeSet, VecDeque};

use delegate::delegate;
use derive_more::From;
use itertools::Itertools;
use relrc::{HistoryGraph, RelRc};
use thiserror::Error;

use super::{Commit, PersistentHugr, PersistentReplacement, PointerEqResolver};
use crate::{Hugr, Node};

/// A copyable handle to a [`Commit`] vertex within a [`CommitStateSpace`]
pub(super) type CommitId = relrc::NodeId;

/// A HUGR node within a commit of the commit state space
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
pub struct PatchNode(pub CommitId, pub Node);

impl std::fmt::Display for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// The data stored in a [`Commit`], either the base [`Hugr`] (on which all
/// other commits apply), or a [`PersistentReplacement`]
#[derive(Debug, Clone, From)]
pub(super) enum CommitData {
    Base(Hugr),
    Replacement(PersistentReplacement),
}

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
pub struct CommitStateSpace {
    /// A set of commits with directed (acyclic) dependencies between them.
    ///
    /// Each commit is stored as a [`RelRc`].
    graph: HistoryGraph<CommitData, (), PointerEqResolver>,
    /// The unique root of the commit graph.
    ///
    /// The only commit in the graph with variant [`CommitData::Base`]. All
    /// other commits are [`CommitData::Replacement`]s, and are descendants
    /// of this.
    base_commit: CommitId,
}

impl CommitStateSpace {
    /// Create a new commit state space with a single base commit.
    pub fn with_base(hugr: Hugr) -> Self {
        let commit = RelRc::new(CommitData::Base(hugr));
        let graph = HistoryGraph::new([commit.clone()], PointerEqResolver);
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
        let graph = HistoryGraph::new(commits.into_iter().map_into(), PointerEqResolver);
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

    /// Add a commit to the state space.
    ///
    /// Returns an [`InvalidCommit::NonUniqueBase`] error if the commit is a
    /// base commit and does not coincide with the existing base commit.
    pub fn try_add_commit(&mut self, commit: Commit) -> Result<CommitId, InvalidCommit> {
        if matches!(commit.value(), CommitData::Base(_) if !commit.0.ptr_eq(&self.base_commit().0))
        {
            return Err(InvalidCommit::NonUniqueBase(2));
        }
        let commit = commit.into();
        Ok(self.graph.insert_node(commit))
    }

    /// Check if `commit` is in the commit state space.
    pub fn contains(&self, commit: &Commit) -> bool {
        self.graph.contains(commit.as_relrc())
    }

    /// Check if `commit_id` is in the commit state space.
    pub fn contains_id(&self, commit_id: CommitId) -> bool {
        self.graph.contains_id(commit_id)
    }

    /// Extract a `PersistentHugr` from this state space, consisting of
    /// `commits` and their ancestors.
    ///
    /// All commits in the resulting `PersistentHugr` are guaranteed to be
    /// compatible. If the selected commits would include two commits which
    /// are incompatible, a [`InvalidCommit::IncompatibleHistory`] error is
    /// returned.
    pub fn try_extract_hugr(
        &self,
        commits: impl IntoIterator<Item = CommitId>,
    ) -> Result<PersistentHugr, InvalidCommit> {
        // Define commits as the set of all ancestors of the given commits
        let commits = get_all_ancestors(&self.graph, commits);

        // Check that all commits are compatible
        for &commit_id in &commits {
            if let Some(node) = find_conflicting_node(self, commit_id, &commits) {
                return Err(InvalidCommit::IncompatibleHistory(commit_id, node));
            }
        }

        let commits = commits
            .into_iter()
            .map(|id| self.get_commit(id).as_relrc().clone());
        let subgraph = HistoryGraph::new(commits, PointerEqResolver);

        Ok(PersistentHugr::from_state_space_unsafe(Self {
            graph: subgraph,
            base_commit: self.base_commit,
        }))
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
    pub fn all_commit_ids(&self) -> impl Iterator<Item = CommitId> + '_ {
        self.graph.all_node_ids()
    }

    /// Get the set of nodes invalidated by `commit_id` in `parent`.
    pub(super) fn invalidation_set(
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

    pub(super) fn as_history_graph(&self) -> &HistoryGraph<CommitData, (), PointerEqResolver> {
        &self.graph
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
}

/// Find a node that is invalidated by more than one child of `commit_id`.
fn find_conflicting_node(
    graph: &CommitStateSpace,
    commit_id: CommitId,
    commits: &BTreeSet<CommitId>,
) -> Option<Node> {
    let mut all_invalidated = BTreeSet::new();
    let mut children = graph
        .children(commit_id)
        .filter(|&child_id| commits.contains(&child_id));

    children.find_map(|child_id| {
        let mut new_invalidated = graph.invalidation_set(child_id, commit_id);
        new_invalidated.find(|&n| !all_invalidated.insert(n))
    })
}

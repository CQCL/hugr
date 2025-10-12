use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    vec,
};

use hugr_core::{
    Hugr, HugrView, Node,
    hugr::patch::{Patch, simple_replace},
};
use itertools::Itertools;
use relrc::HistoryGraph;

use crate::{
    Commit, CommitData, CommitId, CommitStateSpace, InvalidCommit, PatchNode, PersistentReplacement,
};

pub mod serial;

/// A HUGR-like object that tracks its mutation history.
///
/// When mutations are applied to a [`PersistentHugr`], the object is mutated
/// as expected but all references to previous versions of the object remain
/// valid. Furthermore, older versions of the data can be recovered by
/// traversing the object's history with [`Self::state_space`].
///
/// Multiple references to various versions of a Hugr can be maintained in
/// parallel by extracting them from a shared [`CommitStateSpace`].
///
/// ## Supported access and mutation
///
/// [`PersistentHugr`] implements [`HugrView`], so that it can used as
/// a drop-in substitute for a Hugr wherever read-only access is required. It
/// does not implement [`HugrMut`](hugr_core::hugr::hugrmut::HugrMut), however.
/// Mutations must be performed by applying patches (see
/// [`PatchVerification`](hugr_core::hugr::patch::PatchVerification)
/// and [`Patch`]). Currently, only [`SimpleReplacement`] patches are supported.
/// You can use [`Self::add_replacement`] to add a patch to `self`, or use the
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
/// Currently, only patches represented as [`SimpleReplacement`],which apply to
/// subgraphs within dataflow regions are supported.
///
/// [`SimpleReplacement`]: hugr_core::SimpleReplacement
#[derive(Clone, Debug)]
pub struct PersistentHugr {
    /// The state space of all commits.
    ///
    /// Invariants:
    ///  - all commits are "compatible", meaning that no two patches invalidate
    ///    the same node.
    ///  - there is a unique commit of variant [`CommitData::Base`] and its ID
    ///    is `base_commit_id`.
    graph: HistoryGraph<CommitData, ()>,
    /// Cache of the unique root of the commit graph.
    ///
    /// The only commit in the graph with variant [`CommitData::Base`]. All
    /// other commits are [`CommitData::Replacement`]s, and are descendants
    /// of this.
    ///
    /// Invariant: any path from any commit in `self` through ancestors will
    /// always lead to this commit.
    base_commit_id: CommitId,
}

impl PersistentHugr {
    /// Create a [`PersistentHugr`] with `hugr` as its base HUGR.
    ///
    /// All replacements added in the future will apply on top of `hugr`.
    pub fn with_base(hugr: Hugr) -> Self {
        let state_space = CommitStateSpace::new();
        let base = state_space.try_set_base(hugr).expect("empty state space");
        Self::from_commit(base)
    }

    /// Create a [`PersistentHugr`] from a single commit and its ancestors.
    ///
    /// Panics if the commit is invalid.
    // This always defines a valid `PersistentHugr` as the ancestors of a commit
    // are guaranteed to be compatible with each other.
    pub fn from_commit(commit: Commit) -> Self {
        Self::try_new([commit]).expect("invalid commit")
    }

    /// Create a [`PersistentHugr`] from a list of commits.
    ///
    /// `Self` will correspond to the HUGR obtained by applying the patches of
    /// the given commits and of all their ancestors.
    ///
    /// If the state space of the commits would include two commits which are
    /// incompatible, or if the commits do not share a common base HUGR, then
    /// an error is returned.
    pub fn try_new<'a>(
        commits: impl IntoIterator<Item = Commit<'a>>,
    ) -> Result<Self, InvalidCommit> {
        let commits = get_ancestors_while(commits, |_| true);
        let state_space = commits
            .front()
            .ok_or(InvalidCommit::NonUniqueBase(0))?
            .state_space();
        let all_commit_ids = BTreeSet::from_iter(commits.iter().map(|c| c.as_ptr()));
        let mut graph = HistoryGraph::with_registry(state_space.to_registry());

        for commit in commits {
            // Check that all commits are in the same state space
            if commit.state_space() != state_space {
                return Err(InvalidCommit::NonUniqueBase(2));
            }

            // Check that all commits are compatible
            let selected_children = commit
                .children(&state_space)
                .filter(|c| all_commit_ids.contains(&c.as_ptr()));
            if let Some(node) = find_conflicting_node(commit.id(), selected_children) {
                return Err(InvalidCommit::IncompatibleHistory(commit.id(), node));
            }

            graph.insert_node(commit.into());
        }

        let base_commit = graph
            .all_node_ids()
            .filter(|&id| {
                matches!(
                    graph.get_node(id).expect("valid ID").value(),
                    CommitData::Base(_)
                )
            })
            .exactly_one()
            .map_err(|err| InvalidCommit::NonUniqueBase(err.count()))?;

        Ok(Self {
            graph,
            base_commit_id: base_commit,
        })
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
            let invalidation_set = self.deleted_nodes(parent).collect();
            if let Some(&node) = new_invalid_nodes.intersection(&invalidation_set).next() {
                return Err(InvalidCommit::IncompatibleHistory(parent, node));
            }
        }

        let commit = Commit::try_from_replacement(replacement, self.state_space())?;
        // SAFETY: commit does not need to be restrained to state_space's lifetime
        // as it will be added to self.
        let commit = unsafe { commit.upgrade_lifetime() };

        self.try_add_commit(commit)
    }

    /// Add a commit to `self` and all its ancestors.
    ///
    /// The commit and all its ancestors must be compatible with all existing
    /// commits in `self`. If this is not satisfied, an
    /// [`InvalidCommit::IncompatibleHistory`] error is returned. In this case,
    /// as many compatible commits as possible are added to `self`.
    pub fn try_add_commit(&mut self, commit: Commit) -> Result<CommitId, InvalidCommit> {
        self.try_add_commits([commit.clone()])?;
        Ok(commit.id())
    }

    /// Add commits and their ancestors to `self`.
    ///
    /// The commits and all their ancestors must be compatible with all existing
    /// commits in `self`. If this is not satisfied, an
    /// [`InvalidCommit::IncompatibleHistory`] error is returned. In this case,
    /// as many compatible commits as possible are added to `self`.
    pub fn try_add_commits<'a>(
        &mut self,
        commits: impl IntoIterator<Item = Commit<'a>>,
    ) -> Result<(), InvalidCommit> {
        let new_commits = get_ancestors_while(commits, |c| !self.contains(c));

        for new_commit in new_commits.iter().rev() {
            let new_commit_id = new_commit.id();
            if &new_commit.state_space() != self.state_space() {
                return Err(InvalidCommit::NonUniqueBase(2));
            }

            // Check that the new commit is compatible with all its (current and
            // future) children
            let curr_children = self
                .children_commits(new_commit_id)
                .map(|id| self.get_commit(id));
            let new_children = new_commits
                .iter()
                .filter(|&c| c.parents().any(|p| p.as_ptr() == new_commit.as_ptr()));
            if let Some(node) = find_conflicting_node(
                new_commit_id,
                curr_children
                    .chain(new_children)
                    .unique_by(|c| c.as_ptr())
                    .map(|c| c.to_owned()),
            ) {
                return Err(InvalidCommit::IncompatibleHistory(new_commit_id, node));
            }

            self.graph.insert_node(new_commit.clone().into());
        }

        Ok(())
    }

    /// Check the [`PersistentHugr`] invariants.
    pub fn is_valid(&self) -> Result<(), InvalidCommit> {
        let mut found_base = false;
        for id in self.all_commit_ids() {
            let commit = self.get_commit(id);
            if matches!(commit.value(), CommitData::Base(_)) {
                found_base = true;
                if id != self.base_commit_id {
                    return Err(InvalidCommit::NonUniqueBase(2));
                }
            }
            let children = self
                .children_commits(id)
                .map(|child_id| self.get_commit(child_id).clone());
            if let Some(already_invalid) = find_conflicting_node(id, children) {
                return Err(InvalidCommit::IncompatibleHistory(id, already_invalid));
            }
        }

        if !found_base {
            return Err(InvalidCommit::NonUniqueBase(0));
        }

        Ok(())
    }

    /// Get a reference to the underlying state space of `self`.
    pub fn state_space(&self) -> &CommitStateSpace {
        self.graph.registry().into()
    }

    /// Get the base commit ID.
    pub fn base(&self) -> CommitId {
        self.base_commit_id
    }

    /// Get the base [`Hugr`].
    pub fn base_hugr(&self) -> &Hugr {
        let CommitData::Base(hugr) = self.get_commit(self.base_commit_id).value() else {
            panic!("base commit is not a base hugr");
        };
        hugr
    }

    /// Get the commit with ID `commit_id`.
    ///
    /// Panics if `commit_id` is not in `self`.
    pub fn get_commit(&self, commit_id: CommitId) -> &Commit<'_> {
        self.graph
            .get_node(commit_id)
            .expect("invalid commit ID")
            .into()
    }
    /// Check if `commit` is in the PersistentHugr.
    pub fn contains(&self, commit: &Commit) -> bool {
        self.graph.contains(commit.as_relrc())
    }

    /// Check if `commit_id` is in the PersistentHugr.
    pub fn contains_id(&self, commit_id: CommitId) -> bool {
        self.graph.contains_id(commit_id)
    }

    /// Get the base commit.
    pub fn base_commit(&self) -> &Commit<'_> {
        self.get_commit(self.base())
    }

    /// Get an iterator over all commit IDs in the persistent HUGR.
    pub fn all_commit_ids(&self) -> impl Iterator<Item = CommitId> + Clone + '_ {
        self.graph.all_node_ids()
    }

    /// Get all commits in `self` in topological order.
    fn toposort_commits(&self) -> Vec<CommitId> {
        petgraph::algo::toposort(&self.graph, None).expect("history is a DAG")
    }

    pub fn children_commits(&self, commit_id: CommitId) -> impl Iterator<Item = CommitId> + '_ {
        self.graph.children(commit_id)
    }

    pub fn parent_commits(&self, commit_id: CommitId) -> impl Iterator<Item = CommitId> + '_ {
        self.graph.parents(commit_id)
    }

    /// Convert this `PersistentHugr` to a materialized Hugr by applying all
    /// commits in `self`.
    ///
    /// This operation may be expensive and should be avoided in
    /// performance-critical paths. For read-only views into the data, rely
    /// instead on the [`HugrView`] implementation when possible.
    pub fn to_hugr(&self) -> Hugr {
        self.apply_all().0
    }

    /// Apply all commits in `self` to the base HUGR.
    ///
    /// Also returns a map from the nodes of the base HUGR to the nodes of the
    /// materialized HUGR.
    pub fn apply_all(&self) -> (Hugr, HashMap<PatchNode, Node>) {
        let mut hugr = self.base_hugr().clone();
        let mut node_map = HashMap::from_iter(hugr.nodes().map(|n| (PatchNode(self.base(), n), n)));
        for commit_id in self.toposort_commits() {
            let Some(repl) = self.get_commit(commit_id).replacement() else {
                continue;
            };

            let repl = repl
                .map_host_nodes(|n| node_map[&n], &hugr)
                .expect("invalid replacement");

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

    /// Get the set of nodes of `commit_id` that are invalidated by the patches
    /// in the children commits of `commit_id`.
    ///
    /// The invalidation set must include all nodes that are deleted by the
    /// children commits (as returned by [`Self::deleted_nodes`]), but may
    /// also include further nodes to enforce stricter exclusivity constraints
    /// between patches.
    pub fn deleted_nodes<'a>(&'a self, commit_id: CommitId) -> impl Iterator<Item = Node> + 'a {
        self.children_commits(commit_id).flat_map(move |child_id| {
            let all_invalidated = self.get_commit(child_id).deleted_parent_nodes();
            all_invalidated
                .filter_map(move |PatchNode(owner, node)| (owner == commit_id).then_some(node))
        })
    }

    /// Check if a patch node is in the PersistentHugr, that is, it belongs to
    /// a commit in the state space and is not deleted by any child commit.
    pub fn contains_node(&self, PatchNode(commit_id, node): PatchNode) -> bool {
        let is_replacement_io = || {
            let commit = self.get_commit(commit_id);
            commit
                .replacement()
                .is_some_and(|repl| repl.get_replacement_io().contains(&node))
        };
        let is_deleted = || self.deleted_nodes(commit_id).contains(&node);
        self.contains_id(commit_id) && !is_replacement_io() && !is_deleted()
    }
}

/// Get the union of commits with all its ancestors, up to and including the
/// first commits for which `continue_fn` returns false.
///
/// Return all ancestors in reverse topological order.
fn get_ancestors_while<'a>(
    commits: impl IntoIterator<Item = Commit<'a>>,
    continue_fn: impl Fn(&Commit) -> bool,
) -> VecDeque<Commit<'a>> {
    let mut seen_ids = BTreeSet::new();
    let commits = commits.into_iter();
    let mut all_commits = VecDeque::with_capacity(commits.size_hint().0);

    for commit in commits {
        if !seen_ids.insert(commit.as_ptr()) {
            continue;
        }
        let start = all_commits.len();
        let mut ind = start;
        all_commits.push_back(commit);

        while ind < all_commits.len() {
            let commit = all_commits[ind].clone();
            ind += 1;

            if !continue_fn(&commit) {
                continue;
            }

            for commit in commit.parents() {
                if seen_ids.insert(commit.as_ptr()) {
                    all_commits.push_back(commit.clone());
                }
            }
        }
        all_commits.rotate_right(all_commits.len() - start);
    }

    all_commits
}

// non-public methods
impl PersistentHugr {
    /// Convert a node ID specific to a commit HUGR into a patch node in the
    /// [`PersistentHugr`].
    pub(crate) fn to_persistent_node(&self, node: Node, commit_id: CommitId) -> PatchNode {
        PatchNode(commit_id, node)
    }

    /// Get the unique child commit in `self` that deletes `node`.
    pub(crate) fn find_deleting_commit(
        &self,
        node @ PatchNode(commit_id, _): PatchNode,
    ) -> Option<CommitId> {
        let mut children = self.children_commits(commit_id);
        children.find(move |&child_id| {
            let child = self.get_commit(child_id);
            child.deleted_parent_nodes().contains(&node)
        })
    }
}

impl<'a> IntoIterator for &'a PersistentHugr {
    type Item = Commit<'a>;

    type IntoIter = vec::IntoIter<Commit<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.graph
            .all_node_ids()
            .map(|id| self.get_commit(id).clone())
            .collect_vec()
            .into_iter()
    }
}

/// Find a node in `commit` that is invalidated by more than one child commit
/// among `children`.
pub(crate) fn find_conflicting_node<'a>(
    commit_id: CommitId,
    children: impl IntoIterator<Item = Commit<'a>>,
) -> Option<Node> {
    let mut all_invalidated = BTreeSet::new();

    children.into_iter().find_map(|child| {
        let mut new_invalidated =
            child
                .deleted_parent_nodes()
                .filter_map(|PatchNode(del_commit_id, node)| {
                    (del_commit_id == commit_id).then_some(node)
                });
        new_invalidated.find(|&n| !all_invalidated.insert(n))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig},
        extension::prelude::bool_t,
        hugr::views::SiblingSubgraph,
        ops::handle::NodeHandle,
        std_extensions::logic::LogicOp,
    };
    use rstest::*;

    // two NOT gates
    fn notop_hugr() -> (Hugr, Node, Node) {
        let mut builder = DFGBuilder::new(endo_sig(vec![bool_t()])).unwrap();
        let [input] = builder.input_wires_arr();
        let notop = builder.add_dataflow_op(LogicOp::Not, [input]).unwrap();
        let notop2 = builder
            .add_dataflow_op(LogicOp::Not, notop.outputs())
            .unwrap();
        let [out] = notop2.outputs_arr();
        (
            builder.finish_hugr_with_outputs([out]).unwrap(),
            notop.node(),
            notop2.node(),
        )
    }

    fn add_commit(persistent_hugr: &mut PersistentHugr, node: PatchNode) -> (CommitId, PatchNode) {
        let (repl_hugr, repl_not, _) = notop_hugr();
        let repl1 = PersistentReplacement::try_new(
            SiblingSubgraph::from_node(node, &persistent_hugr),
            &persistent_hugr,
            repl_hugr,
        )
        .unwrap();
        let commit = persistent_hugr.add_replacement(repl1);
        (commit, PatchNode(commit, repl_not))
    }

    #[fixture]
    fn linear_commits() -> (PersistentHugr, Vec<CommitId>) {
        let (base_hugr, notop, _) = notop_hugr();

        // Create a linear chain of commits: base -> commit1 -> commit2 -> commit3
        let mut persistent_hugr = PersistentHugr::with_base(base_hugr);

        let base_not = persistent_hugr.base_commit().to_patch_node(notop);

        // Create commit1 (child of base)
        let (cm1, cm1_not) = add_commit(&mut persistent_hugr, base_not);

        // Create commit2 (child of commit1)
        let (cm2, cm2_not) = add_commit(&mut persistent_hugr, cm1_not);

        // Create commit3 (child of commit2)
        let (cm3, _cm3_not) = add_commit(&mut persistent_hugr, cm2_not);

        let base_id = persistent_hugr.base();
        (persistent_hugr, vec![base_id, cm1, cm2, cm3])
    }

    #[fixture]
    fn branching_commits() -> (PersistentHugr, Vec<CommitId>) {
        // Create a branching structure: base -> commit1 -> commit2
        //                                  \-> commit3 -> commit4
        let (base_hugr, notop, notop2) = notop_hugr();
        let mut persistent_hugr = PersistentHugr::with_base(base_hugr);

        let base_commit = persistent_hugr.base_commit();
        let base_not = base_commit.to_patch_node(notop);
        let base_not2 = base_commit.to_patch_node(notop2);
        let base_id = base_commit.id();

        // Create commit1 (child of base)
        let (cm1, cm1_not) = add_commit(&mut persistent_hugr, base_not);

        // Create commit2 (child of commit1)
        let (cm2, _cm2_not) = add_commit(&mut persistent_hugr, cm1_not);

        // Create commit3 (child of base)
        let (cm3, cm3_not) = add_commit(&mut persistent_hugr, base_not2);

        // Create commit4 (child of commit3)
        let (cm4, _cm4_not) = add_commit(&mut persistent_hugr, cm3_not);

        (persistent_hugr, vec![base_id, cm1, cm2, cm3, cm4])
    }

    #[rstest]
    fn test_get_ancestors_while_linear_chain(linear_commits: (PersistentHugr, Vec<CommitId>)) {
        let (persistent_hugr, commit_ids) = linear_commits;
        let commits = commit_ids
            .iter()
            .map(|&id| persistent_hugr.get_commit(id).clone())
            .collect_vec();

        // Starting from commit3, should get ancestors in reverse topological order
        let ancestors = get_ancestors_while([commits[3].clone()], |_| true);
        let ancestor_ids: Vec<_> = ancestors.iter().map(|c| c.id()).collect();

        // Should be in reverse topological order: commit3, commit2, commit1, base
        assert_eq!(
            ancestor_ids,
            vec![commit_ids[3], commit_ids[2], commit_ids[1], commit_ids[0]]
        );
    }

    #[rstest]
    fn test_get_ancestors_while_branching_structure(
        branching_commits: (PersistentHugr, Vec<CommitId>),
    ) {
        let (persistent_hugr, commit_ids) = branching_commits;
        let commits = commit_ids
            .iter()
            .map(|&id| persistent_hugr.get_commit(id).clone())
            .collect_vec();

        // Starting from both commit2 and commit4, should get all ancestors
        let ancestors = get_ancestors_while([commits[2].clone(), commits[4].clone()], |_| true);
        let ancestor_ids: Vec<_> = ancestors.iter().map(|c| c.id()).collect();

        // Should include all commits, with descendants before ancestors
        let valid_orderings = [
            vec![
                commit_ids[4],
                commit_ids[3],
                commit_ids[2],
                commit_ids[1],
                commit_ids[0],
            ],
            vec![
                commit_ids[2],
                commit_ids[1],
                commit_ids[4],
                commit_ids[3],
                commit_ids[0],
            ],
        ];
        assert!(valid_orderings.contains(&ancestor_ids));
    }

    #[rstest]
    fn test_get_ancestors_while_with_filter(linear_commits: (PersistentHugr, Vec<CommitId>)) {
        let (persistent_hugr, commit_ids) = linear_commits;
        let commits = commit_ids
            .iter()
            .map(|&id| persistent_hugr.get_commit(id).clone())
            .collect_vec();
        let [_base, commit1, commit2, commit3] = commits.try_into().unwrap();

        // Use a filter that stops at commit1
        let ancestors = get_ancestors_while([commit3.clone()], |c| c.id() != commit1.id());
        let ancestor_ids: Vec<_> = ancestors.iter().map(|c| c.id()).collect();

        assert_eq!(ancestor_ids, vec![commit3.id(), commit2.id(), commit1.id()]);
    }
}

//! Incremental traversal and construction of [`PersistentHugr`]s from
//! [`CommitStateSpace`]s.
//!
//! This module provides the [`Walker`] type, which enables intuitive
//! exploration of a [`CommitStateSpace`] by traversing wires and gradually
//! pinning (selecting) nodes that match a desired pattern. Unlike direct
//! manipulation of a [`CommitStateSpace`], which may contain many alternative
//! and conflicting versions of the graph data, the Walker presents a familiar
//! nodes-and-wires interface that feels similar to walking a standard (HUGR)
//! graph.
//!
//! The key concept is that of "pinning" nodes - marking specific nodes as fixed
//! points in the exploration. As more nodes are pinned, the space of possible
//! HUGRs that are considered narrows; a Walker instance where all nodes are
//! pinned then corresponds to a [`PersistentHugr`], which is equivalent to a
//! HUGR graph. By choosing which nodes are pinned, the user in effect specifies
//! and extracts the [`PersistentHugr`]s of interest from within the
//! [`CommitStateSpace`].
//!
//! To establish which nodes can be pinned and explore the possible
//! alternatives, walkers are "expanded" along wires. Given a walker and a wire,
//! along with, optionally, a direction of expansion, [`Walker::expand`] returns
//! an iterator over all possible walkers that can be obtained by pinning
//! exactly one additional node from the wire. Repeated expansions result in
//! walkers with an increasing number of pinned nodes, narrowing down the commit
//! state space of interest at each step.
//!
//! ## Typical usage
//!
//! Such incremental exploration is particularly suited for pattern matching
//! and the creation of replacements (rewrite factories).
//!
//! A typical usage flow involves:
//! 1. Create a [`Walker`] over a [`CommitStateSpace`].
//! 2. Pin (at least one) initial node of interest using
//!    [`Walker::try_pin_node`].
//! 3. Traverse wires starting from a pinned node to explore the neighbourhood
//!    in the graph using [`Walker::expand`]. Each new walker thus obtained will
//!    correspond to a different selction of commits, which will result in
//!    several alternative HUGRs that must be considered in parallel. Thus, the
//!    exploration typically branches out into several parallel walkers.
//! 4. As walkers are expanded, more nodes get pinned, narrowing down the space
//!    of possible HUGRs.
//! 5. Once exploration is complete (e.g. a pattern was fully matched), the
//!    walker can be converted into a [`PersistentHugr`] instance using
//!    [`Walker::into_persistent_hugr`]. The matched nodes and ports can then be
//!    used to create a [`SiblingSubgraph`](crate::hugr::views::SiblingSubgraph)
//!    object, which can then be used to create new
//!    [`SimpleReplacement`](crate::SimpleReplacement) instances---and possibly
//!    in turn be added to the commit state space and the exploration of the
//!    state space restarted!
//!
//! This approach allows efficiently finding patterns across many potential
//! versions of the graph simultaneously, without having to materialize
//! each version separately.

mod pinned;
pub use pinned::PinnedWire;

use std::{borrow::Cow, collections::BTreeSet};

use itertools::{Either, Itertools};
use thiserror::Error;

use crate::{
    Direction, Hugr, HugrView, Port, PortIndex as _,
    hugr::views::{RootCheckable, SiblingSubgraph},
    ops::handle::DfgID,
};

use super::{
    Commit, CommitStateSpace, InvalidCommit, PatchNode, PersistentHugr, PersistentReplacement,
    state_space::CommitId,
};

/// A walker over a [`CommitStateSpace`].
///
/// A walker is given by a set of selected commits, along with a set of pinned
/// nodes that belong to those selected commits. The selected commits determine
/// which replacements in the state space are applied. Meanwhile, the pinned
/// nodes within these commits determine which nodes are "frozen" in the
/// exploration: no further commit can be selected that would invalidate any
/// pinned node.
///
/// The set of selected commits of a walker defines a [`PersistentHugr`]
/// instance that can be retrieved by calling [`Walker::into_persistent_hugr`].
/// As the walker is expanded and more of the state space is explored, more
/// commits are selected, and the [`PersistentHugr`] will change accordingly.
/// Pinned nodes (and pinned ports, i.e. ports at pinned nodes) are guaranteed
/// to be valid in all [`PersistentHugr`] instances obtained as a result of
/// expansions of the current walker.
/// current walker.
#[derive(Debug, Clone)]
pub struct Walker<'a> {
    /// The state space being traversed.
    state_space: Cow<'a, CommitStateSpace>,
    /// The subset of compatible commits in `state_space` that are currently
    /// selected.
    // Note that we could store this as a set of `CommitId`s, but it is very
    // convenient to have access to all the methods of PersistentHugr (on top
    // of guaranteeing the compatibility invariant). The tradeoff is more
    // memory consumption.
    selected_commits: PersistentHugr,
    /// The set of nodes that have been traversed by the walker and can no
    /// longer be rewritten.
    pinned_nodes: BTreeSet<PatchNode>,
}

impl<'a> Walker<'a> {
    /// Create a new [`Walker`] over the given state space.
    ///
    /// No nodes are pinned initially. The [`Walker`] starts with only the base
    /// Hugr `state_space.base_hugr()` selected.
    pub fn new(state_space: impl Into<Cow<'a, CommitStateSpace>>) -> Self {
        let state_space = state_space.into();
        let base = state_space.base_commit().clone();
        let selected_commits = PersistentHugr::from_commit(base);
        Self {
            state_space,
            selected_commits,
            pinned_nodes: BTreeSet::new(),
        }
    }

    /// Create a new [`Walker`] with a single pinned node.
    pub fn from_pinned_node(
        node: PatchNode,
        state_space: impl Into<Cow<'a, CommitStateSpace>>,
    ) -> Self {
        let mut walker = Self::new(state_space);
        walker
            .try_pin_node(node)
            .expect("node is valid and not deleted");
        walker
    }

    /// Pin a node in the [`Walker`].
    ///
    /// This method allows pinning a specific node in the Walker, which
    /// restricts the possible paths that can be explored in the Walker.
    ///
    /// Return true if `node` was not pinned already, and false otherwise.
    ///
    /// If the node belongs to a commit that isn't currently selected, the
    /// commit is added to the Walker, or an error is returned if this is
    /// not possible. The node is then added to the set of pinned nodes. If
    /// the node is already pinned, this method is a no-op.
    pub fn try_pin_node(&mut self, node: PatchNode) -> Result<bool, PinNodeError> {
        let commit_id = node.0;
        if self.selected_commits.contains_id(commit_id) {
            if !self.selected_commits.contains_node(node) {
                return Err(PinNodeError::AlreadyDeleted(node));
            }
        } else {
            let commit = self.state_space.get_commit(commit_id).clone();
            // TODO: we should be able to check for an AlreadyPinned error at
            // the same time that we check the ancestors are compatible in
            // `PersistentHugr`, with e.g. a callback, instead of storing a backup
            let backup = self.selected_commits.clone();
            self.selected_commits.try_add_commit(commit)?;
            if let Some(&pinned_node) = self
                .pinned_nodes
                .iter()
                .find(|&&n| !self.selected_commits.contains_node(n))
            {
                self.selected_commits = backup;
                return Err(PinNodeError::AlreadyPinned(pinned_node));
            }
        }
        Ok(self.pinned_nodes.insert(node))
    }

    /// Get the wire connected to a specified port of a pinned node.
    ///
    /// # Panics
    /// Panics if `node` is not already pinned in this Walker.
    pub fn get_wire(&self, node: PatchNode, port: impl Into<Port>) -> PinnedWire {
        PinnedWire::from_pinned_port(node, port, self)
    }

    /// Materialise the [`PersistentHugr`] containing all the compatible commits
    /// that have been selected during exploration.
    pub fn into_persistent_hugr(self) -> PersistentHugr {
        self.selected_commits
    }

    /// View the [`PersistentHugr`] containing all the compatible commits that
    /// have been selected during exploration.
    pub fn as_hugr_view(&self) -> &PersistentHugr {
        &self.selected_commits
    }

    /// Expand the Walker by pinning a node connected to the given wire.
    ///
    /// Return an iterator over all possible [`Walker`]s that can be created by
    /// pinning exactly one additional node connected to `wire`. Each returned
    /// [`Walker`] represents a different alternative Hugr in the exploration
    /// space.
    ///
    /// Optionally, the expansion can be restricted to only ports with the given
    /// direction (incoming or outgoing).
    ///
    /// Repeatedly calling `expand` on a wire will progressively pin all its
    /// endpoints. The multiple alternative [`Walker`]s returned then form
    /// a branching search space to be explored by the user. When a wire is
    /// fully pinned in the specified direction, i.e. `wire.is_complete(dir)` is
    /// true, then an empty iterator is returned.
    pub fn expand<'b>(
        &'b self,
        wire: &'b PinnedWire,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = Walker<'a>> + 'b
    where
        'a: 'b,
    {
        let dir = dir.into();

        // Find unpinned ports on the wire (satisfying the direction constraint)
        let unpinned_ports = wire.unpinned_ports(dir);

        // Obtain set of pinnable nodes by considering all ports (in descendant
        // commits) equivalent to currently unpinned ports.
        let pinnable_nodes = unpinned_ports
            .flat_map(|(node, port)| self.equivalent_descendant_ports(node, port))
            .map(|(n, _, commits)| (n, commits))
            .unique();

        pinnable_nodes.filter_map(|(pinnable_node, new_commits)| {
            debug_assert!(
                !self.is_pinned(pinnable_node),
                "trying to pin already pinned node"
            );

            // Update the selected commits to include the new commits.
            let new_selected_commits = self
                .state_space
                .try_extract_hugr(self.selected_commits.all_commit_ids().chain(new_commits))
                .ok()?;

            // Make sure that the pinned nodes are still valid after including the new
            // selected commits.
            if self
                .pinned_nodes
                .iter()
                .any(|&pnode| !new_selected_commits.contains_node(pnode))
            {
                return None;
            }

            // Construct a new walker and pin `pinnable_node`.
            let mut new_walker = Walker {
                state_space: self.state_space.clone(),
                selected_commits: new_selected_commits,
                pinned_nodes: self.pinned_nodes.clone(),
            };
            new_walker.try_pin_node(pinnable_node).ok()?;

            Some(new_walker)
        })
    }

    /// Create a new commit from a set of complete pinned wires and a
    /// replacement.
    ///
    /// The subgraph of the commit is the subgraph given by the set of edges
    /// in `wires`. `map_boundary` must provide a map from the boundary ports
    /// of the subgraph to the inputs/output ports in `repl`. The returned port
    /// must be of the opposite direction as the port passed as argument:
    ///  - an incoming subgraph port must be mapped to an outgoing port of the
    ///    input node of `repl`
    /// - an outgoing subgraph port must be mapped to an incoming port of the
    ///   output node of `repl`
    ///
    /// ## Panics
    ///
    /// This will panic if repl is not a DFG graph.
    pub fn try_create_commit(
        &self,
        wires: impl IntoIterator<Item = PinnedWire>,
        repl: impl RootCheckable<Hugr, DfgID>,
        map_boundary: impl Fn(PatchNode, Port) -> Port,
    ) -> Result<Commit, InvalidCommit> {
        let mut wire_ports_incoming = BTreeSet::new();
        let mut wire_ports_outgoing = BTreeSet::new();
        let mut additional_parents = Vec::new();

        for w in wires {
            if let Some((n, p)) = w.unpinned_ports(None).next() {
                return Err(InvalidCommit::IncompleteWire(n, p));
            }
            wire_ports_incoming.extend(w.pinned_inports());
            wire_ports_outgoing.extend(w.pinned_outport());
            additional_parents.extend(w.bridge_commits());
        }
        let additional_parents: Vec<_> = additional_parents
            .into_iter()
            .map(|id| {
                if self.state_space.contains_id(id) {
                    Ok(self.state_space.get_commit(id).clone())
                } else {
                    Err(InvalidCommit::UnknownParent(id))
                }
            })
            .try_collect()?;

        let all_nodes: BTreeSet<_> = wire_ports_incoming
            .iter()
            .map(|&(n, _)| n)
            .chain(wire_ports_outgoing.iter().map(|&(n, _)| n))
            .collect();

        // (in/out) boundary: all in/out ports on the nodes of the wire, minus ports
        // that are part of the wires
        let incoming = all_nodes
            .iter()
            .flat_map(|&n| self.as_hugr_view().input_value_ports(n))
            .filter(|node_port| !wire_ports_incoming.contains(node_port))
            .map(|np| vec![np])
            .collect_vec();
        let outgoing = all_nodes
            .iter()
            .flat_map(|&n| self.as_hugr_view().output_value_ports(n))
            .filter(|node_port| !wire_ports_outgoing.contains(node_port))
            .collect_vec();

        let repl = {
            let mut repl = repl.try_into_checked().expect("replacement is not DFG");
            let new_inputs = incoming
                .iter()
                .flatten()
                .map(|&(n, p)| {
                    map_boundary(n, p.into())
                        .as_outgoing()
                        .expect("unexpected port direction returned by map_boundary")
                        .index()
                })
                .collect_vec();
            let new_outputs = outgoing
                .iter()
                .map(|&(n, p)| {
                    map_boundary(n, p.into())
                        .as_incoming()
                        .expect("unexpected port direction returned by map_boundary")
                        .index()
                })
                .collect_vec();
            repl.map_function_type(&new_inputs, &new_outputs)?;
            let subgraph = SiblingSubgraph::try_new(incoming, outgoing, self.as_hugr_view())?;
            PersistentReplacement::try_new(subgraph, self.as_hugr_view(), repl.into_hugr())?
        };

        Commit::try_new(repl, additional_parents, &self.state_space)
    }

    /// Get all equivalent ports among the commits that are descendants of the
    /// current commit.
    ///
    /// The ports in the returned iterator will be in the same direction as
    /// `port`. For each equivalent port, also return the set of empty commits
    /// that were visited to find it.
    fn equivalent_descendant_ports(
        &self,
        node: PatchNode,
        port: Port,
    ) -> Vec<(PatchNode, Port, BTreeSet<CommitId>)> {
        // Now, perform a BFS to find all equivalent ports
        let mut all_ports = vec![(node, port, BTreeSet::new())];
        let mut index = 0;
        while index < all_ports.len() {
            let (node, port, empty_commits) = all_ports[index].clone();
            index += 1;

            for (child_id, (opp_node, opp_port)) in
                self.state_space.children_at_boundary_port(node, port)
            {
                let mut enqueue = |node: PatchNode, port: Port| {
                    let mut empty_commits = empty_commits.clone();
                    if node.0 != child_id {
                        empty_commits.insert(child_id);
                    }
                    all_ports.push((node, port, empty_commits));
                };

                match opp_port.as_directed() {
                    Either::Left(opp_port) => {
                        let Some((n, p)) = self
                            .state_space
                            .linked_child_output(opp_node, opp_port, child_id)
                        else {
                            continue;
                        };
                        enqueue(n, p.into());
                    }
                    Either::Right(opp_port) => {
                        self.state_space
                            .linked_child_inputs(opp_node, opp_port, child_id)
                            .for_each(|(n, p)| enqueue(n, p.into()));
                    }
                }
            }
        }
        all_ports
    }

    fn is_pinned(&self, node: PatchNode) -> bool {
        self.pinned_nodes.contains(&node)
    }
}

impl CommitStateSpace {
    /// Given a node and port, return all children commit of the current `node`
    /// that delete `node` but do not delete a linked port of `(node, port)`.
    /// In other words, (node, port) is a boundary port of the subgraph of
    /// the child replacement.
    ///
    /// Return all tuples of children and linked port of (node, port) that is
    /// outside of the subgraph of the child. The returned ports are opposite
    /// to the direction of `port`.
    fn children_at_boundary_port(
        &self,
        patch_node @ PatchNode(commit_id, node): PatchNode,
        port: Port,
    ) -> impl Iterator<Item = (CommitId, (PatchNode, Port))> + '_ {
        let linked_ports = self
            .commit_hugr(commit_id)
            .linked_ports(node, port)
            .collect_vec();

        self.children(commit_id).flat_map(move |child_id| {
            let deleted_nodes: BTreeSet<_> = self.get_commit(child_id).deleted_nodes().collect();
            if !deleted_nodes.contains(&patch_node) {
                vec![]
            } else {
                linked_ports
                    .iter()
                    .filter_map(move |&(linked_node, linked_port)| {
                        let linked_node = PatchNode(commit_id, linked_node);
                        (!deleted_nodes.contains(&linked_node))
                            .then_some((child_id, (linked_node, linked_port)))
                    })
                    .collect_vec()
            }
        })
    }
}

/// An error that occurs when trying to pin a node in a [`Walker`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum PinNodeError {
    /// Commit is not compatible with the current set of selected commits.
    #[error("cannot add commit to pin node: {0}")]
    InvalidNewCommit(InvalidCommit),
    /// Node to pin is already deleted in the current set of selected commits.
    #[error("cannot pin deleted node: {0}")]
    AlreadyDeleted(PatchNode),
    /// A node that would have to be deleted to pin the given node is already
    /// pinned.
    #[error("cannot delete already pinned node: {0}")]
    AlreadyPinned(PatchNode),
}

impl From<InvalidCommit> for PinNodeError {
    fn from(value: InvalidCommit) -> Self {
        PinNodeError::InvalidNewCommit(value)
    }
}

impl<'a> From<&'a CommitStateSpace> for Cow<'a, CommitStateSpace> {
    fn from(value: &'a CommitStateSpace) -> Self {
        Cow::Borrowed(value)
    }
}

impl From<CommitStateSpace> for Cow<'_, CommitStateSpace> {
    fn from(value: CommitStateSpace) -> Self {
        Cow::Owned(value)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::{
        Direction, HugrView, IncomingPort, OutgoingPort,
        builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig},
        extension::prelude::bool_t,
        hugr::persistent::{
            PersistentHugr, Walker,
            state_space::CommitId,
            tests::{persistent_hugr_empty_child, test_state_space},
        },
        std_extensions::logic::LogicOp,
    };

    use super::*;

    #[rstest]
    fn test_walker_base_or_child_expansion(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [commit1, _commit2, _commit3, _commit4]) = test_state_space;

        // Get an initial node to pin
        let base_and_node = {
            let base_hugr = state_space.base_hugr();
            let and_node = base_hugr
                .nodes()
                .find(|&n| base_hugr.get_optype(n) == &LogicOp::And.into())
                .unwrap();
            PatchNode(state_space.base(), and_node)
        };
        let walker = Walker::from_pinned_node(base_and_node, &state_space);
        assert!(walker.is_pinned(base_and_node));

        let in0 = walker.get_wire(base_and_node, IncomingPort::from(0));

        // a single incoming port (already pinned) => no more expansion
        assert!(walker.expand(&in0, Direction::Incoming).next().is_none());
        // commit 2 cannot be applied, because AND is pinned
        // => only base commit, or commit1
        let out_walkers = walker.expand(&in0, Direction::Outgoing).collect_vec();
        assert_eq!(out_walkers.len(), 2);
        for new_walker in out_walkers {
            // new wire is complete (and thus cannot be expanded)
            let in0 = new_walker.get_wire(base_and_node, IncomingPort::from(0));
            assert!(in0.is_complete(None));
            assert!(new_walker.expand(&in0, None).next().is_none());

            // all nodes on wire are pinned
            let (not_node, _) = in0.pinned_outport().unwrap();
            assert!(new_walker.is_pinned(base_and_node));
            assert!(new_walker.is_pinned(not_node));

            // not node is either in commit1 or the base
            assert!([commit1, state_space.base()].contains(&not_node.0));

            // not node is a NOT gate
            assert_eq!(
                new_walker.as_hugr_view().get_optype(not_node),
                &LogicOp::Not.into()
            );

            let persistent_hugr = new_walker.into_persistent_hugr();
            let hugr = persistent_hugr.commit_hugr(not_node.0);
            assert_eq!(hugr.get_optype(not_node.1), &LogicOp::Not.into());
        }
    }

    /// Test that a walker can be expanded in various ways. Starting from
    /// commit4, it is possible to expand the pinned nodes in three ways:
    ///  - base
    ///  - commit1 without commit2
    ///  - commit1 and commit2
    ///
    /// commit3 is not an option as it conflicts with commit4.
    ///
    /// ```
    ///                         ------- base -------
    ///                        /         |          \
    ///                       /          |           \
    ///                      /           |            \
    ///                  commit1      c̶o̶m̶m̶i̶t̶3̶        commit4 <--- start here
    ///                     |     (incompatible)
    ///                     |
    ///                     |
    ///                  commit2
    /// ```
    #[rstest]
    fn test_walker_disjoint_nephew_expansion(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [commit1, commit2, commit3, commit4]) = test_state_space;

        // Initially, pin the second NOT of commit4
        let not4_node = {
            let repl4 = state_space.replacement(commit4).unwrap();
            let hugr4 = state_space.commit_hugr(commit4);
            let [_, output] = repl4.get_replacement_io();
            let (second_not_node, _) = hugr4.single_linked_output(output, 0).unwrap();
            PatchNode(commit4, second_not_node)
        };
        let walker = Walker::from_pinned_node(not4_node, &state_space);
        assert!(walker.is_pinned(not4_node));

        let not4_out = walker.get_wire(not4_node, OutgoingPort::from(0));
        let expanded_out = walker.expand(&not4_out, Direction::Outgoing).collect_vec();
        // a single outgoing port (already pinned) => no more expansion
        assert!(expanded_out.is_empty());

        // Three options:
        // - AND gate from base
        // - XOR gate from commit3
        // - XOR gate from commit2 (which implies commit1)
        let mut exp_options = BTreeSet::from_iter([
            BTreeSet::from_iter([state_space.base(), commit4]),
            BTreeSet::from_iter([state_space.base(), commit3, commit4]),
            BTreeSet::from_iter([state_space.base(), commit1, commit2, commit4]),
        ]);
        for new_walker in walker.expand(&not4_out, None) {
            // selected commits must be one of the valid options
            let commit_ids = new_walker
                .as_hugr_view()
                .all_commit_ids()
                .collect::<BTreeSet<_>>();
            assert!(
                exp_options.remove(&commit_ids),
                "{:?} not an expected set of commit IDs (or duplicate)",
                commit_ids
            );

            // new wire is complete (and thus cannot be expanded)
            let not4_out = new_walker.get_wire(not4_node, OutgoingPort::from(0));
            assert!(not4_out.is_complete(None));
            assert!(new_walker.expand(&not4_out, None).next().is_none());

            // all nodes on wire are pinned
            let (next_node, _) = not4_out.pinned_inports().exactly_one().ok().unwrap();
            assert!(new_walker.is_pinned(not4_node));
            assert!(new_walker.is_pinned(next_node));

            let persistent_hugr = new_walker.into_persistent_hugr();

            // next_node is either in base (AND gate), in commit3 (XOR gate), or
            // in commit2 (XOR gate)
            let expected_optype = match next_node.0 {
                commit_id if commit_id == state_space.base() => LogicOp::And,
                commit_id if [commit2, commit3].contains(&commit_id) => LogicOp::Xor,
                _ => panic!("neighbour of not4 must be in base, commit2 or commit3"),
            };
            assert_eq!(
                persistent_hugr.get_optype(next_node),
                &expected_optype.into()
            );
        }

        assert!(
            exp_options.is_empty(),
            "missing expected options: {:?}",
            exp_options
        );
    }

    #[rstest]
    fn test_get_wire_endpoints(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [commit1, commit2, _commit3, commit4]) = test_state_space;
        let base_and_node = {
            let base_hugr = state_space.base_hugr();
            let and_node = base_hugr
                .nodes()
                .find(|&n| base_hugr.get_optype(n) == &LogicOp::And.into())
                .unwrap();
            PatchNode(state_space.base(), and_node)
        };

        let hugr = state_space.try_extract_hugr([commit4]).unwrap();
        let (second_not_node, out_port) =
            hugr.get_single_outgoing_port(base_and_node, IncomingPort::from(1));
        assert_eq!(second_not_node.0, commit4);
        assert_eq!(out_port, OutgoingPort::from(0));

        let hugr = state_space
            .try_extract_hugr([commit1, commit2, commit4])
            .unwrap();
        let (new_and_node, in_port) = hugr
            .get_all_incoming_ports(second_not_node, out_port)
            .exactly_one()
            .ok()
            .unwrap();
        assert_eq!(new_and_node.0, commit2);
        assert_eq!(in_port, 1.into());
    }

    /// Test that the walker handles empty replacements correctly.
    ///
    /// The base hugr is a sequence of 3 NOT gates, with a single input/output
    /// boolean. A single replacement exists in the state space, which replaces
    /// the middle NOT gate with nothing.
    #[rstest]
    fn test_walk_over_empty_repls(
        persistent_hugr_empty_child: (PersistentHugr, [CommitId; 2], [PatchNode; 3]),
    ) {
        let (hugr, [_base_commit, empty_commit], [not0, not1, not2]) = persistent_hugr_empty_child;
        let walker = Walker::from_pinned_node(not0, hugr.as_state_space());

        let not0_outwire = walker.get_wire(not0, OutgoingPort::from(0));
        let expanded_wires = walker
            .expand(&not0_outwire, Direction::Incoming)
            .collect_vec();

        assert_eq!(expanded_wires.len(), 2);

        let connected_inports: BTreeSet<_> = expanded_wires
            .iter()
            .map(|new_walker| {
                let wire = new_walker.get_wire(not0, OutgoingPort::from(0));
                wire.pinned_inports().exactly_one().ok().unwrap()
            })
            .collect();

        assert_eq!(
            connected_inports,
            BTreeSet::from_iter([(not1, IncomingPort::from(0)), (not2, IncomingPort::from(0))])
        );

        let bridge_commits: BTreeSet<_> = expanded_wires
            .iter()
            .map(|new_walker| {
                let wire = new_walker.get_wire(not0, OutgoingPort::from(0));
                wire.bridge_commits().collect_vec()
            })
            .collect();

        assert_eq!(
            bridge_commits,
            BTreeSet::from_iter([vec![], vec![empty_commit]])
        );
    }

    #[rstest]
    fn test_create_commit_over_empty(
        persistent_hugr_empty_child: (PersistentHugr, [CommitId; 2], [PatchNode; 3]),
    ) {
        let (hugr, [base_commit, empty_commit], [not0, _not1, not2]) = persistent_hugr_empty_child;
        let mut walker = Walker {
            state_space: hugr.as_state_space().into(),
            selected_commits: hugr.clone(),
            pinned_nodes: BTreeSet::from_iter([not0]),
        };

        // wire: Not0 -> Not2 (bridging over Not1)
        let wire = walker.get_wire(not0, OutgoingPort::from(0));
        walker = walker.expand(&wire, None).exactly_one().ok().unwrap();
        let wire = walker.get_wire(not0, OutgoingPort::from(0));
        assert!(wire.is_complete(None));

        let empty_hugr = {
            let dfg_builder = DFGBuilder::new(endo_sig(bool_t())).unwrap();
            let inputs = dfg_builder.input_wires();
            dfg_builder.finish_hugr_with_outputs(inputs).unwrap()
        };
        let commit = walker
            .try_create_commit(vec![wire], empty_hugr, |node, port| {
                assert_eq!(port.index(), 0);
                assert!([not0, not2].contains(&node));
                match port.direction() {
                    Direction::Incoming => OutgoingPort::from(0).into(),
                    Direction::Outgoing => IncomingPort::from(0).into(),
                }
            })
            .unwrap();

        let mut new_state_space = hugr.as_state_space().to_owned();
        let commit_id = new_state_space.try_add_commit(commit.clone()).unwrap();
        assert_eq!(
            new_state_space.parents(commit_id).collect::<BTreeSet<_>>(),
            BTreeSet::from_iter([base_commit, empty_commit])
        );

        let res_hugr = PersistentHugr::from_commit(commit);
        assert!(res_hugr.validate().is_ok());

        // should be an empty DFG hugr
        // module root + function def + func I/O nodes + DFG entrypoint + I/O nodes
        assert_eq!(res_hugr.num_nodes(), 1 + 1 + 2 + 1 + 2);
    }
}

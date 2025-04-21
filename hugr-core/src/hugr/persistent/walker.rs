//! Exploration of a [`PersistentHugr`] through incremental traversal of
//! connected subgraphs.
//!
//! This module provides the [`Walker`] type, which enables intuitive
//! exploration of a [`CommitStateSpace`] by traversing wires and gradually
//! pinning (selecting) nodes that match a desired pattern. Unlike direct
//! manipulation of a [`PersistentHugr`], the Walker presents a familiar
//! nodes-and-wires interface that feels similar to walking a standard (HUGR)
//! graph.
//!
//! The key concept is that of "pinning" nodes - marking specific nodes as fixed
//! points in the exploration. As more nodes are pinned, the space of possible
//! HUGRs that are considered narrows; a Walker instance where all nodes are
//! pinned corresponds to a unique HUGR in the [`PersistentHugr`].
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
//!    [`Walker::into_hugr`]. The matched nodes and ports can then be used to
//!    create a [`SimpleReplacement`](crate::SimpleReplacement) object, which
//!    can later be added to the commit state space.
//!
//! This approach allows efficiently finding patterns across many potential
//! versions of the graph simultaneously, without having to materialize
//! each version separately.

use std::{
    borrow::Cow,
    collections::{BTreeSet, VecDeque},
    hash::Hash,
};

use itertools::Itertools;
use thiserror::Error;

use crate::{Direction, HugrView, IncomingPort, OutgoingPort, Port};

use super::{CommitStateSpace, InvalidCommit, PatchNode, PersistentHugr};

/// A walker over a [`CommitStateSpace`].
///
/// A walker is given by a set of selected commits, along with a set of pinned
/// nodes that belong to selected commits. The selected commit determine which
/// replacements of the state space are applied; meanwhile, the pinned nodes
/// within selected commits determine which nodes are "frozen" in the
/// exploration; no further commit can be selected that would invalidate any
/// pinned node.
///
/// The set of selected commits of a walker define a [`PersistentHugr`] instance
/// that can be retrieved by calling [`Walker::into_hugr`]. As the walker is
/// expanded and more of the state space is explored, more commits are selected,
/// and the [`PersistentHugr`] will change accordingly. Pinned nodes (and pinned
/// ports, i.e. ports at pinned nodes) are guaranteed to be valid in all
/// [`PersistentHugr`] instances obtained as a result of expansions of the
/// current walker.
#[derive(Debug, Clone)]
pub struct Walker<'a> {
    /// The state space being traversed.
    state_space: Cow<'a, CommitStateSpace>,
    /// The subset of compatible commits in `state_space` that are currently
    /// selected.
    ///
    /// Note that we could store this as a set of `CommitId`s, but it is very
    /// convenient to have access to all the methods of PersistentHugr (on top
    /// of guaranteeing the compatibility invariant). The tradeoff is more
    /// memory consumption.
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
        let selected_commits =
            PersistentHugr::try_new([base]).expect("base is a valid persistent hugr");
        Self {
            state_space,
            selected_commits,
            pinned_nodes: BTreeSet::new(),
        }
    }

    /// Pin a node in the [`Walker`].
    ///
    /// This method allows pinning a specific node in the Walker, which
    /// restricts the possible paths that can be explored in the Walker.
    ///
    /// If the node belongs to a commit that isn't currently selected, the
    /// commit is added to the Walker, or an error is returned if this is
    /// not possible. The node is then added to the set of pinned nodes. If
    /// the node is already pinned, this method is a no-op.
    pub fn try_pin_node(&mut self, node: PatchNode) -> Result<(), PinNodeError> {
        if !self.selected_commits.contains_id(node.0) {
            let commit = self.state_space.get_commit(node.0).clone();
            self.selected_commits.try_add_commit(commit)?;
        }
        if !self.selected_commits.contains_node(node) {
            return Err(PinNodeError::AlreadyDeleted(node));
        }
        if let Some(&pinned_node) = self
            .pinned_nodes
            .iter()
            .find(|&&n| !self.selected_commits.contains_node(n))
        {
            return Err(PinNodeError::AlreadyPinned(pinned_node));
        }
        self.pinned_nodes.insert(node);
        Ok(())
    }

    /// Get the wire connected to a pinned node.
    ///
    /// This method allows accessing the wire attached to a specific port of a
    /// pinned node as a [`PinnedWire`].
    ///
    /// # Panics
    /// Panics if `node` is not already pinned in this Walker.
    pub fn get_wire(&self, node: PatchNode, port: impl Into<Port>) -> PinnedWire {
        PinnedWire::from_pinned_port(node, port, self)
    }

    /// Materialise the [`PersistentHugr`] containing all the compatible commits
    /// that have been selected during exploration.
    pub fn into_hugr(self) -> PersistentHugr {
        self.selected_commits
    }

    /// The [`PersistentHugr`] containing all the compatible commits that have
    /// been selected during exploration.
    pub fn as_hugr(&self) -> &PersistentHugr {
        &self.selected_commits
    }

    /// Expand the Walker by pinning a node connected to the given wire.
    ///
    /// Returns an iterator over all possible [`Walker`]s that can be created by
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
    ) -> impl Iterator<Item = Walker<'a>> + 'b {
        let dir = dir.into();

        // Find unpinned ports on the wire (satisfying the direction constraint)
        let unpinned_ports = wire.unpinned_ports(dir);

        // Obtain set of pinnable nodes by considering  all equivalent ports in
        // descendant commits of currently unpinned ports.
        let pinnable_nodes = unpinned_ports
            .flat_map(|(node, port)| self.equivalent_descendant_ports(node, port))
            .map(|(n, _)| n)
            .unique();

        pinnable_nodes.filter_map(|pinnable_node| {
            debug_assert!(
                !self.is_pinned(pinnable_node),
                "trying to pin already pinned node"
            );

            // Construct a new walker by pinning `pinnable_node` (if possible).
            let mut new_walker = self.clone();
            new_walker.try_pin_node(pinnable_node).ok()?;
            Some(new_walker)
        })
    }

    /// Get all equivalent ports among the commits that are descendants of the
    /// current commit.
    ///
    /// The ports in the returned iterator will be in the same direction as
    /// `port`.
    fn equivalent_descendant_ports(
        &self,
        node: PatchNode,
        port: Port,
    ) -> BTreeSet<(PatchNode, Port)> {
        // First, convert everything into incoming ports (but keep track of whether the
        // port was originally outgoing)
        let port_direction = port.direction();
        use itertools::Either::{Left, Right};
        let mut ports_queue = match port.as_directed() {
            Left(incoming) => VecDeque::from(vec![(node, incoming)]),
            Right(outgoing) => {
                let hugr = self.state_space.commit_hugr(node.0);
                VecDeque::from_iter(
                    hugr.linked_inputs(node.1, outgoing)
                        .map(|(n, p)| (PatchNode(node.0, n), p)),
                )
            }
        };

        // Now, perform a BFS to find all equivalent ports
        let mut all_equivalent_ports = BTreeSet::new();
        while let Some((node, port)) = ports_queue.pop_front() {
            let equivalent_node_port = match port_direction {
                Direction::Incoming => (node, port.into()),
                Direction::Outgoing => {
                    let commit_id = node.0;
                    let hugr = self.state_space.commit_hugr(commit_id);
                    let (node, port) = hugr
                        .single_linked_output(node.1, port)
                        .expect("invalid DFG wire");
                    (PatchNode(commit_id, node), port.into())
                }
            };
            if !all_equivalent_ports.insert(equivalent_node_port) {
                continue;
            }

            match port_direction {
                Direction::Incoming => {
                    ports_queue.extend(self.state_space.children_input_ports(node, port))
                }
                Direction::Outgoing => {
                    ports_queue.extend(self.state_space.children_output_ports(node, port))
                }
            };
        }
        all_equivalent_ports
    }

    fn is_pinned(&self, node: PatchNode) -> bool {
        self.pinned_nodes.contains(&node)
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

/// A wire in the current HUGR of a [`Walker`] with some of its endpoints
/// pinned.
///
/// A [`PinnedWire`] distinguishes itself from a normal HUGR
/// [`Wire`](crate::Wire) in that some of its endpoints may be pinned, while
/// others may not. We say that a port is pinned, if the node it it attached to
/// is pinned in the walker.
///
/// All pinned ports of a [`PinnedWire`] can be retrieved using
/// [`PinnedWire::incoming_ports`] and [`PinnedWire::outgoing_port`]. Unpinned
/// ports, on the other hand, represent undetermined connections, which may
/// still change as the walker is expanded (see [`Walker::expand`]).
///
/// Whether all incoming or outgoing ports are pinned can be checked using
/// [`PinnedWire::is_complete`].
#[derive(Debug, Clone)]
pub struct PinnedWire {
    outgoing: MaybePinned<OutgoingPort>,
    incoming: Vec<MaybePinned<IncomingPort>>,
}

/// A private enum to track whether a port is pinned.
///
/// Encapsulation: only [`MaybePinned::Pinned`] values may be exposed publicly!
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MaybePinned<P> {
    Pinned(PatchNode, P),
    Unpinned(PatchNode, P),
}

impl<P> MaybePinned<P> {
    fn is_pinned(&self) -> bool {
        matches!(self, MaybePinned::Pinned(_, _))
    }

    fn into_unpinned<PP: From<P>>(self) -> Option<(PatchNode, PP)> {
        match self {
            MaybePinned::Pinned(_, _) => None,
            MaybePinned::Unpinned(node, port) => Some((node, port.into())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[error("port ({0:?}, {1:?}) is not pinned")]
struct UnpinnedPort<P>(PatchNode, P);

impl<P> TryFrom<MaybePinned<P>> for (PatchNode, P) {
    type Error = UnpinnedPort<P>;

    fn try_from(value: MaybePinned<P>) -> Result<Self, Self::Error> {
        match value {
            MaybePinned::Pinned(node, port) => Ok((node, port)),
            MaybePinned::Unpinned(node, port) => Err(UnpinnedPort(node, port)),
        }
    }
}

impl PinnedWire {
    /// Create a new pinned wire in `walker` from a pinned node and a port.
    ///
    /// # Panics
    /// Panics if `node` is not pinned in `walker`.
    pub fn from_pinned_port(node: PatchNode, port: impl Into<Port>, walker: &Walker) -> Self {
        assert!(walker.is_pinned(node), "node must be pinned");

        let (outgoing_node, outgoing_port) =
            walker.selected_commits.get_single_outgoing_port(node, port);

        let incoming_nodes_ports = walker
            .selected_commits
            .get_all_incoming_ports(outgoing_node, outgoing_port);

        let outgoing = if walker.is_pinned(outgoing_node) {
            debug_assert!(
                !walker
                    .selected_commits
                    .deleted_nodes(outgoing_node.0)
                    .contains(&outgoing_node.1),
                "pinned node is deleted"
            );
            MaybePinned::Pinned(outgoing_node, outgoing_port)
        } else {
            MaybePinned::Unpinned(outgoing_node, outgoing_port)
        };

        let incoming = incoming_nodes_ports
            .map(|(incoming_node, incoming_ports)| {
                if walker.is_pinned(incoming_node) {
                    debug_assert!(
                        !walker
                            .selected_commits
                            .deleted_nodes(incoming_node.0)
                            .contains(&incoming_node.1),
                        "pinned node is deleted"
                    );
                    MaybePinned::Pinned(incoming_node, incoming_ports)
                } else {
                    MaybePinned::Unpinned(incoming_node, incoming_ports)
                }
            })
            .collect();

        Self { outgoing, incoming }
    }

    /// Check if all ports on the wire in the given direction are pinned.
    ///
    /// A wire is complete in a direction if and only if expanding the wire
    /// in that direction would yield no new walkers. If no direction is
    /// specified, checks if the wire is complete in both directions.
    pub fn is_complete(&self, dir: impl Into<Option<Direction>>) -> bool {
        match dir.into() {
            Some(Direction::Outgoing) => self.outgoing.is_pinned(),
            Some(Direction::Incoming) => self.incoming.iter().all(|p| p.is_pinned()),
            None => self.outgoing.is_pinned() && self.incoming.iter().all(|p| p.is_pinned()),
        }
    }

    /// Get the outgoing port of the wire, if it is pinned.
    ///
    /// Returns `None` if the outgoing port is not pinned.
    pub fn outgoing_port(&self) -> Option<(PatchNode, OutgoingPort)> {
        self.outgoing.try_into().ok()
    }

    /// Get all pinned incoming ports of the wire.
    ///
    /// Returns an iterator over all pinned incoming ports.
    pub fn incoming_ports(&self) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        self.incoming.iter().filter_map(|&p| p.try_into().ok())
    }

    /// Get all pinned ports of the wire.
    pub fn all_ports(&self) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        fn to_port((node, port): (PatchNode, impl Into<Port>)) -> (PatchNode, Port) {
            (node, port.into())
        }
        self.outgoing_port()
            .into_iter()
            .map(to_port)
            .chain(self.incoming_ports().map(to_port))
    }

    /// Get all unpinned ports of the wire in the given direction.
    ///
    /// Not public-facing!
    fn unpinned_ports(
        &self,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        let incoming = self
            .incoming
            .iter()
            .filter_map(|p| p.into_unpinned::<Port>());
        let outgoing = self.outgoing.into_unpinned::<Port>();
        let dir = dir.into();
        (dir != Some(Direction::Outgoing))
            .then_some(incoming)
            .into_iter()
            .flatten()
            .chain(
                (dir != Some(Direction::Incoming))
                    .then_some(outgoing)
                    .into_iter()
                    .flatten(),
            )
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
    use rstest::rstest;

    use crate::hugr::persistent::state_space::CommitId;
    use crate::std_extensions::logic::LogicOp;

    use super::super::tests::test_state_space;
    use super::*;

    #[rstest]
    fn test_walker_base_or_child_expansion(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [commit1, _commit2, _commit3, _commit4]) = test_state_space;
        let mut walker = Walker::new(&state_space);

        // Get an initial node to pin
        let base_and_node = {
            let base_hugr = state_space.base_hugr();
            let and_node = base_hugr
                .nodes()
                .find(|&n| base_hugr.get_optype(n) == &LogicOp::And.into())
                .unwrap();
            PatchNode(state_space.base(), and_node)
        };
        walker.try_pin_node(base_and_node).unwrap();
        assert!(walker.is_pinned(base_and_node));

        let in0 = walker.get_wire(base_and_node, IncomingPort::from(0));
        let mut expanded_in = walker.expand(&in0, Direction::Incoming);
        let expanded_out = walker.expand(&in0, Direction::Outgoing);

        // a single incoming port (already pinned) => no more expansion
        assert!(expanded_in.next().is_none());
        // commit 2 cannot be applied, because AND is pinned
        // => only base commit, or commit1
        let new_walkers = expanded_out.collect_vec();
        assert_eq!(new_walkers.len(), 2);
        for new_walker in new_walkers {
            // new wire is complete (and thus cannot be expanded)
            let in0 = new_walker.get_wire(base_and_node, IncomingPort::from(0));
            assert!(in0.is_complete(None));
            assert!(new_walker.expand(&in0, None).next().is_none());

            // all nodes on wire are pinned
            let (not_node, _) = in0.outgoing_port().unwrap();
            assert!(new_walker.is_pinned(base_and_node));
            assert!(new_walker.is_pinned(not_node));

            // not node is either in commit1 or the base
            assert!([commit1, state_space.base()].contains(&not_node.0));

            // not node is a NOT gate
            let persistent_hugr = new_walker.into_hugr();
            let hugr = persistent_hugr.commit_hugr(not_node.0);
            assert_eq!(hugr.get_optype(not_node.1), &LogicOp::Not.into());
        }
    }

    #[rstest]
    fn test_walker_disjoint_nephew_expansion(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [commit1, commit2, commit3, commit4]) = test_state_space;
        let mut walker = Walker::new(&state_space);

        // Initially, pin the second NOT of commit4
        let not4_node = {
            let repl4 = state_space.replacement(commit4).unwrap();
            let hugr4 = state_space.commit_hugr(commit4);
            let [_, output] = repl4.get_replacement_io().unwrap();
            let (second_not_node, _) = hugr4.single_linked_output(output, 0).unwrap();
            PatchNode(commit4, second_not_node)
        };
        walker.try_pin_node(not4_node).unwrap();
        assert!(walker.is_pinned(not4_node));

        let out0 = walker.get_wire(not4_node, OutgoingPort::from(0));
        let expanded_out = walker.expand(&out0, Direction::Outgoing).collect_vec();
        // a single outgoing port (already pinned) => no more expansion
        assert!(expanded_out.is_empty());

        let expanded_in = walker.expand(&out0, Direction::Incoming);
        // Three options:
        // - AND gate from base
        // - XOR gate from commit3
        // - XOR gate from commit2 (which implies commit1)
        let mut exp_options = BTreeSet::from_iter([
            BTreeSet::from_iter([state_space.base(), commit4]),
            BTreeSet::from_iter([state_space.base(), commit3, commit4]),
            BTreeSet::from_iter([state_space.base(), commit1, commit2, commit4]),
        ]);
        let new_walkers = expanded_in.collect_vec();
        for new_walker in new_walkers {
            // new wire is complete (and thus cannot be expanded)
            let out0 = new_walker.get_wire(not4_node, OutgoingPort::from(0));
            assert!(out0.is_complete(None));
            assert!(new_walker.expand(&out0, None).next().is_none());

            // all nodes on wire are pinned
            let (next_node, _) = out0.incoming_ports().exactly_one().ok().unwrap();
            assert!(new_walker.is_pinned(not4_node));
            assert!(new_walker.is_pinned(next_node));

            let persistent_hugr = new_walker.into_hugr();

            // next_node is either in base (AND gate), in commit3 (XOR gate), or
            // in commit2 (XOR gate)
            let hugr = persistent_hugr.commit_hugr(next_node.0);
            if next_node.0 == state_space.base() {
                assert_eq!(hugr.get_optype(next_node.1), &LogicOp::And.into());
            } else if [commit3, commit2].contains(&next_node.0) {
                assert_eq!(hugr.get_optype(next_node.1), &LogicOp::Xor.into());
            } else {
                panic!("neighbour of not4 must be in base, commit2 or commit3");
            }

            // selected commits must be one of the valid options
            let commit_ids = persistent_hugr.all_commit_ids().collect::<BTreeSet<_>>();
            assert!(
                exp_options.remove(&commit_ids),
                "{:?} not an expected set of commit IDs (or duplicate)",
                commit_ids
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
}

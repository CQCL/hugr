use std::collections::{BTreeSet, VecDeque};

use hugr_core::{
    Direction, HugrView, IncomingPort, OutgoingPort, Port, Wire,
    hugr::patch::simple_replace::BoundaryMode,
};
use itertools::Itertools;

use crate::{CommitId, PatchNode, PersistentHugr, Walker};

/// A wire in a [`PersistentHugr`].
///
/// A wire may be composed of multiple wires in the underlying commits
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistentWire {
    wires: BTreeSet<CommitWire>,
}

/// A wire within a commit HUGR of a [`PersistentHugr`].
///
/// Also stores the ID of the commit that contains the wire;
/// equivalent to (indeed contains) a `Wire<PatchNode>`.
///
/// Note that it does not correspond to a valid wire in a [`PersistentHugr`]
/// (see [`PersistentWire`]): some of its connected ports may be on deleted or
/// IO nodes that are not valid in the [`PersistentHugr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CommitWire(Wire<PatchNode>);

impl CommitWire {
    fn from_connected_port(
        PatchNode(commit_id, node): PatchNode,
        port: impl Into<Port>,
        hugr: &PersistentHugr,
    ) -> Self {
        let commit_hugr = hugr.get_commit(commit_id).commit_hugr();
        let wire = Wire::from_connected_port(node, port, commit_hugr);
        Self(Wire::new(PatchNode(commit_id, wire.node()), wire.source()))
    }

    fn all_connected_ports<'h>(
        &self,
        hugr: &'h PersistentHugr,
    ) -> impl Iterator<Item = (PatchNode, Port)> + use<'h> {
        let wire = Wire::new(self.0.node().1, self.0.source());
        let commit_id = self.commit_id();
        wire.all_connected_ports(hugr.get_commit(commit_id).commit_hugr())
            .map(move |(node, port)| (hugr.to_persistent_node(node, commit_id), port))
    }

    fn commit_id(&self) -> CommitId {
        self.0.node().0
    }

    delegate::delegate! {
        to self.0 {
            fn node(&self) -> PatchNode;
        }
    }
}

/// A node in a commit of a [`PersistentHugr`] is either a valid node of the
/// HUGR, a node deleted by a child commit in that [`PersistentHugr`], or an
/// input or output node in a replacement graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NodeStatus {
    /// A node deleted by a child commit in that [`PersistentHugr`].
    ///
    /// The ID of the child commit is stored in the variant.
    Deleted(CommitId),
    /// An input or output node in the replacement graph of a Commit
    ReplacementIO,
    /// A valid node in the [`PersistentHugr`]
    Valid,
}

impl PersistentHugr {
    pub fn get_wire(&self, node: PatchNode, port: impl Into<Port>) -> PersistentWire {
        PersistentWire::from_port(node, port, self)
    }

    /// Whether a node is valid in `self`, is deleted or is an IO node in a
    /// replacement graph.
    fn node_status(&self, per_node @ PatchNode(commit_id, node): PatchNode) -> NodeStatus {
        debug_assert!(self.contains_id(commit_id), "unknown commit");
        if self
            .get_commit(commit_id)
            .replacement()
            .is_some_and(|repl| repl.get_replacement_io().contains(&node))
        {
            NodeStatus::ReplacementIO
        } else if let Some(commit_id) = self.find_deleting_commit(per_node) {
            NodeStatus::Deleted(commit_id)
        } else {
            NodeStatus::Valid
        }
    }

    /// The unique outgoing port in `self` that `port` is attached to.
    ///
    /// # Panics
    ///
    /// Panics if `node` is not in `self` (in particular if it is deleted) or if
    /// `port` is not a value port in `node`.
    pub(crate) fn single_outgoing_port(
        &self,
        node: PatchNode,
        port: impl Into<IncomingPort>,
    ) -> (PatchNode, OutgoingPort) {
        let w = self.get_wire(node, port.into());
        w.single_outgoing_port(self)
            .expect("found invalid dfg wire")
    }

    /// All incoming ports that the given outgoing port is attached to.
    ///
    /// # Panics
    ///
    /// Panics if `out_node` is not in `self` (in particular if it is deleted)
    /// or if `out_port` is not a value port in `out_node`.
    pub(crate) fn all_incoming_ports(
        &self,
        out_node: PatchNode,
        out_port: OutgoingPort,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        let w = self.get_wire(out_node, out_port);
        w.into_all_ports(self, Direction::Incoming)
            .map(|(node, port)| (node, port.as_incoming().unwrap()))
    }
}

impl PersistentWire {
    /// Get the wire connected to a specified port of a pinned node in `hugr`.
    fn from_port(node: PatchNode, port: impl Into<Port>, per_hugr: &PersistentHugr) -> Self {
        assert!(per_hugr.contains_node(node), "node not in hugr");

        // Queue of wires within each commit HUGR, that combined will form the
        // persistent wire.
        let mut commit_wires =
            BTreeSet::from_iter([CommitWire::from_connected_port(node, port, per_hugr)]);
        let mut queue = VecDeque::from_iter(commit_wires.iter().copied());

        while let Some(wire) = queue.pop_front() {
            let commit_id = wire.commit_id();
            let commit = per_hugr.get_commit(commit_id);
            let commit_hugr = commit.commit_hugr();
            let all_ports = wire.all_connected_ports(per_hugr);

            for (per_node @ PatchNode(_, node), port) in all_ports {
                match per_hugr.node_status(per_node) {
                    NodeStatus::Deleted(deleted_by) => {
                        // If node is deleted, check if there are wires between
                        // ports on the opposite end of the wire and boundary
                        // ports in the child commit that deleted the node.
                        for (opp_node, opp_port) in commit_hugr.linked_ports(node, port) {
                            for (child_node, child_port) in commit.linked_child_ports(
                                opp_node,
                                opp_port,
                                per_hugr.get_commit(deleted_by),
                                BoundaryMode::IncludeIO,
                            ) {
                                debug_assert_eq!(child_node.owner(), deleted_by);
                                let w = CommitWire::from_connected_port(
                                    child_node, child_port, per_hugr,
                                );
                                if commit_wires.insert(w) {
                                    queue.push_back(w);
                                }
                            }
                        }
                    }
                    NodeStatus::ReplacementIO => {
                        // If node is an input (resp. output) node in a replacement graph, there
                        // must be (at least) one wire between the incoming (resp. outgoing)
                        // boundary ports of the commit (i.e. the ports connected to
                        // the input resp. output) and ports in a parent commit.
                        for (opp_node, opp_port) in commit_hugr.linked_ports(node, port) {
                            for (parent_node, parent_port) in
                                commit.linked_parent_ports(opp_node, opp_port)
                            {
                                let w = CommitWire::from_connected_port(
                                    parent_node,
                                    parent_port,
                                    per_hugr,
                                );
                                if commit_wires.insert(w) {
                                    queue.push_back(w);
                                }
                            }
                        }
                    }
                    NodeStatus::Valid => {}
                }
            }
        }

        Self {
            wires: commit_wires,
        }
    }

    /// Get all ports attached to a wire in `hugr`.
    ///
    /// All ports returned are on nodes that are contained in `hugr`.
    pub fn all_ports(
        &self,
        hugr: &PersistentHugr,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        all_ports_impl(self.wires.iter().copied(), dir.into(), hugr)
    }

    /// All commit IDs that the wire traverses.
    pub fn owners(&self) -> impl Iterator<Item = CommitId> {
        self.wires.iter().map(|w| w.node().owner()).unique()
    }

    /// Consume the wire and return all ports attached to a wire in `hugr`.
    ///
    /// All ports returned are on nodes that are contained in `hugr`.
    pub fn into_all_ports(
        self,
        hugr: &PersistentHugr,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        all_ports_impl(self.wires.into_iter(), dir.into(), hugr)
    }

    pub fn single_outgoing_port(&self, hugr: &PersistentHugr) -> Option<(PatchNode, OutgoingPort)> {
        single_outgoing(self.all_ports(hugr, Direction::Outgoing))
    }

    pub fn all_incoming_ports(
        &self,
        hugr: &PersistentHugr,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        self.all_ports(hugr, Direction::Incoming)
            .map(|(node, port)| (node, port.as_incoming().unwrap()))
    }
}

impl Walker<'_> {
    /// Get all ports on a wire that are not pinned in `self`.
    pub(crate) fn wire_unpinned_ports(
        &self,
        wire: &PersistentWire,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        let ports = wire.all_ports(self.as_hugr_view(), dir);
        ports.filter(|(node, _)| !self.is_pinned(*node))
    }

    /// Get the ports of the wire that are on pinned nodes of `self`.
    pub fn wire_pinned_ports(
        &self,
        wire: &PersistentWire,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        let ports = wire.all_ports(self.as_hugr_view(), dir);
        ports.filter(|(node, _)| self.is_pinned(*node))
    }

    /// Get the outgoing port of a wire if it is pinned in `walker`.
    pub fn wire_pinned_outport(&self, wire: &PersistentWire) -> Option<(PatchNode, OutgoingPort)> {
        single_outgoing(self.wire_pinned_ports(wire, Direction::Outgoing))
    }

    /// Get all pinned incoming ports of a wire.
    pub fn wire_pinned_inports(
        &self,
        wire: &PersistentWire,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        self.wire_pinned_ports(wire, Direction::Incoming)
            .map(|(node, port)| (node, port.as_incoming().expect("incoming port")))
    }

    /// Whether a wire is complete in the specified direction, i.e. there are no
    /// unpinned ports left.
    pub fn is_complete(&self, wire: &PersistentWire, dir: impl Into<Option<Direction>>) -> bool {
        self.wire_unpinned_ports(wire, dir).next().is_none()
    }
}

/// Implementation of the (shared) body of [`PersistentWire::all_ports`] and
/// [`PersistentWire::into_all_ports`].
fn all_ports_impl(
    wires: impl Iterator<Item = CommitWire>,
    dir: Option<Direction>,
    per_hugr: &PersistentHugr,
) -> impl Iterator<Item = (PatchNode, Port)> {
    let all_ports = wires.flat_map(move |w| w.all_connected_ports(per_hugr));

    // Filter out invalid and wrong direction ports
    all_ports
        .filter(move |(_, port)| dir.is_none_or(|dir| port.direction() == dir))
        .filter(|&(node, _)| per_hugr.node_status(node) == NodeStatus::Valid)
}

fn single_outgoing<N>(iter: impl Iterator<Item = (N, Port)>) -> Option<(N, OutgoingPort)> {
    let (node, port) = iter.exactly_one().ok()?;
    Some((node, port.as_outgoing().ok()?))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{
        PatchNode, PersistentHugr,
        tests::{TestStateSpace, test_state_space},
    };
    use hugr_core::{HugrView, OutgoingPort};
    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    fn test_all_ports(test_state_space: TestStateSpace) {
        let [_, _, cm3, cm4] = test_state_space.commits();
        let hugr = PersistentHugr::try_new([cm3.clone(), cm4.clone()]).unwrap();
        let cm4_not = {
            let hugr4 = cm4.commit_hugr();
            let out = cm4.replacement().unwrap().get_replacement_io()[1];
            let node = hugr4.input_neighbours(out).exactly_one().ok().unwrap();
            PatchNode(cm4.id(), node)
        };
        let w = hugr.get_wire(cm4_not, OutgoingPort::from(0));
        assert_eq!(
            BTreeSet::from_iter(w.wires.iter().map(|w| w.0.node().0)),
            BTreeSet::from_iter([cm3.id(), cm4.id(), hugr.base(),])
        );
    }
}

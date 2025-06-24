use std::collections::{BTreeSet, VecDeque};

use hugr_core::{Direction, HugrView, IncomingPort, OutgoingPort, Port, Wire};
use itertools::{Either, Itertools};

use crate::{PatchNode, PersistentHugr, Resolver, Walker};

/// A wire in a [`PersistentHugr`].
///
/// A wire may be composed of multiple wires in the underlying commits
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistentWire {
    wires: BTreeSet<Wire<PatchNode>>,
}

/// A node in a commit of a [`PersistentHugr`] is either a valid node of the
/// HUGR, a node deleted by a child commit, or an input or output node in a
/// replacement graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NodeStatus {
    Deleted,
    InOut,
    Valid,
}

impl<R> PersistentHugr<R> {
    pub fn get_wire(&self, node: PatchNode, port: impl Into<Port>) -> PersistentWire {
        PersistentWire::from_port(node, port, self)
    }

    /// Whether a node is valid in `self`, is deleted or is an IO node in a
    /// replacement graph.
    fn node_status(&self, PatchNode(commit_id, node): PatchNode) -> NodeStatus {
        debug_assert!(self.contains_id(commit_id), "unknown commit");
        if self
            .replacement(commit_id)
            .is_some_and(|repl| repl.get_replacement_io().contains(&node))
        {
            NodeStatus::InOut
        } else if self.deleted_nodes(commit_id).contains(&node) {
            NodeStatus::Deleted
        } else {
            NodeStatus::Valid
        }
    }
}

impl PersistentWire {
    /// Get the wire connected to a specified port of a pinned node in `hugr`.
    pub fn from_port<R>(
        node: PatchNode,
        port: impl Into<Port>,
        per_hugr: &PersistentHugr<R>,
    ) -> Self {
        assert!(per_hugr.contains_node(node), "node not in hugr");

        let mut wires = BTreeSet::from_iter([to_wire(node, port, per_hugr)]);
        let mut queue = VecDeque::from_iter(wires.iter().copied());

        while let Some(wire) = queue.pop_front() {
            let all_ports = all_ports(wire, per_hugr);

            for (per_node @ PatchNode(commit_id, node), port) in all_ports {
                let hugr = per_hugr.commit_hugr(commit_id);
                match per_hugr.node_status(per_node) {
                    NodeStatus::Deleted => {
                        // If node is deleted, check if there are wires between
                        // ports on the opposite end of the wire and boundary
                        // ports in the child commit that deleted the node.
                        let deleted_by = per_hugr
                            .find_deleting_commit(per_node)
                            .expect("deleted node has deleting commit");
                        for (opp_node, opp_port) in hugr.linked_ports(node, port) {
                            let opp_node = PatchNode(commit_id, opp_node);
                            for (child_node, child_port) in per_hugr
                                .as_state_space()
                                .linked_child_ports(opp_node, opp_port, deleted_by)
                            {
                                let w = to_wire(child_node, child_port, per_hugr);
                                if wires.insert(w) {
                                    queue.push_back(w);
                                }
                            }
                        }
                    }
                    NodeStatus::InOut => {
                        // If node is an IO node in a replacement graph, there
                        // must be (at least) one wire
                        // between the boundary ports of the
                        // commit (the ports opposite of the IO node) and ports
                        // in a parent commit.
                        for (opp_node, opp_port) in hugr.linked_ports(node, port) {
                            let opp_node = PatchNode(commit_id, opp_node);
                            for (parent_node, parent_port) in per_hugr
                                .as_state_space()
                                .linked_parent_ports(opp_node, opp_port)
                            {
                                let w = to_wire(parent_node, parent_port, per_hugr);
                                if wires.insert(w) {
                                    queue.push_back(w);
                                }
                            }
                        }
                    }
                    NodeStatus::Valid => {}
                }
            }
        }

        Self { wires }
    }

    /// Get all ports attached to a wire in `hugr`.
    ///
    /// All ports returned are on nodes that are contained in `hugr`.
    pub fn all_ports<R>(
        &self,
        hugr: &PersistentHugr<R>,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        let dir = dir.into();
        let all_ports = self.wires.iter().flat_map(|&w| all_ports(w, hugr));
        let dir_ports = all_ports.filter(move |(_, port)| {
            if let Some(dir) = dir {
                port.direction() == dir
            } else {
                true
            }
        });
        dir_ports.filter(|&(node, _)| hugr.node_status(node) == NodeStatus::Valid)
    }

    /// Consume the wire and return all ports attached to a wire in `hugr`.
    ///
    /// All ports returned are on nodes that are contained in `hugr`.
    pub fn into_all_ports<R>(
        self,
        hugr: &PersistentHugr<R>,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> {
        let dir = dir.into();
        let all_ports = self.wires.into_iter().flat_map(|w| all_ports(w, hugr));
        let dir_ports = all_ports.filter(move |(_, port)| {
            if let Some(dir) = dir {
                port.direction() == dir
            } else {
                true
            }
        });
        dir_ports.filter(|&(node, _)| hugr.node_status(node) == NodeStatus::Valid)
    }

    pub fn single_outgoing_port<R>(
        &self,
        hugr: &PersistentHugr<R>,
    ) -> Option<(PatchNode, OutgoingPort)> {
        self.all_ports(hugr, Direction::Outgoing)
            .exactly_one()
            .ok()
            .map(|(node, port)| (node, port.as_outgoing().unwrap()))
    }

    pub fn all_incoming_ports<R>(
        &self,
        hugr: &PersistentHugr<R>,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> {
        self.all_ports(hugr, Direction::Incoming)
            .map(|(node, port)| (node, port.as_incoming().unwrap()))
    }
}

impl<R: Resolver> Walker<'_, R> {
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
        self.wire_pinned_ports(wire, Direction::Outgoing)
            .at_most_one()
            .ok()
            .expect("valid dfg wire")
            .map(|(node, port)| (node, port.as_outgoing().expect("outgoing port")))
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

fn to_wire<R>(
    PatchNode(commit_id, node): PatchNode,
    port: impl Into<Port>,
    per_hugr: &PersistentHugr<R>,
) -> Wire<PatchNode> {
    let (node, out_port) = as_outgoing(node, port, per_hugr.commit_hugr(commit_id));
    Wire::new(PatchNode(commit_id, node), out_port)
}

/// Convert a port to an outgoing port by taking its opposite port if required.
fn as_outgoing<N>(
    node: N,
    port: impl Into<Port>,
    hugr: &impl HugrView<Node = N>,
) -> (N, OutgoingPort) {
    match port.into().as_directed() {
        Either::Left(incoming) => hugr
            .single_linked_output(node, incoming)
            .expect("invalid dfg port"),
        Either::Right(outgoing) => (node, outgoing),
    }
}

/// Get all ports connected to a wire in a persistent HUGR.
fn all_ports<R>(
    wire: Wire<PatchNode>,
    per_hugr: &PersistentHugr<R>,
) -> impl Iterator<Item = (PatchNode, Port)> {
    let PatchNode(commit_id, node) = wire.node();
    let hugr = per_hugr.commit_hugr(commit_id);

    // Get the outgoing port from the wire
    let out_port = wire.source();

    // Return an iterator that yields the original node and port, followed by all
    // linked ports
    std::iter::once((PatchNode(commit_id, node), out_port.into())).chain(
        hugr.linked_ports(node, out_port)
            .map(move |(linked_node, linked_port)| {
                (PatchNode(commit_id, linked_node), linked_port)
            }),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{CommitId, CommitStateSpace, PatchNode, tests::test_state_space};
    use hugr_core::{HugrView, OutgoingPort};
    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    fn test_all_ports(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, [_, _, cm3, cm4]) = test_state_space;
        let hugr = state_space.try_extract_hugr([cm3, cm4]).unwrap();
        let cm4_not = {
            let hugr4 = state_space.commit_hugr(cm4);
            let out = state_space.replacement(cm4).unwrap().get_replacement_io()[1];
            let node = hugr4.input_neighbours(out).exactly_one().ok().unwrap();
            PatchNode(cm4, node)
        };
        let w = hugr.get_wire(cm4_not, OutgoingPort::from(0));
        assert_eq!(
            BTreeSet::from_iter(w.wires.iter().map(|w| w.node().0)),
            BTreeSet::from_iter([cm3, cm4, state_space.base(),])
        );
    }
}

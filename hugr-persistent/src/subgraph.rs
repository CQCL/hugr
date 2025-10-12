use std::collections::BTreeSet;

use hugr_core::{
    IncomingPort, OutgoingPort,
    hugr::views::{
        SiblingSubgraph,
        sibling_subgraph::{IncomingPorts, InvalidSubgraph, OutgoingPorts},
    },
};
use itertools::Itertools;
use thiserror::Error;

use crate::{CommitId, PatchNode, PersistentHugr, PersistentWire, Walker};

/// A set of pinned nodes and wires between them, along with a fixed input
/// and output boundary, simmilar to [`SiblingSubgraph`].
///
/// Unlike [`SiblingSubgraph`], subgraph validity (in particular convexity) is
/// not checked (and cannot be checked), as the same [`PinnedSubgraph`] may
/// represent [`SiblingSubgraph`]s in different HUGRs.
///
/// Obtain a valid [`SiblingSubgraph`] for a specific [`PersistentHugr`] by
/// calling [`PinnedSubgraph::to_sibling_subgraph`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PinnedSubgraph {
    /// The nodes of the induced subgraph.
    nodes: BTreeSet<PatchNode>,
    /// The input ports of the subgraph.
    ///
    /// All ports within the same inner vector of `inputs` must be connected to
    /// the same outgoing port (and thus have the same type). The outer vector
    /// defines the input signature, i.e. the number and types of incoming wires
    /// into the subgraph.
    ///
    /// Multiple input ports of the same type may be grouped within the same
    /// inner vector: this corresponds to an input parameter that is copied and
    /// used multiple times in the subgraph. An inner vector may also be empty,
    /// corresponding to discarding an input parameter.
    ///
    /// Each port must be unique and belong to a node in `nodes`. Input ports of
    /// linear types will always appear as singleton vectors.
    inputs: Vec<Vec<(PatchNode, IncomingPort)>>,
    /// The output ports of the subgraph.
    ///
    /// The types of the output ports define the output signature of the
    /// subgraph. Repeated ports of copyable types are allowed and correspond to
    /// copying a value of the subgraph multiple times.
    ///
    /// Every port must belong to a node in `nodes`.
    outputs: Vec<(PatchNode, OutgoingPort)>,
    /// The commits that must be selected in the host for the subgraph to be
    /// valid.
    selected_commits: BTreeSet<CommitId>,
}

impl From<SiblingSubgraph<PatchNode>> for PinnedSubgraph {
    fn from(subgraph: SiblingSubgraph<PatchNode>) -> Self {
        Self {
            inputs: subgraph.incoming_ports().clone(),
            outputs: subgraph.outgoing_ports().clone(),
            nodes: BTreeSet::from_iter(subgraph.nodes().iter().copied()),
            selected_commits: BTreeSet::new(),
        }
    }
}

impl PinnedSubgraph {
    /// Create a new subgraph from a set of pinned nodes and wires.
    ///
    /// All nodes must be pinned and all wires must be complete in the given
    /// `walker`.
    ///
    /// Nodes that are not isolated, i.e. are attached to at least one wire in
    /// `wires` will be added implicitly to the graph and do not need to be
    /// explicitly listed in `nodes`.
    pub fn try_from_pinned(
        nodes: impl IntoIterator<Item = PatchNode>,
        wires: impl IntoIterator<Item = PersistentWire>,
        walker: &Walker,
    ) -> Result<Self, InvalidPinnedSubgraph> {
        let mut selected_commits = BTreeSet::new();
        let host = walker.as_hugr_view();
        let wires = wires.into_iter().collect_vec();
        let nodes = nodes.into_iter().collect_vec();

        for w in wires.iter() {
            if !walker.is_complete(w, None) {
                return Err(InvalidPinnedSubgraph::IncompleteWire(w.clone()));
            }
            for id in w.owners() {
                if host.contains_id(id) {
                    selected_commits.insert(id);
                } else {
                    return Err(InvalidPinnedSubgraph::InvalidCommit(id));
                }
            }
        }

        if let Some(&unpinned) = nodes.iter().find(|&&n| !walker.is_pinned(n)) {
            return Err(InvalidPinnedSubgraph::UnpinnedNode(unpinned));
        }

        let (inputs, outputs, all_nodes) = Self::compute_io_ports(nodes, wires, host);

        Ok(Self {
            selected_commits,
            nodes: all_nodes,
            inputs,
            outputs,
        })
    }

    /// Create a new subgraph from a set of complete wires in `walker`.
    pub fn try_from_wires(
        wires: impl IntoIterator<Item = PersistentWire>,
        walker: &Walker,
    ) -> Result<Self, InvalidPinnedSubgraph> {
        Self::try_from_pinned(std::iter::empty(), wires, walker)
    }

    /// Compute the input and output ports for the given pinned nodes and wires.
    ///
    /// Return the input boundary ports, output boundary ports as well as the
    /// set of all nodes in the subgraph.
    pub fn compute_io_ports(
        nodes: impl IntoIterator<Item = PatchNode>,
        wires: impl IntoIterator<Item = PersistentWire>,
        host: &PersistentHugr,
    ) -> (
        IncomingPorts<PatchNode>,
        OutgoingPorts<PatchNode>,
        BTreeSet<PatchNode>,
    ) {
        let mut wire_ports_incoming = BTreeSet::new();
        let mut wire_ports_outgoing = BTreeSet::new();

        for w in wires {
            wire_ports_incoming.extend(w.all_incoming_ports(host));
            wire_ports_outgoing.extend(w.single_outgoing_port(host));
        }

        let mut all_nodes = BTreeSet::from_iter(nodes);
        all_nodes.extend(wire_ports_incoming.iter().map(|&(n, _)| n));
        all_nodes.extend(wire_ports_outgoing.iter().map(|&(n, _)| n));

        // (in/out) boundary: all in/out ports on the nodes of the wire, minus ports
        // that are part of the wires
        let inputs = all_nodes
            .iter()
            .flat_map(|&PatchNode(owner, node)| {
                let owner = host.get_commit(owner);
                owner
                    .input_value_ports(node)
                    .map(|(n, p)| (owner.to_patch_node(n), p))
            })
            .filter(|node_port| !wire_ports_incoming.contains(node_port))
            .map(|np| vec![np])
            .collect_vec();
        let outputs = all_nodes
            .iter()
            .flat_map(|&PatchNode(owner, node)| {
                let owner = host.get_commit(owner);
                owner
                    .output_value_ports(node)
                    .map(|(n, p)| (owner.to_patch_node(n), p))
            })
            .filter(|node_port| !wire_ports_outgoing.contains(node_port))
            .collect_vec();

        (inputs, outputs, all_nodes)
    }

    /// Convert the pinned subgraph to a [`SiblingSubgraph`] for the given
    /// `host`.
    ///
    /// This will fail if any of the required selected commits are not in the
    /// host, if any of the nodes are invalid in the host (e.g. deleted by
    /// another commit in host), or if the subgraph is not convex.
    pub fn to_sibling_subgraph(
        &self,
        host: &PersistentHugr,
    ) -> Result<SiblingSubgraph<PatchNode>, InvalidPinnedSubgraph> {
        if let Some(&unselected) = self
            .selected_commits
            .iter()
            .find(|&&id| !host.contains_id(id))
        {
            return Err(InvalidPinnedSubgraph::InvalidCommit(unselected));
        }

        if let Some(invalid) = self.nodes.iter().find(|&&n| !host.contains_node(n)) {
            return Err(InvalidPinnedSubgraph::InvalidNode(*invalid));
        }

        Ok(SiblingSubgraph::try_new(
            self.inputs.clone(),
            self.outputs.clone(),
            host,
        )?)
    }

    /// Iterate over all the commits required by this pinned subgraph.
    pub fn selected_commits(&self) -> impl Iterator<Item = CommitId> + '_ {
        self.selected_commits.iter().copied()
    }

    /// Iterate over all the nodes in this pinned subgraph.
    pub fn nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        self.nodes.iter().copied()
    }

    /// Returns the computed [`IncomingPorts`] of the subgraph.
    #[must_use]
    pub fn incoming_ports(&self) -> &IncomingPorts<PatchNode> {
        &self.inputs
    }

    /// Returns the computed [`OutgoingPorts`] of the subgraph.
    #[must_use]
    pub fn outgoing_ports(&self) -> &OutgoingPorts<PatchNode> {
        &self.outputs
    }
}

#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum InvalidPinnedSubgraph {
    #[error("Invalid subgraph: {0}")]
    InvalidSubgraph(#[from] InvalidSubgraph<PatchNode>),
    #[error("Invalid commit in host: {0:?}")]
    InvalidCommit(CommitId),
    #[error("Wire is not complete: {0:?}")]
    IncompleteWire(PersistentWire),
    #[error("Node is not pinned: {0}")]
    UnpinnedNode(PatchNode),
    #[error("Invalid node in host: {0}")]
    InvalidNode(PatchNode),
}

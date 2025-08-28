//! Methods for traversing wires at commit boundaries

use std::collections::BTreeSet;

use hugr_core::{
    HugrView, IncomingPort, Node, OutgoingPort, Port,
    hugr::patch::{BoundaryPort, simple_replace::BoundaryMode},
};
use itertools::{Either, Itertools};

use crate::{Commit, PatchNode, parents_view::ParentsView};

impl Commit<'_> {
    /// Translate a node in the commit's HUGR into a [`PatchNode`] used for
    /// node indexing within [`PersistentHugr`]s.
    ///
    /// [`PersistentHugr`]: crate::PersistentHugr
    pub fn to_patch_node(&self, node: Node) -> PatchNode {
        PatchNode(self.id(), node)
    }

    /// Whether the wire attached to `(node, port)` in `self` is a boundary edge
    /// into `child`.
    ///
    /// In other words, check if `(node, port)` is outside of the subgraph of
    /// the patch of `child` and at least one opposite node is inside the
    /// subgraph.
    fn has_boundary_edge_into(&self, node: Node, port: impl Into<Port>, child: &Commit) -> bool {
        let deleted_nodes: BTreeSet<_> = child.deleted_parent_nodes().collect();
        if deleted_nodes.contains(&self.to_patch_node(node)) {
            return false;
        }
        let mut opp_nodes = self
            .commit_hugr()
            .linked_ports(node, port)
            .map(|(n, _)| PatchNode(self.id(), n));

        opp_nodes.any(|n| deleted_nodes.contains(&n))
    }

    /// Get the boundary inputs in `child` linked to `(node, port)` in `self`.
    ///
    /// The returned ports will be ports on successors of the input node in the
    /// `child` commit, unless (node, port) is connected to a passthrough wire
    /// in `child` (i.e. a wire from input node to output node), in which
    /// case they will be in one of the parents of `child`.
    ///
    /// `child` should be a child commit of the owner of `node`.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not a boundary edge, if `child` is not
    /// a valid commit ID or if it is the base commit.
    pub(crate) fn linked_child_inputs(
        &self,
        node: Node,
        port: OutgoingPort,
        child: &Commit,
        return_invalid: BoundaryMode,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        assert!(
            self.has_boundary_edge_into(node, port, child),
            "not a boundary edge"
        );

        let parent_hugrs = ParentsView::from_commit(child);
        let repl = child.replacement().expect("valid child commit");
        let child_id = child.id();
        repl.linked_replacement_inputs(
            (self.to_patch_node(node), port),
            &parent_hugrs,
            return_invalid,
        )
        .collect_vec()
        .into_iter()
        .map(move |np| match np {
            BoundaryPort::Host(patch_node, port) => (patch_node, port),
            BoundaryPort::Replacement(node, port) => (PatchNode(child_id, node), port),
        })
    }

    /// Get the single boundary output in `child` linked to `(node, port)` in
    /// `self`.
    ///
    /// The returned port will be a port on a predecessor of the output node in
    /// the `child` commit, unless (node, port) is connected to a passthrough
    /// wire in `child` (i.e. a wire from input node to output node), in
    /// which case it will be in one of the parents of `child`.
    ///
    /// `child` should be a child commit of the owner of `node` (or `None` will
    /// be returned).
    ///
    /// ## Panics
    ///
    /// Panics if `child` is not a valid commit ID.
    pub(crate) fn linked_child_output(
        &self,
        node: Node,
        port: IncomingPort,
        child: &Commit,
        return_invalid: BoundaryMode,
    ) -> Option<(PatchNode, OutgoingPort)> {
        let parent_hugrs = ParentsView::from_commit(child);
        let repl = child.replacement()?;
        match repl.linked_replacement_output(
            (self.to_patch_node(node), port),
            &parent_hugrs,
            return_invalid,
        )? {
            BoundaryPort::Host(patch_node, port) => (patch_node, port),
            BoundaryPort::Replacement(node, port) => (child.to_patch_node(node), port),
        }
        .into()
    }

    /// Get the boundary ports in `child` linked to `(node, port)` in `self`.
    ///
    /// `child` should be a child commit of the owner of `node`.
    ///
    /// See [`Self::linked_child_inputs`] and [`Self::linked_child_output`] for
    /// more details.
    pub(crate) fn linked_child_ports(
        &self,
        node: Node,
        port: impl Into<Port>,
        child: &Commit,
        return_invalid: BoundaryMode,
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

    /// Get the single output port linked to `(node, port)` in a parent of
    /// `self`.
    ///
    /// The returned port belongs to the input boundary of the subgraph in
    /// parent.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not connected to the input node in the
    /// commit of `node`, or if the node is not valid.
    pub fn linked_parent_input(&self, node: Node, port: IncomingPort) -> (PatchNode, OutgoingPort) {
        let repl = self.replacement().expect("valid non-base commit");

        assert!(
            repl.replacement()
                .input_neighbours(node)
                .contains(&repl.get_replacement_io()[0]),
            "not connected to input"
        );

        let parent_hugrs = ParentsView::from_commit(self);
        repl.linked_host_input((node, port), &parent_hugrs).into()
    }

    /// Get the input ports linked to `(node, port)` in a parent of `self`.
    ///
    /// The returned ports belong to the output boundary of the subgraph in
    /// parent.
    ///
    /// ## Panics
    ///
    /// Panics if `(node, port)` is not connected to the output node in the
    /// commit of `node`, or if the node is not valid.
    pub fn linked_parent_outputs(
        &self,
        node: Node,
        port: OutgoingPort,
    ) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        let repl = self.replacement().expect("valid non-base commit");

        assert!(
            repl.replacement()
                .output_neighbours(node)
                .contains(&repl.get_replacement_io()[1]),
            "not connected to output"
        );

        let parent_hugrs = ParentsView::from_commit(self);
        repl.linked_host_outputs((node, port), &parent_hugrs)
            .map_into()
            .collect_vec()
            .into_iter()
    }

    /// Get the ports linked to `(node, port)` in a parent of `self`.
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
        node: Node,
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
}

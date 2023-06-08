//! Wrapper around a portgraph providing multiports via implicit copy nodes

mod iter;

pub use self::iter::{
    Neighbours, NodeConnections, NodeLinks, NodeSubports, Nodes, PortLinks, Ports,
};

use portgraph::{
    portgraph::{NodePortOffsets, NodePorts, PortOperation},
    Direction, LinkError, NodeIndex, PortGraph, PortIndex, PortOffset, SecondaryMap,
};

use bitvec::vec::BitVec;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// An unlabelled port graph that allows multiple links to the same ports.
///
/// A port graph consists of a collection of nodes identified by a [`NodeIndex`].
/// Each node has an ordered sequence of input and output ports, identified by a [`PortIndex`] that is unique within the graph.
/// To optimize for the most common use case, the number of input and output ports of a node must be specified when the node is created.
/// Multiple connections to the same [`PortIndex`] can be distinguished by their [`SubportIndex`].
///
/// When a node and its associated ports are removed their indices will be reused on a best effort basis
/// when a new node is added.
/// The indices of unaffected nodes and ports remain stable.
#[derive(Clone, PartialEq, Default, Debug, Serialize, Deserialize)]
pub struct MultiPortGraph {
    graph: PortGraph,
    /// Flags marking the internal ports of a multiport. That is, the ports connecting the main node and the copy nodes.
    multiport: BitVec,
    /// Flags marking the implicit copy nodes.
    copy_node: BitVec,
    /// Number of implicit copy nodes.
    copy_node_count: usize,
    /// Number of subports in the copy nodes of the graph.
    subport_count: usize,
}

/// Index of a multi port within a `MultiPortGraph`.
///
/// Note that the offsets of the subport indices are not guaranteed to be
/// contiguous nor well-ordered. They are not invalidated by adding or removing
/// other links to the same port.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct SubportIndex {
    port: PortIndex,
    subport_offset: u16,
}

impl MultiPortGraph {
    /// Create a new empty [`MultiPortGraph`].
    pub fn new() -> Self {
        Self {
            graph: PortGraph::new(),
            multiport: BitVec::new(),
            copy_node: BitVec::new(),
            copy_node_count: 0,
            subport_count: 0,
        }
    }

    /// Create a new empty [`MultiPortGraph`] with preallocated capacity.
    pub fn with_capacity(nodes: usize, ports: usize) -> Self {
        Self {
            graph: PortGraph::with_capacity(nodes, ports),
            multiport: BitVec::with_capacity(ports),
            copy_node: BitVec::with_capacity(nodes),
            copy_node_count: 0,
            subport_count: 0,
        }
    }

    /// Returns a reference to the internal plain portgraph.
    ///
    /// This graph exposes the copy nodes as well as the main nodes.
    pub fn as_portgraph(&self) -> &PortGraph {
        // Return the internal graph, exposing the copy nodes
        &self.graph
    }

    /// Adds a node to the portgraph with a given number of input and output ports.
    ///
    /// # Panics
    ///
    /// Panics if the total number of ports exceeds `u16::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr::hugr::multiportgraph::MultiPortGraph;
    /// # use hugr::Direction;
    /// let mut g = MultiPortGraph::new();
    /// let node = g.add_node(4, 3);
    /// assert_eq!(g.inputs(node).count(), 4);
    /// assert_eq!(g.outputs(node).count(), 3);
    /// assert!(g.contains_node(node));
    /// ```
    pub fn add_node(&mut self, incoming: usize, outgoing: usize) -> NodeIndex {
        self.graph.add_node(incoming, outgoing)
    }

    /// Remove a node from the port graph. All ports of the node will be
    /// unlinked and removed as well. Does nothing if the node does not exist.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr::hugr::multiportgraph::MultiPortGraph;
    /// # use hugr::Direction;
    /// let mut g = MultiPortGraph::new();
    /// let node0 = g.add_node(1, 1);
    /// let node1 = g.add_node(1, 1);
    /// g.link_ports(g.outputs(node0).nth(0).unwrap(), g.inputs(node1).nth(0).unwrap());
    /// g.link_ports(g.outputs(node1).nth(0).unwrap(), g.inputs(node0).nth(0).unwrap());
    /// g.remove_node(node0);
    /// assert!(!g.contains_node(node0));
    /// assert!(g.port_link(g.outputs(node1).nth(0).unwrap()).is_none());
    /// assert!(g.port_link(g.inputs(node1).nth(0).unwrap()).is_none());
    /// ```
    pub fn remove_node(&mut self, node: NodeIndex) {
        debug_assert!(!self.copy_node.get(node));
        for port in self.graph.all_ports(node) {
            if *self.multiport.get(port) {
                self.unlink_port(port);
            }
        }
        self.graph.remove_node(node);
    }

    /// Link an output port to an input port.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr::hugr::multiportgraph::MultiPortGraph;
    /// # use hugr::Direction;
    /// let mut g = MultiPortGraph::new();
    /// let node0 = g.add_node(0, 1);
    /// let node1 = g.add_node(1, 0);
    /// let node0_output = g.output(node0, 0).unwrap();
    /// let node1_input = g.input(node1, 0).unwrap();
    /// g.link_ports(node0_output, node1_input).unwrap();
    /// assert_eq!(g.port_link(node0_output).unwrap().port(), node1_input);
    /// assert_eq!(g.port_link(node1_input).unwrap().port(), node0_output);
    /// ```
    ///
    /// # Errors
    ///
    ///  - If `port_a` or `port_b` does not exist.
    ///  - If `port_a` and `port_b` have the same direction.
    pub fn link_ports(
        &mut self,
        port_a: PortIndex,
        port_b: PortIndex,
    ) -> Result<(SubportIndex, SubportIndex), LinkError> {
        let (multiport_a, index_a) = self.get_free_multiport(port_a)?;
        let (multiport_b, index_b) = self.get_free_multiport(port_b)?;
        self.graph.link_ports(index_a, index_b)?;
        Ok((multiport_a, multiport_b))
    }

    /// Link an output subport to an input subport.
    ///
    /// # Errors
    ///
    ///  - If `subport_from` or `subport_to` does not exist.
    ///  - If `subport_a` and `subport_b` have the same direction.
    ///  - If `subport_from` or `subport_to` is already linked.
    pub fn link_subports(
        &mut self,
        subport_from: SubportIndex,
        subport_to: SubportIndex,
    ) -> Result<(), LinkError> {
        // TODO: Custom errors
        let from_index = self
            .get_subport_index(subport_from)
            .expect("subport_from does not exist");
        let to_index = self
            .get_subport_index(subport_to)
            .expect("subport_to does not exist");
        self.graph.link_ports(from_index, to_index)
    }

    /// Unlinks all connections to the `port`. Return `false` if the port was not linked.
    pub fn unlink_port(&mut self, port: PortIndex) -> bool {
        if self.is_multiport(port) {
            self.multiport.set(port, false);
            let link = self
                .graph
                .port_link(port)
                .expect("MultiPortGraph error: a port marked as multiport has no link.");
            let copy_node = self.graph.port_node(link).unwrap();
            self.remove_copy_node(copy_node, link);
            true
        } else {
            self.graph.unlink_port(port).is_some()
        }
    }

    /// Unlinks the `port` and returns the port it was linked to. Returns `None`
    /// when the port was not linked.
    ///
    /// TODO: Remove copy nodes when they are no longer needed?
    pub fn unlink_subport(&mut self, subport: SubportIndex) -> Option<SubportIndex> {
        let subport_index = self.get_subport_index(subport)?;
        let link = self.graph.unlink_port(subport_index)?;
        self.get_subport_from_index(link)
    }

    /// Returns an iterator over every pair of matching ports connecting `from`
    /// with `to`.
    ///
    /// # Example
    /// ```
    /// # use hugr::hugr::multiportgraph::{MultiPortGraph, SubportIndex};
    /// # use portgraph::{NodeIndex, PortIndex, Direction};
    /// # use itertools::Itertools;
    /// let mut g = MultiPortGraph::new();
    /// let a = g.add_node(0, 2);
    /// let b = g.add_node(2, 0);
    ///
    /// g.link_nodes(a, 0, b, 0).unwrap();
    /// g.link_nodes(a, 0, b, 1).unwrap();
    /// g.link_nodes(a, 1, b, 1).unwrap();
    ///
    /// let mut connections = g.get_connections(a, b);
    /// let (out0, out1) = g.outputs(a).collect_tuple().unwrap();
    /// let (in0, in1) = g.inputs(b).collect_tuple().unwrap();
    /// assert_eq!(connections.next().unwrap(), (SubportIndex::new_multi(out0,0), SubportIndex::new_multi(in0,0)));
    /// assert_eq!(connections.next().unwrap(), (SubportIndex::new_multi(out0,1), SubportIndex::new_multi(in1,0)));
    /// assert_eq!(connections.next().unwrap(), (SubportIndex::new_multi(out1,0), SubportIndex::new_multi(in1,1)));
    /// assert_eq!(connections.next(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_connections(&self, from: NodeIndex, to: NodeIndex) -> NodeConnections<'_> {
        NodeConnections::new(self, to, self.output_links(from))
    }

    /// Checks whether there is a directed link between the two nodes and
    /// returns the first matching pair of ports.
    ///
    /// # Example
    /// ```
    /// # use hugr::hugr::multiportgraph::{MultiPortGraph, SubportIndex};
    /// # use portgraph::{NodeIndex, PortIndex, Direction};
    /// let mut g = MultiPortGraph::new();
    /// let a = g.add_node(0, 2);
    /// let b = g.add_node(2, 0);
    ///
    /// g.link_nodes(a, 0, b, 0).unwrap();
    /// g.link_nodes(a, 1, b, 1).unwrap();
    ///
    /// let out0 = g.output(a, 0).unwrap();
    /// let in0 = g.input(b, 0).unwrap();
    /// assert_eq!(g.get_connection(a, b), Some((SubportIndex::new_multi(out0,0), SubportIndex::new_multi(in0,0))));
    /// ```
    #[must_use]
    #[inline]
    pub fn get_connection(
        &self,
        from: NodeIndex,
        to: NodeIndex,
    ) -> Option<(SubportIndex, SubportIndex)> {
        self.get_connections(from, to).next()
    }

    /// Checks whether there is a directed link between the two nodes.
    ///
    /// # Example
    /// ```
    /// # use hugr::hugr::multiportgraph::MultiPortGraph;
    /// # use portgraph::{NodeIndex, PortIndex, Direction};
    /// let mut g = MultiPortGraph::new();
    /// let a = g.add_node(0, 2);
    /// let b = g.add_node(2, 0);
    ///
    /// g.link_nodes(a, 0, b, 0).unwrap();
    ///
    /// assert!(g.connected(a, b));
    /// ```
    #[must_use]
    #[inline]
    pub fn connected(&self, from: NodeIndex, to: NodeIndex) -> bool {
        self.get_connection(from, to).is_some()
    }

    /// Links two nodes at an input and output port offsets.
    pub fn link_nodes(
        &mut self,
        from: NodeIndex,
        from_output: usize,
        to: NodeIndex,
        to_input: usize,
    ) -> Result<(SubportIndex, SubportIndex), LinkError> {
        self.link_offsets(
            from,
            PortOffset::new_outgoing(from_output),
            to,
            PortOffset::new_incoming(to_input),
        )
    }

    /// Links two nodes at an input and output port offsets.
    ///
    /// # Errors
    ///
    ///  - If the ports and nodes do not exist.
    ///  - If the ports have the same direction.
    pub fn link_offsets(
        &mut self,
        node_a: NodeIndex,
        offset_a: PortOffset,
        node_b: NodeIndex,
        offset_b: PortOffset,
    ) -> Result<(SubportIndex, SubportIndex), LinkError> {
        let from_port = self
            .port_index(node_a, offset_a)
            .ok_or(LinkError::UnknownOffset {
                node: node_a,
                offset: offset_a,
            })?;
        let to_port = self
            .port_index(node_b, offset_b)
            .ok_or(LinkError::UnknownOffset {
                node: node_b,
                offset: offset_b,
            })?;
        self.link_ports(from_port, to_port)
    }

    /// Returns the direction of the `port`.
    #[inline]
    pub fn port_direction(&self, port: impl Into<PortIndex>) -> Option<Direction> {
        self.graph.port_direction(port.into())
    }

    /// Returns the node that the `port` belongs to.
    #[inline]
    pub fn port_node(&self, port: impl Into<PortIndex>) -> Option<NodeIndex> {
        self.graph.port_node(port.into())
    }

    /// Returns the index of a `port` within its node's port list.
    pub fn port_offset(&self, port: impl Into<PortIndex>) -> Option<PortOffset> {
        self.graph.port_offset(port.into())
    }

    /// Returns the port index for a given node, direction, and offset.
    #[must_use]
    pub fn port_index(&self, node: NodeIndex, offset: PortOffset) -> Option<PortIndex> {
        self.graph.port_index(node, offset)
    }

    /// Returns the port that the given `port` is linked to.
    #[inline]
    pub fn port_links(&self, port: PortIndex) -> PortLinks {
        PortLinks::new(self, port)
    }

    /// Return the link to the provided port, if not connected return None.
    /// If this port has multiple connected subports, an arbitrary one is returned.
    #[inline]
    pub fn port_link(&self, port: PortIndex) -> Option<SubportIndex> {
        self.port_links(port).next().map(|(_, p)| p)
    }

    /// Return the subport linked to the given `port`. If the port is not
    /// connected, return None.
    pub fn subport_link(&self, subport: SubportIndex) -> Option<SubportIndex> {
        let subport_index = self.get_subport_index(subport)?;
        let link = self.graph.port_link(subport_index)?;
        self.get_subport_from_index(link)
    }

    /// Iterates over all the ports of the `node` in the given `direction`.
    pub fn ports(&self, node: NodeIndex, direction: Direction) -> NodePorts {
        self.graph.ports(node, direction)
    }

    /// Iterates over the input and output ports of the `node` in sequence.
    pub fn all_ports(&self, node: NodeIndex) -> NodePorts {
        self.graph.all_ports(node)
    }

    /// Returns the input port at the given offset in the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::port_index`].
    #[inline]
    pub fn input(&self, node: NodeIndex, offset: usize) -> Option<PortIndex> {
        self.graph.input(node, offset)
    }

    /// Returns the output port at the given offset in the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::ports`].
    #[inline]
    pub fn output(&self, node: NodeIndex, offset: usize) -> Option<PortIndex> {
        self.graph.output(node, offset)
    }

    /// Iterates over all the input ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::ports`].
    #[inline]
    pub fn inputs(&self, node: NodeIndex) -> NodePorts {
        self.graph.inputs(node)
    }

    /// Iterates over all the output ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::ports`].
    #[inline]
    pub fn outputs(&self, node: NodeIndex) -> NodePorts {
        self.graph.outputs(node)
    }

    /// Returns the number of input ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::num_ports`].
    #[inline]
    pub fn num_inputs(&self, node: NodeIndex) -> usize {
        self.graph.num_inputs(node)
    }

    /// Returns the number of output ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::num_ports`].
    #[inline]
    pub fn num_outputs(&self, node: NodeIndex) -> usize {
        self.graph.num_outputs(node)
    }

    /// Returns the number of ports of the `node` in the given `direction`.
    #[inline]
    pub fn num_ports(&self, node: NodeIndex, direction: Direction) -> usize {
        self.graph.num_ports(node, direction)
    }

    /// Iterates over all the ports of the `node` in the given `direction`.
    pub fn subports(&self, node: NodeIndex, direction: Direction) -> NodeSubports {
        NodeSubports::new(self, self.graph.ports(node, direction))
    }

    /// Iterates over the input and output ports of the `node` in sequence.
    pub fn all_subports(&self, node: NodeIndex) -> NodeSubports {
        NodeSubports::new(self, self.graph.all_ports(node))
    }

    /// Iterates over all the input ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::subports`].
    #[inline]
    pub fn subport_inputs(&self, node: NodeIndex) -> NodeSubports {
        self.subports(node, Direction::Incoming)
    }

    /// Iterates over all the output ports of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::subports`].
    #[inline]
    pub fn subport_outputs(&self, node: NodeIndex) -> NodeSubports {
        self.subports(node, Direction::Outgoing)
    }

    /// Iterates over all the port offsets of the `node` in the given `direction`.
    pub fn port_offsets(&self, node: NodeIndex, direction: Direction) -> NodePortOffsets {
        self.graph.port_offsets(node, direction)
    }

    /// Iterates over all the input port offsets of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::port_offsets`].
    #[inline]
    pub fn input_offsets(&self, node: NodeIndex) -> NodePortOffsets {
        self.graph.input_offsets(node)
    }

    /// Iterates over all the output port offsets of the `node`.
    ///
    /// Shorthand for [`MultiPortGraph::port_offsets`].
    #[inline]
    pub fn output_offsets(&self, node: NodeIndex) -> NodePortOffsets {
        self.graph.output_offsets(node)
    }

    /// Iterates over the input and output port offsets of the `node` in sequence.
    #[inline]
    pub fn all_port_offsets(&self, node: NodeIndex) -> NodePortOffsets {
        self.graph.all_port_offsets(node)
    }

    /// Iterates over the connected links of the `node` in the given
    /// `direction`.
    ///
    /// In contrast to [`PortGraph::links`], this iterator only returns linked
    /// subports, and includes the source subport.
    ///
    /// [`PortGraph::links`]: portgraph::PortGraph::links
    #[inline]
    pub fn links(&self, node: NodeIndex, direction: Direction) -> NodeLinks<'_> {
        NodeLinks::new(self, self.ports(node, direction))
    }

    /// Iterates over the connected input links of the `node`. Shorthand for
    /// [`MultiPortGraph::links`].
    #[inline]
    pub fn input_links(&self, node: NodeIndex) -> NodeLinks<'_> {
        self.links(node, Direction::Incoming)
    }

    /// Iterates over the connected output links of the `node`. Shorthand for
    /// [`MultiPortGraph::links`].
    #[inline]
    pub fn output_links(&self, node: NodeIndex) -> NodeLinks<'_> {
        self.links(node, Direction::Outgoing)
    }

    /// Iterates over the connected input and output links of the `node` in sequence.
    #[inline]
    pub fn all_links(&self, node: NodeIndex) -> NodeLinks<'_> {
        NodeLinks::new(self, self.all_ports(node))
    }

    /// Iterates over neighbour nodes in the given `direction`.
    /// May contain duplicates if the graph has multiple links between nodes.
    #[inline]
    pub fn neighbours(&self, node: NodeIndex, direction: Direction) -> Neighbours<'_> {
        Neighbours::new(self, self.subports(node, direction))
    }

    /// Iterates over the input neighbours of the `node`. Shorthand for [`MultiPortGraph::neighbours`].
    #[inline]
    pub fn input_neighbours(&self, node: NodeIndex) -> Neighbours<'_> {
        self.neighbours(node, Direction::Incoming)
    }

    /// Iterates over the output neighbours of the `node`. Shorthand for [`MultiPortGraph::neighbours`].
    #[inline]
    pub fn output_neighbours(&self, node: NodeIndex) -> Neighbours<'_> {
        self.neighbours(node, Direction::Outgoing)
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    #[inline]
    pub fn all_neighbours(&self, node: NodeIndex) -> Neighbours<'_> {
        Neighbours::new(self, self.all_subports(node))
    }

    /// Returns whether the port graph contains the `node`.
    #[inline]
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        self.graph.contains_node(node) && !self.copy_node.get(node)
    }

    /// Returns whether the port graph contains the `port`.
    #[inline]
    pub fn contains_port(&self, port: PortIndex) -> bool {
        self.graph
            .port_node(port)
            .map_or(false, |node| !self.copy_node.get(node))
    }

    /// Returns whether the port graph has no nodes nor ports.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count() == 0
    }

    /// Returns the number of nodes in the port graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.graph.node_count() - self.copy_node_count
    }

    /// Returns the number of ports in the port graph.
    #[inline]
    pub fn port_count(&self) -> usize {
        // Do not count the ports in the copy nodes. We have to subtract one of
        // the two ports connecting the copy nodes with their main nodes, in
        // addition to all the subports.
        self.graph.port_count() - self.subport_count - self.copy_node_count
    }

    /// Returns the number of links between ports.
    #[inline]
    pub fn link_count(&self) -> usize {
        // Do not count the links between copy nodes and their main nodes.
        self.graph.link_count() - self.copy_node_count
    }

    /// Iterates over the nodes in the port graph.
    #[inline]
    pub fn nodes_iter(&self) -> Nodes<'_> {
        self::iter::Nodes {
            multigraph: self,
            iter: self.graph.nodes_iter(),
            len: self.node_count(),
        }
    }

    /// Iterates over the ports in the port graph.
    #[inline]
    pub fn ports_iter(&self) -> Ports<'_> {
        Ports::new(self, self.graph.ports_iter())
    }

    /// Removes all nodes and ports from the port graph.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.multiport.clear();
        self.copy_node.clear();
        self.copy_node_count = 0;
        self.subport_count = 0;
    }

    /// Returns the capacity of the underlying buffer for nodes.
    #[inline]
    pub fn node_capacity(&self) -> usize {
        self.graph.node_capacity() - self.copy_node_count
    }

    /// Returns the capacity of the underlying buffer for ports.
    #[inline]
    pub fn port_capacity(&self) -> usize {
        // See [`MultiPortGraph::port_count`]
        self.graph.port_capacity() - self.subport_count - self.copy_node_count
    }

    /// Returns the allocated port capacity for a specific node.
    ///
    /// Changes to the number of ports of the node will not reallocate
    /// until the number of ports exceeds this capacity.
    #[inline]
    pub fn node_port_capacity(&self, node: NodeIndex) -> usize {
        self.graph.node_port_capacity(node)
    }

    /// Reserves enough capacity to insert at least the given number of additional nodes and ports.
    ///
    /// This method does not take into account the length of the free list and might overallocate speculatively.
    pub fn reserve(&mut self, nodes: usize, ports: usize) {
        self.graph.reserve(nodes, ports);
        self.multiport.reserve(ports);
        self.copy_node.reserve(nodes);
    }

    /// Changes the number of ports of the `node` to the given `incoming` and `outgoing` counts.
    ///
    /// Invalidates the indices of the node's ports. If the number of incoming or outgoing ports
    /// is reduced, the ports are removed from the end of the port list.
    ///
    /// Every time a port is moved, the `rekey` function will be called with its old and new index.
    /// If the port is removed, the new index will be `None`.
    ///
    /// This operation is O(n) where n is the number of ports of the node.
    #[allow(unreachable_code)] // TODO
    pub fn set_num_ports<F>(
        &mut self,
        node: NodeIndex,
        incoming: usize,
        outgoing: usize,
        mut rekey: F,
    ) where
        F: FnMut(PortIndex, PortOperation),
    {
        let mut dropped_ports = Vec::new();
        let rekey_wrapper = |port, op| {
            if let PortOperation::Removed { old_link } = op {
                dropped_ports.push((port, old_link))
            }
            rekey(port, op);
        };
        self.graph
            .set_num_ports(node, incoming, outgoing, rekey_wrapper);
        for (port, old_link) in dropped_ports {
            if self.is_multiport(port) {
                self.multiport.set(port, false);
                let link = old_link.expect("Multiport node has no link");
                let copy_node = self.graph.port_node(link).unwrap();
                self.remove_copy_node(copy_node, link)
            }
        }
    }

    /// Compacts the storage of nodes in the portgraph. Note that indices won't
    /// necessarily be consecutively after this process, as there may be
    /// hidden copy nodes.
    ///
    /// Every time a node is moved, the `rekey` function will be called with its
    /// old and new index.
    pub fn compact_nodes<F>(&mut self, mut rekey: F)
    where
        F: FnMut(NodeIndex, NodeIndex),
    {
        self.graph.compact_nodes(|node, new_node| {
            self.copy_node.swap(node, new_node);
            rekey(node, new_node);
        });
    }

    /// Compacts the storage of ports in the portgraph. Note that indices won't
    /// necessarily be consecutively after this process, as there may be hidden
    /// copy nodes.
    ///
    /// Every time a port is moved, the `rekey` function will be called with is
    /// old and new index.
    pub fn compact_ports<F>(&mut self, mut rekey: F)
    where
        F: FnMut(PortIndex, PortIndex),
    {
        self.graph.compact_ports(|port, new_port| {
            self.multiport.swap(port, new_port);
            rekey(port, new_port);
        });
    }

    /// Shrinks the underlying buffers to the fit the data.
    ///
    /// This does not move nodes or ports, which might prevent freeing up more capacity.
    pub fn shrink_to_fit(&mut self) {
        self.graph.shrink_to_fit();
        self.multiport.shrink_to_fit();
        self.copy_node.shrink_to_fit();
    }
}

/// Internal helper methods
impl MultiPortGraph {
    /// Remove an internal copy node.
    fn remove_copy_node(&mut self, copy_node: NodeIndex, from: PortIndex) {
        let dir = self.port_direction(from).unwrap();
        debug_assert!(self.copy_node.get(copy_node));
        let mut subports = self.graph.ports(copy_node, dir.reverse());
        self.multiport.set(from, false);
        self.copy_node.set(copy_node, false);
        self.graph.remove_node(copy_node);
        self.copy_node_count -= 1;
        self.subport_count -= subports.len();
        debug_assert!(subports.all(|port| !self.multiport.get(port.index())));
    }

    /// Returns a free multiport for the given port, along with its
    /// portgraph-level port index. Allocates a new copy node if needed, and
    /// grows the number of copy ports as needed.
    fn get_free_multiport(
        &mut self,
        port: PortIndex,
    ) -> Result<(SubportIndex, PortIndex), LinkError> {
        let Some(dir) = self.graph.port_direction(port) else {
            return Err(LinkError::UnknownPort{port});
        };
        let multiport = *self.multiport.get(port.index());
        let link = self.graph.port_link(port);
        match (multiport, link) {
            (false, None) => {
                // Port is disconnected, no need to allocate a copy node.
                Ok((SubportIndex::new_unique(port), port))
            }
            (false, Some(link)) => {
                // Port is connected, allocate a copy node.
                let in_out_count = match dir {
                    Direction::Incoming => (2, 1),
                    Direction::Outgoing => (1, 2),
                };
                let copy_node = self.graph.add_node(in_out_count.0, in_out_count.1);
                self.copy_node.set(copy_node, true);
                self.copy_node_count += 1;
                self.subport_count += 2;

                let copy_port = self.graph.ports(copy_node, dir.reverse()).next().unwrap();
                let (old_link, subport) = self.graph.ports(copy_node, dir).collect_tuple().unwrap();

                // Connect the copy node to the original node, and re-connect the old link.
                self.graph.unlink_port(port);
                self.link_ports_directed(port, copy_port, dir)?;
                self.link_ports_directed(old_link, link, dir)?;
                self.multiport.set(copy_port.index(), true);
                self.multiport.set(port.index(), true);

                let subport_offset = 1;
                Ok((SubportIndex::new_multi(port, subport_offset), subport))
            }
            (true, Some(link)) => {
                // Port is already connected to a copy node.
                let copy_node = self.graph.port_node(link).unwrap();
                // We try to reuse an existing disconnected subport, if any.
                for (subport_offset, subport) in self.graph.ports(copy_node, dir).enumerate() {
                    if self.graph.port_link(subport).is_none() {
                        return Ok((SubportIndex::new_multi(port, subport_offset), subport));
                    }
                }
                // No free subport, we need to allocate a new one.
                let subport_offset = self.graph.num_ports(copy_node, dir);
                let subport = self.add_port(copy_node, dir);
                self.subport_count += 1;
                Ok((SubportIndex::new_multi(port, subport_offset), subport))
            }
            (true, None) => {
                // Missing copy node
                // TODO: Write a new error for this
                panic!("Missing copy node")
            }
        }
    }

    /// Adds an extra port to a node, in the specified direction.
    #[inline]
    fn add_port(&mut self, node: NodeIndex, direction: Direction) -> PortIndex {
        let mut incoming = self.graph.num_inputs(node);
        let mut outgoing = self.graph.num_outputs(node);
        let new_offset = match direction {
            Direction::Incoming => {
                incoming += 1;
                incoming - 1
            }
            Direction::Outgoing => {
                outgoing += 1;
                outgoing - 1
            }
        };
        self.graph
            .set_num_ports(node, incoming, outgoing, |_, _| {});
        self.graph
            .port_index(node, PortOffset::new(direction, new_offset))
            .unwrap()
    }

    /// Link two ports, using the direction of `port1` to determine the link.
    ///
    /// Avoids the `UnexpectedDirection` error when passing the ports in the wrong order.
    #[inline]
    fn link_ports_directed(
        &mut self,
        port1: PortIndex,
        port2: PortIndex,
        dir: Direction,
    ) -> Result<(), LinkError> {
        match dir {
            Direction::Incoming => self.graph.link_ports(port2, port1),
            Direction::Outgoing => self.graph.link_ports(port1, port2),
        }
    }

    /// Returns the PortIndex from the main node that connects to this copy node.
    fn copy_node_main_port(&self, copy_node: NodeIndex) -> Option<PortIndex> {
        debug_assert!(self.copy_node.get(copy_node));
        let mut incoming = self.graph.inputs(copy_node);
        let mut outgoing = self.graph.outputs(copy_node);

        let internal_copy_port = match (incoming.len(), outgoing.len()) {
            (1, 1) => {
                // Copy node has one input and one output, we have to check the
                // `multiport` flag to determine on which direction is the main
                // node.
                let in_port = incoming.next().unwrap();
                let out_port = outgoing.next().unwrap();
                match self.multiport.get(in_port) {
                    true => in_port,
                    false => out_port,
                }
            }
            (1, _) => {
                // This is a copy node for an outgoing port.
                incoming.next().unwrap()
            }
            (_, 1) => {
                // This is a copy node for an incoming port.
                outgoing.next().unwrap()
            }
            _ => {
                // TODO: MultiGraph error
                panic!("A copy must have a single port connecting it to the main node. The node had {} inputs and {} outputs", incoming.len(), outgoing.len())
            }
        };
        self.graph.port_link(internal_copy_port)
    }

    /// Returns whether the port is marked as multiport.
    ///
    /// That is, this port is part of the connection between a main port and a copy node.
    #[inline]
    fn is_multiport(&self, port: PortIndex) -> bool {
        *self.multiport.get(port)
    }

    /// Returns whether the node is a copy node.
    #[inline]
    fn is_copy_node(&self, node: NodeIndex) -> bool {
        *self.copy_node.get(node)
    }

    /// Get the copy node for a multiport PortIndex, if it exists.
    #[inline]
    fn get_copy_node(&self, port_index: PortIndex) -> Option<NodeIndex> {
        let link = self.graph.port_link(port_index)?;
        self.graph.port_node(link)
    }

    /// Get the `PortIndex` in the copy node for a SubportIndex.
    ///
    /// If the port is not a multiport, returns the port index in the operation node.
    fn get_subport_index(&self, subport: SubportIndex) -> Option<PortIndex> {
        let port_index = subport.port();
        if self.is_multiport(port_index) {
            let copy_node = self.get_copy_node(port_index)?;
            let dir = self.graph.port_direction(port_index)?;
            let subport_offset = portgraph::PortOffset::new(dir, subport.offset());
            self.graph.port_index(copy_node, subport_offset)
        } else {
            Some(port_index)
        }
    }

    /// Checks if the given `PortIndex` corresponds to a subport, and computes the correct `SubportIndex`.
    /// This should be the inverse of `get_subport_index`.
    fn get_subport_from_index(&self, index: PortIndex) -> Option<SubportIndex> {
        let linked_node = self.graph.port_node(index).unwrap();
        if self.is_copy_node(linked_node) {
            let port = self.copy_node_main_port(linked_node)?;
            let link_offset = self.graph.port_offset(index)?;
            Some(SubportIndex::new_multi(port, link_offset.index()))
        } else {
            Some(SubportIndex::new_unique(index))
        }
    }
}

impl From<PortGraph> for MultiPortGraph {
    fn from(graph: PortGraph) -> Self {
        let node_count = graph.node_count();
        let port_count = graph.port_count();
        Self {
            graph,
            multiport: BitVec::with_capacity(port_count),
            copy_node: BitVec::with_capacity(node_count),
            copy_node_count: 0,
            subport_count: 0,
        }
    }
}

impl From<MultiPortGraph> for PortGraph {
    fn from(multi: MultiPortGraph) -> Self {
        // Return the internal graph, exposing the copy nodes
        multi.graph
    }
}

impl SubportIndex {
    /// Creates a new multiport index for a port without a copy node.
    #[inline]
    pub fn new_unique(port: PortIndex) -> Self {
        Self {
            port,
            subport_offset: 0,
        }
    }

    /// Creates a new multiport index.
    ///
    /// # Panics
    ///
    /// If the subport index is more than 2^16.
    #[inline]
    pub fn new_multi(port: PortIndex, subport_offset: usize) -> Self {
        assert!(
            subport_offset < u16::MAX as usize,
            "Subport index too large"
        );
        Self {
            port,
            subport_offset: subport_offset as u16,
        }
    }

    /// Returns the port index.
    #[inline]
    pub fn port(self) -> PortIndex {
        self.port
    }

    /// Returns the offset of the subport.
    ///
    /// If the port is not a multiport, this will always return 0.
    #[inline]
    pub fn offset(self) -> usize {
        self.subport_offset as usize
    }
}

impl From<SubportIndex> for PortIndex {
    fn from(index: SubportIndex) -> Self {
        PortIndex::new(index.port.index())
    }
}

impl std::fmt::Debug for SubportIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SubportIndex({}:{})", self.port.index(), self.offset())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn create_graph() {
        let graph = MultiPortGraph::new();

        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.port_count(), 0);
        assert_eq!(graph.link_count(), 0);
        assert_eq!(graph.nodes_iter().count(), 0);
        assert_eq!(graph.ports_iter().count(), 0);
    }

    #[test]
    fn link_ports() {
        let mut g = MultiPortGraph::new();
        let node0 = g.add_node(2, 1);
        let node1 = g.add_node(1, 2);
        let node0_output = g.output(node0, 0).unwrap();
        let node1_input = g.input(node1, 0).unwrap();
        assert_eq!(g.link_count(), 0);
        assert!(!g.connected(node0, node1));
        assert!(!g.connected(node1, node0));
        assert_eq!(g.get_connections(node0, node1).count(), 0);
        assert_eq!(g.get_connection(node0, node1), None);

        // Link the same ports thrice
        let (from0, to0) = g.link_ports(node0_output, node1_input).unwrap();
        let (from1, to1) = g.link_ports(node0_output, node1_input).unwrap();
        let (from2, to2) = g.link_ports(node0_output, node1_input).unwrap();
        assert_eq!(from0, SubportIndex::new_multi(node0_output, 0));
        assert_eq!(from1, SubportIndex::new_multi(node0_output, 1));
        assert_eq!(from2, SubportIndex::new_multi(node0_output, 2));
        assert_eq!(to0, SubportIndex::new_multi(node1_input, 0));
        assert_eq!(to1, SubportIndex::new_multi(node1_input, 1));
        assert_eq!(to2, SubportIndex::new_multi(node1_input, 2));
        assert_eq!(g.link_count(), 3);
        assert_eq!(g.subport_link(from0), Some(to0));
        assert_eq!(g.subport_link(to1), Some(from1));
        assert_eq!(
            g.port_links(node0_output).collect_vec(),
            vec![(from0, to0), (from1, to1), (from2, to2)]
        );
        assert_eq!(
            g.get_connections(node0, node1).collect_vec(),
            vec![(from0, to0), (from1, to1), (from2, to2)]
        );
        assert_eq!(g.get_connection(node0, node1), Some((from0, to0)));
        assert!(g.connected(node0, node1));
        assert!(!g.connected(node1, node0));

        let unlinked_to0 = g.unlink_subport(from0).unwrap();
        assert_eq!(unlinked_to0, to0);
        assert_eq!(g.link_count(), 2);
        assert_eq!(
            g.get_connections(node0, node1).collect_vec(),
            vec![(from1, to1), (from2, to2)]
        );
        assert_eq!(g.get_connection(node0, node1), Some((from1, to1)));
        assert!(g.connected(node0, node1));
    }

    #[test]
    fn link_iterators() {
        let mut g = MultiPortGraph::new();
        let node0 = g.add_node(1, 2);
        let node1 = g.add_node(2, 1);
        let (node0_output0, node0_output1) = g.outputs(node0).collect_tuple().unwrap();
        let (node1_input0, node1_input1) = g.inputs(node1).collect_tuple().unwrap();

        assert!(g.input_links(node0).eq([]));
        assert!(g.output_links(node0).eq([]));
        assert!(g.all_links(node0).eq([]));
        assert!(g.input_neighbours(node0).eq([]));
        assert!(g.output_neighbours(node0).eq([]));
        assert!(g.all_neighbours(node0).eq([]));

        g.link_nodes(node0, 0, node1, 0).unwrap();
        g.link_nodes(node0, 0, node1, 1).unwrap();
        g.link_nodes(node0, 1, node1, 1).unwrap();

        assert_eq!(
            g.subport_outputs(node0).collect_vec(),
            [
                SubportIndex::new_multi(node0_output0, 0),
                SubportIndex::new_multi(node0_output0, 1),
                SubportIndex::new_unique(node0_output1),
            ]
        );
        assert_eq!(
            g.subport_inputs(node1).collect_vec(),
            [
                SubportIndex::new_unique(node1_input0),
                SubportIndex::new_multi(node1_input1, 0),
                SubportIndex::new_multi(node1_input1, 1),
            ]
        );

        let links = [
            (
                SubportIndex::new_multi(node0_output0, 0),
                SubportIndex::new_unique(node1_input0),
            ),
            (
                SubportIndex::new_multi(node0_output0, 1),
                SubportIndex::new_multi(node1_input1, 0),
            ),
            (
                SubportIndex::new_unique(node0_output1),
                SubportIndex::new_multi(node1_input1, 1),
            ),
        ];
        assert_eq!(g.input_links(node0).collect_vec(), []);
        assert_eq!(g.output_links(node0).collect_vec(), links);
        assert_eq!(g.all_links(node0).collect_vec(), links);
        assert_eq!(g.input_neighbours(node0).collect_vec(), []);
        assert_eq!(
            g.output_neighbours(node0).collect_vec(),
            [node1, node1, node1]
        );
        assert_eq!(g.all_neighbours(node0).collect_vec(), [node1, node1, node1]);
        assert_eq!(g.port_links(node0_output0).collect_vec(), links[0..2]);
    }
}

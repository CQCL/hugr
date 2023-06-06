//! The Hugr data structure, and its basic component handles.

mod hugrmut;

pub mod multiportgraph;
pub mod serialize;
pub mod validate;
pub mod view;

use std::collections::HashMap;

pub use self::hugrmut::HugrMut;
pub use self::validate::ValidationError;

use derive_more::From;
use itertools::Itertools;
use portgraph::dot::{hier_graph_dot_string_with, DotEdgeStyle};
use portgraph::{Hierarchy, NodeIndex, PortGraph, UnmanagedDenseMap};
use thiserror::Error;

pub use self::view::HugrView;
use crate::ops::tag::OpTag;
use crate::ops::{ModuleOp, OpType};
use crate::replacement::{SimpleReplacement, SimpleReplacementError};
use crate::rewrite::{Rewrite, RewriteError};
use crate::types::EdgeKind;

use html_escape::encode_text_to_string;

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    graph: PortGraph,

    /// The node hierarchy.
    hierarchy: Hierarchy,

    /// The single root node in the hierarchy.
    /// It must correspond to a [`ModuleOp::Root`] node.
    ///
    /// [`ModuleOp::Root`]: crate::ops::ModuleOp::Root.
    root: portgraph::NodeIndex,

    /// Operation types for each node.
    op_types: UnmanagedDenseMap<portgraph::NodeIndex, OpType>,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new(ModuleOp::Root)
    }
}

/// A handle to a node in the HUGR.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, From)]
pub struct Node {
    index: portgraph::NodeIndex,
}

/// A handle to a port for a node in the HUGR.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug, From)]
pub struct Port {
    offset: portgraph::PortOffset,
}

/// The direction of a port.
pub type Direction = portgraph::Direction;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

/// Public API for HUGRs.
impl Hugr {
    /// Returns an immutable view over the graph.
    pub fn view(&self) {
        unimplemented!()
    }

    /// Apply a simple replacement operation to the HUGR.
    pub fn apply_simple_replacement(
        &mut self,
        r: SimpleReplacement,
    ) -> Result<(), SimpleReplacementError> {
        // 1. Check the parent node exists.
        if !self.graph.contains_node(r.region.index)
            || self.get_optype(r.region).tag() != OpTag::Dfg
        {
            return Err(SimpleReplacementError::InvalidParentNode());
        }
        // 2. Check that all the to-be-removed nodes are children of it and are leaves.
        for node in &r.removal {
            if self.hierarchy.is_root(node.index)
                || self.hierarchy.parent(node.index).unwrap() != r.region.index
                || self.hierarchy.has_children(node.index)
            {
                return Err(SimpleReplacementError::InvalidRemovedNode());
            }
        }
        // 3. Do the replacement.
        // First locate the DFG in r.n. TODO this won't be necessary when we have DFG-rooted HUGRs.
        let n_dfg_node = r
            .replacement
            .nodes()
            .find(|node: &Node| r.replacement.get_optype(*node).tag() == OpTag::Dfg)
            .unwrap();
        // 3.1. Add copies of all children of n_dfg_node to self. Exclude Input/Output nodes.
        // Create map from old NodeIndex (in r.n) to new NodeIndex (in self).
        let mut index_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let replacement_nodes = r
            .replacement
            .hierarchy
            .children(n_dfg_node.index)
            .map_into::<Node>()
            .collect::<Vec<Node>>();
        let replacement_sz = replacement_nodes.len(); // number of replacement nodes including Input and Output
        let replacement_inner_nodes = &replacement_nodes[1..replacement_sz - 1]; // omit Input and Output
        for &node in replacement_inner_nodes {
            // 3.1.1. Check there are no const inputs.
            if !r
                .replacement
                .get_optype(node)
                .signature()
                .const_input
                .is_empty()
            {
                return Err(SimpleReplacementError::InvalidReplacementNode());
            }
        }
        let self_input_node_index = self.hierarchy.first(r.region.index).unwrap();
        let replacement_output_node = replacement_nodes[replacement_sz - 1];
        for &node in replacement_inner_nodes {
            // 3.1.2. Add the nodes.
            let op: &OpType = r.replacement.get_optype(node);
            let sig = op.signature();
            let new_node_index = self.graph.add_node(sig.input.len(), sig.output.len());
            self.op_types[new_node_index] = op.clone();
            // Make r.p the parent
            self.hierarchy
                .insert_after(new_node_index, self_input_node_index)
                .ok();
            index_map.insert(node.index, new_node_index);
        }
        // 3.2. Add edges between all newly added nodes matching those in n_dfg_node.
        // TODO This will probably change when implicit copies are implemented.
        for &node in replacement_inner_nodes {
            let new_node_index = index_map.get(&node.index).unwrap();
            for node_successor in r.replacement.output_neighbours(node) {
                if r.replacement.get_optype(node_successor).tag() != OpTag::Output {
                    let new_node_successor_index = index_map.get(&node_successor.index).unwrap();
                    for connection in r
                        .replacement
                        .graph
                        .get_connections(node.index, node_successor.index)
                    {
                        let src_offset = r
                            .replacement
                            .graph
                            .port_offset(connection.0)
                            .unwrap()
                            .index();
                        let tgt_offset = r
                            .replacement
                            .graph
                            .port_offset(connection.1)
                            .unwrap()
                            .index();
                        self.graph
                            .link_nodes(
                                *new_node_index,
                                src_offset,
                                *new_node_successor_index,
                                tgt_offset,
                            )
                            .ok();
                    }
                }
            }
        }
        // 3.3. For each p in inp(n_dfg_node), add an edge from the predecessor of r.nu_inp[p] to (new copy of) p.
        for ((rep_inp_node, rep_inp_port), (rem_inp_node, rem_inp_port)) in r.nu_inp {
            let new_inp_node_index = index_map.get(&rep_inp_node.index).unwrap();
            // add edge from predecessor of (s_inp_node, s_inp_port) to (new_inp_node, n_inp_port)
            let rem_inp_port_index = self
                .graph
                .port_index(rem_inp_node.index, rem_inp_port.offset)
                .unwrap();
            let rem_inp_predecessor_port_index = self.graph.port_link(rem_inp_port_index).unwrap();
            let new_inp_port_index = self
                .graph
                .port_index(*new_inp_node_index, rep_inp_port.offset)
                .unwrap();
            self.graph.unlink_port(rem_inp_predecessor_port_index);
            self.graph
                .link_ports(rem_inp_predecessor_port_index, new_inp_port_index)
                .ok();
        }
        // 3.4. For each p in out(r.s), add an edge from (new copy of) the predecessor of r.nu_out[p] to p.
        for ((rem_out_node, rem_out_port), rep_out_port) in r.nu_out {
            let rem_out_port_index = self
                .graph
                .port_index(rem_out_node.index, rem_out_port.offset)
                .unwrap();
            let rep_out_port_index = r
                .replacement
                .graph
                .port_index(replacement_output_node.index, rep_out_port.offset)
                .unwrap();
            let rep_out_predecessor_port_index =
                r.replacement.graph.port_link(rep_out_port_index).unwrap();
            let rep_out_predecessor_node_index = r
                .replacement
                .graph
                .port_node(rep_out_predecessor_port_index)
                .unwrap();
            let rep_out_predecessor_port_offset = r
                .replacement
                .graph
                .port_offset(rep_out_predecessor_port_index)
                .unwrap();
            let new_out_node_index = index_map.get(&rep_out_predecessor_node_index).unwrap();
            let new_out_port_index = self
                .graph
                .port_index(*new_out_node_index, rep_out_predecessor_port_offset)
                .unwrap();
            self.graph.unlink_port(rem_out_port_index);
            self.graph
                .link_ports(new_out_port_index, rem_out_port_index)
                .ok();
        }
        // 3.5. Remove all nodes in r.s and edges between them.
        for node in &r.removal {
            self.graph.remove_node(node.index);
            self.hierarchy.remove(node.index);
        }
        Ok(())
    }

    /// Applies a rewrite to the graph.
    pub fn apply_rewrite(mut self, rewrite: Rewrite) -> Result<(), RewriteError> {
        // Get the open graph for the rewrites, and a HUGR with the additional components.
        let (rewrite, mut replacement, parents) = rewrite.into_parts();

        // TODO: Use `parents` to update the hierarchy, and keep the internal hierarchy from `replacement`.
        let _ = parents;

        let node_inserted = |old, new| {
            std::mem::swap(&mut self.op_types[new], &mut replacement.op_types[old]);
            // TODO: metadata (Fn parameter ?)
        };
        rewrite.apply_with_callbacks(
            &mut self.graph,
            |_| {},
            |_| {},
            node_inserted,
            |_, _| {},
            |_, _| {},
        )?;

        // TODO: Check types

        Ok(())
    }

    /// Return dot string showing underlying graph and hierarchy side by side.
    pub fn dot_string(&self) -> String {
        hier_graph_dot_string_with(
            &self.graph,
            &self.hierarchy,
            |n| {
                format!(
                    "({ni}) {name}",
                    name = self.op_types[n].name(),
                    ni = n.index()
                )
            },
            |p| {
                let src = self.graph.port_node(p).unwrap();
                let Some(tgt_port) = self.graph.port_link(p) else {
                        return ("".into(), DotEdgeStyle::None);
                    };
                let tgt = self.graph.port_node(tgt_port).unwrap();
                let style = if self.hierarchy.parent(src) != self.hierarchy.parent(tgt) {
                    DotEdgeStyle::Some("dashed".into())
                } else if self
                    .get_optype(src.into())
                    .port_kind(self.graph.port_offset(p).unwrap())
                    == Some(EdgeKind::StateOrder)
                {
                    DotEdgeStyle::Some("dotted".into())
                } else {
                    DotEdgeStyle::None
                };

                let optype = self.op_types.get(src);
                let mut label = String::new();
                let offset = self.graph.port_offset(p).unwrap();
                let type_string = match optype.port_kind(offset) {
                    Some(EdgeKind::Const(ty)) => format!("{}", ty),
                    Some(EdgeKind::Value(ty)) => format!("{}", ty),
                    _ => String::new(),
                };
                encode_text_to_string(type_string, &mut label);

                (label, style)
            },
        )
    }
}

/// Internal API for HUGRs, not intended for use by users.
impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new(root_op: impl Into<OpType>) -> Self {
        Self::with_capacity(root_op, 0, 0)
    }

    /// Create a new Hugr, with a single root node and preallocated capacity.
    pub(crate) fn with_capacity(root_op: impl Into<OpType>, nodes: usize, ports: usize) -> Self {
        let mut graph = PortGraph::with_capacity(nodes, ports);
        let hierarchy = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::with_capacity(nodes);
        let root = graph.add_node(0, 0);
        op_types[root] = root_op.into();

        Self {
            graph,
            hierarchy,
            root,
            op_types,
        }
    }
}

impl Port {
    /// Creates a new port.
    #[inline]
    pub fn new(direction: Direction, port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new(direction, port),
        }
    }

    /// Creates a new incoming port.
    #[inline]
    pub fn new_incoming(port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new_incoming(port),
        }
    }

    /// Creates a new outgoing port.
    #[inline]
    pub fn new_outgoing(port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new_outgoing(port),
        }
    }

    /// Returns the direction of the port.
    #[inline]
    pub fn direction(self) -> Direction {
        self.offset.direction()
    }

    /// Returns the offset of the port.
    #[inline(always)]
    pub fn index(self) -> usize {
        self.offset.index()
    }
}

impl Wire {
    /// Create a new wire from a node and a port.
    #[inline]
    pub fn new(node: Node, port: Port) -> Self {
        Self(node, port.index())
    }

    /// The node that this wire is connected to.
    #[inline]
    pub fn node(&self) -> Node {
        self.0
    }

    /// The output port that this wire is connected to.
    #[inline]
    pub fn source(&self) -> Port {
        Port::new_outgoing(self.1)
    }
}

/// Errors that can occur while manipulating a Hugr.
///
/// TODO: Better descriptions, not just re-exporting portgraph errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum HugrError {
    /// An error occurred while connecting nodes.
    #[error("An error occurred while connecting the nodes.")]
    ConnectionError(#[from] portgraph::LinkError),
    /// An error occurred while manipulating the hierarchy.
    #[error("An error occurred while manipulating the hierarchy.")]
    HierarchyError(#[from] portgraph::hierarchy::AttachError),
}

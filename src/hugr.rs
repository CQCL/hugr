//! The Hugr data structure.
//!
//! TODO: metadata
#![allow(dead_code)]

pub mod validate;

use portgraph::dot::{hier_graph_dot_string_with, DotEdgeStyle};
use portgraph::portgraph::NodePorts;
use portgraph::{Hierarchy, NodeIndex, PortGraph, SecondaryMap};
use thiserror::Error;

use crate::ops::{ModuleOp, OpType};
use crate::rewrite::{Rewrite, RewriteError};

pub use validate::ValidationError;
mod hugrmut;
pub mod serialize;
pub use hugrmut::{BuildError, HugrMut};

/// The Hugr data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    pub(crate) graph: PortGraph,

    /// The node hierarchy.
    hierarchy: Hierarchy,

    /// The single root node in the hierarchy.
    /// It must correspond to a [`ModuleOp::Root`] node.
    ///
    /// [`ModuleOp::Root`]: crate::ops::ModuleOp::Root
    root: NodeIndex,

    /// Operation types for each node.
    op_types: SecondaryMap<NodeIndex, OpType>,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new()
    }
}

impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub(crate) fn new() -> Self {
        let mut graph = PortGraph::default();
        let hierarchy = Hierarchy::new();
        let mut op_types = SecondaryMap::new();
        let root = graph.add_node(0, 0);
        op_types[root] = OpType::Module(ModuleOp::Root);

        Self {
            graph,
            hierarchy,
            root,
            op_types,
        }
    }

    /// Returns the parent of a node.
    #[inline]
    pub fn get_parent(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.hierarchy.parent(node)
    }

    /// Returns the operation type of a node.
    #[inline]
    pub fn get_optype(&self, node: NodeIndex) -> &OpType {
        self.op_types.get(node)
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

    /// Iterator over outputs of node
    #[inline]
    pub fn node_outputs(&self, node: NodeIndex) -> NodePorts {
        self.graph.outputs(node)
    }

    /// Iterator over inputs of node
    #[inline]
    pub fn node_inputs(&self, node: NodeIndex) -> NodePorts {
        self.graph.inputs(node)
    }

    /// Return dot string showing underlying graph and hierarchy side by side
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
            |_| ("".into(), DotEdgeStyle::None),
        )
    }

    /// Number of inputs to node
    #[inline]
    pub fn num_inputs(&self, node: NodeIndex) -> usize {
        self.graph.num_inputs(node)
    }

    /// Number of outputs to node
    #[inline]
    pub fn num_outputs(&self, node: NodeIndex) -> usize {
        self.graph.num_outputs(node)
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

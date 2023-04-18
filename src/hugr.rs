//! The Hugr data structure.
//!
//! TODO: metadata
#![allow(dead_code)]

pub mod validate;

use portgraph::{Hierarchy, NodeIndex, PortGraph, SecondaryMap};
use thiserror::Error;

use crate::ops::{ModuleOp, OpType};
use crate::rewrite::{Rewrite, RewriteError};

pub use validate::ValidationError;
mod base;
pub mod serialize;
pub use base::BaseBuilder;

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
    pub fn get_parent(&self, node: NodeIndex) -> Option<NodeIndex> {
        self.hierarchy.parent(node)
    }

    /// Returns the operation type of a node.
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

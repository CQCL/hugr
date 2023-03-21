//! The Hugr data structure.
//!
//! TODO: metadata
#![allow(dead_code)]

use portgraph::{Hierarchy, NodeIndex, PortGraph, PortIndex, SecondaryMap};

use crate::ops::{ModuleOp, OpType};
use crate::rewrite::{Rewrite, RewriteError};
use crate::types::Type;

/// The Hugr data structure.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    pub(crate) graph: PortGraph,
    hierarchy: Hierarchy,

    /// The single root node in the hierarchy.
    /// It must correspond to a [`ModuleOp::Root`] node.
    root: NodeIndex,

    op_types: SecondaryMap<NodeIndex, OpType>,
    port_types: SecondaryMap<PortIndex, Type>,
}

impl Default for Hugr {
    fn default() -> Self {
        Self::new()
    }
}

impl Hugr {
    /// Create a new Hugr, with a single root node.
    pub fn new() -> Self {
        let mut graph = PortGraph::default();
        let hierarchy = Hierarchy::new();
        let mut op_types = SecondaryMap::new();
        let port_types = SecondaryMap::new();

        let root = graph.add_node(0, 0);
        op_types[root] = OpType::Module(ModuleOp::Root);

        Self {
            graph,
            hierarchy,
            root,
            op_types,
            port_types,
        }
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
        let port_inserted = |old, new| {
            std::mem::swap(&mut self.port_types[new], &mut replacement.port_types[old]);
        };
        rewrite.apply_with_callbacks(
            &mut self.graph,
            |_| {},
            |_| {},
            node_inserted,
            port_inserted,
            |_, _| {},
        )?;

        // TODO: Check types

        Ok(())
    }
}

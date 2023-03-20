//! The Hugr data structure.
#![allow(dead_code)]

use portgraph::{Hierarchy, NodeIndex, PortGraph, PortIndex, SecondaryMap};

use crate::ops::OpType;
use crate::rewrite::{Rewrite, RewriteError};
use crate::types::Type;

/// The Hugr data structure.
#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    pub(crate) graph: PortGraph,
    hierarchy: Hierarchy,

    op_types: SecondaryMap<NodeIndex, OpType>,
    port_types: SecondaryMap<PortIndex, Type>,
}

impl Hugr {
    /// TODO: metadata operations

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

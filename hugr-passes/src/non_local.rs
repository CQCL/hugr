//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
//!
//! TODO Add `remove_nonlocal_edges` and `add_nonlocal_edges` functions
use itertools::Itertools as _;
use thiserror::Error;

use hugr_core::{HugrView, IncomingPort, Node};

/// Returns an iterator over all non local edges in a Hugr.
///
/// All `(node, in_port)` pairs are returned where `in_port` is connected to a
/// node with a parent other than the parent of `node`.
pub fn nonlocal_edges(hugr: &impl HugrView) -> impl Iterator<Item = (Node, IncomingPort)> + '_ {
    hugr.nodes().flat_map(move |node| {
        hugr.in_value_types(node).filter_map(move |(in_p, _)| {
            hugr.linked_outputs(node, in_p)
                .any(|(neighbour_node, _)| hugr.get_parent(node) == hugr.get_parent(neighbour_node))
                .then_some((node, in_p))
        })
    })
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NonLocalEdgesError {
    #[error("Found {} nonlocal edges", .0.len())]
    Edges(Vec<(Node, IncomingPort)>),
}

/// Verifies that there are no non local edges in the Hugr.
pub fn ensure_no_nonlocal_edges(hugr: &impl HugrView) -> Result<(), NonLocalEdgesError> {
    let non_local_edges: Vec<_> = nonlocal_edges(hugr).collect_vec();
    if non_local_edges.is_empty() {
        Ok(())
    } else {
        Err(NonLocalEdgesError::Edges(non_local_edges))?
    }
}

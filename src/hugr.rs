//! The Hugr data structure.
#![allow(dead_code)]

use portgraph::{Hierarchy, NodeIndex, PortGraph, PortIndex, SecondaryMap};

use crate::ops::OpType;
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

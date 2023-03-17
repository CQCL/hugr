//! The Hugr data structure.
#![allow(dead_code)]

use portgraph::{Hierarchy, PortGraph, PortIndex, SecondaryMap};

use crate::types::Type;

/// The Hugr data structure.
#[derive(Clone, Default, Debug)]
pub struct Hugr {
    /// The graph encoding the adjacency structure of the HUGR.
    pub(crate) graph: PortGraph,
    hierarchy: Hierarchy,

    //op_types: SecondaryMap<NodeIndex, Op>,
    port_types: SecondaryMap<PortIndex, Type>,
    // TODO: Node and port metadata
}

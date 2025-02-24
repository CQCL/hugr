//! Types to distinguish ports in the host and replacement graphs.

use std::collections::HashMap;

use crate::{IncomingPort, Node, NodeIndex, OutgoingPort, Port};

use derive_more::From;

/// A port in either the host or replacement graph.
///
/// This is used to represent boundary edges that will be added between the host and
/// replacement graphs when applying a rewrite.
#[derive(Debug, Clone, Copy)]
pub enum BoundaryPort<HostNode, P> {
    /// A port in the host graph.
    Host(HostNode, P),
    /// A port in the replacement graph.
    Replacement(Node, P),
}

/// A port in the host graph.
#[derive(Debug, Clone, Copy, From)]
pub struct HostPort<N, P>(pub N, pub P);

/// A port in the replacement graph.
#[derive(Debug, Clone, Copy, From)]
pub struct ReplacementPort<P>(pub Node, pub P);

impl<HostNode: NodeIndex, P> BoundaryPort<HostNode, P> {
    /// Maps a boundary port according to the insertion mapping.
    /// Host ports are unchanged, while Replacement ports are mapped according to the index_map.
    pub fn map_replacement(self, index_map: &HashMap<Node, HostNode>) -> (HostNode, P) {
        match self {
            BoundaryPort::Host(node, port) => (node, port),
            BoundaryPort::Replacement(node, port) => (*index_map.get(&node).unwrap(), port),
        }
    }
}

impl<N, P> From<HostPort<N, P>> for BoundaryPort<N, P> {
    fn from(HostPort(node, port): HostPort<N, P>) -> Self {
        BoundaryPort::Host(node, port)
    }
}

impl<N, P> From<ReplacementPort<P>> for BoundaryPort<N, P> {
    fn from(ReplacementPort(node, port): ReplacementPort<P>) -> Self {
        BoundaryPort::Replacement(node, port)
    }
}

impl<HostNode> From<HostPort<HostNode, OutgoingPort>> for HostPort<HostNode, Port> {
    fn from(HostPort(node, port): HostPort<HostNode, OutgoingPort>) -> Self {
        HostPort(node, port.into())
    }
}

impl<HostNode> From<HostPort<HostNode, IncomingPort>> for HostPort<HostNode, Port> {
    fn from(HostPort(node, port): HostPort<HostNode, IncomingPort>) -> Self {
        HostPort(node, port.into())
    }
}

impl From<ReplacementPort<OutgoingPort>> for ReplacementPort<Port> {
    fn from(ReplacementPort(node, port): ReplacementPort<OutgoingPort>) -> Self {
        ReplacementPort(node, port.into())
    }
}

impl From<ReplacementPort<IncomingPort>> for ReplacementPort<Port> {
    fn from(ReplacementPort(node, port): ReplacementPort<IncomingPort>) -> Self {
        ReplacementPort(node, port.into())
    }
}

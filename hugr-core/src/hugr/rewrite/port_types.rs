//! Types to distinguish ports in the host and replacement graphs.

use std::collections::HashMap;

use crate::{IncomingPort, Node, OutgoingPort, Port};

use derive_more::From;

/// A port in either the host or replacement graph.
///
/// This is used to represent boundary edges that will be added between the host and
/// replacement graphs when applying a rewrite.
#[derive(Debug, Clone, Copy)]
pub enum BoundaryPort<P> {
    /// A port in the host graph.
    Host(Node, P),
    /// A port in the replacement graph.
    Replacement(Node, P),
}

/// A port in the host graph.
#[derive(Debug, Clone, Copy, From)]
pub struct HostPort<P>(pub Node, pub P);

/// A port in the replacement graph.
#[derive(Debug, Clone, Copy, From)]
pub struct ReplacementPort<P>(pub Node, pub P);

impl<P> BoundaryPort<P> {
    /// Maps a boundary port according to the insertion mapping.
    /// Host ports are unchanged, while Replacement ports are mapped according to the index_map.
    pub fn map_replacement(self, index_map: &HashMap<Node, Node>) -> (Node, P) {
        match self {
            BoundaryPort::Host(node, port) => (node, port),
            BoundaryPort::Replacement(node, port) => (*index_map.get(&node).unwrap(), port),
        }
    }
}

impl<P> From<HostPort<P>> for BoundaryPort<P> {
    fn from(HostPort(node, port): HostPort<P>) -> Self {
        BoundaryPort::Host(node, port)
    }
}

impl<P> From<ReplacementPort<P>> for BoundaryPort<P> {
    fn from(ReplacementPort(node, port): ReplacementPort<P>) -> Self {
        BoundaryPort::Replacement(node, port)
    }
}

macro_rules! impl_port_conversion {
    ($from_type:ty, $to_type:ty, $wrapper:ident) => {
        impl From<$from_type> for $to_type {
            fn from($wrapper(node, port): $from_type) -> Self {
                $wrapper(node, port.into())
            }
        }
    };
}

// impl From<HostPort<OutgoingPort>> for HostPort<Port>
impl_port_conversion!(HostPort<OutgoingPort>, HostPort<Port>, HostPort);
// impl From<HostPort<IncomingPort>> for HostPort<Port>
impl_port_conversion!(HostPort<IncomingPort>, HostPort<Port>, HostPort);
// impl From<ReplacementPort<OutgoingPort>> for ReplacementPort<Port>
impl_port_conversion!(
    ReplacementPort<OutgoingPort>,
    ReplacementPort<Port>,
    ReplacementPort
);
// impl From<ReplacementPort<IncomingPort>> for ReplacementPort<Port>
impl_port_conversion!(
    ReplacementPort<IncomingPort>,
    ReplacementPort<Port>,
    ReplacementPort
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IncomingPort, OutgoingPort, Port};

    #[test]
    fn test_port_conversions() {
        let node = Node::new(1);

        // Test HostPort conversions
        let host_out = HostPort(node, OutgoingPort::from(0));
        let host_port: HostPort<Port> = host_out.into();
        assert_eq!(host_port.0, node);
        assert_eq!(host_port.1, Port::from(OutgoingPort::from(0)));

        let host_in = HostPort(node, IncomingPort::from(1));
        let host_port: HostPort<Port> = host_in.into();
        assert_eq!(host_port.0, node);
        assert_eq!(host_port.1, Port::from(IncomingPort::from(1)));

        // Test ReplacementPort conversions
        let repl_out = ReplacementPort(node, OutgoingPort::from(0));
        let repl_port: ReplacementPort<Port> = repl_out.into();
        assert_eq!(repl_port.0, node);
        assert_eq!(repl_port.1, Port::from(OutgoingPort::from(0)));

        let repl_in = ReplacementPort(node, IncomingPort::from(1));
        let repl_port: ReplacementPort<Port> = repl_in.into();
        assert_eq!(repl_port.0, node);
        assert_eq!(repl_port.1, Port::from(IncomingPort::from(1)));
    }
}

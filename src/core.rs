//! Definitions for the core types used in the Hugr.
//!
//! These types are re-exported in the root of the crate.

use derive_more::From;

#[cfg(feature = "pyo3")]
use pyo3::pyclass;

use crate::hugr::HugrError;

/// A handle to a node in the HUGR.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    From,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Node {
    index: portgraph::NodeIndex,
}

/// A handle to a port for a node in the HUGR.
#[derive(
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Hash,
    Default,
    Debug,
    From,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct Port {
    offset: portgraph::PortOffset,
}

/// A trait for getting the undirected index of a port.
pub trait PortIndex {
    /// Returns the offset of the port.
    fn index(self) -> usize;
}

/// A trait for getting the index of a node.
pub trait NodeIndex {
    /// Returns the index of the node.
    fn index(self) -> usize;
}

/// A port in the incoming direction.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug)]
pub struct IncomingPort {
    index: u16,
}

/// A port in the outgoing direction.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, Debug)]
pub struct OutgoingPort {
    index: u16,
}

/// The direction of a port.
pub type Direction = portgraph::Direction;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

impl Node {
    /// Returns the node as a portgraph `NodeIndex`.
    #[inline]
    pub(crate) fn pg_index(self) -> portgraph::NodeIndex {
        self.index
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
    pub fn new_incoming(port: impl Into<IncomingPort>) -> Self {
        Self::try_new_incoming(port).unwrap()
    }

    /// Creates a new outgoing port.
    #[inline]
    pub fn new_outgoing(port: impl Into<OutgoingPort>) -> Self {
        Self::try_new_outgoing(port).unwrap()
    }

    /// Creates a new incoming port.
    #[inline]
    pub fn try_new_incoming(port: impl TryInto<IncomingPort>) -> Result<Self, HugrError> {
        let Ok(port) = port.try_into() else {
            return Err(HugrError::InvalidPortDirection(Direction::Outgoing));
        };
        Ok(Self {
            offset: portgraph::PortOffset::new_incoming(port.index()),
        })
    }

    /// Creates a new outgoing port.
    #[inline]
    pub fn try_new_outgoing(port: impl TryInto<OutgoingPort>) -> Result<Self, HugrError> {
        let Ok(port) = port.try_into() else {
            return Err(HugrError::InvalidPortDirection(Direction::Incoming));
        };
        Ok(Self {
            offset: portgraph::PortOffset::new_outgoing(port.index()),
        })
    }

    /// Returns the direction of the port.
    #[inline]
    pub fn direction(self) -> Direction {
        self.offset.direction()
    }

    /// Returns the port as a portgraph `PortOffset`.
    #[inline]
    pub(crate) fn pg_offset(self) -> portgraph::PortOffset {
        self.offset
    }
}

impl PortIndex for Port {
    #[inline(always)]
    fn index(self) -> usize {
        self.offset.index()
    }
}

impl PortIndex for usize {
    #[inline(always)]
    fn index(self) -> usize {
        self
    }
}

impl PortIndex for IncomingPort {
    #[inline(always)]
    fn index(self) -> usize {
        self.index as usize
    }
}

impl PortIndex for OutgoingPort {
    #[inline(always)]
    fn index(self) -> usize {
        self.index as usize
    }
}

impl From<usize> for IncomingPort {
    #[inline(always)]
    fn from(index: usize) -> Self {
        Self {
            index: index as u16,
        }
    }
}

impl From<usize> for OutgoingPort {
    #[inline(always)]
    fn from(index: usize) -> Self {
        Self {
            index: index as u16,
        }
    }
}

impl TryFrom<Port> for IncomingPort {
    type Error = HugrError;
    #[inline(always)]
    fn try_from(port: Port) -> Result<Self, Self::Error> {
        match port.direction() {
            Direction::Incoming => Ok(Self {
                index: port.index() as u16,
            }),
            dir @ Direction::Outgoing => Err(HugrError::InvalidPortDirection(dir)),
        }
    }
}

impl TryFrom<Port> for OutgoingPort {
    type Error = HugrError;
    #[inline(always)]
    fn try_from(port: Port) -> Result<Self, Self::Error> {
        match port.direction() {
            Direction::Outgoing => Ok(Self {
                index: port.index() as u16,
            }),
            dir @ Direction::Incoming => Err(HugrError::InvalidPortDirection(dir)),
        }
    }
}

impl NodeIndex for Node {
    fn index(self) -> usize {
        self.index.into()
    }
}

impl Wire {
    /// Create a new wire from a node and a port.
    #[inline]
    pub fn new(node: Node, port: impl TryInto<OutgoingPort>) -> Self {
        Self(node, Port::try_new_outgoing(port).unwrap().index())
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

/// Enum for uniquely identifying the origin of linear wires in a circuit-like
/// dataflow region.
///
/// Falls back to [`Wire`] if the wire is not linear or if it's not possible to
/// track the origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CircuitUnit {
    /// Arbitrary input wire.
    Wire(Wire),
    /// Index to region input.
    Linear(usize),
}

impl CircuitUnit {
    /// Check if this is a wire.
    pub fn is_wire(&self) -> bool {
        matches!(self, CircuitUnit::Wire(_))
    }

    /// Check if this is a linear unit.
    pub fn is_linear(&self) -> bool {
        matches!(self, CircuitUnit::Linear(_))
    }
}

impl From<usize> for CircuitUnit {
    fn from(value: usize) -> Self {
        CircuitUnit::Linear(value)
    }
}

impl From<Wire> for CircuitUnit {
    fn from(value: Wire) -> Self {
        CircuitUnit::Wire(value)
    }
}

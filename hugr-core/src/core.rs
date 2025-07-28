//! Definitions for the core types used in the Hugr.
//!
//! These types are re-exported in the root of the crate.

pub use itertools::Either;

use derive_more::From;
use itertools::Either::{Left, Right};

use crate::{HugrView, hugr::HugrError};

/// A handle to a node in the HUGR.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
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
    From,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct Port {
    offset: portgraph::PortOffset<u32>,
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

/// A trait for nodes in the Hugr.
pub trait HugrNode: Copy + Ord + std::fmt::Debug + std::fmt::Display + std::hash::Hash {}

impl<T: Copy + Ord + std::fmt::Debug + std::fmt::Display + std::hash::Hash> HugrNode for T {}

/// A port in the incoming direction.
#[derive(
    Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct IncomingPort {
    index: u16,
}

/// A port in the outgoing direction.
#[derive(
    Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct OutgoingPort {
    index: u16,
}

/// The direction of a port.
pub type Direction = portgraph::Direction;

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
/// A `DataFlow` wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire<N = Node>(N, OutgoingPort);

impl Node {
    /// Returns the node as a portgraph `NodeIndex`.
    #[inline]
    pub(crate) fn into_portgraph(self) -> portgraph::NodeIndex {
        self.index
    }
}

impl Port {
    /// Creates a new port.
    #[inline]
    #[must_use]
    pub fn new(direction: Direction, port: usize) -> Self {
        Self {
            offset: portgraph::PortOffset::new(direction, port),
        }
    }

    /// Converts to an [`IncomingPort`] if this port is one; else fails with
    /// [`HugrError::InvalidPortDirection`]
    #[inline]
    pub fn as_incoming(&self) -> Result<IncomingPort, HugrError> {
        self.as_directed()
            .left()
            .ok_or(HugrError::InvalidPortDirection(self.direction()))
    }

    /// Converts to an [`OutgoingPort`] if this port is one; else fails with
    /// [`HugrError::InvalidPortDirection`]
    #[inline]
    pub fn as_outgoing(&self) -> Result<OutgoingPort, HugrError> {
        self.as_directed()
            .right()
            .ok_or(HugrError::InvalidPortDirection(self.direction()))
    }

    /// Converts to either an [`IncomingPort`] or an [`OutgoingPort`], as appropriate.
    #[inline]
    #[must_use]
    pub fn as_directed(&self) -> Either<IncomingPort, OutgoingPort> {
        match self.direction() {
            Direction::Incoming => Left(IncomingPort {
                index: self.index() as u16,
            }),
            Direction::Outgoing => Right(OutgoingPort {
                index: self.index() as u16,
            }),
        }
    }

    /// Returns the direction of the port.
    #[inline]
    #[must_use]
    pub fn direction(self) -> Direction {
        self.offset.direction()
    }

    /// Returns the port as a portgraph `PortOffset`.
    #[inline]
    pub(crate) fn pg_offset(self) -> portgraph::PortOffset<u32> {
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

impl From<IncomingPort> for Port {
    fn from(value: IncomingPort) -> Self {
        Self {
            offset: portgraph::PortOffset::new_incoming(value.index()),
        }
    }
}

impl From<OutgoingPort> for Port {
    fn from(value: OutgoingPort) -> Self {
        Self {
            offset: portgraph::PortOffset::new_outgoing(value.index()),
        }
    }
}

impl NodeIndex for Node {
    fn index(self) -> usize {
        self.index.into()
    }
}

impl<N: HugrNode> Wire<N> {
    /// Create a new wire from a node and a port.
    #[inline]
    pub fn new(node: N, port: impl Into<OutgoingPort>) -> Self {
        Self(node, port.into())
    }

    /// Create a new wire from a node and a port that is connected to the wire.
    ///
    /// If `port` is an incoming port, the wire is traversed to find the unique
    /// outgoing port that is connected to the wire. Otherwise, this is
    /// equivalent to constructing a wire using [`Wire::new`].
    ///
    /// ## Panics
    ///
    /// This will panic if the wire is not connected to a unique outgoing port.
    #[inline]
    pub fn from_connected_port(
        node: N,
        port: impl Into<Port>,
        hugr: &impl HugrView<Node = N>,
    ) -> Self {
        let (node, outgoing) = match port.into().as_directed() {
            Either::Left(incoming) => hugr
                .single_linked_output(node, incoming)
                .expect("invalid dfg port"),
            Either::Right(outgoing) => (node, outgoing),
        };
        Self::new(node, outgoing)
    }

    /// The node of the unique outgoing port that the wire is connected to.
    #[inline]
    pub fn node(&self) -> N {
        self.0
    }

    /// The unique outgoing port that the wire is connected to.
    #[inline]
    pub fn source(&self) -> OutgoingPort {
        self.1
    }

    /// Get all ports connected to the wire.
    ///
    /// Return a chained iterator of the unique outgoing port, followed by all
    /// incoming ports connected to the wire.
    pub fn all_connected_ports<'h, H: HugrView<Node = N>>(
        &self,
        hugr: &'h H,
    ) -> impl Iterator<Item = (N, Port)> + use<'h, N, H> {
        let node = self.node();
        let out_port = self.source();

        std::iter::once((node, out_port.into())).chain(hugr.linked_ports(node, out_port))
    }
}

impl<N: HugrNode> std::fmt::Display for Wire<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wire({}, {})", self.0, self.1.index)
    }
}

/// Marks [FuncDefn](crate::ops::FuncDefn)s and [FuncDecl](crate::ops::FuncDecl)s as
/// to whether they should be considered for linking.
#[derive(
    Clone,
    Debug,
    derive_more::Display,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub enum Visibility {
    /// Function is visible or exported
    Public,
    /// Function is hidden, for use within the hugr only
    Private,
}

impl From<hugr_model::v0::Visibility> for Visibility {
    fn from(value: hugr_model::v0::Visibility) -> Self {
        match value {
            hugr_model::v0::Visibility::Private => Self::Private,
            hugr_model::v0::Visibility::Public => Self::Public,
        }
    }
}

impl From<Visibility> for hugr_model::v0::Visibility {
    fn from(value: Visibility) -> Self {
        match value {
            Visibility::Public => hugr_model::v0::Visibility::Public,
            Visibility::Private => hugr_model::v0::Visibility::Private,
        }
    }
}

/// Enum for uniquely identifying the origin of linear wires in a circuit-like
/// dataflow region.
///
/// Falls back to [`Wire`] if the wire is not linear or if it's not possible to
/// track the origin.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CircuitUnit<N = Node> {
    /// Arbitrary input wire.
    Wire(Wire<N>),
    /// Index to region input.
    Linear(usize),
}

impl CircuitUnit {
    /// Check if this is a wire.
    #[must_use]
    pub fn is_wire(&self) -> bool {
        matches!(self, CircuitUnit::Wire(_))
    }

    /// Check if this is a linear unit.
    #[must_use]
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

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Node").field(&self.index()).finish()
    }
}

impl std::fmt::Debug for Port {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Port")
            .field(&self.offset.direction())
            .field(&self.index())
            .finish()
    }
}

impl std::fmt::Debug for IncomingPort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IncomingPort").field(&self.index).finish()
    }
}

impl std::fmt::Debug for OutgoingPort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OutgoingPort").field(&self.index).finish()
    }
}

impl<N: HugrNode> std::fmt::Debug for Wire<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wire")
            .field("node", &self.0)
            .field("port", &self.1)
            .finish()
    }
}

impl std::fmt::Debug for CircuitUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wire(w) => f
                .debug_struct("WireUnit")
                .field("node", &w.0.index())
                .field("port", &w.1)
                .finish(),
            Self::Linear(id) => f.debug_tuple("LinearUnit").field(id).finish(),
        }
    }
}

macro_rules! impl_display_from_debug {
    ($($t:ty),*) => {
        $(
            impl std::fmt::Display for $t {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    <Self as std::fmt::Debug>::fmt(self, f)
                }
            }
        )*
    };
}
impl_display_from_debug!(Node, Port, IncomingPort, OutgoingPort, CircuitUnit);

//! Utilities for pinned ports and pinned wires.
//!
//! Encapsulation: we only ever expose pinned values publicly.

use itertools::Either;

use crate::PatchNode;
use hugr_core::{Direction, IncomingPort, OutgoingPort, Port};

use super::Walker;

/// A wire in the current HUGR of a [`Walker`] with some of its endpoints
/// pinned.
///
/// Just like a normal HUGR [`Wire`](hugr_core::Wire), a [`PinnedWire`] has
/// endpoints: the ports that are linked together by the wire.  A [`PinnedWire`]
/// however distinguishes itself in that each of its ports is specified either
/// as "pinned" or "unpinned". A port is pinned if and only if the node it is
/// attached to is pinned in the walker.
///
/// A [`PinnedWire`] always has at least one pinned port.
///
/// All pinned ports of a [`PinnedWire`] can be retrieved using
/// [`PinnedWire::pinned_inports`] and [`PinnedWire::pinned_outport`]. Unpinned
/// ports, on the other hand, represent undetermined connections, which may
/// still change as the walker is expanded (see [`Walker::expand`]).
///
/// Whether all incoming or outgoing ports are pinned can be checked using
/// [`PinnedWire::is_complete`].
#[derive(Debug, Clone)]
pub struct PinnedWire {
    outgoing: MaybePinned<OutgoingPort>,
    incoming: Vec<MaybePinned<IncomingPort>>,
}

/// A private enum to track whether a port is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MaybePinned<P> {
    Pinned(PatchNode, P),
    Unpinned(PatchNode, P),
}

impl<P> MaybePinned<P> {
    fn new<R: Clone>(node: PatchNode, port: P, walker: &Walker<R>) -> Self {
        debug_assert!(
            walker.selected_commits.contains_node(node),
            "pinned node not in walker"
        );
        if walker.is_pinned(node) {
            MaybePinned::Pinned(node, port)
        } else {
            MaybePinned::Unpinned(node, port)
        }
    }

    fn is_pinned(&self) -> bool {
        matches!(self, MaybePinned::Pinned(_, _))
    }

    fn into_unpinned<PP: From<P>>(self) -> Option<(PatchNode, PP)> {
        match self {
            MaybePinned::Pinned(_, _) => None,
            MaybePinned::Unpinned(node, port) => Some((node, port.into())),
        }
    }

    fn into_pinned<PP: From<P>>(self) -> Option<(PatchNode, PP)> {
        match self {
            MaybePinned::Pinned(node, port) => Some((node, port.into())),
            MaybePinned::Unpinned(_, _) => None,
        }
    }
}

impl PinnedWire {
    /// Create a new pinned wire in `walker` from a pinned node and a port.
    ///
    /// # Panics
    /// Panics if `node` is not pinned in `walker`.
    pub fn from_pinned_port<R: Clone>(
        node: PatchNode,
        port: impl Into<Port>,
        walker: &Walker<R>,
    ) -> Self {
        assert!(walker.is_pinned(node), "node must be pinned");

        let (outgoing_node, outgoing_port) = match port.into().as_directed() {
            Either::Left(incoming) => walker
                .selected_commits
                .get_single_outgoing_port(node, incoming),
            Either::Right(outgoing) => (node, outgoing),
        };

        let outgoing = MaybePinned::new(outgoing_node, outgoing_port, walker);

        let incoming = walker
            .selected_commits
            .get_all_incoming_ports(outgoing_node, outgoing_port)
            .map(|(n, p)| MaybePinned::new(n, p, walker))
            .collect();

        Self { outgoing, incoming }
    }

    /// Check if all ports on the wire in the given direction are pinned.
    ///
    /// A wire is complete in a direction if and only if expanding the wire
    /// in that direction would yield no new walkers. If no direction is
    /// specified, checks if the wire is complete in both directions.
    pub fn is_complete(&self, dir: impl Into<Option<Direction>>) -> bool {
        match dir.into() {
            Some(Direction::Outgoing) => self.outgoing.is_pinned(),
            Some(Direction::Incoming) => self.incoming.iter().all(|p| p.is_pinned()),
            None => self.outgoing.is_pinned() && self.incoming.iter().all(|p| p.is_pinned()),
        }
    }

    /// Get the outgoing port of the wire, if it is pinned.
    ///
    /// Returns `None` if the outgoing port is not pinned.
    pub fn pinned_outport(&self) -> Option<(PatchNode, OutgoingPort)> {
        self.outgoing.into_pinned()
    }

    /// Get all pinned incoming ports of the wire.
    ///
    /// Returns an iterator over all pinned incoming ports.
    pub fn pinned_inports(&self) -> impl Iterator<Item = (PatchNode, IncomingPort)> + '_ {
        self.incoming.iter().filter_map(|&p| p.into_pinned())
    }

    /// Get all pinned ports of the wire.
    pub fn all_pinned_ports(&self) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        fn to_port((node, port): (PatchNode, impl Into<Port>)) -> (PatchNode, Port) {
            (node, port.into())
        }
        self.pinned_outport()
            .into_iter()
            .map(to_port)
            .chain(self.pinned_inports().map(to_port))
    }

    /// Get all unpinned ports of the wire, optionally filtering to only those
    /// in the given direction.
    pub(crate) fn unpinned_ports(
        &self,
        dir: impl Into<Option<Direction>>,
    ) -> impl Iterator<Item = (PatchNode, Port)> + '_ {
        let incoming = self
            .incoming
            .iter()
            .filter_map(|p| p.into_unpinned::<Port>());
        let outgoing = self.outgoing.into_unpinned::<Port>();
        let dir = dir.into();
        mask_iter(incoming, dir != Some(Direction::Outgoing))
            .chain(mask_iter(outgoing, dir != Some(Direction::Incoming)))
    }
}

/// Return an iterator over the items in `iter` if `mask` is true, otherwise
/// return an empty iterator.
#[inline]
fn mask_iter<I>(iter: impl IntoIterator<Item = I>, mask: bool) -> impl Iterator<Item = I> {
    match mask {
        true => Either::Left(iter.into_iter()),
        false => Either::Right(std::iter::empty()),
    }
    .into_iter()
}

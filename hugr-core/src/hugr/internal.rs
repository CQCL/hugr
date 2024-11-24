//! Internal traits, not exposed in the public `hugr` API.

use std::ops::Range;

use delegate::delegate;
use portgraph::{LinkView, MultiPortGraph, PortMut, PortView};

use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Direction, Hugr, Node};

use super::hugrmut::{panic_invalid_node, panic_invalid_non_root};
use super::{HugrError, OpType, RootTagged};

/// Trait for accessing the internals of a Hugr(View).
///
/// Specifically, this trait provides access to the underlying portgraph
/// view.
pub trait HugrInternals {
    /// The underlying portgraph view type.
    type Portgraph<'p>: LinkView + Clone + 'p
    where
        Self: 'p;

    /// Returns a reference to the underlying portgraph.
    fn portgraph(&self) -> Self::Portgraph<'_>;

    /// Returns the Hugr at the base of a chain of views.
    fn base_hugr(&self) -> &Hugr;

    /// Return the root node of this view.
    fn root_node(&self) -> Node;
}

impl HugrInternals for Hugr {
    type Portgraph<'p> = &'p MultiPortGraph where Self: 'p;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        &self.as_ref().graph
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self.as_ref()
    }

    #[inline]
    fn root_node(&self) -> Node {
        self.as_ref().root.into()
    }
}

impl<T: HugrInternals> HugrInternals for &T {
    type Portgraph<'p> = T::Portgraph<'p> where Self: 'p;
    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Node;
        }
    }
}

/// Trait for accessing the mutable internals of a Hugr(Mut).
///
/// Specifically, this trait lets you apply arbitrary modifications that may
/// invalidate the HUGR.
pub trait HugrMutInternals: RootTagged {
    /// Returns the Hugr at the base of a chain of views.
    fn hugr_mut(&mut self) -> &mut Hugr;

    /// Set the number of ports on a node. This may invalidate the node's `PortIndex`.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        panic_invalid_node(self, node);
        self.hugr_mut().set_num_ports(node, incoming, outgoing)
    }

    /// Alter the number of ports on a node and returns a range with the new
    /// port offsets, if any. This may invalidate the node's `PortIndex`.
    ///
    /// The `direction` parameter specifies whether to add ports to the incoming
    /// or outgoing list.
    ///
    /// Returns the range of newly created ports.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
        panic_invalid_node(self, node);
        self.hugr_mut().add_ports(node, direction, amount)
    }

    /// Sets the parent of a node.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If either the node or the parent is not in the graph.
    fn set_parent(&mut self, node: Node, parent: Node) {
        panic_invalid_node(self, parent);
        panic_invalid_non_root(self, node);
        self.hugr_mut().set_parent(node, parent);
    }

    /// Move a node in the hierarchy to be the subsequent sibling of another
    /// node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If either node is not in the graph, or if it is a root.
    fn move_after_sibling(&mut self, node: Node, after: Node) {
        panic_invalid_non_root(self, node);
        panic_invalid_non_root(self, after);
        self.hugr_mut().move_after_sibling(node, after);
    }

    /// Move a node in the hierarchy to be the prior sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If either node is not in the graph, or if it is a root.
    fn move_before_sibling(&mut self, node: Node, before: Node) {
        panic_invalid_non_root(self, node);
        panic_invalid_non_root(self, before);
        self.hugr_mut().move_before_sibling(node, before)
    }

    /// Replace the OpType at node and return the old OpType.
    /// In general this invalidates the ports, which may need to be resized to
    /// match the OpType signature.
    ///
    /// Returns the old OpType.
    ///
    /// TODO: Add a version which ignores input extensions
    ///
    /// # Errors
    ///
    /// Returns a [`HugrError::InvalidTag`] if this would break the bound
    /// (`Self::RootHandle`) on the root node's OpTag.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> Result<OpType, HugrError> {
        panic_invalid_node(self, node);
        let op = op.into();
        if node == self.root() && !Self::RootHandle::TAG.is_superset(op.tag()) {
            return Err(HugrError::InvalidTag {
                required: Self::RootHandle::TAG,
                actual: op.tag(),
            });
        }
        self.hugr_mut().replace_op(node, op)
    }
}

/// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
impl<T: RootTagged<RootHandle = Node> + AsMut<Hugr>> HugrMutInternals for T {
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.as_mut()
    }

    #[inline]
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        self.hugr_mut()
            .graph
            .set_num_ports(node.pg_index(), incoming, outgoing, |_, _| {})
    }

    fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
        let mut incoming = self.hugr_mut().graph.num_inputs(node.pg_index());
        let mut outgoing = self.hugr_mut().graph.num_outputs(node.pg_index());
        let increment = |num: &mut usize| {
            let new = num.saturating_add_signed(amount);
            let range = *num..new;
            *num = new;
            range
        };
        let range = match direction {
            Direction::Incoming => increment(&mut incoming),
            Direction::Outgoing => increment(&mut outgoing),
        };
        self.hugr_mut()
            .graph
            .set_num_ports(node.pg_index(), incoming, outgoing, |_, _| {});
        range
    }

    fn set_parent(&mut self, node: Node, parent: Node) {
        self.hugr_mut().hierarchy.detach(node.pg_index());
        self.hugr_mut()
            .hierarchy
            .push_child(node.pg_index(), parent.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_after_sibling(&mut self, node: Node, after: Node) {
        self.hugr_mut().hierarchy.detach(node.pg_index());
        self.hugr_mut()
            .hierarchy
            .insert_after(node.pg_index(), after.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_before_sibling(&mut self, node: Node, before: Node) {
        self.hugr_mut().hierarchy.detach(node.pg_index());
        self.hugr_mut()
            .hierarchy
            .insert_before(node.pg_index(), before.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> Result<OpType, HugrError> {
        // We know RootHandle=Node here so no need to check
        let cur = self.hugr_mut().op_types.get_mut(node.pg_index());
        Ok(std::mem::replace(cur, op.into()))
    }
}

//! Internal traits, not exposed in the public `hugr` API.

use std::borrow::Cow;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use delegate::delegate;
use itertools::Itertools;
use portgraph::{LinkMut, LinkView, MultiPortGraph, PortMut, PortOffset, PortView};

use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Direction, Hugr, Node, NodeIndex};

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

    /// The type of nodes in the Hugr.
    type Node: NodeIndex;

    /// Returns a reference to the underlying portgraph.
    fn portgraph(&self) -> Self::Portgraph<'_>;

    /// Returns the Hugr at the base of a chain of views.
    fn base_hugr(&self) -> &Hugr;

    /// Return the root node of this view.
    fn root_node(&self) -> Self::Node;
}

impl HugrInternals for Hugr {
    type Portgraph<'p>
        = &'p MultiPortGraph
    where
        Self: 'p;

    type Node = Node;

    #[inline]
    fn portgraph(&self) -> Self::Portgraph<'_> {
        &self.graph
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self
    }

    #[inline]
    fn root_node(&self) -> Self::Node {
        self.root.into()
    }
}

impl<T: HugrInternals> HugrInternals for &T {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}

impl<T: HugrInternals> HugrInternals for &mut T {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}

impl<T: HugrInternals> HugrInternals for Rc<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}

impl<T: HugrInternals> HugrInternals for Arc<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}

impl<T: HugrInternals> HugrInternals for Box<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to (**self) {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}

impl<T: HugrInternals + ToOwned> HugrInternals for Cow<'_, T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    delegate! {
        to self.as_ref() {
            fn portgraph(&self) -> Self::Portgraph<'_>;
            fn base_hugr(&self) -> &Hugr;
            fn root_node(&self) -> Self::Node;
        }
    }
}
/// Trait for accessing the mutable internals of a Hugr(Mut).
///
/// Specifically, this trait lets you apply arbitrary modifications that may
/// invalidate the HUGR.
pub trait HugrMutInternals: RootTagged<Node = Node> {
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

    /// Insert `amount` new ports for a node, starting at `index`.  The
    /// `direction` parameter specifies whether to add ports to the incoming or
    /// outgoing list. Links from this node are preserved, even when ports are
    /// renumbered by the insertion.
    ///
    /// Returns the range of newly created ports.
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn insert_ports(
        &mut self,
        node: Node,
        direction: Direction,
        index: usize,
        amount: usize,
    ) -> Range<usize> {
        panic_invalid_node(self, node);
        self.hugr_mut().insert_ports(node, direction, index, amount)
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
impl<T: RootTagged<RootHandle = Node, Node = Node> + AsMut<Hugr>> HugrMutInternals for T {
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.as_mut()
    }

    #[inline]
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        self.hugr_mut()
            .graph
            .set_num_ports(node.pg_index(), incoming, outgoing, |_, _| {})
    }

    fn insert_ports(
        &mut self,
        node: Node,
        direction: Direction,
        index: usize,
        amount: usize,
    ) -> Range<usize> {
        let old_num_ports = self.base_hugr().graph.num_ports(node.pg_index(), direction);

        self.add_ports(node, direction, amount as isize);

        for swap_from_port in (index..old_num_ports).rev() {
            let swap_to_port = swap_from_port + amount;
            let [from_port_index, to_port_index] = [swap_from_port, swap_to_port].map(|p| {
                self.base_hugr()
                    .graph
                    .port_index(node.pg_index(), PortOffset::new(direction, p))
                    .unwrap()
            });
            let linked_ports = self
                .base_hugr()
                .graph
                .port_links(from_port_index)
                .map(|(_, to_subport)| to_subport.port())
                .collect_vec();
            self.hugr_mut().graph.unlink_port(from_port_index);
            for linked_port_index in linked_ports {
                let _ = self
                    .hugr_mut()
                    .graph
                    .link_ports(to_port_index, linked_port_index)
                    .expect("Ports exist");
            }
        }
        index..index + amount
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

#[cfg(test)]
mod test {
    use crate::{
        builder::{Container, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::Noop,
        hugr::internal::HugrMutInternals as _,
        ops::handle::NodeHandle,
        types::{Signature, Type},
        Direction, HugrView as _,
    };

    #[test]
    fn insert_ports() {
        let (nop, mut hugr) = {
            let mut builder =
                DFGBuilder::new(Signature::new_endo(Type::UNIT).with_prelude()).unwrap();
            let [nop_in] = builder.input_wires_arr();
            let nop = builder
                .add_dataflow_op(Noop::new(Type::UNIT), [nop_in])
                .unwrap();
            builder.add_other_wire(nop.node(), builder.output().node());
            let [nop_out] = nop.outputs_arr();
            (
                nop.node(),
                builder.finish_hugr_with_outputs([nop_out]).unwrap(),
            )
        };
        let [i, o] = hugr.get_io(hugr.root()).unwrap();
        assert_eq!(0..2, hugr.insert_ports(nop, Direction::Incoming, 0, 2));
        assert_eq!(1..3, hugr.insert_ports(nop, Direction::Outgoing, 1, 2));

        assert_eq!(hugr.single_linked_input(i, 0), Some((nop, 2.into())));
        assert_eq!(hugr.single_linked_output(o, 0), Some((nop, 0.into())));
        assert_eq!(hugr.single_linked_output(o, 1), Some((nop, 3.into())));
    }
}

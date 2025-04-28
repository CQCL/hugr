//! Internal traits, not exposed in the public `hugr` API.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::OnceLock;

use itertools::Itertools;
use portgraph::{LinkMut, LinkView, MultiPortGraph, PortMut, PortOffset, PortView};

use crate::extension::ExtensionRegistry;
use crate::ops::handle::NodeHandle;
use crate::{Direction, Hugr, Node};

use super::hugrmut::{panic_invalid_node, panic_invalid_non_root};
use super::{HugrView, NodeMetadataMap, OpType};

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
    type Node: Copy + Ord + std::fmt::Debug + std::fmt::Display + std::hash::Hash;

    /// Returns a reference to the underlying portgraph.
    fn portgraph(&self) -> Self::Portgraph<'_>;

    /// Returns the portgraph [Hierarchy](portgraph::Hierarchy) of the graph
    /// returned by [`HugrInternals::portgraph`].
    #[inline]
    fn hierarchy(&self) -> Cow<'_, portgraph::Hierarchy> {
        Cow::Borrowed(&self.base_hugr().hierarchy)
    }

    /// Returns the Hugr at the base of a chain of views.
    fn base_hugr(&self) -> &Hugr;

    /// Return the root node of this view.
    fn root_node(&self) -> Self::Node;

    /// Convert a node to a portgraph node index.
    fn get_pg_index(&self, node: impl NodeHandle<Self::Node>) -> portgraph::NodeIndex;

    /// Convert a portgraph node index to a node.
    fn get_node(&self, index: portgraph::NodeIndex) -> Self::Node;

    /// Returns a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap;
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
    fn hierarchy(&self) -> Cow<'_, portgraph::Hierarchy> {
        Cow::Borrowed(&self.hierarchy)
    }

    #[inline]
    fn base_hugr(&self) -> &Hugr {
        self
    }

    #[inline]
    fn root_node(&self) -> Self::Node {
        self.root.into()
    }

    #[inline]
    fn get_pg_index(&self, node: impl NodeHandle<Self::Node>) -> portgraph::NodeIndex {
        node.node().pg_index()
    }

    #[inline]
    fn get_node(&self, index: portgraph::NodeIndex) -> Self::Node {
        index.into()
    }

    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap {
        static EMPTY: OnceLock<NodeMetadataMap> = OnceLock::new();
        panic_invalid_node(self, node);
        let map = self.metadata.get(node.pg_index()).as_ref();
        map.unwrap_or(EMPTY.get_or_init(Default::default))
    }
}

/// Trait for accessing the mutable internals of a Hugr(Mut).
///
/// Specifically, this trait lets you apply arbitrary modifications that may
/// invalidate the HUGR.
pub trait HugrMutInternals: HugrView {
    /// Set root node of the HUGR.
    ///
    /// This should be an existing node in the HUGR. Most operations use the
    /// root node as a starting point for traversal.
    fn set_root(&mut self, root: Self::Node);

    /// Set the number of ports on a node. This may invalidate the node's `PortIndex`.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_num_ports(&mut self, node: Self::Node, incoming: usize, outgoing: usize);

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
    fn add_ports(&mut self, node: Self::Node, direction: Direction, amount: isize) -> Range<usize>;

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
        node: Self::Node,
        direction: Direction,
        index: usize,
        amount: usize,
    ) -> Range<usize>;

    /// Sets the parent of a node.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If either the node or the parent is not in the graph.
    fn set_parent(&mut self, node: Self::Node, parent: Self::Node);

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
    fn move_after_sibling(&mut self, node: Self::Node, after: Self::Node);

    /// Move a node in the hierarchy to be the prior sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If either node is not in the graph, or if it is a root.
    fn move_before_sibling(&mut self, node: Self::Node, before: Self::Node);

    /// Replace the OpType at node and return the old OpType.
    /// In general this invalidates the ports, which may need to be resized to
    /// match the OpType signature.
    ///
    /// Returns the old OpType.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn replace_op(&mut self, node: Self::Node, op: impl Into<OpType>) -> OpType;

    /// Gets a mutable reference to the optype.
    ///
    /// Changing this may invalidate the ports, which may need to be resized to
    /// match the OpType signature.
    ///
    /// Mutating the root node operation may invalidate the root tag.
    ///
    /// Mutating the module root into a non-module operation will invalidate the hugr.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn optype_mut(&mut self, node: Self::Node) -> &mut OpType;

    /// Returns a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut NodeMetadataMap;

    /// Returns a mutable reference to the extension registry for this HUGR.
    ///
    /// This set contains all extensions required to define the operations and
    /// types in the HUGR.
    fn extensions_mut(&mut self) -> &mut ExtensionRegistry;
}

/// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
impl HugrMutInternals for Hugr {
    fn set_root(&mut self, root: Node) {
        panic_invalid_node(self, root);
        self.root = self.get_pg_index(root);
    }

    #[inline]
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        panic_invalid_node(self, node);
        self.graph
            .set_num_ports(node.pg_index(), incoming, outgoing, |_, _| {})
    }

    fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
        panic_invalid_node(self, node);
        let mut incoming = self.graph.num_inputs(node.pg_index());
        let mut outgoing = self.graph.num_outputs(node.pg_index());
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
        self.graph
            .set_num_ports(node.pg_index(), incoming, outgoing, |_, _| {});
        range
    }

    fn insert_ports(
        &mut self,
        node: Node,
        direction: Direction,
        index: usize,
        amount: usize,
    ) -> Range<usize> {
        panic_invalid_node(self, node);
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
            self.graph.unlink_port(from_port_index);
            for linked_port_index in linked_ports {
                let _ = self
                    .graph
                    .link_ports(to_port_index, linked_port_index)
                    .expect("Ports exist");
            }
        }
        index..index + amount
    }

    fn set_parent(&mut self, node: Node, parent: Node) {
        panic_invalid_node(self, parent);
        panic_invalid_node(self, node);
        self.hierarchy.detach(node.pg_index());
        self.hierarchy
            .push_child(node.pg_index(), parent.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_after_sibling(&mut self, node: Node, after: Node) {
        panic_invalid_non_root(self, node);
        panic_invalid_non_root(self, after);
        self.hierarchy.detach(node.pg_index());
        self.hierarchy
            .insert_after(node.pg_index(), after.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_before_sibling(&mut self, node: Node, before: Node) {
        panic_invalid_non_root(self, node);
        panic_invalid_non_root(self, before);
        self.hierarchy.detach(node.pg_index());
        self.hierarchy
            .insert_before(node.pg_index(), before.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> OpType {
        panic_invalid_node(self, node);
        std::mem::replace(self.optype_mut(node), op.into())
    }

    fn optype_mut(&mut self, node: Self::Node) -> &mut OpType {
        panic_invalid_node(self, node);
        let node = self.get_pg_index(node);
        self.op_types.get_mut(node)
    }

    fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut NodeMetadataMap {
        panic_invalid_node(self, node);
        self.metadata
            .get_mut(node.pg_index())
            .get_or_insert_with(Default::default)
    }

    fn extensions_mut(&mut self) -> &mut ExtensionRegistry {
        &mut self.extensions
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

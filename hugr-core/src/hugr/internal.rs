//! Internal traits, not exposed in the public `hugr` API.

use std::ops::Range;
use std::sync::OnceLock;

use itertools::Itertools;
use portgraph::{LinkMut, LinkView, MultiPortGraph, PortMut, PortOffset, PortView};

use crate::core::HugrNode;
use crate::extension::ExtensionRegistry;
use crate::{Direction, Hugr, Node};

use super::HugrView;
use super::views::{panic_invalid_node, panic_invalid_non_entrypoint};
use super::{NodeMetadataMap, OpType};
use crate::ops::handle::NodeHandle;

/// Trait for accessing the internals of a Hugr(View).
///
/// Specifically, this trait provides access to the underlying portgraph
/// view.
pub trait HugrInternals {
    /// The portgraph graph structure returned by [`HugrInternals::region_portgraph`].
    type RegionPortgraph<'p>: LinkView<LinkEndpoint: Eq, PortOffsetBase = u32> + Clone + 'p
    where
        Self: 'p;

    /// The type of nodes in the Hugr.
    type Node: Copy + Ord + std::fmt::Debug + std::fmt::Display + std::hash::Hash;

    /// A mapping between HUGR nodes and portgraph nodes in the graph returned by
    /// [`HugrInternals::region_portgraph`].
    type RegionPortgraphNodes: PortgraphNodeMap<Self::Node>;

    /// Returns a flat portgraph view of a region in the HUGR, and a mapping between
    /// HUGR nodes and portgraph nodes in the graph.
    //
    // NOTE: Ideally here we would just return `Self::RegionPortgraph<'_>`, but
    // when doing so we are unable to restrict the type to implement petgraph's
    // traits over references (e.g. `&MyGraph : IntoNodeIdentifiers`, which is
    // needed if we want to use petgraph's algorithms on the region graph).
    // This won't be solvable until we do the big petgraph refactor -.-
    // In the meantime, just wrap the portgraph in a `FlatRegion` as needed.
    fn region_portgraph(
        &self,
        parent: Self::Node,
    ) -> (
        portgraph::view::FlatRegion<'_, Self::RegionPortgraph<'_>>,
        Self::RegionPortgraphNodes,
    );

    /// Returns a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap;
}

/// A map between hugr nodes and portgraph nodes in the graph returned by
/// [`HugrInternals::region_portgraph`].
pub trait PortgraphNodeMap<N>: Clone + Sized + std::fmt::Debug {
    /// Returns the portgraph index of a HUGR node in the associated region
    /// graph.
    ///
    /// If the node is not in the region, the result is undefined.
    fn to_portgraph(&self, node: N) -> portgraph::NodeIndex;

    /// Returns the HUGR node for a portgraph node in the associated region
    /// graph.
    ///
    /// If the node is not in the region, the result is undefined.
    #[allow(clippy::wrong_self_convention)]
    fn from_portgraph(&self, node: portgraph::NodeIndex) -> N;
}

/// An identity map between HUGR nodes and portgraph nodes.
#[derive(
    Copy, Clone, Debug, Default, Eq, PartialEq, Hash, PartialOrd, Ord, derive_more::Display,
)]
pub struct DefaultPGNodeMap;

impl PortgraphNodeMap<Node> for DefaultPGNodeMap {
    #[inline]
    fn to_portgraph(&self, node: Node) -> portgraph::NodeIndex {
        node.into_portgraph()
    }

    #[inline]
    fn from_portgraph(&self, node: portgraph::NodeIndex) -> Node {
        node.into()
    }
}

impl<N: HugrNode> PortgraphNodeMap<N> for std::collections::HashMap<N, Node> {
    #[inline]
    fn to_portgraph(&self, node: N) -> portgraph::NodeIndex {
        self[&node].into_portgraph()
    }

    #[inline]
    fn from_portgraph(&self, node: portgraph::NodeIndex) -> N {
        let node = node.into();
        self.iter()
            .find_map(|(&k, &v)| (v == node).then_some(k))
            .expect("Portgraph node not found in map")
    }
}

impl HugrInternals for Hugr {
    type RegionPortgraph<'p>
        = &'p MultiPortGraph<u32, u32, u32>
    where
        Self: 'p;

    type Node = Node;

    type RegionPortgraphNodes = DefaultPGNodeMap;

    #[inline]
    fn region_portgraph(
        &self,
        parent: Self::Node,
    ) -> (
        portgraph::view::FlatRegion<'_, Self::RegionPortgraph<'_>>,
        Self::RegionPortgraphNodes,
    ) {
        let root = parent.into_portgraph();
        let region =
            portgraph::view::FlatRegion::new_without_root(&self.graph, &self.hierarchy, root);
        (region, DefaultPGNodeMap)
    }

    #[inline]
    fn node_metadata_map(&self, node: Self::Node) -> &NodeMetadataMap {
        static EMPTY: OnceLock<NodeMetadataMap> = OnceLock::new();
        panic_invalid_node(self, node);
        let map = self.metadata.get(node.into_portgraph()).as_ref();
        map.unwrap_or(EMPTY.get_or_init(Default::default))
    }
}

/// Trait for accessing the mutable internals of a Hugr(Mut).
///
/// Specifically, this trait lets you apply arbitrary modifications that may
/// invalidate the HUGR.
pub trait HugrMutInternals: HugrView {
    /// Set the node at the root of the HUGR hierarchy.
    ///
    /// Any node not reachable from this root should be manually removed from
    /// the HUGR.
    ///
    /// To set the working entrypoint of the HUGR, use
    /// [`HugrMut::set_entrypoint`][crate::hugr::HugrMut::set_entrypoint]
    /// instead.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_module_root(&mut self, root: Self::Node);

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

    /// Replace the `OpType` at node and return the old `OpType`.
    /// In general this invalidates the ports, which may need to be resized to
    /// match the `OpType` signature.
    ///
    /// Returns the old `OpType`.
    ///
    /// If the module root is set to a non-module operation the hugr will
    /// become invalid.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn replace_op(&mut self, node: Self::Node, op: impl Into<OpType>) -> OpType;

    /// Gets a mutable reference to the optype.
    ///
    /// Changing this may invalidate the ports, which may need to be resized to
    /// match the `OpType` signature.
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
    fn set_module_root(&mut self, root: Node) {
        panic_invalid_node(self, root.node());
        let root = root.into_portgraph();
        self.hierarchy.detach(root);
        self.module_root = root;
    }

    #[inline]
    fn set_num_ports(&mut self, node: Node, incoming: usize, outgoing: usize) {
        panic_invalid_node(self, node);
        self.graph
            .set_num_ports(node.into_portgraph(), incoming, outgoing, |_, _| {});
    }

    fn add_ports(&mut self, node: Node, direction: Direction, amount: isize) -> Range<usize> {
        panic_invalid_node(self, node);
        let mut incoming = self.graph.num_inputs(node.into_portgraph());
        let mut outgoing = self.graph.num_outputs(node.into_portgraph());
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
            .set_num_ports(node.into_portgraph(), incoming, outgoing, |_, _| {});
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
        let old_num_ports = self.graph.num_ports(node.into_portgraph(), direction);

        self.add_ports(node, direction, amount as isize);

        for swap_from_port in (index..old_num_ports).rev() {
            let swap_to_port = swap_from_port + amount;
            let [from_port_index, to_port_index] = [swap_from_port, swap_to_port].map(|p| {
                self.graph
                    .port_index(node.into_portgraph(), PortOffset::new(direction, p))
                    .unwrap()
            });
            let linked_ports = self
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
        self.hierarchy.detach(node.into_portgraph());
        self.hierarchy
            .push_child(node.into_portgraph(), parent.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_after_sibling(&mut self, node: Node, after: Node) {
        panic_invalid_non_entrypoint(self, node);
        panic_invalid_non_entrypoint(self, after);
        self.hierarchy.detach(node.into_portgraph());
        self.hierarchy
            .insert_after(node.into_portgraph(), after.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn move_before_sibling(&mut self, node: Node, before: Node) {
        panic_invalid_non_entrypoint(self, node);
        panic_invalid_non_entrypoint(self, before);
        self.hierarchy.detach(node.into_portgraph());
        self.hierarchy
            .insert_before(node.into_portgraph(), before.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    fn replace_op(&mut self, node: Node, op: impl Into<OpType>) -> OpType {
        panic_invalid_node(self, node);
        std::mem::replace(self.optype_mut(node), op.into())
    }

    fn optype_mut(&mut self, node: Node) -> &mut OpType {
        panic_invalid_node(self, node);
        let node = node.into_portgraph();
        self.op_types.get_mut(node)
    }

    fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut NodeMetadataMap {
        panic_invalid_node(self, node);
        self.metadata
            .get_mut(node.into_portgraph())
            .get_or_insert_with(Default::default)
    }

    fn extensions_mut(&mut self) -> &mut ExtensionRegistry {
        &mut self.extensions
    }
}

impl Hugr {
    /// Consumes the HUGR and return a flat portgraph view of the region rooted
    /// at `parent`.
    #[inline]
    pub fn into_region_portgraph(
        self,
        parent: Node,
    ) -> portgraph::view::FlatRegion<'static, MultiPortGraph<u32, u32, u32>> {
        let root = parent.into_portgraph();
        let Self {
            graph, hierarchy, ..
        } = self;
        portgraph::view::FlatRegion::new_without_root(graph, hierarchy, root)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Direction, HugrView as _,
        builder::{Container, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::Noop,
        hugr::internal::HugrMutInternals as _,
        ops::handle::NodeHandle,
        types::{Signature, Type},
    };

    #[test]
    fn insert_ports() {
        let (nop, mut hugr) = {
            let mut builder = DFGBuilder::new(Signature::new_endo(Type::UNIT)).unwrap();
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
        let [i, o] = hugr.get_io(hugr.entrypoint()).unwrap();
        assert_eq!(0..2, hugr.insert_ports(nop, Direction::Incoming, 0, 2));
        assert_eq!(1..3, hugr.insert_ports(nop, Direction::Outgoing, 1, 2));

        assert_eq!(hugr.single_linked_input(i, 0), Some((nop, 2.into())));
        assert_eq!(hugr.single_linked_output(o, 0), Some((nop, 0.into())));
        assert_eq!(hugr.single_linked_output(o, 1), Some((nop, 3.into())));
    }
}

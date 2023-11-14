//! Read-only access into HUGR graphs and subgraphs.

pub mod descendants;
pub mod petgraph;
mod root_checked;
pub mod sibling;
pub mod sibling_subgraph;

#[cfg(test)]
mod tests;

use std::iter::Map;

pub use self::petgraph::PetgraphWrapper;
pub use descendants::DescendantsGraph;
pub use root_checked::RootChecked;
pub use sibling::SiblingGraph;
pub use sibling_subgraph::SiblingSubgraph;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::dot::{DotFormat, EdgeStyle, NodeStyle, PortStyle};
use portgraph::{multiportgraph, LinkView, MultiPortGraph, PortView};

use super::{Hugr, HugrError, NodeMetadata, NodeMetadataMap, NodeType, DEFAULT_NODETYPE};
use crate::ops::handle::NodeHandle;
use crate::ops::{FuncDecl, FuncDefn, OpName, OpTag, OpTrait, OpType, DFG};
#[rustversion::since(1.75)] // uses impl in return position
use crate::types::Type;
use crate::types::{EdgeKind, FunctionType};
use crate::{Direction, IncomingPort, Node, OutgoingPort, Port};

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView: sealed::HugrInternals {
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a>: Iterator<Item = Port>
    where
        Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a>: Iterator<Item = Node>
    where
        Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a>: Iterator<Item = (Node, Port)>
    where
        Self: 'a;

    /// Iterator over the links between two nodes.
    type NodeConnections<'a>: Iterator<Item = [Port; 2]>
    where
        Self: 'a;

    /// Return the root node of this view.
    #[inline]
    fn root(&self) -> Node {
        self.root_node()
    }

    /// Return the type of the HUGR root node.
    #[inline]
    fn root_type(&self) -> &NodeType {
        let node_type = self.get_nodetype(self.root());
        // Sadly no way to do this at present
        // debug_assert!(Self::RootHandle::can_hold(node_type.tag()));
        node_type
    }

    /// Returns whether the node exists.
    fn contains_node(&self, node: Node) -> bool;

    /// Validates that a node is valid in the graph.
    ///
    /// Returns a [`HugrError::InvalidNode`] otherwise.
    #[inline]
    fn valid_node(&self, node: Node) -> Result<(), HugrError> {
        match self.contains_node(node) {
            true => Ok(()),
            false => Err(HugrError::InvalidNode(node)),
        }
    }

    /// Validates that a node is a valid root descendant in the graph.
    ///
    /// To include the root node use [`HugrView::valid_node`] instead.
    ///
    /// Returns a [`HugrError::InvalidNode`] otherwise.
    #[inline]
    fn valid_non_root(&self, node: Node) -> Result<(), HugrError> {
        match self.root() == node {
            true => Err(HugrError::InvalidNode(node)),
            false => self.valid_node(node),
        }
    }

    /// Returns the parent of a node.
    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        self.valid_non_root(node).ok()?;
        self.base_hugr()
            .hierarchy
            .parent(node.pg_index())
            .map(Into::into)
    }

    /// Returns the operation type of a node.
    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        &self.get_nodetype(node).op
    }

    /// Returns the type of a node.
    #[inline]
    fn get_nodetype(&self, node: Node) -> &NodeType {
        match self.contains_node(node) {
            true => self.base_hugr().op_types.get(node.pg_index()),
            false => &DEFAULT_NODETYPE,
        }
    }

    /// Returns the metadata associated with a node.
    #[inline]
    fn get_metadata(&self, node: Node, key: impl AsRef<str>) -> Option<&NodeMetadata> {
        match self.contains_node(node) {
            true => self.get_node_metadata(node)?.get(key.as_ref()),
            false => None,
        }
    }

    /// Retrieve the complete metadata map for a node.
    fn get_node_metadata(&self, node: Node) -> Option<&NodeMetadataMap> {
        self.valid_node(node).ok()?;
        self.base_hugr().metadata.get(node.pg_index()).as_ref()
    }

    /// Returns the number of nodes in the hugr.
    fn node_count(&self) -> usize;

    /// Returns the number of edges in the hugr.
    fn edge_count(&self) -> usize;

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> Self::Nodes<'_>;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_>;

    /// Iterator over output ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`
    /// but preserves knowledge that the ports are [OutgoingPort]s.
    #[inline]
    fn node_outputs(&self, node: Node) -> OutgoingPorts<Self::NodePorts<'_>> {
        self.node_ports(node, Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Iterator over inputs ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`
    /// but preserves knowledge that the ports are [IncomingPort]s.
    #[inline]
    fn node_inputs(&self, node: Node) -> IncomingPorts<Self::NodePorts<'_>> {
        self.node_ports(node, Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Iterator over both the input and output ports of node.
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_>;

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(&self, node: Node, port: impl Into<Port>) -> Self::PortLinks<'_>;

    #[rustversion::since(1.75)] // uses impl in return position
    /// Iterator over all the nodes and ports connected to a node's inputs..
    fn all_linked_outputs(&self, node: Node) -> impl Iterator<Item = (Node, OutgoingPort)> {
        self.node_inputs(node)
            .flat_map(move |port| self.linked_outputs(node, port))
    }

    #[rustversion::since(1.75)] // uses impl in return position
    /// Iterator over all the nodes and ports connected to a node's outputs..
    fn all_linked_inputs(&self, node: Node) -> impl Iterator<Item = (Node, IncomingPort)> {
        self.node_outputs(node)
            .flat_map(move |port| self.linked_inputs(node, port))
    }

    /// If there is exactly one OutgoingPort connected to this IncomingPort, return
    /// it and its node.
    fn single_source(
        &self,
        node: Node,
        port: impl Into<IncomingPort>,
    ) -> Option<(Node, OutgoingPort)> {
        self.linked_ports(node, port.into())
            .exactly_one()
            .ok()
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// If there is exactly one IncomingPort connected to this OutgoingPort, return
    /// it and its node.
    fn single_target(
        &self,
        node: Node,
        port: impl Into<OutgoingPort>,
    ) -> Option<(Node, IncomingPort)> {
        self.linked_ports(node, port.into())
            .exactly_one()
            .ok()
            .map(|(n, p)| (n, p.as_incoming().unwrap()))
    }
    /// Iterator over the nodes and output ports connected to a given *input* port.
    /// Like [`linked_ports`][HugrView::linked_ports] but preserves knowledge
    /// that the linked ports are [OutgoingPort]s.
    fn linked_outputs(
        &self,
        node: Node,
        port: impl Into<IncomingPort>,
    ) -> OutgoingNodePorts<Self::PortLinks<'_>> {
        self.linked_ports(node, port.into())
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// Iterator over the nodes and input ports connected to a given *output* port
    /// Like [`linked_ports`][HugrView::linked_ports] but preserves knowledge
    /// that the linked ports are [IncomingPort]s.
    fn linked_inputs(
        &self,
        node: Node,
        port: impl Into<OutgoingPort>,
    ) -> IncomingNodePorts<Self::PortLinks<'_>> {
        self.linked_ports(node, port.into())
            .map(|(n, p)| (n, p.as_incoming().unwrap()))
    }

    /// Iterator the links between two nodes.
    fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_>;

    /// Returns whether a port is connected.
    fn is_linked(&self, node: Node, port: impl Into<Port>) -> bool {
        self.linked_ports(node, port).next().is_some()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Node, dir: Direction) -> usize;

    /// Number of inputs to a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn num_inputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Incoming)
    }

    /// Number of outputs from a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn num_outputs(&self, node: Node) -> usize {
        self.num_ports(node, Direction::Outgoing)
    }

    /// Return iterator over children of node.
    fn children(&self, node: Node) -> Self::Children<'_>;

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_>;

    /// Iterates over the input neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Incoming)`.
    #[inline]
    fn input_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Incoming)
    }

    /// Iterates over the output neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Outgoing)`.
    #[inline]
    fn output_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.neighbours(node, Direction::Outgoing)
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_>;

    /// Get the input and output child nodes of a dataflow parent.
    /// If the node isn't a dataflow parent, then return None
    #[inline]
    fn get_io(&self, node: Node) -> Option<[Node; 2]> {
        let op = self.get_nodetype(node);
        // Nodes outside the view have no children (and a non-DataflowParent NodeType::default())
        if OpTag::DataflowParent.is_superset(op.tag()) {
            self.children(node).take(2).collect_vec().try_into().ok()
        } else {
            None
        }
    }

    /// For function-like HUGRs (DFG, FuncDefn, FuncDecl), report the function
    /// type. Otherwise return None.
    fn get_function_type(&self) -> Option<&FunctionType> {
        let op = self.get_nodetype(self.root());
        match &op.op {
            OpType::DFG(DFG { signature })
            | OpType::FuncDecl(FuncDecl { signature, .. })
            | OpType::FuncDefn(FuncDefn { signature, .. }) => Some(signature),
            _ => None,
        }
    }

    /// Return a wrapper over the view that can be used in petgraph algorithms.
    #[inline]
    fn as_petgraph(&self) -> PetgraphWrapper<'_, Self>
    where
        Self: Sized,
    {
        PetgraphWrapper { hugr: self }
    }

    /// Return dot string showing underlying graph and hierarchy side by side.
    fn dot_string(&self) -> String {
        let hugr = self.base_hugr();
        let graph = self.portgraph();
        graph
            .dot_format()
            .with_hierarchy(&hugr.hierarchy)
            .with_node_style(|n| {
                NodeStyle::Box(format!(
                    "({ni}) {name}",
                    ni = n.index(),
                    name = self.get_optype(n.into()).name()
                ))
            })
            .with_port_style(|port| {
                let node = graph.port_node(port).unwrap();
                let optype = self.get_optype(node.into());
                let offset = graph.port_offset(port).unwrap();
                match optype.port_kind(offset).unwrap() {
                    EdgeKind::Static(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::Value(ty) => {
                        PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
                    }
                    EdgeKind::StateOrder => match graph.port_links(port).count() > 0 {
                        true => PortStyle::text("", false),
                        false => PortStyle::Hidden,
                    },
                    _ => PortStyle::text("", true),
                }
            })
            .with_edge_style(|src, tgt| {
                let src_node = graph.port_node(src).unwrap();
                let src_optype = self.get_optype(src_node.into());
                let src_offset = graph.port_offset(src).unwrap();
                let tgt_node = graph.port_node(tgt).unwrap();

                if hugr.hierarchy.parent(src_node) != hugr.hierarchy.parent(tgt_node) {
                    EdgeStyle::Dashed
                } else if src_optype.port_kind(src_offset) == Some(EdgeKind::StateOrder) {
                    EdgeStyle::Dotted
                } else {
                    EdgeStyle::Solid
                }
            })
            .finish()
    }

    /// If a node has a static input, return the source node.
    fn static_source(&self, node: Node) -> Option<Node> {
        self.linked_outputs(node, self.get_optype(node).static_input_port()?)
            .next()
            .map(|(n, _)| n)
    }

    #[rustversion::since(1.75)] // uses impl in return position
    /// If a node has a static output, return the targets.
    fn static_targets(&self, node: Node) -> Option<impl Iterator<Item = (Node, IncomingPort)>> {
        Some(self.linked_inputs(node, self.get_optype(node).static_output_port()?))
    }

    /// Get the "signature" (incoming and outgoing types) of a node, non-Value
    /// kind edges will be missing.
    fn signature(&self, node: Node) -> Option<FunctionType> {
        self.get_optype(node).signature()
    }

    #[rustversion::since(1.75)] // uses impl in return position
    /// Iterator over all ports in a given direction that have Value type, along
    /// with corresponding types.
    fn value_types(&self, node: Node, dir: Direction) -> impl Iterator<Item = (Port, Type)> {
        let sig = self.signature(node).unwrap_or_default();
        self.node_ports(node, dir)
            .flat_map(move |port| sig.port_type(port).map(|typ| (port, typ.clone())))
    }

    #[rustversion::since(1.75)] // uses impl in return position
    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn in_value_types(&self, node: Node) -> impl Iterator<Item = (IncomingPort, Type)> {
        self.value_types(node, Direction::Incoming)
            .map(|(p, t)| (p.as_incoming().unwrap(), t))
    }

    #[rustversion::since(1.75)] // uses impl in return position
    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn out_value_types(&self, node: Node) -> impl Iterator<Item = (OutgoingPort, Type)> {
        self.value_types(node, Direction::Outgoing)
            .map(|(p, t)| (p.as_outgoing().unwrap(), t))
    }
}

/// Wraps an iterator over [Port]s that are known to be [OutgoingPort]s
pub type OutgoingPorts<I> = Map<I, fn(Port) -> OutgoingPort>;

/// Wraps an iterator over [Port]s that are known to be [IncomingPort]s
pub type IncomingPorts<I> = Map<I, fn(Port) -> IncomingPort>;

/// Wraps an iterator over `(`[`Node`],[`Port`]`)` when the ports are known to be [OutgoingPort]s
pub type OutgoingNodePorts<I> = Map<I, fn((Node, Port)) -> (Node, OutgoingPort)>;

/// Wraps an iterator over `(`[`Node`],[`Port`]`)` when the ports are known to be [IncomingPort]s
pub type IncomingNodePorts<I> = Map<I, fn((Node, Port)) -> (Node, IncomingPort)>;

/// Trait for views that provides a guaranteed bound on the type of the root node.
pub trait RootTagged: HugrView {
    /// The kind of handle that can be used to refer to the root node.
    ///
    /// The handle is guaranteed to be able to contain the operation returned by
    /// [`HugrView::root_type`].
    type RootHandle: NodeHandle;
}

/// A common trait for views of a HUGR hierarchical subgraph.
pub trait HierarchyView<'a>: RootTagged + Sized {
    /// Create a hierarchical view of a HUGR given a root node.
    ///
    /// # Errors
    /// Returns [`HugrError::InvalidTag`] if the root isn't a node of the required [OpTag]
    fn try_new(hugr: &'a impl HugrView, root: Node) -> Result<Self, HugrError>;
}

fn check_tag<Required: NodeHandle>(hugr: &impl HugrView, node: Node) -> Result<(), HugrError> {
    hugr.valid_node(node)?;
    let actual = hugr.get_optype(node).tag();
    let required = Required::TAG;
    if !required.is_superset(actual) {
        return Err(HugrError::InvalidTag { required, actual });
    }
    Ok(())
}

impl RootTagged for Hugr {
    type RootHandle = Node;
}

impl RootTagged for &Hugr {
    type RootHandle = Node;
}

impl RootTagged for &mut Hugr {
    type RootHandle = Node;
}

impl<T: AsRef<Hugr>> HugrView for T {
    /// An Iterator over the nodes in a Hugr(View)
    type Nodes<'a> = MapInto<multiportgraph::Nodes<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) ports of a node
    type NodePorts<'a> = MapInto<portgraph::portgraph::NodePortOffsets, Port> where Self: 'a;

    /// An Iterator over the children of a node
    type Children<'a> = MapInto<portgraph::hierarchy::Children<'a>, Node> where Self: 'a;

    /// An Iterator over (some or all) the nodes neighbouring a node
    type Neighbours<'a> = MapInto<multiportgraph::Neighbours<'a>, Node> where Self: 'a;

    /// Iterator over the children of a node
    type PortLinks<'a> = MapWithCtx<multiportgraph::PortLinks<'a>, &'a Hugr, (Node, Port)>
    where
        Self: 'a;

    type NodeConnections<'a> = MapWithCtx<multiportgraph::NodeConnections<'a>,&'a Hugr, [Port; 2]> where Self: 'a;

    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.as_ref().graph.contains_node(node.pg_index())
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.as_ref().graph.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.as_ref().graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> Self::Nodes<'_> {
        self.as_ref().graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> Self::NodePorts<'_> {
        self.as_ref()
            .graph
            .port_offsets(node.pg_index(), dir)
            .map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> Self::NodePorts<'_> {
        self.as_ref()
            .graph
            .all_port_offsets(node.pg_index())
            .map_into()
    }

    #[inline]
    fn linked_ports(&self, node: Node, port: impl Into<Port>) -> Self::PortLinks<'_> {
        let port = port.into();
        let hugr = self.as_ref();
        let port = hugr
            .graph
            .port_index(node.pg_index(), port.pg_offset())
            .unwrap();
        hugr.graph
            .port_links(port)
            .with_context(hugr)
            .map_with_context(|(_, link), hugr| {
                let port = link.port();
                let node = hugr.graph.port_node(port).unwrap();
                let offset = hugr.graph.port_offset(port).unwrap();
                (node.into(), offset.into())
            })
    }

    #[inline]
    fn node_connections(&self, node: Node, other: Node) -> Self::NodeConnections<'_> {
        let hugr = self.as_ref();

        hugr.graph
            .get_connections(node.pg_index(), other.pg_index())
            .with_context(hugr)
            .map_with_context(|(p1, p2), hugr| {
                [p1, p2].map(|link| {
                    let offset = hugr.graph.port_offset(link.port()).unwrap();
                    offset.into()
                })
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.as_ref().graph.num_ports(node.pg_index(), dir)
    }

    #[inline]
    fn children(&self, node: Node) -> Self::Children<'_> {
        self.as_ref().hierarchy.children(node.pg_index()).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> Self::Neighbours<'_> {
        self.as_ref()
            .graph
            .neighbours(node.pg_index(), dir)
            .map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> Self::Neighbours<'_> {
        self.as_ref()
            .graph
            .all_neighbours(node.pg_index())
            .map_into()
    }
}

/// Filter an iterator of node-ports to only dataflow dependency specifying
/// ports (Value and StateOrder)
pub fn dataflow_ports_only<'i, 'a: 'i, P: Into<Port> + Copy>(
    hugr: &'a impl HugrView,
    it: impl Iterator<Item = (Node, P)> + 'i,
) -> impl Iterator<Item = (Node, P)> + 'i {
    it.filter(move |(n, p)| {
        matches!(
            hugr.get_optype(*n).port_kind(*p),
            Some(EdgeKind::Value(_) | EdgeKind::StateOrder)
        )
    })
}
pub(crate) mod sealed {
    use super::*;

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

    impl<T: AsRef<Hugr>> HugrInternals for T {
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
}

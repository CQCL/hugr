//! Read-only access into HUGR graphs and subgraphs.

pub mod descendants;
pub mod petgraph;
pub mod render;
mod root_checked;
pub mod sibling;
pub mod sibling_subgraph;

#[cfg(test)]
mod tests;

use std::iter::Map;

pub use self::petgraph::PetgraphWrapper;
use self::render::RenderConfig;
pub use descendants::DescendantsGraph;
pub use root_checked::RootChecked;
pub use sibling::SiblingGraph;
pub use sibling_subgraph::SiblingSubgraph;

use context_iterators::{ContextIterator, IntoContextIterator, MapWithCtx};
use itertools::{Itertools, MapInto};
use portgraph::render::{DotFormat, MermaidFormat};
use portgraph::{multiportgraph, LinkView, PortView};

use super::internal::HugrInternals;
use super::{
    Hugr, HugrError, NodeMetadata, NodeMetadataMap, NodeType, ValidationError, DEFAULT_NODETYPE,
};
use crate::extension::ExtensionRegistry;
use crate::ops::handle::NodeHandle;
use crate::ops::{OpParent, OpTag, OpTrait, OpType};

use crate::types::{EdgeKind, FunctionType, Signature};
use crate::types::{PolyFuncType, Type};
use crate::{Direction, IncomingPort, Node, OutgoingPort, Port};

use itertools::Either;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView: HugrInternals {
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
    #[inline]
    fn valid_node(&self, node: Node) -> bool {
        self.contains_node(node)
    }

    /// Validates that a node is a valid root descendant in the graph.
    ///
    /// To include the root node use [`HugrView::valid_node`] instead.
    #[inline]
    fn valid_non_root(&self, node: Node) -> bool {
        self.root() != node && self.valid_node(node)
    }

    /// Returns the parent of a node.
    #[inline]
    fn get_parent(&self, node: Node) -> Option<Node> {
        if !self.valid_non_root(node) {
            return None;
        };
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
        if !self.valid_node(node) {
            return None;
        }
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

    /// Iterator over all the nodes and ports connected to a node in a given direction.
    fn all_linked_ports(
        &self,
        node: Node,
        dir: Direction,
    ) -> Either<
        impl Iterator<Item = (Node, OutgoingPort)>,
        impl Iterator<Item = (Node, IncomingPort)>,
    > {
        match dir {
            Direction::Incoming => Either::Left(
                self.node_inputs(node)
                    .flat_map(move |port| self.linked_outputs(node, port)),
            ),
            Direction::Outgoing => Either::Right(
                self.node_outputs(node)
                    .flat_map(move |port| self.linked_inputs(node, port)),
            ),
        }
    }

    /// Iterator over all the nodes and ports connected to a node's inputs.
    fn all_linked_outputs(&self, node: Node) -> impl Iterator<Item = (Node, OutgoingPort)> {
        self.all_linked_ports(node, Direction::Incoming)
            .left()
            .unwrap()
    }

    /// Iterator over all the nodes and ports connected to a node's outputs.
    fn all_linked_inputs(&self, node: Node) -> impl Iterator<Item = (Node, IncomingPort)> {
        self.all_linked_ports(node, Direction::Outgoing)
            .right()
            .unwrap()
    }

    /// If there is exactly one port connected to this port, return
    /// it and its node.
    fn single_linked_port(&self, node: Node, port: impl Into<Port>) -> Option<(Node, Port)> {
        self.linked_ports(node, port).exactly_one().ok()
    }

    /// If there is exactly one OutgoingPort connected to this IncomingPort, return
    /// it and its node.
    fn single_linked_output(
        &self,
        node: Node,
        port: impl Into<IncomingPort>,
    ) -> Option<(Node, OutgoingPort)> {
        self.single_linked_port(node, port.into())
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// If there is exactly one IncomingPort connected to this OutgoingPort, return
    /// it and its node.
    fn single_linked_input(
        &self,
        node: Node,
        port: impl Into<OutgoingPort>,
    ) -> Option<(Node, IncomingPort)> {
        self.single_linked_port(node, port.into())
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

    /// Return iterator over the direct children of node.
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

    /// Returns the function type defined by this dataflow HUGR.
    ///
    /// If the root of the Hugr is a
    /// [`DataflowParent`][crate::ops::DataflowParent] operation, report the
    /// signature corresponding to the input and output node of its sibling
    /// graph. Otherwise, returns `None`.
    ///
    /// In contrast to [`get_function_type`][HugrView::get_function_type], this
    /// method always return a concrete [`FunctionType`].
    fn get_df_function_type(&self) -> Option<FunctionType<false>> {
        let op = self.get_optype(self.root());
        op.inner_function_type()
    }

    /// Returns the function type defined by this HUGR.
    ///
    /// For HUGRs with a [`DataflowParent`][crate::ops::DataflowParent] root
    /// operation, report the signature of the inner dataflow sibling graph.
    ///
    /// For HUGRS with a [`FuncDecl`][crate::ops::FuncDecl] or
    /// [`FuncDefn`][crate::ops::FuncDefn] root operation, report the signature
    /// of the function.
    ///
    /// Otherwise, returns `None`.
    fn get_function_type(&self) -> Option<PolyFuncType<false>> {
        let op = self.get_optype(self.root());
        match op {
            OpType::FuncDecl(decl) => Some(decl.signature.clone()),
            OpType::FuncDefn(defn) => Some(defn.signature.clone()),
            _ => op.inner_function_type().map(PolyFuncType::from),
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

    /// Return the mermaid representation of the underlying hierarchical graph.
    ///
    /// The hierarchy is represented using subgraphs. Edges are labelled with
    /// their source and target ports.
    ///
    /// For a more detailed representation, use the [`HugrView::dot_string`]
    /// format instead.
    fn mermaid_string(&self) -> String {
        self.mermaid_string_with_config(RenderConfig {
            node_indices: true,
            port_offsets_in_edges: true,
            type_labels_in_edges: true,
        })
    }

    /// Return the mermaid representation of the underlying hierarchical graph.
    ///
    /// The hierarchy is represented using subgraphs. Edges are labelled with
    /// their source and target ports.
    ///
    /// For a more detailed representation, use the [`HugrView::dot_string`]
    /// format instead.
    fn mermaid_string_with_config(&self, config: RenderConfig) -> String {
        let hugr = self.base_hugr();
        let graph = self.portgraph();
        graph
            .mermaid_format()
            .with_hierarchy(&hugr.hierarchy)
            .with_node_style(render::node_style(self, config))
            .with_edge_style(render::edge_style(self, config))
            .finish()
    }

    /// Return the graphviz representation of the underlying graph and hierarchy side by side.
    ///
    /// For a simpler representation, use the [`HugrView::mermaid_string`] format instead.
    fn dot_string(&self) -> String
    where
        Self: Sized,
    {
        let hugr = self.base_hugr();
        let graph = self.portgraph();
        let config = RenderConfig::default();
        graph
            .dot_format()
            .with_hierarchy(&hugr.hierarchy)
            .with_node_style(render::node_style(self, config))
            .with_port_style(render::port_style(self, config))
            .with_edge_style(render::edge_style(self, config))
            .finish()
    }

    /// If a node has a static input, return the source node.
    fn static_source(&self, node: Node) -> Option<Node> {
        self.linked_outputs(node, self.get_optype(node).static_input_port()?)
            .next()
            .map(|(n, _)| n)
    }

    /// If a node has a static output, return the targets.
    fn static_targets(&self, node: Node) -> Option<impl Iterator<Item = (Node, IncomingPort)>> {
        Some(self.linked_inputs(node, self.get_optype(node).static_output_port()?))
    }

    /// Get the "signature" (incoming and outgoing types) of a node, non-Value
    /// kind ports will be missing.
    fn signature(&self, node: Node) -> Option<Signature> {
        self.get_optype(node).dataflow_signature()
    }

    /// Iterator over all outgoing ports that have Value type, along
    /// with corresponding types.
    fn value_types(&self, node: Node, dir: Direction) -> impl Iterator<Item = (Port, Type)> {
        let sig = self.signature(node).unwrap_or_default();
        self.node_ports(node, dir)
            .flat_map(move |port| sig.port_type(port).map(|typ| (port, typ.clone())))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn in_value_types(&self, node: Node) -> impl Iterator<Item = (IncomingPort, Type)> {
        self.value_types(node, Direction::Incoming)
            .map(|(p, t)| (p.as_incoming().unwrap(), t))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn out_value_types(&self, node: Node) -> impl Iterator<Item = (OutgoingPort, Type)> {
        self.value_types(node, Direction::Outgoing)
            .map(|(p, t)| (p.as_outgoing().unwrap(), t))
    }

    /// Check the validity of the underlying HUGR.
    fn validate(&self, reg: &ExtensionRegistry) -> Result<(), ValidationError> {
        self.base_hugr().validate(reg)
    }

    /// Check the validity of the underlying HUGR, but don't check consistency
    /// of extension requirements between connected nodes or between parents and
    /// children.
    fn validate_no_extensions(&self, reg: &ExtensionRegistry) -> Result<(), ValidationError> {
        self.base_hugr().validate_no_extensions(reg)
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

/// Trait implementing methods on port iterators.
pub trait PortIterator<P>: Iterator<Item = (Node, P)>
where
    P: Into<Port> + Copy,
    Self: Sized,
{
    /// Filter an iterator of node-ports to only dataflow dependency specifying
    /// ports (Value and StateOrder)
    fn dataflow_ports_only(self, hugr: &impl HugrView) -> impl Iterator<Item = (Node, P)> {
        self.filter(move |(n, p)| {
            matches!(
                hugr.get_optype(*n).port_kind(*p),
                Some(EdgeKind::Value(_) | EdgeKind::StateOrder)
            )
        })
    }
}

impl<I, P> PortIterator<P> for I
where
    I: Iterator<Item = (Node, P)>,
    P: Into<Port> + Copy,
{
}

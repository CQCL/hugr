//! Read-only access into HUGR graphs and subgraphs.

pub mod descendants;
mod impls;
pub mod petgraph;
pub mod render;
mod root_checked;
pub mod sibling;
pub mod sibling_subgraph;

#[cfg(test)]
mod tests;

use std::borrow::Cow;

pub use self::petgraph::PetgraphWrapper;
use self::render::RenderConfig;
pub use descendants::DescendantsGraph;
pub use root_checked::{check_tag, RootCheckable, RootChecked};
pub use sibling::SiblingGraph;
pub use sibling_subgraph::SiblingSubgraph;

use itertools::Itertools;
use portgraph::render::{DotFormat, MermaidFormat};
use portgraph::{LinkView, PortView};

use super::internal::HugrInternals;
use super::{
    Hugr, HugrError, HugrMut, Node, NodeMetadata, NodeMetadataMap, ValidationError, DEFAULT_OPTYPE,
};
use crate::extension::ExtensionRegistry;
use crate::ops::{OpParent, OpTag, OpTrait, OpType};

use crate::types::{EdgeKind, PolyFuncType, Signature, Type};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

use itertools::Either;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView: HugrInternals {
    /// Return the root node of this view.
    #[inline]
    fn root(&self) -> Self::Node {
        self.root_node()
    }

    /// Return the type of the HUGR root node.
    #[inline]
    fn root_type(&self) -> &OpType {
        let node_type = self.get_optype(self.root());
        // Sadly no way to do this at present
        // debug_assert!(Self::RootHandle::can_hold(node_type.tag()));
        node_type
    }

    /// Returns whether the node exists.
    fn contains_node(&self, node: Self::Node) -> bool;

    /// Validates that a node is valid in the graph.
    #[inline]
    fn valid_node(&self, node: Self::Node) -> bool {
        self.contains_node(node)
    }

    /// Validates that a node is a valid root descendant in the graph.
    ///
    /// To include the root node use [`HugrView::valid_node`] instead.
    #[inline]
    fn valid_non_root(&self, node: Self::Node) -> bool {
        self.root() != node && self.valid_node(node)
    }

    /// Returns the parent of a node.
    #[inline]
    fn get_parent(&self, node: Self::Node) -> Option<Self::Node> {
        if !self.valid_non_root(node) {
            return None;
        };
        self.base_hugr()
            .hierarchy
            .parent(self.get_pg_index(node))
            .map(|index| self.get_node(index))
    }

    /// Returns the operation type of a node.
    #[inline]
    fn get_optype(&self, node: Self::Node) -> &OpType {
        match self.contains_node(node) {
            true => self.base_hugr().op_types.get(self.get_pg_index(node)),
            false => &DEFAULT_OPTYPE,
        }
    }

    /// Returns the metadata associated with a node.
    #[inline]
    fn get_metadata(&self, node: Self::Node, key: impl AsRef<str>) -> Option<&NodeMetadata> {
        match self.contains_node(node) {
            true => self.get_node_metadata(node)?.get(key.as_ref()),
            false => None,
        }
    }

    /// Retrieve the complete metadata map for a node.
    fn get_node_metadata(&self, node: Self::Node) -> Option<&NodeMetadataMap> {
        if !self.valid_node(node) {
            return None;
        }
        self.base_hugr()
            .metadata
            .get(self.get_pg_index(node))
            .as_ref()
    }

    /// Returns the number of nodes in the hugr.
    fn node_count(&self) -> usize;

    /// Returns the number of edges in the hugr.
    fn edge_count(&self) -> usize;

    /// Iterates over the nodes in the port graph.
    fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = Port> + Clone;

    /// Iterator over output ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`
    /// but preserves knowledge that the ports are [OutgoingPort]s.
    #[inline]
    fn node_outputs(&self, node: Self::Node) -> impl Iterator<Item = OutgoingPort> + Clone {
        self.node_ports(node, Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Iterator over inputs ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`
    /// but preserves knowledge that the ports are [IncomingPort]s.
    #[inline]
    fn node_inputs(&self, node: Self::Node) -> impl Iterator<Item = IncomingPort> + Clone {
        self.node_ports(node, Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Iterator over both the input and output ports of node.
    fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = Port> + Clone;

    /// Iterator over the nodes and ports connected to a port.
    fn linked_ports(
        &self,
        node: Self::Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Self::Node, Port)> + Clone;

    /// Iterator over all the nodes and ports connected to a node in a given direction.
    fn all_linked_ports(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> Either<
        impl Iterator<Item = (Self::Node, OutgoingPort)>,
        impl Iterator<Item = (Self::Node, IncomingPort)>,
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
    fn all_linked_outputs(
        &self,
        node: Self::Node,
    ) -> impl Iterator<Item = (Self::Node, OutgoingPort)> {
        self.all_linked_ports(node, Direction::Incoming)
            .left()
            .unwrap()
    }

    /// Iterator over all the nodes and ports connected to a node's outputs.
    fn all_linked_inputs(
        &self,
        node: Self::Node,
    ) -> impl Iterator<Item = (Self::Node, IncomingPort)> {
        self.all_linked_ports(node, Direction::Outgoing)
            .right()
            .unwrap()
    }

    /// If there is exactly one port connected to this port, return
    /// it and its node.
    fn single_linked_port(
        &self,
        node: Self::Node,
        port: impl Into<Port>,
    ) -> Option<(Self::Node, Port)> {
        self.linked_ports(node, port).exactly_one().ok()
    }

    /// If there is exactly one OutgoingPort connected to this IncomingPort, return
    /// it and its node.
    fn single_linked_output(
        &self,
        node: Self::Node,
        port: impl Into<IncomingPort>,
    ) -> Option<(Self::Node, OutgoingPort)> {
        self.single_linked_port(node, port.into())
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// If there is exactly one IncomingPort connected to this OutgoingPort, return
    /// it and its node.
    fn single_linked_input(
        &self,
        node: Self::Node,
        port: impl Into<OutgoingPort>,
    ) -> Option<(Self::Node, IncomingPort)> {
        self.single_linked_port(node, port.into())
            .map(|(n, p)| (n, p.as_incoming().unwrap()))
    }
    /// Iterator over the nodes and output ports connected to a given *input* port.
    /// Like [`linked_ports`][HugrView::linked_ports] but preserves knowledge
    /// that the linked ports are [OutgoingPort]s.
    fn linked_outputs(
        &self,
        node: Self::Node,
        port: impl Into<IncomingPort>,
    ) -> impl Iterator<Item = (Self::Node, OutgoingPort)> {
        self.linked_ports(node, port.into())
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// Iterator over the nodes and input ports connected to a given *output* port
    /// Like [`linked_ports`][HugrView::linked_ports] but preserves knowledge
    /// that the linked ports are [IncomingPort]s.
    fn linked_inputs(
        &self,
        node: Self::Node,
        port: impl Into<OutgoingPort>,
    ) -> impl Iterator<Item = (Self::Node, IncomingPort)> {
        self.linked_ports(node, port.into())
            .map(|(n, p)| (n, p.as_incoming().unwrap()))
    }

    /// Iterator the links between two nodes.
    fn node_connections(
        &self,
        node: Self::Node,
        other: Self::Node,
    ) -> impl Iterator<Item = [Port; 2]> + Clone;

    /// Returns whether a port is connected.
    fn is_linked(&self, node: Self::Node, port: impl Into<Port>) -> bool {
        self.linked_ports(node, port).next().is_some()
    }

    /// Number of ports in node for a given direction.
    fn num_ports(&self, node: Self::Node, dir: Direction) -> usize;

    /// Number of inputs to a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Incoming)`.
    #[inline]
    fn num_inputs(&self, node: Self::Node) -> usize {
        self.num_ports(node, Direction::Incoming)
    }

    /// Number of outputs from a node.
    /// Shorthand for [`num_ports`][HugrView::num_ports]`(node, Direction::Outgoing)`.
    #[inline]
    fn num_outputs(&self, node: Self::Node) -> usize {
        self.num_ports(node, Direction::Outgoing)
    }

    /// Return iterator over the direct children of node.
    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone;

    /// Returns the first child of the specified node (if it is a parent).
    /// Useful because `x.children().next()` leaves x borrowed.
    fn first_child(&self, node: Self::Node) -> Option<Self::Node> {
        self.children(node).next()
    }

    /// Iterates over neighbour nodes in the given direction.
    /// May contain duplicates if the graph has multiple links between nodes.
    fn neighbours(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Self::Node> + Clone;

    /// Iterates over the input neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Incoming)`.
    #[inline]
    fn input_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        self.neighbours(node, Direction::Incoming)
    }

    /// Iterates over the output neighbours of the `node`.
    /// Shorthand for [`neighbours`][HugrView::neighbours]`(node, Direction::Outgoing)`.
    #[inline]
    fn output_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        self.neighbours(node, Direction::Outgoing)
    }

    /// Iterates over the input and output neighbours of the `node` in sequence.
    fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;

    /// Get the input and output child nodes of a dataflow parent.
    /// If the node isn't a dataflow parent, then return None
    #[inline]
    fn get_io(&self, node: Self::Node) -> Option<[Self::Node; 2]> {
        let op = self.get_optype(node);
        // Nodes outside the view have no children (and a non-DataflowParent OpType::default())
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
    /// In contrast to [`poly_func_type`][HugrView::poly_func_type], this
    /// method always return a concrete [`Signature`].
    fn inner_function_type(&self) -> Option<Cow<'_, Signature>> {
        self.root_type().inner_function_type()
    }

    /// Returns the function type defined by this HUGR, i.e. `Some` iff the root is
    /// a [`FuncDecl`][crate::ops::FuncDecl] or [`FuncDefn`][crate::ops::FuncDefn].
    fn poly_func_type(&self) -> Option<PolyFuncType> {
        match self.root_type() {
            OpType::FuncDecl(decl) => Some(decl.signature.clone()),
            OpType::FuncDefn(defn) => Some(defn.signature.clone()),
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
    fn static_source(&self, node: Self::Node) -> Option<Self::Node> {
        self.linked_outputs(node, self.get_optype(node).static_input_port()?)
            .next()
            .map(|(n, _)| n)
    }

    /// If a node has a static output, return the targets.
    fn static_targets(
        &self,
        node: Self::Node,
    ) -> Option<impl Iterator<Item = (Self::Node, IncomingPort)>> {
        Some(self.linked_inputs(node, self.get_optype(node).static_output_port()?))
    }

    /// Get the "signature" (incoming and outgoing types) of a node, non-Value
    /// kind ports will be missing.
    fn signature(&self, node: Self::Node) -> Option<Cow<'_, Signature>> {
        self.get_optype(node).dataflow_signature()
    }

    /// Iterator over all outgoing ports that have Value type, along
    /// with corresponding types.
    fn value_types(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = (Port, Type)> {
        let sig = self.signature(node).unwrap_or_default();
        self.node_ports(node, dir)
            .flat_map(move |port| sig.port_type(port).map(|typ| (port, typ.clone())))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn in_value_types(&self, node: Self::Node) -> impl Iterator<Item = (IncomingPort, Type)> {
        self.value_types(node, Direction::Incoming)
            .map(|(p, t)| (p.as_incoming().unwrap(), t))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn out_value_types(&self, node: Self::Node) -> impl Iterator<Item = (OutgoingPort, Type)> {
        self.value_types(node, Direction::Outgoing)
            .map(|(p, t)| (p.as_outgoing().unwrap(), t))
    }

    /// Returns the set of extensions used by the HUGR.
    ///
    /// This set may contain extensions that are no longer required by the HUGR.
    fn extensions(&self) -> &ExtensionRegistry {
        &self.base_hugr().extensions
    }

    /// Check the validity of the underlying HUGR.
    ///
    /// This includes checking consistency of extension requirements between
    /// connected nodes and between parents and children.
    /// See [`HugrView::validate_no_extensions`] for a version that doesn't check
    /// extension requirements.
    fn validate(&self) -> Result<(), ValidationError> {
        self.base_hugr().validate()
    }

    /// Check the validity of the underlying HUGR, but don't check consistency
    /// of extension requirements between connected nodes or between parents and
    /// children.
    ///
    /// For a more thorough check, use [`HugrView::validate`].
    fn validate_no_extensions(&self) -> Result<(), ValidationError> {
        self.base_hugr().validate_no_extensions()
    }
}

/// A common trait for views of a HUGR hierarchical subgraph.
pub trait HierarchyView<'a>: HugrView + Sized {
    /// Create a hierarchical view of a HUGR given a root node.
    ///
    /// # Errors
    /// Returns [`HugrError::InvalidTag`] if the root isn't a node of the required [OpTag]
    fn try_new(
        hugr: &'a impl HugrView<Node = Self::Node>,
        root: Self::Node,
    ) -> Result<Self, HugrError>;
}

/// A trait for [`HugrView`]s that can be extracted into a valid HUGR containing
/// only the nodes and edges of the view.
pub trait ExtractHugr: HugrView + Sized {
    /// Extracts the view into an owned HUGR, rooted at the view's root node
    /// and containing only the nodes and edges of the view.
    fn extract_hugr(self) -> Hugr {
        let mut hugr = Hugr::default();
        let old_root = hugr.root();
        let new_root = hugr.insert_from_view(old_root, &self).new_root;
        hugr.set_root(new_root);
        hugr.remove_node(old_root);
        hugr
    }
}

// Explicit implementation to avoid cloning the Hugr.
impl ExtractHugr for Hugr {
    fn extract_hugr(self) -> Hugr {
        self
    }
}

impl ExtractHugr for &Hugr {
    fn extract_hugr(self) -> Hugr {
        self.clone()
    }
}

impl ExtractHugr for &mut Hugr {
    fn extract_hugr(self) -> Hugr {
        self.clone()
    }
}

impl HugrView for Hugr {
    #[inline]
    fn contains_node(&self, node: Node) -> bool {
        self.graph.contains_node(node.pg_index())
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    fn edge_count(&self) -> usize {
        self.graph.link_count()
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = Node> + Clone {
        self.graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        self.graph.port_offsets(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone {
        self.graph.all_port_offsets(node.pg_index()).map_into()
    }

    #[inline]
    fn linked_ports(
        &self,
        node: Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Node, Port)> + Clone {
        let port = port.into();

        let port = self
            .graph
            .port_index(node.pg_index(), port.pg_offset())
            .unwrap();
        self.graph.port_links(port).map(|(_, link)| {
            let port = link.port();
            let node = self.graph.port_node(port).unwrap();
            let offset = self.graph.port_offset(port).unwrap();
            (node.into(), offset.into())
        })
    }

    #[inline]
    fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone {
        self.graph
            .get_connections(node.pg_index(), other.pg_index())
            .map(|(p1, p2)| {
                [p1, p2].map(|link| self.graph.port_offset(link.port()).unwrap().into())
            })
    }

    #[inline]
    fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.graph.num_ports(node.pg_index(), dir)
    }

    #[inline]
    fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone {
        self.hierarchy.children(node.pg_index()).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.graph.neighbours(node.pg_index(), dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.graph.all_neighbours(node.pg_index()).map_into()
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
    fn dataflow_ports_only(
        self,
        hugr: &impl HugrView<Node = Node>,
    ) -> impl Iterator<Item = (Node, P)> {
        self.filter_edge_kind(
            |kind| matches!(kind, Some(EdgeKind::Value(..) | EdgeKind::StateOrder)),
            hugr,
        )
    }

    /// Filter an iterator of node-ports based on the port kind.
    fn filter_edge_kind(
        self,
        predicate: impl Fn(Option<EdgeKind>) -> bool,
        hugr: &impl HugrView<Node = Node>,
    ) -> impl Iterator<Item = (Node, P)> {
        self.filter(move |(n, p)| {
            let kind = hugr.get_optype(*n).port_kind(*p);
            predicate(kind)
        })
    }
}

impl<I, P> PortIterator<P> for I
where
    I: Iterator<Item = (Node, P)>,
    P: Into<Port> + Copy,
{
}

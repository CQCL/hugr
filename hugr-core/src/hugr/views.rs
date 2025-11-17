//! Read-only access into HUGR graphs and subgraphs.

mod impls;
mod nodes_iter;
pub mod petgraph;
pub mod render;
mod rerooted;
mod root_checked;
pub mod sibling_subgraph;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;

pub use self::petgraph::PetgraphWrapper;
#[expect(deprecated)]
use self::render::{MermaidFormatter, RenderConfig};
pub use nodes_iter::NodesIter;
pub use rerooted::Rerooted;
pub use root_checked::{InvalidSignature, RootCheckable, RootChecked, check_tag};
pub use sibling_subgraph::SiblingSubgraph;

use itertools::Itertools;
use portgraph::render::{DotFormat, MermaidFormat};
use portgraph::{LinkView, PortView};

use super::internal::{HugrInternals, HugrMutInternals};
use super::validate::ValidationContext;
use super::{Hugr, HugrMut, Node, NodeMetadata, ValidationError};
use crate::core::HugrNode;
use crate::extension::ExtensionRegistry;
use crate::ops::handle::NodeHandle;
use crate::ops::{OpParent, OpTag, OpTrait, OpType};

use crate::types::{EdgeKind, PolyFuncType, Signature, Type};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

use itertools::Either;

/// A trait for inspecting HUGRs.
/// For end users we intend this to be superseded by region-specific APIs.
pub trait HugrView: HugrInternals {
    /// The distinguished node from where operations are applied, commonly
    /// defining a region of interest.
    ///
    /// This node represents the execution entrypoint of the HUGR. When running
    /// local graph analysis or optimizations, the region defined under this
    /// node will be used as the starting point.
    fn entrypoint(&self) -> Self::Node;

    /// Returns the operation type of the entrypoint node.
    #[inline]
    fn entrypoint_optype(&self) -> &OpType {
        self.get_optype(self.entrypoint())
    }

    /// An operation tag that is guaranteed to represent the
    /// [`HugrView::entrypoint`] node operation.
    ///
    /// The specificity of the tag may vary depending on the HUGR view.
    /// [`OpTag::Any`] may be returned for any node, but more specific tags may
    /// be used instead.
    ///
    /// The tag returned may vary if the entrypoint node's operation is modified,
    /// or if the entrypoint node is replaced with another node.
    #[inline]
    fn entrypoint_tag(&self) -> OpTag {
        self.entrypoint_optype().tag()
    }

    /// Returns a non-mutable view of the HUGR with a different entrypoint.
    ///
    /// For a mutable view, use [`HugrMut::with_entrypoint_mut`] instead.
    ///
    /// # Panics
    ///
    /// Panics if the entrypoint node is not valid in the HUGR.
    fn with_entrypoint(&self, entrypoint: Self::Node) -> Rerooted<&Self>
    where
        Self: Sized,
    {
        Rerooted::new(self, entrypoint)
    }

    /// A pointer to the module region defined at the root of the HUGR.
    ///
    /// This node is the root node of the node hierarchy. It is the ancestor of
    /// all other nodes in the HUGR.
    ///
    /// Operations applied to a hugr normally start at the
    /// [`HugrView::entrypoint`] instead.
    fn module_root(&self) -> Self::Node;

    /// Returns `true` if the node exists in the HUGR.
    fn contains_node(&self, node: Self::Node) -> bool;

    /// Returns the parent of a node.
    fn get_parent(&self, node: Self::Node) -> Option<Self::Node>;

    /// Returns the metadata associated with a node.
    #[inline]
    fn get_metadata(&self, node: Self::Node, key: impl AsRef<str>) -> Option<&NodeMetadata> {
        if self.contains_node(node) {
            self.node_metadata_map(node).get(key.as_ref())
        } else {
            None
        }
    }

    /// Returns the operation type of a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn get_optype(&self, node: Self::Node) -> &OpType;

    /// Returns the number of nodes in the HUGR.
    fn num_nodes(&self) -> usize;

    /// Returns the number of edges in the HUGR.
    fn num_edges(&self) -> usize;

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

    /// Iterates over the all the nodes in the HUGR.
    ///
    /// This iterator returns every node in the HUGR. In most cases, you will
    /// want to use [`HugrView::entry_descendants`] instead to get the nodes
    /// that are reachable from the entrypoint.
    ///
    /// See also [`HugrView::descendants`] and [`HugrView::children`] for more
    /// general iterators.
    fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone;

    /// Iterator over ports of node in a given direction.
    fn node_ports(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = Port> + Clone;

    /// Iterator over output ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Outgoing)`
    /// but preserves knowledge that the ports are [`OutgoingPort`]s.
    #[inline]
    fn node_outputs(&self, node: Self::Node) -> impl Iterator<Item = OutgoingPort> + Clone {
        self.node_ports(node, Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Iterator over inputs ports of node.
    /// Like [`node_ports`][HugrView::node_ports]`(node, Direction::Incoming)`
    /// but preserves knowledge that the ports are [`IncomingPort`]s.
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

    /// If there is exactly one `OutgoingPort` connected to this `IncomingPort`, return
    /// it and its node.
    fn single_linked_output(
        &self,
        node: Self::Node,
        port: impl Into<IncomingPort>,
    ) -> Option<(Self::Node, OutgoingPort)> {
        self.single_linked_port(node, port.into())
            .map(|(n, p)| (n, p.as_outgoing().unwrap()))
    }

    /// If there is exactly one `IncomingPort` connected to this `OutgoingPort`, return
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
    /// that the linked ports are [`OutgoingPort`]s.
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
    /// that the linked ports are [`IncomingPort`]s.
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

    /// Returns an iterator over the direct children of node.
    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone;

    /// Returns an iterator over all the descendants of a node,
    /// including the node itself.
    ///
    /// Yields the node itself first, followed by its children in breath-first order.
    fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;

    /// Returns an iterator over all the descendants of the hugr entrypoint,
    /// including the node itself.
    ///
    /// Yields the node itself first, followed by its children in breath-first order.
    fn entry_descendants(&self) -> impl Iterator<Item = Self::Node> + Clone {
        self.descendants(self.entrypoint())
    }

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
        self.entrypoint_optype().inner_function_type()
    }

    /// Returns the function type defined by this HUGR, i.e. `Some` iff the root is
    /// a [`FuncDecl`][crate::ops::FuncDecl] or [`FuncDefn`][crate::ops::FuncDefn].
    fn poly_func_type(&self) -> Option<PolyFuncType> {
        match self.entrypoint_optype() {
            OpType::FuncDecl(decl) => Some(decl.signature().clone()),
            OpType::FuncDefn(defn) => Some(defn.signature().clone()),
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
        self.mermaid_string_with_formatter(self.mermaid_format())
    }

    /// Return the mermaid representation of the underlying hierarchical graph.
    ///
    /// The hierarchy is represented using subgraphs. Edges are labelled with
    /// their source and target ports.
    ///
    /// For a more detailed representation, use the [`HugrView::dot_string`]
    /// format instead.
    #[deprecated(note = "Use `mermaid_format` instead", since = "0.20.2")]
    #[expect(deprecated)]
    fn mermaid_string_with_config(&self, config: RenderConfig<Self::Node>) -> String;

    /// Return the mermaid representation of the underlying hierarchical graph
    /// according to the provided [`MermaidFormatter`] formatting options.
    ///
    /// The hierarchy is represented using subgraphs. Edges are labelled with
    /// their source and target ports.
    ///
    /// For a more detailed representation, use the [`HugrView::dot_string`]
    /// format instead.
    ///
    /// ## Deprecation of [`RenderConfig`]
    /// While the deprecated [HugrView::mermaid_string_with_config] exists, this
    /// will by default try to convert the formatter options to a [`RenderConfig`],
    /// but this may panic if the configuration is not supported. Users are
    /// encouraged to provide an implementation of this method overriding the default
    /// and no longer rely on [HugrView::mermaid_string_with_config].
    fn mermaid_string_with_formatter(&self, formatter: MermaidFormatter<Self>) -> String {
        #[expect(deprecated)]
        let config = match RenderConfig::try_from(formatter) {
            Ok(config) => config,
            Err(e) => {
                panic!("Unsupported format option: {e}");
            }
        };
        #[expect(deprecated)]
        self.mermaid_string_with_config(config)
    }

    /// Construct a mermaid representation of the underlying hierarchical graph.
    ///
    /// Options can be set on the returned [`MermaidFormatter`] struct, before
    /// generating the String with [`MermaidFormatter::finish`].
    ///
    /// The hierarchy is represented using subgraphs. Edges are labelled with
    /// their source and target ports.
    ///
    /// For a more detailed representation, use the [`HugrView::dot_string`]
    /// format instead.
    fn mermaid_format(&self) -> MermaidFormatter<'_, Self> {
        MermaidFormatter::new(self).with_entrypoint(self.entrypoint())
    }

    /// Return the graphviz representation of the underlying graph and hierarchy side by side.
    ///
    /// For a simpler representation, use the [`HugrView::mermaid_string`] format instead.
    fn dot_string(&self) -> String
    where
        Self: Sized;

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
            .filter_map(move |port| sig.port_type(port).map(|typ| (port, typ.clone())))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    fn in_value_types(&self, node: Self::Node) -> impl Iterator<Item = (IncomingPort, Type)> {
        self.value_types(node, Direction::Incoming)
            .map(|(p, t)| (p.as_incoming().unwrap(), t))
    }

    /// Iterator over all outgoing ports that have Value type, along
    /// with corresponding types.
    fn out_value_types(&self, node: Self::Node) -> impl Iterator<Item = (OutgoingPort, Type)> {
        self.value_types(node, Direction::Outgoing)
            .map(|(p, t)| (p.as_outgoing().unwrap(), t))
    }

    /// Returns the set of extensions used by the HUGR.
    ///
    /// This set contains all extensions required to define the operations and
    /// types in the HUGR.
    fn extensions(&self) -> &ExtensionRegistry;

    /// Check the validity of the underlying HUGR.
    fn validate(&self) -> Result<(), ValidationError<Self::Node>>
    where
        Self: Sized,
    {
        let mut validator = ValidationContext::new(self);
        validator.validate()
    }

    /// Extracts a HUGR containing the parent node and all its descendants.
    ///
    /// Returns a new HUGR and a map from the nodes in the source HUGR to the
    /// nodes in the extracted HUGR. The new HUGR entrypoint corresponds to the
    /// extracted `parent` node.
    ///
    /// Edges that connected to nodes outside the parent node are not
    /// included in the new HUGR.
    ///
    /// If the parent is not a module, the returned HUGR will contain some
    /// additional nodes to contain the new entrypoint. E.g. if the optype must
    /// be contained in a dataflow region, a module with a function definition
    /// will be created to contain it.
    fn extract_hugr(
        &self,
        parent: Self::Node,
    ) -> (Hugr, impl ExtractionResult<Self::Node> + 'static);
}

/// Records the result of extracting a Hugr via [`HugrView::extract_hugr`].
///
/// Contains a map from the nodes in the source HUGR to the nodes in the extracted
/// HUGR, using their respective `Node` types.
pub trait ExtractionResult<SourceN> {
    /// Returns the node in the extracted HUGR that corresponds to the given
    /// node in the source HUGR.
    ///
    /// If the source node was not a descendant of the entrypoint, the result
    /// is undefined.
    fn extracted_node(&self, node: SourceN) -> Node;
}

/// A node map that defaults to the identity function if the node is not found.
struct DefaultNodeMap(HashMap<Node, Node>);

impl ExtractionResult<Node> for DefaultNodeMap {
    #[inline]
    fn extracted_node(&self, node: Node) -> Node {
        self.0.get(&node).copied().unwrap_or(node)
    }
}

impl<S: HugrNode> ExtractionResult<S> for HashMap<S, Node> {
    #[inline]
    fn extracted_node(&self, node: S) -> Node {
        self[&node]
    }
}

impl HugrView for Hugr {
    #[inline]
    fn entrypoint(&self) -> Self::Node {
        self.entrypoint.into()
    }

    #[inline]
    fn module_root(&self) -> Self::Node {
        let node: Self::Node = self.module_root.into();
        let handle = node.try_cast();
        debug_assert!(
            handle.is_some(),
            "The root node in a HUGR must be a module."
        );
        handle.unwrap()
    }

    #[inline]
    fn contains_node(&self, node: Self::Node) -> bool {
        self.graph.contains_node(node.into_portgraph())
    }

    #[inline]
    fn get_parent(&self, node: Self::Node) -> Option<Self::Node> {
        if !check_valid_non_root(self, node) {
            return None;
        }
        self.hierarchy.parent(node.into_portgraph()).map(Into::into)
    }

    #[inline]
    fn get_optype(&self, node: Node) -> &OpType {
        panic_invalid_node(self, node);
        self.op_types.get(node.into_portgraph())
    }

    #[inline]
    fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    #[inline]
    fn num_edges(&self) -> usize {
        self.graph.link_count()
    }

    #[inline]
    fn num_ports(&self, node: Self::Node, dir: Direction) -> usize {
        self.graph.num_ports(node.into_portgraph(), dir)
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = Node> + Clone {
        self.graph.nodes_iter().map_into()
    }

    #[inline]
    fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        self.graph
            .port_offsets(node.into_portgraph(), dir)
            .map_into()
    }

    #[inline]
    fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone {
        self.graph
            .all_port_offsets(node.into_portgraph())
            .map_into()
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
            .port_index(node.into_portgraph(), port.pg_offset())
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
            .get_connections(node.into_portgraph(), other.into_portgraph())
            .map(|(p1, p2)| {
                [p1, p2].map(|link| self.graph.port_offset(link.port()).unwrap().into())
            })
    }

    #[inline]
    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone {
        self.hierarchy.children(node.into_portgraph()).map_into()
    }

    #[inline]
    fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        self.hierarchy.descendants(node.into_portgraph()).map_into()
    }

    #[inline]
    fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone {
        self.graph.neighbours(node.into_portgraph(), dir).map_into()
    }

    #[inline]
    fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone {
        self.graph.all_neighbours(node.into_portgraph()).map_into()
    }

    #[expect(deprecated)]
    fn mermaid_string_with_config(&self, config: RenderConfig) -> String {
        self.mermaid_string_with_formatter(MermaidFormatter::from_render_config(config, self))
    }

    fn mermaid_string_with_formatter(&self, formatter: MermaidFormatter<Self>) -> String {
        self.graph
            .mermaid_format()
            .with_hierarchy(&self.hierarchy)
            .with_node_style(render::node_style(self, formatter.clone()))
            .with_edge_style(render::edge_style(self, formatter))
            .finish()
    }

    fn dot_string(&self) -> String
    where
        Self: Sized,
    {
        let formatter = MermaidFormatter::new(self).with_entrypoint(self.entrypoint());
        self.graph
            .dot_format()
            .with_hierarchy(&self.hierarchy)
            .with_node_style(render::node_style(self, formatter.clone()))
            .with_port_style(render::port_style(self))
            .with_edge_style(render::edge_style(self, formatter))
            .finish()
    }

    #[inline]
    fn extensions(&self) -> &ExtensionRegistry {
        &self.extensions
    }

    #[inline]
    fn extract_hugr(&self, target: Node) -> (Hugr, impl ExtractionResult<Node> + 'static) {
        // Shortcircuit if the extracted HUGR is the same as the original
        if target == self.module_root().node() {
            return (self.clone(), DefaultNodeMap(HashMap::new()));
        }

        // Initialize a new HUGR with the desired entrypoint operation.
        // If we cannot create a new hugr with the parent's optype (e.g. if it's a `BasicBlock`),
        // find the first ancestor that can be extracted and use that instead.
        //
        // The final entrypoint will be set to the original `parent`.
        let mut parent = target;
        let mut extracted = loop {
            let parent_op = self.get_optype(parent).clone();
            if let Ok(hugr) = Hugr::new_with_entrypoint(parent_op) {
                break hugr;
            }
            // If the operation is not extractable, try the parent.
            // This loop always terminates, since at least the module root is extractable.
            parent = self
                .get_parent(parent)
                .expect("The module root is always extractable");
        };

        // The entrypoint and its parent in the newly created HUGR.
        // These will be replaced with nodes from the original HUGR.
        let old_entrypoint = extracted.entrypoint();
        let old_parent = extracted.get_parent(old_entrypoint);

        let inserted = extracted.insert_from_view(old_entrypoint, &self.with_entrypoint(parent));
        let new_entrypoint = inserted.inserted_entrypoint;

        match old_parent {
            Some(old_parent) => {
                // Depending on the entrypoint operation, the old entrypoint may
                // be connected to other nodes (dataflow region input/outputs).
                let old_ins = extracted
                    .node_inputs(old_entrypoint)
                    .flat_map(|inp| {
                        extracted
                            .linked_outputs(old_entrypoint, inp)
                            .map(move |link| (inp, link))
                    })
                    .collect_vec();
                let old_outs = extracted
                    .node_outputs(old_entrypoint)
                    .flat_map(|out| {
                        extracted
                            .linked_inputs(old_entrypoint, out)
                            .map(move |link| (out, link))
                    })
                    .collect_vec();
                // Replace the node
                extracted.set_entrypoint(inserted.node_map[&target]);
                extracted.remove_node(old_entrypoint);
                extracted.set_parent(new_entrypoint, old_parent);
                // Reconnect the inputs and outputs to the new entrypoint
                for (inp, (neigh, neigh_out)) in old_ins {
                    extracted.connect(neigh, neigh_out, new_entrypoint, inp);
                }
                for (out, (neigh, neigh_in)) in old_outs {
                    extracted.connect(new_entrypoint, out, neigh, neigh_in);
                }
            }
            // The entrypoint a module op
            None => {
                extracted.set_entrypoint(inserted.node_map[&target]);
                extracted.set_module_root(new_entrypoint);
                extracted.remove_node(old_entrypoint);
            }
        }
        (extracted, DefaultNodeMap(inserted.node_map))
    }
}

/// Trait implementing methods on port iterators.
pub trait PortIterator<P>: Iterator<Item = (Node, P)>
where
    P: Into<Port> + Copy,
    Self: Sized,
{
    /// Filter an iterator of node-ports to only dataflow dependency specifying
    /// ports (Value and `StateOrder`)
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
            let kind = HugrView::get_optype(hugr, *n).port_kind(*p);
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

/// Returns `true` if the node exists in the graph and is not the entrypoint node.
pub(super) fn check_valid_non_entrypoint<H: HugrView + ?Sized>(hugr: &H, node: H::Node) -> bool {
    hugr.contains_node(node) && node != hugr.entrypoint()
}

/// Returns `true` if the node exists in the graph and is not the module at the hierarchy root.
pub(super) fn check_valid_non_root<H: HugrView + ?Sized>(hugr: &H, node: H::Node) -> bool {
    hugr.contains_node(node) && node != hugr.module_root().node()
}

/// Panic if [`HugrView::contains_node`] fails.
#[track_caller]
pub(super) fn panic_invalid_node<H: HugrView + ?Sized>(hugr: &H, node: H::Node) {
    assert!(hugr.contains_node(node), "Received an invalid node {node}.",);
}

/// Panic if [`check_valid_non_entrypoint`] fails.
#[track_caller]
pub(super) fn panic_invalid_non_entrypoint<H: HugrView + ?Sized>(hugr: &H, node: H::Node) {
    assert!(
        check_valid_non_entrypoint(hugr, node),
        "Received an invalid non-entrypoint node {node}.",
    );
}

/// Panic if [`HugrView::valid_node`] fails.
#[track_caller]
pub(super) fn panic_invalid_port(hugr: &Hugr, node: Node, port: impl Into<Port>) {
    let port = port.into();
    if hugr
        .graph
        .port_index(node.into_portgraph(), port.pg_offset())
        .is_none()
    {
        panic!("Received an invalid {port} for {node} while mutating a HUGR");
    }
}

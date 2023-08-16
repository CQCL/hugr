//! Views for HUGR sibling subgraphs.
//!
//! Views into subgraphs of HUGRs within a single level of the
//! hierarchy, i.e. within a sibling graph. Such a subgraph is
//! represented by a parent node, of which all nodes in the sibling subgraph are
//! children, as well as a set of edges forming the subgraph boundary.
//! The boundary must be fully contained within the sibling graph of the parent.
//!
//! Sibling subgraphs complement [`super::HierarchyView`]s in the sense that the
//! latter provide views for subgraphs defined by hierarchical relationships,
//! while the former provide views for subgraphs within a single level of the
//! hierarchy.
//!

// TODO:
// //! This module exposes the [`SiblingView`] trait, which is currently
// //! implemented by the [`SiblingSubgraph`] struct in this module, as well as
// //! the [`SiblingGraph`] hierarchical view.

use itertools::Itertools;
use portgraph::{view::Subgraph, LinkView, PortIndex, PortView};
use thiserror::Error;

use crate::{
    ops::{OpTag, OpTrait},
    Direction, Hugr, Node, Port, SimpleReplacement,
};

use super::{sealed::HugrInternals, HugrView};

/// A boundary edge of a sibling subgraph.
///
/// Boundary edges come in two types: incoming and outgoing. The target of an
/// incoming edge is a node in the subgraph, and the source of an outgoing
/// edge is a node in the subgraph. The other ends are typically not in the
/// subgraph, except in the special where an edge is both an incoming and
/// outgoing boundary edge.
///
/// We uniquely identify a boundary edge by the node and port of the target of
/// the edge. Source ports do not uniquely identify edges as they can be copied
/// and be linked to multiple targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundaryEdge {
    target: (Node, Port),
    direction: Direction,
}

impl BoundaryEdge {
    /// Construct a new boundary edge.
    ///
    /// Only a target port can be used to create a new edge. Also ensures that
    /// the edge exists.
    fn try_new<H: HugrView>(
        node: Node,
        port: Port,
        direction: Direction,
        hugr: &H,
    ) -> Result<Self, InvalidEdge> {
        if port.direction() != Direction::Incoming {
            Err(InvalidEdge::ExpectedTargetPort)
        } else if hugr.linked_ports(node, port).next().is_none() {
            Err(InvalidEdge::DisconnectedPort)
        } else {
            Ok(Self {
                target: (node, port),
                direction,
            })
        }
    }

    /// Target port of the edge in a portgraph.
    fn portgraph_target<G: PortView>(&self, g: &G) -> PortIndex {
        let (node, port) = self.target;
        g.port_index(node.index, port.offset).unwrap()
    }

    /// Source port of the edge in a portgraph.
    fn portgraph_source<G: LinkView>(&self, g: &G) -> PortIndex {
        let tgt = self.portgraph_target(g);
        g.port_link(tgt).expect("invalid boundary edge").into()
    }

    /// Port Index in a portgraph that is within the boundary.
    ///
    /// If the edge is incoming, this is the target port, otherwise it is the
    /// source port.
    fn portgraph_internal_port<G: LinkView>(&self, g: &G) -> PortIndex {
        match self.direction {
            Direction::Incoming => self.portgraph_target(g),
            Direction::Outgoing => self.portgraph_source(g),
        }
    }

    /// Create boundary edges incident at a port, either incoming or outgoing.
    fn new_boundary<H: HugrView>(node: Node, port: Port, dir: Direction, hugr: &H) -> Vec<Self> {
        if port.direction() == Direction::Incoming {
            BoundaryEdge::try_new(node, port, dir, hugr)
                .ok()
                .into_iter()
                .collect()
        } else {
            hugr.linked_ports(node, port)
                .flat_map(|(n, p)| BoundaryEdge::try_new(n, p, dir, hugr).ok())
                .collect()
        }
    }

    /// Create incoming boundary edges incident at a port
    fn new_boundary_incoming<H: HugrView>(node: Node, port: Port, hugr: &H) -> Vec<Self> {
        Self::new_boundary(node, port, Direction::Incoming, hugr)
    }

    /// Create outgoing boundary edges incident at a port.
    fn new_boundary_outgoing<H: HugrView>(node: Node, port: Port, hugr: &H) -> Vec<Self> {
        Self::new_boundary(node, port, Direction::Outgoing, hugr)
    }

    /// Create an outgoing boundary from the incoming edges of a node.
    ///
    /// Pass it an Output node to obtain the incoming boundary.
    ///
    /// Filters out any non-DFG edges.
    fn from_output_node<H: HugrView>(node: Node, hugr: &H) -> impl Iterator<Item = Self> + '_ {
        hugr.node_inputs(node)
            .filter(move |&p| hugr.get_optype(node).signature().get(p).is_some())
            .flat_map(move |p| BoundaryEdge::new_boundary_outgoing(node, p, hugr))
    }

    /// Create an incoming boundary from the outgoing edges of a node.
    ///
    /// Pass it an Input node to obtain the outgoing boundary.
    ///
    /// Filters out any non-DFG edges.
    fn from_input_node<H: HugrView>(node: Node, hugr: &H) -> impl Iterator<Item = Self> + '_ {
        hugr.node_outputs(node)
            .filter(move |&p| hugr.get_optype(node).signature().get(p).is_some())
            .flat_map(move |p| BoundaryEdge::new_boundary_incoming(node, p, hugr))
    }
}

/// A subgraph of a HUGR sibling graph.
///
/// A HUGR region in which all nodes share the same parent. Unlike
/// [`super::SiblingGraph`],  not all nodes of the sibling graph must be
/// included.
///
/// Given a node in a HUGR, let E be the set of edges of its sibling graph. A
/// sibling subgraph is described by a set of boundary edges B ⊂ E that can be
/// marked as incoming boundary edge, outgoing boundary edge or both.
///
/// The sibling subgraph defined by B is the graph given by the connected
/// components of the graph with edges E\B which contain at least one
/// node that is either
///  - the target of an incoming boundary edge, or
///  - the source of an outgoing boundary edge.
///
/// A subgraph is well-formed if every edge in B into the subgraph is an
/// incoming boundary edge and every edge in B pointing out of the subgraph is
/// an outgoing boundary edge.
///
/// The parent node itself is not included in the subgraph. If the boundary set
/// is empty, the subgraph is taken to be all children of the parent and is
/// equivalent to a [`super::SiblingGraph`].
///
/// In this implementation, the order of the boundary edges is used to determine
/// the ordered boundary of the subgraph. When using [`SiblingSubgraph`] for
/// rewriting, this must thus match the boundary ordering of the replacement
/// graph.
#[derive(Clone, Debug)]
pub struct SiblingSubgraph<'g, Base: HugrInternals> {
    parent: Node,
    boundary: Vec<BoundaryEdge>,
    sibling_graph: Subgraph<'g, Base::Portgraph>,
}

impl<'g, Base: HugrInternals> SiblingSubgraph<'g, Base> {
    /// A sibling subgraph from a HUGR.
    ///
    /// The subgraph is given by the sibling graph of the root. If you wish to
    /// create a subgraph from another root, wrap the argument `region` in a
    /// [`super::SiblingGraph`].
    pub fn from_sibling_graph(region: &'g Base) -> Self
    where
        Base: HugrView + HugrInternals,
    {
        let parent = region.root();
        Self::new(region, parent, [], [])
    }

    /// A sibling subgraph from a [`crate::ops::OpTag::DataflowParent`]-rooted HUGR.
    ///
    /// The subgraph is given by the nodes between the input and output
    /// children nodes of the parent node. If you wish to create a subgraph
    /// from another root, wrap the `region` argument in a [`super::SiblingGraph`].
    pub fn from_dataflow_graph(region: &'g Base) -> Self
    where
        Base: HugrView,
    {
        let parent = region.root();
        let (inp, out) = region.children(parent).take(2).collect_tuple().unwrap();
        let incoming = BoundaryEdge::from_input_node(inp, region);
        let outgoing = BoundaryEdge::from_output_node(out, region);
        let boundary = incoming.chain(outgoing).collect_vec();
        let sibling_graph = compute_subgraph(region, boundary.iter().copied());
        Self {
            parent,
            boundary,
            sibling_graph,
        }
    }

    /// Creates a new sibling subgraph.
    ///
    /// The incoming and outgoing edges can be expressed as either source or
    /// target ports. If source ports are given, they will be converted to
    /// target ports, and, in the presence of copies, the signature will be
    /// expanded accordingly.
    pub fn new(
        hugr: &'g Base,
        parent: Node,
        incoming: impl IntoIterator<Item = (Node, Port)>,
        outgoing: impl IntoIterator<Item = (Node, Port)>,
    ) -> Self
    where
        Base: HugrView,
    {
        let incoming = incoming
            .into_iter()
            .flat_map(|(n, p)| BoundaryEdge::new_boundary_incoming(n, p, hugr));
        let outgoing = outgoing
            .into_iter()
            .flat_map(|(n, p)| BoundaryEdge::new_boundary_outgoing(n, p, hugr));
        let boundary = incoming.chain(outgoing).collect_vec();
        let sibling_graph = compute_subgraph(hugr, boundary.iter().copied());
        Self {
            parent,
            boundary,
            sibling_graph,
        }
    }

    /// An iterator over the nodes in the subgraph.
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_
    where
        Base: HugrView,
    {
        self.sibling_graph.nodes_iter().map_into()
    }

    /// The number of incoming and outgoing wires of the subgraph.
    pub fn boundary_size(&self) -> (usize, usize)
    where
        Base: HugrView,
    {
        let incoming = self
            .boundary
            .iter()
            .filter(|e| e.direction == Direction::Incoming)
            .count();
        let outgoing = self
            .boundary
            .iter()
            .filter(|e| e.direction == Direction::Outgoing)
            .count();
        (incoming, outgoing)
    }

    /// Construct a [`SimpleReplacement`] to replace `self` with `replacement`.
    ///
    /// `replacement` must be a hugr with DFG root and its signature must
    /// match the signature of the subgraph.
    ///
    /// May return one of the following four errors
    ///  - [`InvalidReplacement::InvalidDataflowGraph`]: the replacement
    ///    graph is not a [`crate::ops::OpTag::DataflowParent`]-rooted graph,
    ///  - [`InvalidReplacement::InvalidDataflowParent`]: the replacement does
    ///    not have an input and output node,
    ///  - [`InvalidReplacement::InvalidBoundarySize`]: the number of incoming
    ///    and outgoing ports in replacement does not match the subgraph boundary
    ///    signature, or
    ///  - [`InvalidReplacement::NonConvexSubgrah`]: the sibling subgraph is not
    ///    convex.
    pub fn create_simple_replacement(
        &self,
        replacement: Hugr,
    ) -> Result<SimpleReplacement, InvalidReplacement>
    where
        Base: HugrView,
    {
        let removal = self.nodes().collect();

        let rep_root = replacement.root();
        if replacement.get_optype(rep_root).tag() != OpTag::Dfg {
            return Err(InvalidReplacement::InvalidDataflowGraph);
        }
        let Some((rep_input, rep_output)) = replacement
            .children(rep_root)
            .take(2)
            .collect_tuple()
        else { return Err(InvalidReplacement::InvalidDataflowParent) };
        let rep_inputs = BoundaryEdge::from_input_node(rep_input, &replacement).collect_vec();
        let rep_outputs = BoundaryEdge::from_output_node(rep_output, &replacement).collect_vec();
        if (rep_inputs.len(), rep_outputs.len()) != self.boundary_size() {
            return Err(InvalidReplacement::InvalidBoundarySize);
        }
        let incoming = self
            .boundary
            .iter()
            .copied()
            .filter(|e| e.direction == Direction::Incoming);
        let outgoing = self
            .boundary
            .iter()
            .copied()
            .filter(|e| e.direction == Direction::Outgoing);
        let nu_inp = rep_inputs
            .into_iter()
            .map(|e| e.target)
            .zip_eq(incoming.map(|e| e.target))
            .collect();
        let nu_out = outgoing
            .map(|e| e.target)
            .zip_eq(rep_outputs.into_iter().map(|e| e.target.1))
            .collect();

        Ok(SimpleReplacement::new(
            self.parent,
            removal,
            replacement,
            nu_inp,
            nu_out,
        ))
    }

    /// Whether the sibling subgraph is convex.
    pub fn is_convex(&self) -> bool {
        self.sibling_graph.is_convex()
    }
}

/// Errors that can occur while constructing a [`SimpleReplacement`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidReplacement {
    /// No DataflowParent root in replacement graph.
    #[error("No DataflowParent root in replacement graph.")]
    InvalidDataflowGraph,
    /// Malformed DataflowParent in replacement graph.
    #[error("Malformed DataflowParent in replacement graph.")]
    InvalidDataflowParent,
    /// Replacement graph boundary size mismatch.
    #[error("Replacement graph boundary size mismatch.")]
    InvalidBoundarySize,
    /// SiblingSubgraph is not convex.
    #[error("SiblingSubgraph is not convex.")]
    NonConvexSubgrah,
}

/// Errors that can occur while constructing a [`BoundaryEdge`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
enum InvalidEdge {
    /// The port is not connected to an edge.
    #[error("Port must be connected to an edge.")]
    DisconnectedPort,
    /// Edges must be defined through their target port.
    #[error("Edges must be defined through their target port.")]
    ExpectedTargetPort,
}

fn compute_subgraph<Base: HugrInternals>(
    hugr: &Base,
    boundary: impl IntoIterator<Item = BoundaryEdge>,
) -> Subgraph<'_, Base::Portgraph> {
    let graph = hugr.portgraph();
    let boundary = boundary
        .into_iter()
        .map(|e| e.portgraph_internal_port(graph));
    Subgraph::new_subgraph(graph, boundary)
}

// /// A common trait for views of a HUGR sibling subgraph.
// pub trait SiblingView: HugrView {}

// impl<'g, Base: HugrView> SiblingView for SiblingSubgraph<'g, Base> {}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{
            BuildError, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        hugr::views::{HierarchyView, SiblingGraph},
        ops::{handle::NodeHandle, LeafOp},
        resource::prelude::QB_T,
        type_row,
        types::AbstractSignature,
    };

    use super::*;

    fn build_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            AbstractSignature::new_linear(type_row![QB_T, QB_T]).pure(),
        )?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let outs = dfg.add_dataflow_op(LeafOp::CX, dfg.input_wires())?;
            dfg.finish_with_outputs(outs.outputs())?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    #[test]
    fn construct_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let sibling_graph: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let from_root = SiblingSubgraph::from_sibling_graph(&sibling_graph);
        let region: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let from_region = SiblingSubgraph::from_sibling_graph(&region);
        assert_eq!(from_root.parent, from_region.parent);
        assert_eq!(from_root.boundary, from_region.boundary);
    }

    #[test]
    fn construct_simple_replacement() {
        let (mut hugr, func_root) = build_hugr().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let sub = SiblingSubgraph::from_dataflow_graph(&func);

        let empty_dfg = {
            let builder =
                DFGBuilder::new(AbstractSignature::new_linear(type_row![QB_T, QB_T])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        let rep = sub.create_simple_replacement(empty_dfg).unwrap();

        assert_eq!(rep.removal.len(), 1);

        hugr.apply_rewrite(rep).unwrap();

        assert_eq!(hugr.node_count(), 4); // Module + Def + In + Out
    }

    #[test]
    fn construct_simple_replacement_signature_panics() {
        let (hugr, dfg) = build_hugr().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, dfg);
        let sub = SiblingSubgraph::from_sibling_graph(&func);

        let empty_dfg = {
            let builder = DFGBuilder::new(AbstractSignature::new_linear(type_row![QB_T])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        assert_eq!(
            sub.create_simple_replacement(empty_dfg).unwrap_err(),
            InvalidReplacement::InvalidBoundarySize
        )
    }

    #[test]
    fn convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let sub = SiblingSubgraph::from_dataflow_graph(&func);
        assert!(sub.is_convex());
    }

    #[test]
    fn convex_subgraph_2() {
        let (hugr, func_root) = build_hugr().unwrap();
        let (inp, out) = hugr.children(func_root).take(2).collect_tuple().unwrap();
        // All graph except input/output nodes
        let sub = SiblingSubgraph::new(
            &hugr,
            func_root,
            hugr.node_outputs(inp).map(|p| (inp, p)),
            hugr.node_inputs(out).map(|p| (out, p)),
        );
        assert!(sub.is_convex());
    }

    #[test]
    fn non_convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let (inp, _) = hugr.children(func_root).take(2).collect_tuple().unwrap();
        let first_cx_edge = hugr.node_outputs(inp).next().unwrap();
        // All graph but one edge
        let sub = SiblingSubgraph::new(
            &hugr,
            func_root,
            [(inp, first_cx_edge)],
            [(inp, first_cx_edge)],
        );
        assert!(!sub.is_convex());
    }
}

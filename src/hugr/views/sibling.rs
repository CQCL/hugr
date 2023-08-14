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

use std::cell::OnceCell;

use itertools::Itertools;
use portgraph::{view::Subgraph, LinkView, PortIndex, PortView};

use crate::{
    ops::{handle::NodeHandle, OpTag, OpTrait},
    Direction, Hugr, Node, Port, SimpleReplacement,
};

use super::{sealed::HugrInternals, HierarchyView, HugrView, SiblingGraph};

/// A boundary edge of a sibling subgraph.
///
/// Boundary edges come in two types: incoming and outgoing. The target of an
/// incoming edge is a node in the subgraph, and the source of an outgoing
/// edge is a node in the subgraph. The other ends are typically not in the
/// subgraph, except in the special where an edge is both an incoming and
/// outgoing boundary edge.
///
/// We uniquely identify a boundary edge by the node and port of the target of
/// the edge. Source edges are not unique as they can involve copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundaryEdge {
    target: (Node, Port),
    direction: Direction,
}

impl BoundaryEdge {
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

    /// Create incoming boundary edges incident at a port
    fn new_boundary_incoming<H: HugrView>(node: Node, port: Port, hugr: &H) -> Vec<Self> {
        if port.direction() == Direction::Incoming {
            vec![BoundaryEdge {
                target: (node, port),
                direction: Direction::Incoming,
            }]
        } else {
            hugr.linked_ports(node, port)
                .map(|(n, p)| BoundaryEdge {
                    target: (n, p),
                    direction: Direction::Incoming,
                })
                .collect()
        }
    }

    /// Create outgoing boundary edges incident at a port
    fn new_boundary_outgoing<H: HugrView>(node: Node, port: Port, hugr: &H) -> Vec<Self> {
        if port.direction() == Direction::Incoming {
            vec![BoundaryEdge {
                target: (node, port),
                direction: Direction::Outgoing,
            }]
        } else {
            hugr.linked_ports(node, port)
                .map(|(n, p)| BoundaryEdge {
                    target: (n, p),
                    direction: Direction::Outgoing,
                })
                .collect()
        }
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
/// The subgraph is described by the following data:
///  - a HugrView to the underlying HUGR,
///  - a parent node, of which all nodes in the [`SiblingSubgraph`] must be
///    children,
///  - incoming/outgoing boundary edges, pointing into/out of the subgraph.
/// The incoming/outgoing edges must be contained within the sibling graph of the
/// parent node. Their ordering matters when using the [`SiblingSubgraph`] for
/// replacements, as it will match the ordering of the ports in the replacement.
/// The list of incoming and/or outgoing ports may be empty.
///
/// A subgraph is well-defined if the incoming/outgoing edges partition the
/// edges of the sibling graph into three sets:
///  - boundary edges: either incoming boundary or outgoing boundary edges,
///  - interior edges: edges such that all the successor edges are either
///    outgoing boundary edges or interior edges AND all the predecessor edges
///    are either incoming boundary edges or interior edges,
///  - exterior edges: edges such that all the successor edges are either
///    incoming boundary edges or exterior edges AND all the predecessor edges
///    are either outgoing boundary edges or exterior edges.
///
/// Then the subgraph contains all nodes of the sibling graph of the parent that
/// are
///  - adjacent to an interior edge, or
///  - the target of an incoming boundary edge and the source of an outgoing
///    boundary edge.
///
/// The parent node itself is not included in the subgraph. If both incoming and
/// outgoing ports are empty, the subgraph is taken to be all children of the
/// parent and is equivalent to a [`SiblingGraph`].
///
/// This does not implement Sync as we use a `OnceCell` to cache the sibling
/// graph.
#[derive(Clone, Debug)]
pub struct SiblingSubgraph<'g, Base: HugrInternals> {
    hugr: &'g Base,
    parent: Node,
    boundary: Vec<BoundaryEdge>,
    sibling_graph: OnceCell<Subgraph<'g, Base::Portgraph>>,
}

impl<'g> SiblingSubgraph<'g, Hugr> {
    /// A sibling subgraph from a [`SiblingGraph`] object.
    ///
    /// The subgraph is given by the entire sibling graph.
    pub fn from_sibling_graph<Base, Root>(region: &'g SiblingGraph<'g, Root, Base>) -> Self
    where
        Base: HugrView + HugrInternals,
        Root: NodeHandle,
    {
        let parent = region.root();
        Self::new(region.base_hugr(), parent, [], [])
    }
}

impl<'g, Base: HugrInternals> SiblingSubgraph<'g, Base> {
    /// A sibling subgraph given by a HUGR and a parent node.
    ///
    /// The subgraph is given by the entire sibling graph.
    pub fn from_parent(hugr: &'g Base, parent: Node) -> Self {
        Self {
            hugr,
            parent,
            boundary: Vec::new(),
            sibling_graph: OnceCell::new(),
        }
    }

    /// A sibling subgraph from a [`crate::ops::OpTag::DataflowParent`]-rooted HUGR.
    ///
    /// The subgraph is given by the nodes between the input and output
    /// children nodes of the parent node.
    pub fn from_dfg(hugr: &'g Base) -> Self
    where
        Base: HugrView,
    {
        let parent = hugr.root();
        Self::from_dfg_parent(hugr, parent)
    }

    /// A sibling subgraph from a [`crate::ops::OpTag::DataflowParent`] node in a HUGR.
    ///
    /// The subgraph is given by the nodes between the input and output
    /// children nodes of the parent node.
    ///
    /// Panics if it could not find an input and an output node.
    pub fn from_dfg_parent(hugr: &'g Base, parent: Node) -> Self
    where
        Base: HugrView,
    {
        let (inp, out) = hugr.children(parent).take(2).collect_tuple().unwrap();
        let incoming = BoundaryEdge::from_input_node(inp, hugr);
        let outgoing = BoundaryEdge::from_output_node(out, hugr);
        let boundary = incoming.chain(outgoing).collect();
        Self {
            hugr,
            parent,
            boundary,
            sibling_graph: OnceCell::new(),
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
            .flat_map(|(n, p)| BoundaryEdge::new_boundary_incoming(n, p, hugr));
        let boundary = incoming.chain(outgoing).collect();
        Self {
            hugr,
            parent,
            boundary,
            sibling_graph: OnceCell::new(),
        }
    }

    fn get_sibling_graph(&self) -> &Subgraph<'g, Base::Portgraph> {
        self.sibling_graph.get_or_init(|| {
            let graph = self.hugr.portgraph();
            let boundary = self
                .boundary
                .iter()
                .copied()
                .map(|e| e.portgraph_internal_port(graph));
            Subgraph::new_subgraph(graph, boundary)
        })
    }

    /// An iterator over the nodes in the subgraph.
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_
    where
        Base: HugrView,
    {
        self.get_sibling_graph().nodes_iter().flat_map(|index| {
            let region: SiblingGraph<'_, Node, Base> = SiblingGraph::new(self.hugr, Node { index });
            region.nodes().collect_vec()
        })
    }

    /// The number of incoming and outgoing wires of the subgraph.
    pub fn boundary_signature(&self) -> (usize, usize)
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
    /// Panics if
    ///  - `replacement` is not a hugr with DFG root, or
    ///  - the DFG root does not have an input and output node, or
    ///  - the number of incoming and outgoing ports in replacement does not
    ///    match the subgraph boundary signature.
    pub fn create_simple_replacement(&self, replacement: Hugr) -> SimpleReplacement
    where
        Base: AsRef<Hugr>,
    {
        let removal = self.nodes().collect();

        let rep_root = replacement.root();
        if replacement.get_optype(rep_root).tag() != OpTag::Dfg {
            panic!("Replacement must have DFG root");
        }
        let Some((rep_input, rep_output)) = replacement
            .children(rep_root)
            .take(2)
            .collect_tuple() else { panic!("Invalid DFG node") };
        let rep_inputs = BoundaryEdge::from_input_node(rep_input, &replacement);
        let rep_outputs = BoundaryEdge::from_output_node(rep_output, &replacement);
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
            .map(|e| e.target)
            .zip_eq(incoming.map(|e| e.target))
            .collect();
        let nu_out = outgoing
            .map(|e| e.target)
            .zip_eq(rep_outputs.map(|e| e.target.1))
            .collect();

        SimpleReplacement::new(self.parent, removal, replacement, nu_inp, nu_out)
    }

    /// Whether the sibling subgraph is convex.
    pub fn is_convex(&self) -> bool {
        self.get_sibling_graph().is_convex()
    }
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
        ops::{handle::NodeHandle, LeafOp},
        type_row,
        types::{AbstractSignature, SimpleType},
    };

    use super::*;

    const QB: SimpleType = SimpleType::Qubit;

    fn build_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            AbstractSignature::new_linear(type_row![QB, QB]).pure(),
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
        let from_root = SiblingSubgraph::from_parent(&hugr, func_root);
        let region: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let from_region = SiblingSubgraph::from_sibling_graph(&region);
        assert_eq!(from_root.parent, from_region.parent);
        assert_eq!(from_root.boundary, from_region.boundary);
    }

    #[test]
    fn construct_simple_replacement() {
        let (mut hugr, func_root) = build_hugr().unwrap();
        let sub = SiblingSubgraph::from_dfg_parent(&hugr, func_root);

        let empty_dfg = {
            let builder =
                DFGBuilder::new(AbstractSignature::new_linear(type_row![QB, QB])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        let rep = sub.create_simple_replacement(empty_dfg);

        assert_eq!(rep.removal.len(), 1);

        hugr.apply_rewrite(rep).unwrap();

        assert_eq!(hugr.node_count(), 4); // Module + Def + In + Out
    }

    #[test]
    #[should_panic(expected = "zip_eq")]
    fn construct_simple_replacement_signature_panics() {
        let (hugr, dfg) = build_hugr().unwrap();
        let sub = SiblingSubgraph::from_dfg_parent(&hugr, dfg);

        let empty_dfg = {
            let builder = DFGBuilder::new(AbstractSignature::new_linear(type_row![QB])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        sub.create_simple_replacement(empty_dfg);
    }

    #[test]
    fn convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let sub = SiblingSubgraph::from_dfg_parent(&hugr, func_root);
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

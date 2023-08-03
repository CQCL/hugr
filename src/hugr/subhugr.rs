//! Views into subgraphs of HUGRs.
//!
//! Subgraphs are a generalisation of a [`Region`], in which there can be
//! more than one root node. However, all root nodes must belong to the same
//! sibling graph.
//!
//! TODO: Subgraphs implement [`HugrView`] as well as petgraph's _visit_ traits.

use std::cell::OnceCell;

use derive_more::Into;
use itertools::Itertools;
use portgraph::{view::Subgraph, LinkView, PortIndex, PortView};

use crate::{
    ops::{OpTag, OpTrait},
    Direction, Hugr, Node, Port, SimpleReplacement,
};

use super::{
    region::{Region, RegionView},
    view::sealed::HugrInternals,
    HugrView,
};

/// A boundary edge of a subhugr.
///
/// It is always uniquely identified (any Hugr edge is) by the node and port
/// of the target of the edge. Source edges are not unique as they can involve
/// copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Into)]
struct BoundaryEdge(Node, Port);

impl BoundaryEdge {
    fn target_port_index<G: PortView>(&self, g: &G) -> PortIndex {
        let Node { index: node } = self.0;
        let Port { offset: port } = self.1;
        g.port_index(node, port).unwrap()
    }

    fn source_port_index<G: LinkView>(&self, g: &G) -> PortIndex {
        let tgt = self.target_port_index(g);
        g.port_link(tgt).unwrap().into()
    }

    fn get_boundary_edges<H: HugrView>(
        node: Node,
        port: Port,
        hugr: &H,
    ) -> impl Iterator<Item = Self> {
        if port.direction() == Direction::Incoming {
            vec![BoundaryEdge(node, port)].into_iter()
        } else {
            hugr.linked_ports(node, port)
                .map(|(n, p)| BoundaryEdge(n, p))
                .collect_vec()
                .into_iter()
        }
    }

    /// The DFG input ports of a `node`
    fn get_boundary_incoming<H: HugrView>(node: Node, hugr: &H) -> impl Iterator<Item = Self> + '_ {
        hugr.node_inputs(node)
            .filter(move |&p| hugr.get_optype(node).signature().get(p).is_some())
            .map(move |p| BoundaryEdge(node, p))
    }

    /// The linked ports of DFG output ports of a `node`
    ///
    /// We always consider target ports, as they are unique
    /// (source ports can be copied)
    fn get_boundary_outgoing<H: HugrView>(node: Node, hugr: &H) -> impl Iterator<Item = Self> + '_ {
        hugr.node_outputs(node)
            .filter(move |&p| hugr.get_optype(node).signature().get(p).is_some())
            .flat_map(move |p| hugr.linked_ports(node, p))
            .map(|(n, p)| BoundaryEdge(n, p))
    }
}

/// A subgraph of a HUGR.
///
/// The subgraph is described by the following data:
///  - a HugrView to the underlying HUGR,
///  - a root node, of which all nodes in the [`SubHugr`] must be descendants,
///  - incoming/outgoing boundary edges, pointing into/out of the subgraph.
/// The incoming/outgoing edges must link direct children of the
/// root node. Their ordering matters when using the [`SubHugr`] for replacements,
/// as it will match the ordering of the ports in the replacement.
/// The list of incoming and/or outgoing ports may be empty.
///
/// A subhugr is well-defined if the incoming/outgoing edges partition the
/// edges between the children of the root node into three sets:
///  - boundary edges: either incoming boundary or outgoing boundary edges,
///  - interior edges: edges such that all the successor edges are either
///    outgoing boundary edges or interior edges AND all the predecessor edges
///    are either incoming boundary edges or interior edges,
///  - exterior edges: edges such that all the successor edges are either
///    incoming boundary edges or exterior edges AND all the predecessor edges
///    are either outgoing boundary edges or exterior edges.
///
/// Then the subhugr contains all children nodes of the root that are
///  - adjacent to an interior edge, or
///  - in-adjacent to an incoming edge and out-adjacent to an outgoing edge.
///
/// The root node itself is not included in the subhugr. Any node that is a
/// descendant of a node in the subhugr is itself also in the subhugr.
/// If both incoming and outgoing ports are empty, the subhugr is taken to be
/// all children of the root.
///
/// This does not implement Sync as we use a `OnceCell` to cache the sibling
/// graph.
#[derive(Clone, Debug)]
pub struct SubHugr<'g, Base: HugrInternals> {
    hugr: &'g Base,
    root: Node,
    incoming: Vec<BoundaryEdge>,
    outgoing: Vec<BoundaryEdge>,
    sibling_graph: OnceCell<Subgraph<'g, Base::Portgraph>>,
}

impl<'g, Base: HugrView> SubHugr<'g, RegionView<'g, Base>> {
    /// Creates a new subhugr from a region.
    pub fn from_region(region: &'g RegionView<'g, Base>) -> Self {
        let root = region.root();
        Self::new(region, root, [], [])
    }
}

impl<'g, Base: HugrInternals> SubHugr<'g, Base> {
    /// Creates a new subhugr from a root node
    pub fn from_root(hugr: &'g Base, root: Node) -> Self {
        Self {
            hugr,
            root,
            incoming: Vec::new(),
            outgoing: Vec::new(),
            sibling_graph: OnceCell::new(),
        }
    }

    /// Create a new subhugr from a DFG-rooted hugr
    pub fn from_dfg(hugr: &'g Base) -> Self
    where
        Base: HugrView,
    {
        let root = hugr.root();
        Self::from_dfg_root(hugr, root)
    }

    /// Create a new subhugr from a DFG node in a Hugr
    ///
    /// This also works for e.g. FuncDefn nodes, i.e. roots where the first
    /// two children correspond to inputs and outputs.
    ///
    /// Panics if it could not find an input and an output node.
    pub fn from_dfg_root(hugr: &'g Base, root: Node) -> Self
    where
        Base: HugrView,
    {
        let (inp, out) = hugr.children(root).take(2).collect_tuple().unwrap();
        let incoming = BoundaryEdge::get_boundary_outgoing(inp, hugr).collect();
        let outgoing = BoundaryEdge::get_boundary_incoming(out, hugr).collect();
        Self {
            hugr,
            root,
            incoming,
            outgoing,
            sibling_graph: OnceCell::new(),
        }
    }

    /// Creates a new subhugr.
    ///
    /// The incoming and outgoing edges can be expressed as either source or
    /// target ports. If source ports are given, they will be converted to
    /// target ports, and, in the presence of copies, the signature will be
    /// expanded accordingly.
    pub fn new(
        hugr: &'g Base,
        root: Node,
        incoming: impl IntoIterator<Item = (Node, Port)>,
        outgoing: impl IntoIterator<Item = (Node, Port)>,
    ) -> Self
    where
        Base: HugrView,
    {
        let incoming = incoming
            .into_iter()
            .flat_map(|(n, p)| BoundaryEdge::get_boundary_edges(n, p, hugr))
            .collect();
        let outgoing = outgoing
            .into_iter()
            .flat_map(|(n, p)| BoundaryEdge::get_boundary_edges(n, p, hugr))
            .collect();
        Self {
            hugr,
            root,
            incoming,
            outgoing,
            sibling_graph: OnceCell::new(),
        }
    }

    fn get_sibling_graph(&self) -> &Subgraph<'g, Base::Portgraph> {
        self.sibling_graph.get_or_init(|| {
            let graph = self.hugr.portgraph();
            let incoming = self
                .incoming
                .iter()
                .copied()
                .map(|e| e.target_port_index(graph));
            let outgoing = self
                .outgoing
                .iter()
                .copied()
                .map(|e| e.source_port_index(graph));
            Subgraph::new_subgraph(graph, incoming.chain(outgoing))
        })
    }

    /// An iterator over the nodes in the subhugr.
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_
    where
        Base: HugrView,
    {
        self.get_sibling_graph().nodes_iter().flat_map(|index| {
            let region = RegionView::new(self.hugr, Node { index });
            region.nodes().collect_vec()
        })
    }

    /// The number of incoming and outgoing wires of the subhugr.
    pub fn boundary_signature(&self) -> (usize, usize)
    where
        Base: HugrView,
    {
        let incoming = self.incoming.len();
        let outgoing = self.outgoing.len();
        (incoming, outgoing)
    }

    /// Construct a [`SimpleReplacement`] to replace `self` with `replacement`.
    ///
    /// `replacement` must be a hugr with DFG root and its signature must
    /// match the signature of the subhugr.
    ///
    /// Panics if
    ///  - `replacement` is not a hugr with DFG root, or
    ///  - the DFG root does not have an input and output node, or
    ///  - the number of incoming and outgoing ports in replacement does not
    ///    match the subhugr boundary signature.
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
        let rep_inputs = BoundaryEdge::get_boundary_outgoing(rep_input, &replacement);
        let rep_outputs = BoundaryEdge::get_boundary_incoming(rep_output, &replacement);
        let nu_inp = rep_inputs
            .map_into()
            .zip_eq(self.incoming.iter().copied().map_into())
            .collect();
        let nu_out = self
            .outgoing
            .iter()
            .copied()
            .map_into()
            .zip_eq(rep_outputs.map(|BoundaryEdge(_, p)| p))
            .collect();

        SimpleReplacement::new(self.root, removal, replacement, nu_inp, nu_out)
    }

    /// Whether the subhugr is convex.
    pub fn is_convex(&self) -> bool {
        self.get_sibling_graph().is_convex()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{
            BuildError, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        hugr::region::Region,
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
    fn construct_subhugr() {
        let (hugr, func_root) = build_hugr().unwrap();
        let from_root = SubHugr::from_root(&hugr, func_root);
        let region = RegionView::new(&hugr, func_root);
        let from_region = SubHugr::from_region(&region);
        assert_eq!(from_root.root, from_region.root);
        assert_eq!(from_root.incoming, from_region.incoming);
        assert_eq!(from_root.outgoing, from_region.outgoing);
    }

    #[test]
    fn construct_simple_replacement() {
        let (mut hugr, func_root) = build_hugr().unwrap();
        let sub = SubHugr::from_dfg_root(&hugr, func_root);

        let empty_dfg = {
            let builder = DFGBuilder::new(type_row![QB, QB], type_row![QB, QB]).unwrap();
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
        let sub = SubHugr::from_dfg_root(&hugr, dfg);

        let empty_dfg = {
            let builder = DFGBuilder::new(type_row![QB], type_row![QB]).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        sub.create_simple_replacement(empty_dfg);
    }

    #[test]
    fn convex_subhugr() {
        let (hugr, func_root) = build_hugr().unwrap();
        let sub = SubHugr::from_dfg_root(&hugr, func_root);
        assert!(sub.is_convex());
    }

    #[test]
    fn non_convex_subhugr() {
        let (hugr, func_root) = build_hugr().unwrap();
        let (inp, out) = hugr.children(func_root).take(2).collect_tuple().unwrap();
        let sub = SubHugr::new(
            &hugr,
            func_root,
            hugr.node_outputs(inp).map(|p| (inp, p)),
            hugr.node_inputs(out).map(|p| (out, p)),
        );
        assert!(!sub.is_convex());
    }
}

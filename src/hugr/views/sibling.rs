//! Views for HUGR sibling subgraphs.
//!
//! Views into convex subgraphs of HUGRs within a single level of the
//! hierarchy, i.e. within a sibling graph. Convex subgraph are always
//! induced subgraphs, i.e. they are defined by a subset of the sibling nodes.
//!
//! Sibling subgraphs complement [`super::HierarchyView`]s in the sense that the
//! latter provide views for subgraphs defined by hierarchical relationships,
//! while the former provide views for subgraphs within a single level of the
//! hierarchy.

use itertools::Itertools;
use portgraph::{algorithms::ConvexChecker, view::Subgraph, Direction, PortView};
use thiserror::Error;

use crate::{
    ops::{
        handle::{ContainerHandle, DataflowOpID},
        OpTag, OpTrait,
    },
    types::FunctionType,
    Hugr, Node, Port, SimpleReplacement,
};

use super::{sealed::HugrInternals, HugrView};

/// A non-empty convex subgraph of a HUGR sibling graph.
///
/// A HUGR region in which all nodes share the same parent. Unlike
/// [`super::SiblingGraph`],  not all nodes of the sibling graph must be
/// included. A convex subgraph is always an induced subgraph, i.e. it is defined
/// by a set of nodes and all edges between them.

/// The incoming boundary (resp. outgoing boundary) is given by the input (resp.
/// output) ports of the subgraph that are linked to nodes outside of the subgraph.
/// The signature of the subgraph is then given by the types of the incoming
/// and outgoing boundary ports. Given a replacement with the same signature,
/// a [`SimpleReplacement`] can be constructed to rewrite the subgraph with the
/// replacement.
///
/// The ordering of the nodes in the subgraph is irrelevant to define the convex
/// subgraph, but it determines the ordering of the boundary signature.
///
/// At the moment we do not support state order edges at the subgraph boundary.
/// The `boundary_port` and `signature` methods will panic if any are found.
/// State order edges are also unsupported in replacements in
/// `create_simple_replacement`.
#[derive(Clone, Debug)]
pub struct SiblingSubgraph<'g, Base> {
    /// The underlying Hugr.
    base: &'g Base,
    /// The nodes of the induced subgraph.
    nodes: Vec<Node>,
}

impl<'g, Base: HugrInternals> SiblingSubgraph<'g, Base> {
    /// A sibling subgraph from a [`crate::ops::OpTag::DataflowParent`]-rooted HUGR.
    ///
    /// The subgraph is given by the nodes between the input and output
    /// children nodes of the parent node. If you wish to create a subgraph
    /// from another root, wrap the `region` argument in a [`super::SiblingGraph`].
    ///
    /// This will return an [`InvalidSubgraph::EmptySubgraph`] error if the
    /// subgraph is empty.
    pub fn from_dataflow_graph<Root>(dfg_graph: &'g Base) -> Result<Self, InvalidSubgraph>
    where
        Base: HugrView<RootHandle = Root>,
        Root: ContainerHandle<ChildrenHandle = DataflowOpID>,
    {
        let parent = dfg_graph.root();
        let nodes = dfg_graph.children(parent).skip(2).collect_vec();
        if nodes.is_empty() {
            Err(InvalidSubgraph::EmptySubgraph)
        } else {
            Ok(Self {
                base: dfg_graph,
                nodes,
            })
        }
    }

    /// Create a new sibling subgraph from some boundary edges.
    ///
    /// Any sibling subgraph can be defined using two sets of boundary edges
    /// $B_I$ and $B_O$, the incoming and outgoing boundary edges respectively.
    /// Intuitively, the sibling subgraph is all the edges and nodes between
    /// an edge of $B_I$ and an edge of $B_O$.
    ///
    /// The `incoming` and `outgoing` arguments give $B_I$ and $B_O$ respectively.
    /// They can be either source or target ports. We currently assume that if
    /// the source port of an outgoing boundary edge is linked to multiple
    /// target ports, then all edges from the same source port are outgoing
    /// boundary edges.
    ///
    /// More formally, the sibling subgraph of a graph $G = (V, E)$ given
    /// by sets of incoming and outoing boundary edges $B_I, B_O \subseteq E$
    /// is the graph given by the connected components of the graph
    /// $G' = (V, E \ B_I \ B_O)$ that contain at least one node that is either
    ///  - the target of an incoming boundary edge, or
    ///  - the source of an outgoing boundary edge.
    ///
    /// A subgraph is well-formed if for every edge in the HUGR
    ///  - it is in $B_I$ if and only if it has a source outside of the subgraph
    ///    and a target inside of it, and
    ///  - it is in $B_O$ if and only if it has a source inside of the subgraph
    ///    and a target outside of it.
    ///
    /// This function fails if the subgraph is not convex, if the nodes
    /// do not share a common parent or if the subgraph is empty.
    ///
    /// The order of the boundary edges is used to determine the order of the
    /// signature.
    pub fn from_boundary_edges(
        base: &'g Base,
        incoming: impl IntoIterator<Item = (Node, Port)>,
        outgoing: impl IntoIterator<Item = (Node, Port)>,
    ) -> Result<Self, InvalidSubgraph>
    where
        Base: HugrView,
    {
        let mut checker = ConvexChecker::new(base.portgraph());
        Self::from_boundary_edges_with_checker(base, incoming, outgoing, &mut checker)
    }

    /// Create a new sibling subgraph from some boundary edges.
    ///
    /// Provide a [`ConvexChecker`] instance to avoid constructing one for
    /// faster convexity check. If you do not have one, use
    /// [`SiblingSubgraph::from_boundary_edges`].
    ///
    /// Refer to [`SiblingSubgraph::from_boundary_edges`] for the full
    /// documentation.
    pub fn from_boundary_edges_with_checker(
        base: &'g Base,
        incoming: impl IntoIterator<Item = (Node, Port)>,
        outgoing: impl IntoIterator<Item = (Node, Port)>,
        checker: &mut ConvexChecker<&'g Base::Portgraph>,
    ) -> Result<Self, InvalidSubgraph>
    where
        Base: HugrView,
    {
        let pg = base.portgraph();
        let incoming = incoming.into_iter().flat_map(|(n, p)| match p.direction() {
            Direction::Outgoing => base.linked_ports(n, p).collect(),
            Direction::Incoming => vec![(n, p)],
        });
        let outgoing = outgoing.into_iter().flat_map(|(n, p)| match p.direction() {
            Direction::Incoming => base.linked_ports(n, p).collect(),
            Direction::Outgoing => vec![(n, p)],
        });
        let to_pg = |(n, p): (Node, Port)| pg.port_index(n.index, p.offset).expect("invalid port");
        // Ordering of the edges here is preserved and becomes ordering of the signature.
        let subpg = Subgraph::new_subgraph(pg, incoming.chain(outgoing).map(to_pg));
        if !subpg.is_convex_with_checker(checker) {
            return Err(InvalidSubgraph::NotConvex);
        }
        let nodes = subpg.nodes_iter().map_into().collect_vec();
        if nodes.is_empty() {
            return Err(InvalidSubgraph::EmptySubgraph);
        }
        if !nodes.iter().map(|&n| base.get_parent(n)).all_equal() {
            return Err(InvalidSubgraph::NoSharedParent);
        }
        Ok(Self { base, nodes })
    }

    /// Create a new convex sibling subgraph from a set of nodes.
    ///
    /// This fails if the set of nodes is not convex, nodes do not share a
    /// common parent or the subgraph is empty.
    pub fn try_new(base: &'g Base, nodes: Vec<Node>) -> Result<Self, InvalidSubgraph>
    where
        Base: HugrView,
    {
        let mut checker = ConvexChecker::new(base.portgraph());
        Self::try_new_with_checker(base, nodes, &mut checker)
    }

    /// Create a new convex sibling subgraph from a set of nodes.
    ///
    /// Provide a [`ConvexChecker`] instance to avoid constructing one for
    /// faster convexity check. If you do not have one, use [`SiblingSubgraph::try_new`].
    ///
    /// This fails if the set of nodes is not convex, nodes do not share a
    /// common parent or the subgraph is empty.
    pub fn try_new_with_checker(
        base: &'g Base,
        nodes: Vec<Node>,
        checker: &mut ConvexChecker<&'g Base::Portgraph>,
    ) -> Result<Self, InvalidSubgraph>
    where
        Base: HugrView,
    {
        if !checker.is_node_convex(nodes.iter().map(|n| n.index)) {
            return Err(InvalidSubgraph::NotConvex);
        }
        if nodes.is_empty() {
            return Err(InvalidSubgraph::EmptySubgraph);
        }
        if !nodes.iter().map(|&n| base.get_parent(n)).all_equal() {
            return Err(InvalidSubgraph::NoSharedParent);
        }
        Ok(Self { base, nodes })
    }

    /// An iterator over the nodes in the subgraph.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Whether a port is at the subgraph boundary.
    fn is_boundary_port(&self, n: Node, p: Port) -> bool
    where
        Base: HugrView,
    {
        // TODO: handle state order edges
        if is_order_edge(self.base, n, p) {
            unimplemented!("State order edges not supported at boundary")
        }
        self.nodes.contains(&n)
            && self
                .base
                .linked_ports(n, p)
                .any(|(n, _)| !self.nodes.contains(&n))
    }

    /// An iterator of the incoming boundary ports.
    pub fn incoming_ports(&self) -> impl Iterator<Item = (Node, Port)> + '_
    where
        Base: HugrView,
    {
        self.boundary_ports(Direction::Incoming)
    }

    /// An iterator of the outgoing boundary ports.
    pub fn outgoing_ports(&self) -> impl Iterator<Item = (Node, Port)> + '_
    where
        Base: HugrView,
    {
        self.boundary_ports(Direction::Outgoing)
    }

    /// An iterator of the boundary ports, either incoming or outgoing.
    pub fn boundary_ports(&self, dir: Direction) -> impl Iterator<Item = (Node, Port)> + '_
    where
        Base: HugrView,
    {
        self.nodes.iter().flat_map(move |&n| {
            self.base
                .node_ports(n, dir)
                .filter(move |&p| self.is_boundary_port(n, p))
                .map(move |p| (n, p))
        })
    }

    /// The signature of the subgraph.
    pub fn signature(&self) -> FunctionType
    where
        Base: HugrView,
    {
        let input = self
            .incoming_ports()
            .map(|(n, p)| {
                let sig = self.base.get_optype(n).signature();
                sig.get(p).cloned().expect("must be dataflow edge")
            })
            .collect_vec();
        let output = self
            .outgoing_ports()
            .map(|(n, p)| {
                let sig = self.base.get_optype(n).signature();
                sig.get(p).cloned().expect("must be dataflow edge")
            })
            .collect_vec();
        FunctionType::new(input, output)
    }

    /// The parent of the sibling subgraph.
    pub fn get_parent(&self) -> Node
    where
        Base: HugrView,
    {
        self.base
            .get_parent(self.nodes[0])
            .expect("invalid subgraph")
    }

    /// Construct a [`SimpleReplacement`] to replace `self` with `replacement`.
    ///
    /// `replacement` must be a hugr with DFG root and its signature must
    /// match the signature of the subgraph.
    ///
    /// May return one of the following five errors
    ///  - [`InvalidReplacement::InvalidDataflowGraph`]: the replacement
    ///    graph is not a [`crate::ops::OpTag::DataflowParent`]-rooted graph,
    ///  - [`InvalidReplacement::InvalidDataflowParent`]: the replacement does
    ///    not have an input and output node,
    ///  - [`InvalidReplacement::InvalidSignature`]: the signature of the
    ///    replacement DFG does not match the subgraph signature, or
    ///  - [`InvalidReplacement::NonConvexSubgraph`]: the sibling subgraph is not
    ///    convex.
    ///
    /// At the moment we do not support state order edges. If any are found in
    /// the replacement graph, this will panic.
    pub fn create_simple_replacement(
        &self,
        replacement: Hugr,
    ) -> Result<SimpleReplacement, InvalidReplacement>
    where
        Base: HugrView,
    {
        let removal = self.nodes().iter().copied().collect();

        let rep_root = replacement.root();
        let dfg_optype = replacement.get_optype(rep_root);
        if !OpTag::Dfg.is_superset(dfg_optype.tag()) {
            return Err(InvalidReplacement::InvalidDataflowGraph);
        }
        let Some((rep_input, rep_output)) = replacement.children(rep_root).take(2).collect_tuple()
        else {
            return Err(InvalidReplacement::InvalidDataflowParent);
        };
        if dfg_optype.signature() != self.signature() {
            return Err(InvalidReplacement::InvalidSignature);
        }

        // TODO: handle state order edges. For now panic if any are present.
        // See https://github.com/CQCL-DEV/hugr/discussions/432
        let rep_inputs = replacement.node_outputs(rep_input).map(|p| (rep_input, p));
        let rep_outputs = replacement.node_inputs(rep_output).map(|p| (rep_output, p));
        let (rep_inputs, in_order_ports): (Vec<_>, Vec<_>) =
            rep_inputs.partition(|&(n, p)| replacement.get_optype(n).signature().get(p).is_some());
        let (rep_outputs, out_order_ports): (Vec<_>, Vec<_>) =
            rep_outputs.partition(|&(n, p)| replacement.get_optype(n).signature().get(p).is_some());
        let mut order_ports = in_order_ports.into_iter().chain(out_order_ports);
        if order_ports.any(|(n, p)| is_order_edge(&replacement, n, p)) {
            unimplemented!("Found state order edges in replacement graph");
        }

        let self_inputs = self.incoming_ports();
        let self_outputs = self.outgoing_ports();
        let nu_inp = rep_inputs
            .into_iter()
            .zip_eq(self_inputs)
            .flat_map(|((rep_source_n, rep_source_p), self_target)| {
                replacement
                    .linked_ports(rep_source_n, rep_source_p)
                    .map(move |rep_target| (rep_target, self_target))
            })
            .collect();
        let nu_out = self_outputs
            .zip_eq(rep_outputs)
            .flat_map(|((self_source_n, self_source_p), (_, rep_target_p))| {
                self.base
                    .linked_ports(self_source_n, self_source_p)
                    .map(move |self_target| (self_target, rep_target_p))
            })
            .collect();

        Ok(SimpleReplacement::new(
            self.get_parent(),
            removal,
            replacement,
            nu_inp,
            nu_out,
        ))
    }
}

/// Whether a port is linked to a state order edge.
fn is_order_edge<H: HugrView>(hugr: &H, node: Node, port: Port) -> bool {
    hugr.get_optype(node).signature().get(port).is_none()
        && hugr.linked_ports(node, port).count() > 0
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
    InvalidSignature,
    /// SiblingSubgraph is not convex.
    #[error("SiblingSubgraph is not convex.")]
    NonConvexSubgraph,
}

/// Errors that can occur while constructing a [`SiblingSubgraph`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidSubgraph {
    /// The subgraph is not convex.
    #[error("The subgraph is not convex.")]
    NotConvex,
    /// Not all nodes have the same parent.
    #[error("Not a sibling subgraph.")]
    NoSharedParent,
    /// Empty subgraphs are not supported.
    #[error("Empty subgraphs are not supported.")]
    EmptySubgraph,
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{
            BuildError, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        extension::prelude::{BOOL_T, QB_T},
        hugr::views::{HierarchyView, SiblingGraph},
        ops::{
            handle::{FuncID, NodeHandle},
            OpType,
        },
        std_extensions::{logic::test::and_op, quantum::test::cx_gate},
        type_row,
    };

    use super::*;

    impl<'g, Base: HugrInternals> SiblingSubgraph<'g, Base> {
        /// A sibling subgraph from a HUGR.
        ///
        /// The subgraph is given by the sibling graph of the root. If you wish to
        /// create a subgraph from another root, wrap the argument `region` in a
        /// [`super::SiblingGraph`].
        ///
        /// This will return an [`InvalidSubgraph::EmptySubgraph`] error if the
        /// subgraph is empty.
        fn from_sibling_graph(sibling_graph: &'g Base) -> Result<Self, InvalidSubgraph>
        where
            Base: HugrView,
        {
            let root = sibling_graph.root();
            let nodes = sibling_graph.children(root).collect_vec();
            if nodes.is_empty() {
                Err(InvalidSubgraph::EmptySubgraph)
            } else {
                Ok(Self {
                    base: sibling_graph,
                    nodes,
                })
            }
        }
    }

    fn build_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            FunctionType::new_linear(type_row![QB_T, QB_T]).pure(),
        )?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let outs = dfg.add_dataflow_op(cx_gate(), dfg.input_wires())?;
            dfg.finish_with_outputs(outs.outputs())?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    /// A HUGR with a copy
    fn build_hugr_classical() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            FunctionType::new(type_row![BOOL_T], type_row![BOOL_T]).pure(),
        )?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let in_wire = dfg.input_wires().exactly_one().unwrap();
            let outs = dfg.add_dataflow_op(and_op(), [in_wire, in_wire])?;
            dfg.finish_with_outputs(outs.outputs())?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    #[test]
    fn construct_subgraph() -> Result<(), InvalidSubgraph> {
        let (hugr, func_root) = build_hugr().unwrap();
        let sibling_graph: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let from_root = SiblingSubgraph::from_sibling_graph(&sibling_graph)?;
        let region: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let from_region = SiblingSubgraph::from_sibling_graph(&region)?;
        assert_eq!(from_root.get_parent(), from_region.get_parent());
        assert_eq!(from_root.signature(), from_region.signature());
        Ok(())
    }

    #[test]
    fn construct_simple_replacement() -> Result<(), InvalidSubgraph> {
        let (mut hugr, func_root) = build_hugr().unwrap();
        let func: SiblingGraph<'_, FuncID<true>> = SiblingGraph::new(&hugr, func_root);
        let sub = SiblingSubgraph::from_dataflow_graph(&func)?;

        let empty_dfg = {
            let builder = DFGBuilder::new(FunctionType::new_linear(type_row![QB_T, QB_T])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        let rep = sub.create_simple_replacement(empty_dfg).unwrap();

        assert_eq!(rep.removal.len(), 1);

        assert_eq!(hugr.node_count(), 5); // Module + Def + In + CX + Out
        hugr.apply_rewrite(rep).unwrap();
        assert_eq!(hugr.node_count(), 4); // Module + Def + In + Out

        Ok(())
    }

    #[test]
    fn test_signature() -> Result<(), InvalidSubgraph> {
        let (hugr, dfg) = build_hugr().unwrap();
        let func: SiblingGraph<'_, FuncID<true>> = SiblingGraph::new(&hugr, dfg);
        let sub = SiblingSubgraph::from_dataflow_graph(&func)?;
        assert_eq!(
            sub.signature(),
            FunctionType::new_linear(type_row![QB_T, QB_T])
        );
        Ok(())
    }

    #[test]
    fn construct_simple_replacement_invalid_signature() -> Result<(), InvalidSubgraph> {
        let (hugr, dfg) = build_hugr().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, dfg);
        let sub = SiblingSubgraph::from_sibling_graph(&func)?;

        let empty_dfg = {
            let builder = DFGBuilder::new(FunctionType::new_linear(type_row![QB_T])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        assert_eq!(
            sub.create_simple_replacement(empty_dfg).unwrap_err(),
            InvalidReplacement::InvalidSignature
        );
        Ok(())
    }

    #[test]
    fn convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func: SiblingGraph<'_, FuncID<true>> = SiblingGraph::new(&hugr, func_root);
        assert_eq!(
            SiblingSubgraph::from_dataflow_graph(&func)
                .unwrap()
                .nodes()
                .len(),
            1
        )
    }

    #[test]
    fn convex_subgraph_2() {
        let (hugr, func_root) = build_hugr().unwrap();
        let (inp, out) = hugr.children(func_root).take(2).collect_tuple().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        // All graph except input/output nodes
        SiblingSubgraph::from_boundary_edges(
            &func,
            hugr.node_outputs(inp).map(|p| (inp, p)),
            hugr.node_inputs(out).map(|p| (out, p)),
        )
        .unwrap();
    }

    #[test]
    fn non_convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func: SiblingGraph<'_> = SiblingGraph::new(&hugr, func_root);
        let (inp, _) = hugr.children(func_root).take(2).collect_tuple().unwrap();
        let first_cx_edge = hugr.node_outputs(inp).next().unwrap();
        // All graph but one edge
        assert!(matches!(
            SiblingSubgraph::from_boundary_edges(
                &func,
                [(inp, first_cx_edge)],
                [(inp, first_cx_edge)],
            ),
            Err(InvalidSubgraph::NotConvex)
        ));
    }

    #[test]
    fn preserve_signature() {
        let (hugr, func_root) = build_hugr_classical().unwrap();
        let func: SiblingGraph<'_, FuncID<true>> = SiblingGraph::new(&hugr, func_root);
        let func = SiblingSubgraph::from_dataflow_graph(&func).unwrap();
        let OpType::FuncDefn(func_defn) = hugr.get_optype(func_root) else { panic!()};
        assert_eq!(func_defn.signature, func.signature())
    }
}

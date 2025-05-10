//! Views for HUGR sibling subgraphs.
//!
//! Views into convex subgraphs of HUGRs within a single level of the
//! hierarchy, i.e. within a sibling graph. Convex subgraph are always
//! induced subgraphs, i.e. they are defined by a subset of the sibling nodes.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::mem;

use itertools::Itertools;
use portgraph::algorithms::ConvexChecker;
use portgraph::boundary::Boundary;
use portgraph::LinkView;
use portgraph::{view::Subgraph, Direction, PortView};
use thiserror::Error;

use crate::builder::{Container, FunctionBuilder};
use crate::core::HugrNode;
use crate::hugr::internal::{HugrInternals, PortgraphNodeMap};
use crate::hugr::{HugrMut, HugrView};
use crate::ops::dataflow::DataflowOpTrait;
use crate::ops::handle::{ContainerHandle, DataflowOpID};
use crate::ops::{NamedOp, OpTag, OpTrait, OpType};
use crate::types::{Signature, Type};
use crate::{Hugr, IncomingPort, Node, OutgoingPort, Port, SimpleReplacement};

use super::root_checked::RootCheckable;

/// A non-empty convex subgraph of a HUGR sibling graph.
///
/// A HUGR region in which all nodes share the same parent. A convex subgraph is
/// always an induced subgraph, i.e. it is defined by a set of nodes and all
/// edges between them.
///
/// The incoming boundary (resp. outgoing boundary) is given by the input (resp.
/// output) ports of the subgraph that are linked to nodes outside of the
/// subgraph. The signature of the subgraph is then given by the types of the
/// incoming and outgoing boundary ports. Given a replacement with the same
/// signature, a [`SimpleReplacement`] can be constructed to rewrite the
/// subgraph with the replacement.
///
/// The ordering of the nodes in the subgraph is irrelevant to define the convex
/// subgraph, but it determines the ordering of the boundary signature.
///
/// No reference to the underlying graph is kept. Thus most of the subgraph
/// methods expect a reference to the Hugr as an argument.
///
/// At the moment we do not support state order edges at the subgraph boundary.
/// The `boundary_port` and `signature` methods will panic if any are found.
/// State order edges are also unsupported in replacements in
/// `create_simple_replacement`.
#[derive(Clone, Debug)]
pub struct SiblingSubgraph<N = Node> {
    /// The nodes of the induced subgraph.
    nodes: Vec<N>,
    /// The input ports of the subgraph.
    ///
    /// Grouped by input parameter. Each port must be unique and belong to a
    /// node in `nodes`.
    inputs: Vec<Vec<(N, IncomingPort)>>,
    /// The output ports of the subgraph.
    ///
    /// Repeated ports are allowed and correspond to copying the output. Every
    /// port must belong to a node in `nodes`.
    outputs: Vec<(N, OutgoingPort)>,
}

/// The type of the incoming boundary of [`SiblingSubgraph`].
///
/// The nested vec represents a partition of the incoming boundary ports by
/// input parameter. A set in the partition that has more than one element
/// corresponds to an input parameter that is copied and useful multiple times
/// in the subgraph.
pub type IncomingPorts<N = Node> = Vec<Vec<(N, IncomingPort)>>;
/// The type of the outgoing boundary of [`SiblingSubgraph`].
pub type OutgoingPorts<N = Node> = Vec<(N, OutgoingPort)>;

impl<N: HugrNode> SiblingSubgraph<N> {
    /// A sibling subgraph from a [`crate::ops::OpTag::DataflowParent`]-rooted
    /// HUGR.
    ///
    /// The subgraph is given by the nodes between the input and output children
    /// nodes of the root node.
    /// Wires connecting the input and output nodes are ignored. Note that due
    /// to this the resulting subgraph's signature may not match the signature
    /// of the dataflow parent.
    ///
    /// This will return an [`InvalidSubgraph::EmptySubgraph`] error if the
    /// subgraph is empty.
    pub fn try_new_dataflow_subgraph<'h, H, Root>(
        dfg_graph: impl RootCheckable<&'h H, Root>,
    ) -> Result<Self, InvalidSubgraph<N>>
    where
        H: 'h + Clone + HugrView<Node = N>,
        Root: ContainerHandle<N, ChildrenHandle = DataflowOpID>,
    {
        let Ok(dfg_graph) = dfg_graph.try_into_checked() else {
            return Err(InvalidSubgraph::NonDataflowRegion);
        };
        let dfg_graph = dfg_graph.into_hugr();

        let parent = HugrView::entrypoint(&dfg_graph);
        let nodes = dfg_graph.children(parent).skip(2).collect_vec();
        let (inputs, outputs) = get_input_output_ports(dfg_graph);

        validate_subgraph(dfg_graph, &nodes, &inputs, &outputs)?;

        if nodes.is_empty() {
            Err(InvalidSubgraph::EmptySubgraph)
        } else {
            Ok(Self {
                nodes,
                inputs,
                outputs,
            })
        }
    }

    /// Create a new convex sibling subgraph from input and output boundaries.
    ///
    /// Any sibling subgraph can be defined using two sets of boundary edges
    /// $B_I$ and $B_O$, the incoming and outgoing boundary edges respectively.
    /// Intuitively, the sibling subgraph is all the edges and nodes "between"
    /// an edge of $B_I$ and an edge of $B_O$.
    ///
    /// ## Definition
    ///
    /// More formally, the sibling subgraph of a graph $G = (V, E)$ given
    /// by sets of incoming and outgoing boundary edges $B_I, B_O \subseteq E$
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
    /// ## Arguments
    ///
    /// The `incoming` and `outgoing` arguments give $B_I$ and $B_O$ respectively.
    /// Incoming edges must be given by incoming ports and outgoing edges by
    /// outgoing ports. The ordering of the incoming and outgoing ports defines
    /// the signature of the subgraph.
    ///
    /// Incoming boundary ports must be unique and partitioned by input
    /// parameter: two ports within the same set of the partition must be
    /// copyable and will result in the input being copied. Outgoing
    /// boundary ports are given in a list and can appear multiple times if
    /// they are copyable, in which case the output will be copied.
    ///
    /// ## Errors
    ///
    /// This function fails if the subgraph is not convex, if the nodes
    /// do not share a common parent or if the subgraph is empty.
    pub fn try_new(
        incoming: IncomingPorts<N>,
        outgoing: OutgoingPorts<N>,
        hugr: &impl HugrView<Node = N>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let parent = pick_parent(hugr, &incoming, &outgoing)?;
        let checker = TopoConvexChecker::new(hugr, parent);
        Self::try_new_with_checker(incoming, outgoing, hugr, &checker)
    }

    /// Create a new convex sibling subgraph from input and output boundaries.
    ///
    /// Provide a [`ConvexChecker`] instance to avoid constructing one for
    /// faster convexity check. If you do not have one, use
    /// [`SiblingSubgraph::try_new`].
    ///
    /// Refer to [`SiblingSubgraph::try_new`] for the full
    /// documentation.
    pub fn try_new_with_checker<H: HugrView<Node = N>>(
        inputs: IncomingPorts<N>,
        outputs: OutgoingPorts<N>,
        hugr: &H,
        checker: &impl ConvexChecker,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let parent = pick_parent(hugr, &inputs, &outputs)?;
        let (region, node_map) = hugr.region_portgraph(parent);

        // Ordering of the edges here is preserved and becomes ordering of the signature.
        let boundary = make_boundary::<H>(&region, &node_map, &inputs, &outputs);
        let subpg = Subgraph::new_subgraph(region, boundary);
        let nodes = subpg
            .nodes_iter()
            .map(|index| node_map.from_portgraph(index))
            .collect_vec();
        validate_subgraph(hugr, &nodes, &inputs, &outputs)?;

        if nodes.len() > 1 && !subpg.is_convex_with_checker(checker) {
            return Err(InvalidSubgraph::NotConvex);
        }

        Ok(Self {
            nodes,
            inputs,
            outputs,
        })
    }

    /// Create a subgraph from a set of nodes.
    ///
    /// The incoming boundary is given by the set of edges with a source
    /// not in nodes and a target in nodes. Conversely, the outgoing boundary
    /// is given by the set of edges with a source in nodes and a target not
    /// in nodes.
    ///
    /// The subgraph signature will be given by the types of the incoming and
    /// outgoing edges ordered by the node order in `nodes` and within each node
    /// by the port order.
    ///
    /// The in- and out-arity of the signature will match the
    /// number of incoming and outgoing edges respectively. In particular, the
    /// assumption is made that no two incoming edges have the same source
    /// (no copy nodes at the input boundary).
    pub fn try_from_nodes(
        nodes: impl Into<Vec<N>>,
        hugr: &impl HugrView<Node = N>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let nodes = nodes.into();
        let Some(node) = nodes.first() else {
            return Err(InvalidSubgraph::EmptySubgraph);
        };
        let parent = hugr
            .get_parent(*node)
            .ok_or(InvalidSubgraph::OrphanNode { orphan: *node })?;

        let checker = TopoConvexChecker::new(hugr, parent);
        Self::try_from_nodes_with_checker(nodes, hugr, &checker)
    }

    /// Create a subgraph from a set of nodes.
    ///
    /// Provide a [`ConvexChecker`] instance to avoid constructing one for
    /// faster convexity check. If you do not have one, use
    /// [`SiblingSubgraph::try_from_nodes`].
    ///
    /// Refer to [`SiblingSubgraph::try_from_nodes`] for the full
    /// documentation.
    pub fn try_from_nodes_with_checker<'c, 'h: 'c>(
        nodes: impl Into<Vec<N>>,
        hugr: &'h impl HugrView<Node = N>,
        checker: &impl ConvexChecker,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let nodes = nodes.into();

        if nodes.is_empty() {
            return Err(InvalidSubgraph::EmptySubgraph);
        }

        let nodes_set = nodes.iter().copied().collect::<HashSet<_>>();
        let incoming_edges = nodes
            .iter()
            .flat_map(|&n| hugr.node_inputs(n).map(move |p| (n, p)));
        let outgoing_edges = nodes
            .iter()
            .flat_map(|&n| hugr.node_outputs(n).map(move |p| (n, p)));
        let inputs = incoming_edges
            .filter(|&(n, p)| {
                if !hugr.is_linked(n, p) {
                    return false;
                }
                let (out_n, _) = hugr.single_linked_output(n, p).unwrap();
                !nodes_set.contains(&out_n)
            })
            // Every incoming edge is its own input.
            .map(|p| vec![p])
            .collect_vec();
        let outputs = outgoing_edges
            .filter(|&(n, p)| {
                hugr.linked_ports(n, p)
                    .any(|(n1, _)| !nodes_set.contains(&n1))
            })
            .collect_vec();
        Self::try_new_with_checker(inputs, outputs, hugr, checker)
    }

    /// Create a subgraph containing a single node.
    ///
    /// The subgraph signature will be given by signature of the node.
    pub fn from_node(node: N, hugr: &impl HugrView<Node = N>) -> Self {
        let nodes = vec![node];
        let inputs = hugr
            .node_inputs(node)
            .filter(|&p| hugr.is_linked(node, p))
            .map(|p| vec![(node, p)])
            .collect_vec();
        let outputs = hugr
            .node_outputs(node)
            .filter_map(|p| {
                // accept linked outputs or unlinked value outputs
                {
                    hugr.is_linked(node, p)
                        || HugrView::get_optype(&hugr, node)
                            .port_kind(p)
                            .is_some_and(|k| k.is_value())
                }
                .then_some((node, p))
            })
            .collect_vec();

        Self {
            nodes,
            inputs,
            outputs,
        }
    }

    /// An iterator over the nodes in the subgraph.
    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }

    /// The number of nodes in the subgraph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the computed [`IncomingPorts`] of the subgraph.
    pub fn incoming_ports(&self) -> &IncomingPorts<N> {
        &self.inputs
    }

    /// Returns the computed [`OutgoingPorts`] of the subgraph.
    pub fn outgoing_ports(&self) -> &OutgoingPorts<N> {
        &self.outputs
    }

    /// The signature of the subgraph.
    pub fn signature(&self, hugr: &impl HugrView<Node = N>) -> Signature {
        let input = self
            .inputs
            .iter()
            .map(|part| {
                let &(n, p) = part.iter().next().expect("is non-empty");
                let sig = hugr.signature(n).expect("must have dataflow signature");
                sig.port_type(p).cloned().expect("must be dataflow edge")
            })
            .collect_vec();
        let output = self
            .outputs
            .iter()
            .map(|&(n, p)| {
                let sig = hugr.signature(n).expect("must have dataflow signature");
                sig.port_type(p).cloned().expect("must be dataflow edge")
            })
            .collect_vec();
        Signature::new(input, output)
    }

    /// The parent of the sibling subgraph.
    pub fn get_parent(&self, hugr: &impl HugrView<Node = N>) -> N {
        hugr.get_parent(self.nodes[0]).expect("invalid subgraph")
    }

    /// Map the nodes in the subgraph according to `node_map`.
    ///
    /// This does not check convexity. It is up to the caller to ensure that
    /// the mapped subgraph obtained from this is convex in the new Hugr.
    pub(crate) fn map_nodes<N2: HugrNode>(
        &self,
        node_map: impl Fn(N) -> N2,
    ) -> SiblingSubgraph<N2> {
        let nodes = self.nodes.iter().map(|&n| node_map(n)).collect_vec();
        let inputs = self
            .inputs
            .iter()
            .map(|part| part.iter().map(|&(n, p)| (node_map(n), p)).collect_vec())
            .collect_vec();
        let outputs = self
            .outputs
            .iter()
            .map(|&(n, p)| (node_map(n), p))
            .collect_vec();
        SiblingSubgraph {
            nodes,
            inputs,
            outputs,
        }
    }

    /// Construct a [`SimpleReplacement`] to replace `self` with `replacement`.
    ///
    /// `replacement` must be a hugr with DFG root and its signature must
    /// match the signature of the subgraph.
    ///
    /// May return one of the following five errors
    ///  - [`InvalidReplacement::InvalidDataflowGraph`]: the replacement
    ///    graph is not a [`crate::ops::OpTag::DataflowParent`]-rooted graph,
    ///  - [`InvalidReplacement::InvalidSignature`]: the signature of the
    ///    replacement DFG does not match the subgraph signature, or
    ///  - [`InvalidReplacement::NonConvexSubgraph`]: the sibling subgraph is not
    ///    convex.
    ///
    /// At the moment we do not support state order edges. If any are found in
    /// the replacement graph, this will panic.
    pub fn create_simple_replacement(
        &self,
        hugr: &impl HugrView<Node = N>,
        replacement: Hugr,
    ) -> Result<SimpleReplacement<N>, InvalidReplacement> {
        let rep_root = replacement.entrypoint();
        let dfg_optype = replacement.get_optype(rep_root);
        if !OpTag::Dfg.is_superset(dfg_optype.tag()) {
            return Err(InvalidReplacement::InvalidDataflowGraph {
                node: rep_root,
                op: dfg_optype.clone(),
            });
        }
        let [rep_input, rep_output] = replacement
            .get_io(rep_root)
            .expect("DFG root in the replacement does not have input and output nodes.");

        let current_signature = self.signature(hugr);
        let new_signature = dfg_optype.dataflow_signature();
        if new_signature.as_ref().map(|s| &s.input) != Some(&current_signature.input)
            || new_signature.as_ref().map(|s| &s.output) != Some(&current_signature.output)
        {
            return Err(InvalidReplacement::InvalidSignature {
                expected: self.signature(hugr),
                actual: dfg_optype.dataflow_signature().map(|s| s.into_owned()),
            });
        }

        // TODO: handle state order edges. For now panic if any are present.
        // See https://github.com/CQCL/hugr/discussions/432
        let rep_inputs = replacement.node_outputs(rep_input).map(|p| (rep_input, p));
        let rep_outputs = replacement.node_inputs(rep_output).map(|p| (rep_output, p));
        let (rep_inputs, in_order_ports): (Vec<_>, Vec<_>) = rep_inputs.partition(|&(n, p)| {
            replacement
                .signature(n)
                .is_some_and(|s| s.port_type(p).is_some())
        });
        let (rep_outputs, out_order_ports): (Vec<_>, Vec<_>) = rep_outputs.partition(|&(n, p)| {
            replacement
                .signature(n)
                .is_some_and(|s| s.port_type(p).is_some())
        });

        if iter_io(&vec![out_order_ports], &in_order_ports)
            .any(|(n, p)| is_order_edge(&replacement, n, p))
        {
            unimplemented!("Found state order edges in replacement graph");
        }

        let nu_inp = rep_inputs
            .into_iter()
            .zip_eq(&self.inputs)
            .flat_map(|((rep_source_n, rep_source_p), self_targets)| {
                replacement
                    .linked_inputs(rep_source_n, rep_source_p)
                    .flat_map(move |rep_target| {
                        self_targets
                            .iter()
                            .map(move |&self_target| (rep_target, self_target))
                    })
            })
            .collect();
        let nu_out: HashMap<_, _> = self
            .outputs
            .iter()
            .zip_eq(rep_outputs)
            .map(|(&self_target, (_, rep_target_p))| (self_target, rep_target_p))
            .collect();

        Ok(SimpleReplacement::new(
            self.clone(),
            replacement,
            nu_inp,
            nu_out,
        ))
    }

    /// Create a new Hugr containing only the subgraph.
    ///
    /// The new Hugr will contain a [FuncDefn][crate::ops::FuncDefn] root
    /// with the same signature as the subgraph and the specified `name`
    pub fn extract_subgraph(
        &self,
        hugr: &impl HugrView<Node = N>,
        name: impl Into<String>,
    ) -> Hugr {
        let mut builder = FunctionBuilder::new(name, self.signature(hugr)).unwrap();
        // Take the unfinished Hugr from the builder, to avoid unnecessary
        // validation checks that require connecting the inputs and outputs.
        let mut extracted = mem::take(builder.hugr_mut());
        let node_map = extracted.insert_subgraph(extracted.entrypoint(), hugr, self);

        // Connect the inserted nodes in-between the input and output nodes.
        let [inp, out] = extracted.get_io(extracted.entrypoint()).unwrap();
        let inputs = extracted.node_outputs(inp).zip(self.inputs.iter());
        let outputs = extracted.node_inputs(out).zip(self.outputs.iter());
        let mut connections = Vec::with_capacity(inputs.size_hint().0 + outputs.size_hint().0);

        for (inp_port, repl_ports) in inputs {
            for (repl_node, repl_port) in repl_ports {
                connections.push((inp, inp_port, node_map[repl_node], *repl_port));
            }
        }
        for (out_port, (repl_node, repl_port)) in outputs {
            connections.push((node_map[repl_node], *repl_port, out, out_port));
        }

        for (src, src_port, dst, dst_port) in connections {
            extracted.connect(src, src_port, dst, dst_port);
        }

        extracted
    }
}

/// Returns an iterator over the input ports.
fn iter_incoming<N: HugrNode>(
    inputs: &IncomingPorts<N>,
) -> impl Iterator<Item = (N, IncomingPort)> + '_ {
    inputs.iter().flat_map(|part| part.iter().copied())
}

/// Returns an iterator over the output ports.
fn iter_outgoing<N: HugrNode>(
    outputs: &OutgoingPorts<N>,
) -> impl Iterator<Item = (N, OutgoingPort)> + '_ {
    outputs.iter().copied()
}

/// Returns an iterator over both incoming and outgoing ports.
fn iter_io<'a, N: HugrNode>(
    inputs: &'a IncomingPorts<N>,
    outputs: &'a OutgoingPorts<N>,
) -> impl Iterator<Item = (N, Port)> + 'a {
    iter_incoming(inputs)
        .map(|(n, p)| (n, Port::from(p)))
        .chain(iter_outgoing(outputs).map(|(n, p)| (n, Port::from(p))))
}

/// Pick a parent node from the set of incoming and outgoing ports.
///
/// This *does not* validate that all nodes have the same parent, but just picks
/// the first one found.
///
/// # Errors
///
/// If there are no nodes in the subgraph, or if the first node does not have a
/// parent, this will return an error.
fn pick_parent<'a, N: HugrNode>(
    hugr: &impl HugrView<Node = N>,
    inputs: &'a IncomingPorts<N>,
    outputs: &'a OutgoingPorts<N>,
) -> Result<N, InvalidSubgraph<N>> {
    // Pick an arbitrary node so we know the shared parent.
    let Some(node) = iter_incoming(inputs)
        .map(|(n, _)| n)
        .chain(iter_outgoing(outputs).map(|(n, _)| n))
        .next()
    else {
        return Err(InvalidSubgraph::EmptySubgraph);
    };
    let Some(parent) = hugr.get_parent(node) else {
        return Err(InvalidSubgraph::OrphanNode { orphan: node });
    };

    Ok(parent)
}

fn make_boundary<'a, H: HugrView>(
    region: &impl LinkView,
    node_map: &H::RegionPortgraphNodes,
    inputs: &'a IncomingPorts<H::Node>,
    outputs: &'a OutgoingPorts<H::Node>,
) -> Boundary {
    let to_pg_index = |n: H::Node, p: Port| {
        region
            .port_index(node_map.to_portgraph(n), p.pg_offset())
            .unwrap()
    };
    Boundary::new(
        iter_incoming(inputs).map(|(n, p)| to_pg_index(n, p.into())),
        iter_outgoing(outputs).map(|(n, p)| to_pg_index(n, p.into())),
    )
}

type CheckerRegion<'g, Base> =
    portgraph::view::FlatRegion<'g, <Base as HugrInternals>::RegionPortgraph<'g>>;

/// Precompute convexity information for a HUGR.
///
/// This can be used when constructing multiple sibling subgraphs to speed up
/// convexity checking.
pub struct TopoConvexChecker<'g, Base: 'g + HugrView> {
    /// The base HUGR to check convexity on.
    base: &'g Base,
    /// The parent of the region where we are checking convexity.
    region_parent: Base::Node,
    /// A lazily initialized convexity checker, along with a map from nodes in the region to `Base` nodes.
    checker: OnceCell<(
        portgraph::algorithms::TopoConvexChecker<CheckerRegion<'g, Base>>,
        Base::RegionPortgraphNodes,
    )>,
}

impl<'g, Base: HugrView> TopoConvexChecker<'g, Base> {
    /// Create a new convexity checker.
    pub fn new(base: &'g Base, region_parent: Base::Node) -> Self {
        Self {
            base,
            region_parent,
            checker: OnceCell::new(),
        }
    }

    /// Returns the portgraph convexity checker, initializing it if necessary.
    fn init_checker(
        &self,
    ) -> &(
        portgraph::algorithms::TopoConvexChecker<CheckerRegion<'g, Base>>,
        Base::RegionPortgraphNodes,
    ) {
        self.checker.get_or_init(|| {
            let (region, node_map) = self.base.region_portgraph(self.region_parent);
            let checker = portgraph::algorithms::TopoConvexChecker::new(region);
            (checker, node_map)
        })
    }

    /// Returns the portgraph convexity checker, initializing it if necessary.
    fn get_checker(
        &self,
    ) -> &portgraph::algorithms::TopoConvexChecker<
        portgraph::view::FlatRegion<'g, Base::RegionPortgraph<'g>>,
    > {
        &self.init_checker().0
    }

    /// Returns the node map from the region to the base HUGR.
    #[expect(dead_code)]
    fn get_node_map(&self) -> &Base::RegionPortgraphNodes {
        &self.init_checker().1
    }
}

impl<Base: HugrView> ConvexChecker for TopoConvexChecker<'_, Base> {
    fn is_convex(
        &self,
        nodes: impl IntoIterator<Item = portgraph::NodeIndex>,
        inputs: impl IntoIterator<Item = portgraph::PortIndex>,
        outputs: impl IntoIterator<Item = portgraph::PortIndex>,
    ) -> bool {
        let mut nodes = nodes.into_iter().multipeek();
        // If the node iterator contains less than two nodes, the subgraph is
        // trivially convex.
        if nodes.peek().is_none() || nodes.peek().is_none() {
            return true;
        };
        self.get_checker().is_convex(nodes, inputs, outputs)
    }
}

/// The type of all ports in the iterator.
///
/// If the array is empty or a port does not exist, returns `None`.
fn get_edge_type<H: HugrView, P: Into<Port> + Copy>(
    hugr: &H,
    ports: &[(H::Node, P)],
) -> Option<Type> {
    let &(n, p) = ports.first()?;
    let edge_t = hugr.signature(n)?.port_type(p)?.clone();
    ports
        .iter()
        .all(|&(n, p)| {
            hugr.signature(n)
                .is_some_and(|s| s.port_type(p) == Some(&edge_t))
        })
        .then_some(edge_t)
}

/// Whether a subgraph is valid.
///
/// Verifies that input and output ports are valid subgraph boundaries, i.e. they belong
/// to nodes within the subgraph and are linked to at least one node outside of the subgraph.
/// This does NOT check convexity proper, i.e. whether the set of nodes form a convex
/// induced graph.
fn validate_subgraph<H: HugrView>(
    hugr: &H,
    nodes: &[H::Node],
    inputs: &IncomingPorts<H::Node>,
    outputs: &OutgoingPorts<H::Node>,
) -> Result<(), InvalidSubgraph<H::Node>> {
    // Copy of the nodes for fast lookup.
    let node_set = nodes.iter().copied().collect::<HashSet<_>>();

    // Check nodes is not empty
    if nodes.is_empty() {
        return Err(InvalidSubgraph::EmptySubgraph);
    }
    // Check all nodes share parent
    if !nodes.iter().map(|&n| hugr.get_parent(n)).all_equal() {
        let first_node = nodes[0];
        let first_parent = hugr
            .get_parent(first_node)
            .ok_or(InvalidSubgraph::OrphanNode { orphan: first_node })?;
        let other_node = *nodes
            .iter()
            .skip(1)
            .find(|&&n| hugr.get_parent(n) != Some(first_parent))
            .unwrap();
        let other_parent = hugr
            .get_parent(other_node)
            .ok_or(InvalidSubgraph::OrphanNode { orphan: other_node })?;
        return Err(InvalidSubgraph::NoSharedParent {
            first_node,
            first_parent,
            other_node,
            other_parent,
        });
    }

    // Check there are no linked "other" ports
    if iter_io(inputs, outputs).any(|(n, p)| is_order_edge(hugr, n, p)) {
        unimplemented!("Connected order edges not supported at the boundary")
    }

    let boundary_ports = iter_io(inputs, outputs).collect_vec();
    // Check that the boundary ports are all in the subgraph.
    if let Some(&(n, p)) = boundary_ports.iter().find(|(n, _)| !node_set.contains(n)) {
        Err(InvalidSubgraphBoundary::PortNodeNotInSet(n, p))?;
    };
    // Check that every inside port has at least one linked port outside.
    if let Some(&(n, p)) = boundary_ports.iter().find(|&&(n, p)| {
        hugr.linked_ports(n, p)
            .all(|(n1, _)| node_set.contains(&n1))
    }) {
        Err(InvalidSubgraphBoundary::DisconnectedBoundaryPort(n, p))?;
    };

    // Check that every incoming port of a node in the subgraph whose source is not in the subgraph
    // belongs to inputs.
    if nodes.iter().any(|&n| {
        hugr.node_inputs(n).any(|p| {
            hugr.linked_ports(n, p).any(|(n1, _)| {
                !node_set.contains(&n1) && !inputs.iter().any(|nps| nps.contains(&(n, p)))
            })
        })
    }) {
        return Err(InvalidSubgraph::NotConvex);
    }
    // Check that every outgoing port of a node in the subgraph whose target is not in the subgraph
    // belongs to outputs.
    if nodes.iter().any(|&n| {
        hugr.node_outputs(n).any(|p| {
            hugr.linked_ports(n, p)
                .any(|(n1, _)| !node_set.contains(&n1) && !outputs.contains(&(n, p)))
        })
    }) {
        return Err(InvalidSubgraph::NotConvex);
    }

    // Check inputs are unique
    if !inputs.iter().flatten().all_unique() {
        return Err(InvalidSubgraphBoundary::NonUniqueInput.into());
    }

    // Check no incoming partition is empty
    if inputs.iter().any(|p| p.is_empty()) {
        return Err(InvalidSubgraphBoundary::EmptyPartition.into());
    }

    // Check edge types are equal within partition and copyable if partition size > 1
    if let Some((i, _)) = inputs.iter().enumerate().find(|(_, ports)| {
        let Some(edge_t) = get_edge_type(hugr, ports) else {
            return true;
        };
        let require_copy = ports.len() > 1;
        require_copy && !edge_t.copyable()
    }) {
        Err(InvalidSubgraphBoundary::MismatchedTypes(i))?;
    };

    Ok(())
}

fn get_input_output_ports<H: HugrView>(
    hugr: &H,
) -> (IncomingPorts<H::Node>, OutgoingPorts<H::Node>) {
    let [inp, out] = hugr
        .get_io(HugrView::entrypoint(&hugr))
        .expect("invalid DFG");
    if has_other_edge(hugr, inp, Direction::Outgoing) {
        unimplemented!("Non-dataflow output not supported at input node")
    }
    let dfg_inputs = HugrView::get_optype(&hugr, inp)
        .as_input()
        .unwrap()
        .signature()
        .output_ports();
    if has_other_edge(hugr, out, Direction::Incoming) {
        unimplemented!("Non-dataflow input not supported at output node")
    }
    let dfg_outputs = HugrView::get_optype(&hugr, out)
        .as_output()
        .unwrap()
        .signature()
        .input_ports();

    // Collect for each port in the input the set of target ports, filtering
    // direct wires to the output.
    let inputs = dfg_inputs
        .into_iter()
        .map(|p| {
            hugr.linked_inputs(inp, p)
                .filter(|&(n, _)| n != out)
                .collect_vec()
        })
        .filter(|v| !v.is_empty())
        .collect();
    // Collect for each port in the output the set of source ports, filtering
    // direct wires to the input.
    let outputs = dfg_outputs
        .into_iter()
        .filter_map(|p| hugr.linked_outputs(out, p).find(|&(n, _)| n != inp))
        .collect();
    (inputs, outputs)
}

/// Whether a port is linked to a state order edge.
fn is_order_edge<H: HugrView>(hugr: &H, node: H::Node, port: Port) -> bool {
    let op = hugr.get_optype(node);
    op.other_port(port.direction()) == Some(port) && hugr.is_linked(node, port)
}

/// Whether node has a non-df linked port in the given direction.
fn has_other_edge<H: HugrView>(hugr: &H, node: H::Node, dir: Direction) -> bool {
    let op = hugr.get_optype(node);
    op.other_port_kind(dir).is_some() && hugr.is_linked(node, op.other_port(dir).unwrap())
}

/// Errors that can occur while constructing a [`SimpleReplacement`].
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum InvalidReplacement {
    /// No DataflowParent root in replacement graph.
    #[error("The root of the replacement {node} is a {}, but only OpType::DFGs are supported.", op.name())]
    InvalidDataflowGraph {
        /// The node ID of the root node.
        node: Node,
        /// The op type of the root node.
        op: OpType,
    },
    /// Replacement graph type mismatch.
    #[error(
        "Replacement graph type mismatch. Expected {expected}, got {}.",
        actual.clone().map_or("none".to_string(), |t| t.to_string()))
    ]
    InvalidSignature {
        /// The expected signature.
        expected: Signature,
        /// The actual signature.
        actual: Option<Signature>,
    },
    /// SiblingSubgraph is not convex.
    #[error("SiblingSubgraph is not convex.")]
    NonConvexSubgraph,
}

/// Errors that can occur while constructing a [`SiblingSubgraph`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidSubgraph<N: HugrNode = Node> {
    /// The subgraph is not convex.
    #[error("The subgraph is not convex.")]
    NotConvex,
    /// Not all nodes have the same parent.
    #[error(
        "Not a sibling subgraph. {first_node} has parent {first_parent}, but {other_node} has parent {other_parent}.",
    )]
    NoSharedParent {
        /// The first node.
        first_node: N,
        /// The parent of the first node.
        first_parent: N,
        /// The other node.
        other_node: N,
        /// The parent of the other node.
        other_parent: N,
    },
    /// Not all nodes have the same parent.
    #[error("Not a sibling subgraph. {orphan} has no parent")]
    OrphanNode {
        /// The orphan node.
        orphan: N,
    },
    /// Empty subgraphs are not supported.
    #[error("Empty subgraphs are not supported.")]
    EmptySubgraph,
    /// An invalid boundary port was found.
    #[error("Invalid boundary port.")]
    InvalidBoundary(#[from] InvalidSubgraphBoundary<N>),
    /// The hugr region is not a dataflow graph.
    #[error("SiblingSubgraphs may only be defined on dataflow regions.")]
    NonDataflowRegion,
}

/// Errors that can occur while constructing a [`SiblingSubgraph`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidSubgraphBoundary<N: HugrNode = Node> {
    /// A boundary port's node is not in the set of nodes.
    #[error("(node {0}, port {1}) is in the boundary, but node {0} is not in the set.")]
    PortNodeNotInSet(N, Port),
    /// A boundary port has no connections outside the subgraph.
    #[error("(node {0}, port {1}) is in the boundary, but the port is not connected to a node outside the subgraph.")]
    DisconnectedBoundaryPort(N, Port),
    /// There's a non-unique input-boundary port.
    #[error("A port in the input boundary is used multiple times.")]
    NonUniqueInput,
    /// There's an empty partition in the input boundary.
    #[error("A partition in the input boundary is empty.")]
    EmptyPartition,
    /// Different types in a partition of the input boundary.
    #[error("The partition {0} in the input boundary has ports with different types.")]
    MismatchedTypes(usize),
}

#[cfg(test)]
mod tests {
    use cool_asserts::assert_matches;

    use crate::builder::inout_sig;
    use crate::hugr::Patch;
    use crate::ops::Const;
    use crate::std_extensions::arithmetic::float_types::ConstF64;
    use crate::std_extensions::logic::LogicOp;
    use crate::type_row;
    use crate::utils::test_quantum_extension::{cx_gate, rz_f64};
    use crate::{
        builder::{
            BuildError, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        extension::prelude::{bool_t, qb_t},
        ops::handle::{DfgID, FuncID, NodeHandle},
        std_extensions::logic::test::and_op,
    };

    use super::*;

    impl<N: HugrNode> SiblingSubgraph<N> {
        /// Create a sibling subgraph containing every node in a HUGR region.
        ///
        /// This will return an [`InvalidSubgraph::EmptySubgraph`] error if the
        /// subgraph is empty.
        fn from_sibling_graph(
            hugr: &impl HugrView<Node = N>,
            parent: N,
        ) -> Result<Self, InvalidSubgraph<N>> {
            let nodes = hugr.children(parent).collect_vec();
            if nodes.is_empty() {
                Err(InvalidSubgraph::EmptySubgraph)
            } else {
                Ok(Self {
                    nodes,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                })
            }
        }
    }

    /// A Module with a single function from three qubits to three qubits.
    /// The function applies a CX gate to the first two qubits and a Rz gate (with a constant angle) to the last qubit.
    fn build_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]).into(),
        )?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let [w0, w1, w2] = dfg.input_wires_arr();
            let [w0, w1] = dfg.add_dataflow_op(cx_gate(), [w0, w1])?.outputs_arr();
            let c = dfg.add_load_const(Const::new(ConstF64::new(0.5).into()));
            let [w2] = dfg.add_dataflow_op(rz_f64(), [w2, c])?.outputs_arr();
            dfg.finish_with_outputs([w0, w1, w2])?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    /// A bool to bool hugr with three subsequent NOT gates.
    fn build_3not_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare("test", Signature::new_endo(vec![bool_t()]).into())?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let outs1 = dfg.add_dataflow_op(LogicOp::Not, dfg.input_wires())?;
            let outs2 = dfg.add_dataflow_op(LogicOp::Not, outs1.outputs())?;
            let outs3 = dfg.add_dataflow_op(LogicOp::Not, outs2.outputs())?;
            dfg.finish_with_outputs(outs3.outputs())?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    /// A bool to (bool, bool) with multiports.
    fn build_multiport_hugr() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare(
            "test",
            Signature::new(bool_t(), vec![bool_t(), bool_t()]).into(),
        )?;
        let func_id = {
            let mut dfg = mod_builder.define_declaration(&func)?;
            let [b0] = dfg.input_wires_arr();
            let [b1] = dfg.add_dataflow_op(LogicOp::Not, [b0])?.outputs_arr();
            let [b2] = dfg.add_dataflow_op(LogicOp::Not, [b1])?.outputs_arr();
            dfg.finish_with_outputs([b1, b2])?
        };
        let hugr = mod_builder
            .finish_hugr()
            .map_err(|e| -> BuildError { e.into() })?;
        Ok((hugr, func_id.node()))
    }

    /// A HUGR with a copy
    fn build_hugr_classical() -> Result<(Hugr, Node), BuildError> {
        let mut mod_builder = ModuleBuilder::new();
        let func = mod_builder.declare("test", Signature::new_endo(bool_t()).into())?;
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
    fn construct_simple_replacement() -> Result<(), InvalidSubgraph> {
        let (mut hugr, func_root) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        let sub = SiblingSubgraph::try_new_dataflow_subgraph::<_, FuncID<true>>(&func)?;

        let empty_dfg = {
            let builder =
                DFGBuilder::new(Signature::new_endo(vec![qb_t(), qb_t(), qb_t()])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        let rep = sub.create_simple_replacement(&func, empty_dfg).unwrap();

        assert_eq!(rep.subgraph().nodes().len(), 4);

        assert_eq!(hugr.num_nodes(), 8); // Module + Def + In + CX + Rz + Const + LoadConst + Out
        hugr.apply_patch(rep).unwrap();
        assert_eq!(hugr.num_nodes(), 4); // Module + Def + In + Out

        Ok(())
    }

    #[test]
    fn test_signature() -> Result<(), InvalidSubgraph> {
        let (hugr, dfg) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(dfg);
        let sub = SiblingSubgraph::try_new_dataflow_subgraph::<_, FuncID<true>>(&func)?;
        assert_eq!(
            sub.signature(&func),
            Signature::new_endo(vec![qb_t(), qb_t(), qb_t()])
        );
        Ok(())
    }

    #[test]
    fn construct_simple_replacement_invalid_signature() -> Result<(), InvalidSubgraph> {
        let (hugr, dfg) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(dfg);
        let sub = SiblingSubgraph::from_sibling_graph(&hugr, dfg)?;

        let empty_dfg = {
            let builder = DFGBuilder::new(Signature::new_endo(vec![qb_t()])).unwrap();
            let inputs = builder.input_wires();
            builder.finish_hugr_with_outputs(inputs).unwrap()
        };

        assert_matches!(
            sub.create_simple_replacement(&func, empty_dfg).unwrap_err(),
            InvalidReplacement::InvalidSignature { .. }
        );
        Ok(())
    }

    #[test]
    fn convex_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        assert_eq!(
            SiblingSubgraph::try_new_dataflow_subgraph::<_, FuncID<true>>(&func)
                .unwrap()
                .nodes()
                .len(),
            4
        )
    }

    #[test]
    fn convex_subgraph_2() {
        let (hugr, func_root) = build_hugr().unwrap();
        let [inp, out] = hugr.get_io(func_root).unwrap();
        let func = hugr.with_entrypoint(func_root);
        // All graph except input/output nodes
        SiblingSubgraph::try_new(
            hugr.node_outputs(inp)
                .take(2)
                .map(|p| hugr.linked_inputs(inp, p).collect_vec())
                .filter(|ps| !ps.is_empty())
                .collect(),
            hugr.node_inputs(out)
                .take(2)
                .filter_map(|p| hugr.single_linked_output(out, p))
                .collect(),
            &func,
        )
        .unwrap();
    }

    #[test]
    fn degen_boundary() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        let [inp, _] = hugr.get_io(func_root).unwrap();
        let first_cx_edge = hugr.node_outputs(inp).next().unwrap();
        // All graph but one edge
        assert_matches!(
            SiblingSubgraph::try_new(
                vec![hugr
                    .linked_ports(inp, first_cx_edge)
                    .map(|(n, p)| (n, p.as_incoming().unwrap()))
                    .collect()],
                vec![(inp, first_cx_edge)],
                &func,
            ),
            Err(InvalidSubgraph::InvalidBoundary(
                InvalidSubgraphBoundary::DisconnectedBoundaryPort(_, _)
            ))
        );
    }

    #[test]
    fn non_convex_subgraph() {
        let (hugr, func_root) = build_3not_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        let [inp, _out] = hugr.get_io(func_root).unwrap();
        let not1 = hugr.output_neighbours(inp).exactly_one().ok().unwrap();
        let not2 = hugr.output_neighbours(not1).exactly_one().ok().unwrap();
        let not3 = hugr.output_neighbours(not2).exactly_one().ok().unwrap();
        let not1_inp = hugr.node_inputs(not1).next().unwrap();
        let not1_out = hugr.node_outputs(not1).next().unwrap();
        let not3_inp = hugr.node_inputs(not3).next().unwrap();
        let not3_out = hugr.node_outputs(not3).next().unwrap();
        assert_matches!(
            SiblingSubgraph::try_new(
                vec![vec![(not1, not1_inp)], vec![(not3, not3_inp)]],
                vec![(not1, not1_out), (not3, not3_out)],
                &func
            ),
            Err(InvalidSubgraph::NotConvex)
        );
    }

    /// A subgraphs mixed with multiports caused a NonConvex error.
    /// https://github.com/CQCL/hugr/issues/1294
    #[test]
    fn convex_multiports() {
        let (hugr, func_root) = build_multiport_hugr().unwrap();
        let [inp, out] = hugr.get_io(func_root).unwrap();
        let not1 = hugr.output_neighbours(inp).exactly_one().ok().unwrap();
        let not2 = hugr
            .output_neighbours(not1)
            .filter(|&n| n != out)
            .exactly_one()
            .ok()
            .unwrap();

        let subgraph = SiblingSubgraph::try_from_nodes([not1, not2], &hugr).unwrap();
        assert_eq!(subgraph.nodes(), [not1, not2]);
    }

    #[test]
    fn invalid_boundary() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        let [inp, out] = hugr.get_io(func_root).unwrap();
        let cx_edges_in = hugr.node_outputs(inp);
        let cx_edges_out = hugr.node_inputs(out);
        // All graph but the CX
        assert_matches!(
            SiblingSubgraph::try_new(
                cx_edges_out.map(|p| vec![(out, p)]).collect(),
                cx_edges_in.map(|p| (inp, p)).collect(),
                &func,
            ),
            Err(InvalidSubgraph::InvalidBoundary(
                InvalidSubgraphBoundary::DisconnectedBoundaryPort(_, _)
            ))
        );
    }

    #[test]
    fn preserve_signature() {
        let (hugr, func_root) = build_hugr_classical().unwrap();
        let func_graph = hugr.with_entrypoint(func_root);
        let func =
            SiblingSubgraph::try_new_dataflow_subgraph::<_, FuncID<true>>(&func_graph).unwrap();
        let func_defn = hugr.get_optype(func_root).as_func_defn().unwrap();
        assert_eq!(func_defn.signature, func.signature(&func_graph).into());
    }

    #[test]
    fn extract_subgraph() {
        let (hugr, func_root) = build_hugr().unwrap();
        let func_graph = hugr.with_entrypoint(func_root);
        let subgraph =
            SiblingSubgraph::try_new_dataflow_subgraph::<_, FuncID<true>>(&func_graph).unwrap();
        let extracted = subgraph.extract_subgraph(&hugr, "region");

        extracted.validate().unwrap();
    }

    #[test]
    fn edge_both_output_and_copy() {
        // https://github.com/CQCL/hugr/issues/518
        let one_bit = vec![bool_t()];
        let two_bit = vec![bool_t(), bool_t()];

        let mut builder = DFGBuilder::new(inout_sig(one_bit.clone(), two_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw1 = builder
            .add_dataflow_op(LogicOp::Not, [inw])
            .unwrap()
            .out_wire(0);
        let outw2 = builder
            .add_dataflow_op(and_op(), [inw, outw1])
            .unwrap()
            .outputs();
        let outw = [outw1].into_iter().chain(outw2);
        let h = builder.finish_hugr_with_outputs(outw).unwrap();
        let subg = SiblingSubgraph::try_new_dataflow_subgraph::<_, DfgID>(&h).unwrap();
        assert_eq!(subg.nodes().len(), 2);
    }

    #[test]
    fn test_unconnected() {
        // test a replacement on a subgraph with a discarded output
        let mut b = DFGBuilder::new(Signature::new(bool_t(), type_row![])).unwrap();
        let inw = b.input_wires().exactly_one().unwrap();
        let not_n = b.add_dataflow_op(LogicOp::Not, [inw]).unwrap();
        // Unconnected output, discarded
        let mut h = b.finish_hugr_with_outputs([]).unwrap();

        let subg = SiblingSubgraph::from_node(not_n.node(), &h);

        assert_eq!(subg.nodes().len(), 1);
        //  TODO create a valid replacement
        let replacement = {
            let mut rep_b = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
            let inw = rep_b.input_wires().exactly_one().unwrap();

            let not_n = rep_b.add_dataflow_op(LogicOp::Not, [inw]).unwrap();

            rep_b.finish_hugr_with_outputs(not_n.outputs()).unwrap()
        };
        let rep = subg.create_simple_replacement(&h, replacement).unwrap();
        rep.apply(&mut h).unwrap();
    }

    /// Test the behaviour of the sibling subgraph when built from a single node.
    #[test]
    fn single_node_subgraph() {
        // A hugr with a single NOT operation, with disconnected output.
        let mut b = DFGBuilder::new(Signature::new(bool_t(), type_row![])).unwrap();
        let inw = b.input_wires().exactly_one().unwrap();
        let not_n = b.add_dataflow_op(LogicOp::Not, [inw]).unwrap();
        // Unconnected output, discarded
        let h = b.finish_hugr_with_outputs([]).unwrap();

        // When built with `from_node`, the subgraph's signature is the same as the node's.
        // (bool input, bool output)
        let subg = SiblingSubgraph::from_node(not_n.node(), &h);
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(vec![bool_t()], vec![bool_t()]).io()
        );

        // `from_nodes` is different, is it only uses incoming and outgoing edges to compute the signature.
        // In this case, the output is disconnected, so it is not part of the subgraph signature.
        let subg = SiblingSubgraph::try_from_nodes([not_n.node()], &h).unwrap();
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(vec![bool_t()], vec![]).io()
        );
    }
}

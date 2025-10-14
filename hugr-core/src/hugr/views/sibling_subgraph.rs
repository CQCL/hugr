//! Views for HUGR sibling subgraphs.
//!
//! Views into convex subgraphs of HUGRs within a single level of the
//! hierarchy, i.e. within a sibling graph. Convex subgraph are always
//! induced subgraphs, i.e. they are defined by a subset of the sibling nodes.

use std::cell::OnceCell;
use std::collections::HashSet;
use std::mem;

use itertools::Itertools;
use portgraph::LinkView;
use portgraph::PortView;
use portgraph::algorithms::CreateConvexChecker;
use portgraph::algorithms::convex::{LineIndex, LineIntervals, Position};
use portgraph::boundary::Boundary;
use rustc_hash::FxHashSet;
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
/// At the moment the only supported non-value edges at the subgraph boundary
/// are static function calls. The existence of any other edges at the boundary
/// will cause the fallible constructors to return an [`InvalidSubgraph`] error
/// and will cause [`SiblingSubgraph::from_node`] and
/// [`SiblingSubgraph::create_simple_replacement`] to panic.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SiblingSubgraph<N = Node> {
    /// The nodes of the induced subgraph.
    nodes: Vec<N>,
    /// The input ports of the subgraph.
    ///
    /// All ports within the same inner vector of `inputs` must be connected to
    /// the same outgoing port (and thus have the same type). The outer vector
    /// defines the input signature, i.e. the number and types of incoming wires
    /// into the subgraph.
    ///
    /// Multiple input ports of the same type may be grouped within the same
    /// inner vector: this corresponds to an input parameter that is copied and
    /// used multiple times in the subgraph. An inner vector may also be empty,
    /// corresponding to discarding an input parameter.
    ///
    /// Each port must be unique and belong to a node in `nodes`. Input ports of
    /// linear types will always appear as singleton vectors.
    inputs: IncomingPorts<N>,
    /// The output ports of the subgraph.
    ///
    /// The types of the output ports define the output signature of the
    /// subgraph. Repeated ports of copyable types are allowed and correspond to
    /// copying a value of the subgraph multiple times.
    ///
    /// Every port must belong to a node in `nodes`.
    outputs: OutgoingPorts<N>,
    /// The function called in the subgraph but defined outside of it.
    ///
    /// The incoming ports of call nodes are grouped by the function definition
    /// that is called.
    function_calls: IncomingPorts<N>,
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

        if nodes.is_empty() {
            return Err(InvalidSubgraph::EmptySubgraph);
        }

        let (inputs, outputs) = get_input_output_ports(dfg_graph)?;
        let non_local = get_non_local_edges(&nodes, &dfg_graph);
        let function_calls = group_into_function_calls(non_local, &dfg_graph)?;

        validate_subgraph(dfg_graph, &nodes, &inputs, &outputs, &function_calls)?;

        Ok(Self {
            nodes,
            inputs,
            outputs,
            function_calls,
        })
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
    /// The `incoming` and `outgoing` arguments give $B_I$ and $B_O$
    /// respectively. Incoming edges must be given by incoming ports and
    /// outgoing edges by outgoing ports. The ordering of the incoming and
    /// outgoing ports defines the signature of the subgraph.
    ///
    /// Incoming boundary ports must be unique and partitioned by input
    /// parameter: two ports within the same set of the partition must be
    /// copyable and will result in the input being copied. Outgoing
    /// boundary ports are given in a list and can appear multiple times if
    /// they are copyable, in which case the output will be copied.
    ///
    /// ## Errors
    ///
    /// This function will return an error in the following cases
    ///  - An [`InvalidSubgraph::NotConvex`] error if the subgraph is not
    ///    convex,
    ///  - An [`InvalidSubgraph::NoSharedParent`] or
    ///    [`InvalidSubgraph::OrphanNode`] error if the boundary nodes do not
    ///    share a common parent or have no parent,
    ///  - An [`InvalidSubgraph::EmptySubgraph`] error if the subgraph is empty,
    ///  - An [`InvalidSubgraph::InvalidBoundary`] error if a boundary port is
    ///    invalid, i.e. if the boundary port is not in the subgraph or if none
    ///    of the linked ports are outside of the subgraph.
    pub fn try_new(
        inputs: IncomingPorts<N>,
        outputs: OutgoingPorts<N>,
        hugr: &impl HugrView<Node = N>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let parent = pick_parent(hugr, &inputs, &outputs)?;
        let checker = TopoConvexChecker::new(hugr, parent);
        Self::try_new_with_checker(inputs, outputs, hugr, &checker)
    }

    /// Create a new [`SiblingSubgraph`], bypassing all validity checks.
    ///
    /// You MUST make sure that the boundary ports and nodes provided satisfy
    /// the SiblingSubgraph validity conditions described in
    /// [`SiblingSubgraph::try_new`] and which can be checked using
    /// [`SiblingSubgraph::validate`].
    ///
    /// See [`SiblingSubgraph::try_new`] for the full documentation.
    ///
    /// # Arguments
    ///
    /// - `inputs`: The incoming boundary ports that are value edges.
    /// - `outputs`: The outgoing boundary ports (must be value edges).
    /// - `function_calls`: The incoming boundary ports that are static function
    ///   call edges.
    /// - `nodes`: The nodes in the subgraph.
    pub fn new_unchecked(
        inputs: IncomingPorts<N>,
        outputs: OutgoingPorts<N>,
        function_calls: IncomingPorts<N>,
        nodes: Vec<N>,
    ) -> Self {
        Self {
            nodes,
            inputs,
            outputs,
            function_calls,
        }
    }

    /// Create a new convex sibling subgraph from input and output boundaries.
    ///
    /// Provide a [`ConvexChecker`] instance to avoid constructing one for
    /// faster convexity check. If you do not have one, use
    /// [`SiblingSubgraph::try_new`].
    ///
    /// Refer to [`SiblingSubgraph::try_new`] for the full
    /// documentation.
    // TODO(breaking): generalize to any convex checker
    pub fn try_new_with_checker<H: HugrView<Node = N>>(
        mut inputs: IncomingPorts<N>,
        outputs: OutgoingPorts<N>,
        hugr: &H,
        checker: &TopoConvexChecker<H>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let (subpg, node_map) = make_pg_subgraph(hugr, &inputs, &outputs);
        let nodes = subpg
            .nodes_iter()
            .map(|index| node_map.from_portgraph(index))
            .collect_vec();

        let function_calls = drain_function_calls(&mut inputs, hugr);

        validate_subgraph(hugr, &nodes, &inputs, &outputs, &function_calls)?;

        if nodes.len() > 1 && !subpg.is_convex_with_checker(checker) {
            return Err(InvalidSubgraph::NotConvex);
        }

        Ok(Self {
            nodes,
            inputs,
            outputs,
            function_calls,
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
    // TODO(breaking): generalize to any convex checker
    pub fn try_from_nodes_with_checker<H: HugrView<Node = N>>(
        nodes: impl Into<Vec<N>>,
        hugr: &H,
        checker: &TopoConvexChecker<H>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        let mut nodes: Vec<N> = nodes.into();
        let num_nodes = nodes.len();

        if nodes.is_empty() {
            return Err(InvalidSubgraph::EmptySubgraph);
        }

        let (inputs, outputs) = get_boundary_from_nodes(hugr, &mut nodes);

        // If there are no input/output edges, the set is always convex so we can initialize the subgraph directly.
        // If we called `try_new_with_checker` with no boundaries, we'd select every node in the region instead of the expected subset.
        if inputs.is_empty() && outputs.is_empty() {
            return Ok(Self {
                nodes,
                inputs,
                outputs,
                function_calls: vec![],
            });
        }

        let mut subgraph = Self::try_new_with_checker(inputs, outputs, hugr, checker)?;

        // If some nodes formed a fully connected component, they won't be included in the subgraph generated from the boundaries.
        // We re-add them here.
        if subgraph.node_count() < num_nodes {
            subgraph.nodes = nodes;
        }

        Ok(subgraph)
    }

    /// Create a subgraph from a set of nodes, using a line convexity checker
    /// and pre-computed line intervals.
    ///
    /// Note that passing intervals that do not correspond to the intervals of
    /// the nodes is undefined behaviour.
    ///
    /// This function is equivalent to [`SiblingSubgraph::try_from_nodes`]
    /// or [`SiblingSubgraph::try_new_with_checker`] but is provided for
    /// performance reasons, in cases where line intervals for the nodes have
    /// already been computed.
    ///
    /// Refer to [`SiblingSubgraph::try_from_nodes`] for the full
    /// documentation.
    pub fn try_from_nodes_with_intervals(
        nodes: impl Into<Vec<N>>,
        intervals: &LineIntervals,
        line_checker: &LineConvexChecker<impl HugrView<Node = N>>,
    ) -> Result<Self, InvalidSubgraph<N>> {
        if !line_checker.get_checker().is_convex_by_intervals(intervals) {
            return Err(InvalidSubgraph::NotConvex);
        }

        let nodes: Vec<N> = nodes.into();
        let hugr = line_checker.hugr();

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
        let mut inputs = incoming_edges
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
        let function_calls = drain_function_calls(&mut inputs, hugr);

        Ok(Self {
            nodes,
            inputs,
            outputs,
            function_calls,
        })
    }

    /// Create a subgraph containing a single node.
    ///
    /// The subgraph signature will be given by signature of the node.
    ///
    /// # Panics
    ///
    /// If the node has incoming or outgoing state order edges.
    pub fn from_node(node: N, hugr: &impl HugrView<Node = N>) -> Self {
        let nodes = vec![node];
        let mut inputs = hugr
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
        let function_calls = drain_function_calls(&mut inputs, hugr);

        let state_order_at_input = hugr
            .get_optype(node)
            .other_output_port()
            .is_some_and(|p| hugr.is_linked(node, p));
        let state_order_at_output = hugr
            .get_optype(node)
            .other_input_port()
            .is_some_and(|p| hugr.is_linked(node, p));
        if state_order_at_input || state_order_at_output {
            unimplemented!("Order edges in {node:?} not supported");
        }

        Self {
            nodes,
            inputs,
            outputs,
            function_calls,
        }
    }

    /// Check the validity of the subgraph, as described in the docs of
    /// [`SiblingSubgraph::try_new`].
    ///
    /// The `mode` parameter controls the convexity check:
    ///  - [`ValidationMode::CheckConvexity`] will create a new convexity
    ///    checker for the subgraph and use it to check convexity of the
    ///    subgraph.
    ///  - [`ValidationMode::WithChecker`] will use the given convexity checker
    ///    to check convexity of the subgraph.
    ///  - [`ValidationMode::SkipConvexity`] will skip the convexity check.
    pub fn validate<'h, H: HugrView<Node = N>>(
        &self,
        hugr: &'h H,
        mode: ValidationMode<'_, 'h, H>,
    ) -> Result<(), InvalidSubgraph<N>> {
        let mut exp_nodes = {
            let (subpg, node_map) = make_pg_subgraph(hugr, &self.inputs, &self.outputs);
            subpg
                .nodes_iter()
                .map(|n| node_map.from_portgraph(n))
                .collect_vec()
        };
        let mut nodes = self.nodes.clone();

        exp_nodes.sort_unstable();
        nodes.sort_unstable();

        if exp_nodes != nodes {
            return Err(InvalidSubgraph::InvalidNodeSet);
        }

        validate_subgraph(
            hugr,
            &self.nodes,
            &self.inputs,
            &self.outputs,
            &self.function_calls,
        )?;

        let checker;
        let checker_ref = match mode {
            ValidationMode::WithChecker(c) => Some(c),
            ValidationMode::CheckConvexity => {
                checker = ConvexChecker::new(hugr, self.get_parent(hugr));
                Some(&checker)
            }
            ValidationMode::SkipConvexity => None,
        };
        if let Some(checker) = checker_ref {
            let (subpg, _) = make_pg_subgraph(hugr, &self.inputs, &self.outputs);
            if !subpg.is_convex_with_checker(&checker.init_checker().0) {
                return Err(InvalidSubgraph::NotConvex);
            }
        }
        Ok(())
    }

    /// An iterator over the nodes in the subgraph.
    #[must_use]
    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }

    /// The number of nodes in the subgraph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the computed [`IncomingPorts`] of the subgraph.
    #[must_use]
    pub fn incoming_ports(&self) -> &IncomingPorts<N> {
        &self.inputs
    }

    /// Returns the computed [`OutgoingPorts`] of the subgraph.
    #[must_use]
    pub fn outgoing_ports(&self) -> &OutgoingPorts<N> {
        &self.outputs
    }

    /// Return the static incoming ports of the subgraph that are calls to
    /// functions defined outside of the subgraph.
    #[must_use]
    pub fn function_calls(&self) -> &IncomingPorts<N> {
        &self.function_calls
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
        let function_calls = self
            .function_calls
            .iter()
            .map(|calls| calls.iter().map(|&(n, p)| (node_map(n), p)).collect_vec())
            .collect_vec();
        SiblingSubgraph {
            nodes,
            inputs,
            outputs,
            function_calls,
        }
    }

    /// Construct a [`SimpleReplacement`] to replace `self` with `replacement`.
    ///
    /// `replacement` must be a hugr with dataflow graph and its signature must
    /// match the signature of the subgraph.
    ///
    /// May return one of the following five errors
    ///  - [`InvalidReplacement::InvalidDataflowGraph`]: the replacement graph
    ///    is not a [`crate::ops::OpTag::DataflowParent`]-rooted graph,
    ///  - [`InvalidReplacement::InvalidSignature`]: the signature of the
    ///    replacement DFG does not match the subgraph signature, or
    ///  - [`InvalidReplacement::NonConvexSubgraph`]: the sibling subgraph is
    ///    not convex.
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
        if !OpTag::DataflowParent.is_superset(dfg_optype.tag()) {
            return Err(InvalidReplacement::InvalidDataflowGraph {
                node: rep_root,
                op: Box::new(dfg_optype.clone()),
            });
        }
        let [rep_input, rep_output] = replacement
            .get_io(rep_root)
            .expect("DFG root in the replacement does not have input and output nodes.");

        // TODO: handle state order edges. For now panic if any are present.
        // See https://github.com/CQCL/hugr/discussions/432
        let state_order_at_input = replacement
            .get_optype(rep_input)
            .other_output_port()
            .is_some_and(|p| replacement.is_linked(rep_input, p));
        let state_order_at_output = replacement
            .get_optype(rep_output)
            .other_input_port()
            .is_some_and(|p| replacement.is_linked(rep_output, p));
        if state_order_at_input || state_order_at_output {
            unimplemented!("Found state order edges in replacement graph");
        }

        SimpleReplacement::try_new(self.clone(), hugr, replacement)
    }

    /// Create a new Hugr containing only the subgraph.
    ///
    /// The new Hugr will contain a [`FuncDefn`][crate::ops::FuncDefn] root
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

    /// Change the output boundary of the subgraph.
    ///
    /// All ports of the new boundary must be contained in the old boundary,
    /// i.e. the only changes that are allowed are copying, discarding and
    /// shuffling existing ports in the output boundary.
    ///
    /// Returns an error if the new boundary is invalid (contains ports not in
    /// the old boundary or has non-unique linear ports). In this case,
    /// `self` is left unchanged.
    pub fn set_outgoing_ports(
        &mut self,
        ports: OutgoingPorts<N>,
        host: &impl HugrView<Node = N>,
    ) -> Result<(), InvalidOutputPorts<N>> {
        let old_boundary: HashSet<_> = iter_outgoing(&self.outputs).collect();

        // Check for unknown ports
        if let Some((node, port)) =
            iter_outgoing(&ports).find(|(n, p)| !old_boundary.contains(&(*n, *p)))
        {
            return Err(InvalidOutputPorts::UnknownOutput { port, node });
        }

        // Check for non-unique linear ports
        if !has_unique_linear_ports(host, &ports) {
            return Err(InvalidOutputPorts::NonUniqueLinear);
        }

        self.outputs = ports;
        Ok(())
    }
}

/// Specify the checks to perform for [`SiblingSubgraph::validate`].
#[derive(Default)]
pub enum ValidationMode<'t, 'h, H: HugrView> {
    /// Check convexity with the given checker.
    WithChecker(&'t TopoConvexChecker<'h, H>),
    /// Construct a checker and check convexity.
    #[default]
    CheckConvexity,
    /// Skip convexity check.
    SkipConvexity,
}

fn make_pg_subgraph<'h, H: HugrView>(
    hugr: &'h H,
    inputs: &IncomingPorts<H::Node>,
    outputs: &OutgoingPorts<H::Node>,
) -> (
    portgraph::view::Subgraph<CheckerRegion<'h, H>>,
    H::RegionPortgraphNodes,
) {
    let (region, node_map) = hugr.region_portgraph(hugr.entrypoint());

    // Ordering of the edges here is preserved and becomes ordering of the
    // signature.
    let boundary = make_boundary::<H>(&region, &node_map, inputs, outputs);
    (
        portgraph::view::Subgraph::new_subgraph(region, boundary),
        node_map,
    )
}

/// Returns the input and output boundary ports for a given set of nodes.
///
/// Removes duplicates in `nodes` while preserving the order.
fn get_boundary_from_nodes<N: HugrNode>(
    hugr: &impl HugrView<Node = N>,
    nodes: &mut Vec<N>,
) -> (IncomingPorts<N>, OutgoingPorts<N>) {
    // remove duplicates in `nodes` while preserving the order
    // simultaneously build a set for fast lookup
    let mut nodes_set = FxHashSet::default();
    nodes.retain(|&n| nodes_set.insert(n));

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

    (inputs, outputs)
}

/// Remove all function calls from `inputs` and return them in a new vector.
fn drain_function_calls<N: HugrNode, H: HugrView<Node = N>>(
    inputs: &mut IncomingPorts<N>,
    hugr: &H,
) -> IncomingPorts<N> {
    let mut function_calls = Vec::new();
    inputs.retain_mut(|calls| {
        let Some(&(n, p)) = calls.first() else {
            return true;
        };
        let op = hugr.get_optype(n);
        if op.static_input_port() == Some(p)
            && op
                .static_input()
                .expect("static input exists")
                .is_function()
        {
            function_calls.extend(mem::take(calls));
            false
        } else {
            true
        }
    });

    group_into_function_calls(function_calls.into_iter().map(|(n, p)| (n, p.into())), hugr)
        .expect("valid function calls")
}

/// Group incoming ports by the function definition/declaration that they are
/// connected to.
///
/// Fails if not all ports passed as incoming are of kind [EdgeKind::Function].
fn group_into_function_calls<N: HugrNode>(
    ports: impl IntoIterator<Item = (N, Port)>,
    hugr: &impl HugrView<Node = N>,
) -> Result<Vec<Vec<(N, IncomingPort)>>, InvalidSubgraph<N>> {
    let incoming_ports: Vec<_> = ports
        .into_iter()
        .map(|(n, p)| {
            let p = p
                .as_incoming()
                .map_err(|_| InvalidSubgraph::UnsupportedEdgeKind(n, p))?;
            let op = hugr.get_optype(n);
            if Some(p) != op.static_input_port() {
                return Err(InvalidSubgraph::UnsupportedEdgeKind(n, p.into()));
            }
            if !op
                .static_input()
                .expect("static input exists")
                .is_function()
            {
                return Err(InvalidSubgraph::UnsupportedEdgeKind(n, p.into()));
            }
            Ok::<_, InvalidSubgraph<N>>((n, p))
        })
        .try_collect()?;
    let grouped_non_local = incoming_ports
        .into_iter()
        .into_group_map_by(|&(n, p)| hugr.single_linked_output(n, p).expect("valid dfg wire"));
    Ok(grouped_non_local
        .into_iter()
        .sorted_unstable_by(|(n1, _), (n2, _)| n1.cmp(n2))
        .map(|(_, v)| v)
        .collect())
}

/// Returns an iterator over the non-local edges in `nodes`.
///
/// `nodes` must be non-empty and all nodes within must have the same parent
fn get_non_local_edges<'a, N: HugrNode>(
    nodes: &'a [N],
    hugr: &'a impl HugrView<Node = N>,
) -> impl Iterator<Item = (N, Port)> + 'a {
    let parent = hugr.get_parent(nodes[0]);
    let is_non_local = move |n, p| {
        hugr.linked_ports(n, p)
            .any(|(n, _)| hugr.get_parent(n) != parent)
    };
    nodes
        .iter()
        .flat_map(move |&n| hugr.all_node_ports(n).map(move |p| (n, p)))
        .filter(move |&(n, p)| is_non_local(n, p))
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
    region: &impl LinkView<PortOffsetBase = u32>,
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
///
/// This a good default choice for most convexity checking use cases.
pub type TopoConvexChecker<'g, Base> =
    ConvexChecker<'g, Base, portgraph::algorithms::TopoConvexChecker<CheckerRegion<'g, Base>>>;

/// Precompute convexity information for a HUGR.
///
/// This can be used when constructing multiple sibling subgraphs to speed up
/// convexity checking.
///
/// This is a good choice for checking convexity of circuit-like graphs,
/// particularly when many checks must be performed.
pub type LineConvexChecker<'g, Base> =
    ConvexChecker<'g, Base, portgraph::algorithms::LineConvexChecker<CheckerRegion<'g, Base>>>;

/// Precompute convexity information for a HUGR.
///
/// This can be used when constructing multiple sibling subgraphs to speed up
/// convexity checking.
///
/// This type is generic over the convexity checker used. If checking convexity
/// for circuit-like graphs, use [`LineConvexChecker`], otherwise use
/// [`TopoConvexChecker`].
pub struct ConvexChecker<'g, Base: HugrView, Checker> {
    /// The base HUGR to check convexity on.
    base: &'g Base,
    /// The parent of the region where we are checking convexity.
    region_parent: Base::Node,
    /// A lazily initialized convexity checker, along with a map from nodes in
    /// the region to `Base` nodes.
    checker: OnceCell<(Checker, Base::RegionPortgraphNodes)>,
}

impl<'g, Base: HugrView, Checker: Clone> Clone for ConvexChecker<'g, Base, Checker> {
    fn clone(&self) -> Self {
        Self {
            base: self.base,
            region_parent: self.region_parent,
            checker: self.checker.clone(),
        }
    }
}

impl<'g, Base: HugrView, Checker> ConvexChecker<'g, Base, Checker> {
    /// Create a new convexity checker.
    pub fn new(base: &'g Base, region_parent: Base::Node) -> Self {
        Self {
            base,
            region_parent,
            checker: OnceCell::new(),
        }
    }

    /// Create a new convexity checker for the region of the entrypoint.
    #[inline(always)]
    pub fn from_entrypoint(base: &'g Base) -> Self {
        let region_parent = base.entrypoint();
        Self::new(base, region_parent)
    }

    /// The base HUGR to check convexity on.
    pub fn hugr(&self) -> &'g Base {
        self.base
    }
}

impl<'g, Base, Checker> ConvexChecker<'g, Base, Checker>
where
    Base: HugrView,
    Checker: CreateConvexChecker<CheckerRegion<'g, Base>>,
{
    /// Returns the portgraph convexity checker, initializing it if necessary.
    fn init_checker(&self) -> &(Checker, Base::RegionPortgraphNodes) {
        self.checker.get_or_init(|| {
            let (region, node_map) = self.base.region_portgraph(self.region_parent);
            let checker = Checker::new_convex_checker(region);
            (checker, node_map)
        })
    }

    /// Returns the node map from the region to the base HUGR.
    #[expect(dead_code)]
    fn get_node_map(&self) -> &Base::RegionPortgraphNodes {
        &self.init_checker().1
    }

    /// Returns the portgraph convexity checker, initializing it if necessary.
    fn get_checker(&self) -> &Checker {
        &self.init_checker().0
    }
}

impl<'g, Base, Checker> portgraph::algorithms::ConvexChecker for ConvexChecker<'g, Base, Checker>
where
    Base: HugrView,
    Checker: CreateConvexChecker<CheckerRegion<'g, Base>>,
{
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
        }
        self.get_checker().is_convex(nodes, inputs, outputs)
    }
}

impl<'g, Base: HugrView> LineConvexChecker<'g, Base> {
    /// Return the line intervals occupied by the given nodes in the region.
    pub fn get_intervals_from_nodes(
        &self,
        nodes: impl IntoIterator<Item = Base::Node>,
    ) -> Option<LineIntervals> {
        let (checker, node_map) = self.init_checker();
        let nodes = nodes
            .into_iter()
            .map(|n| node_map.to_portgraph(n))
            .collect_vec();
        checker.get_intervals_from_nodes(nodes)
    }

    /// Return the line intervals defined by the given boundary ports in the
    /// region.
    ///
    /// Incoming ports correspond to boundary inputs, outgoing ports correspond
    /// to boundary outputs.
    pub fn get_intervals_from_boundary_ports(
        &self,
        ports: impl IntoIterator<Item = (Base::Node, Port)>,
    ) -> Option<LineIntervals> {
        let (checker, node_map) = self.init_checker();
        let ports = ports
            .into_iter()
            .map(|(n, p)| {
                let node = node_map.to_portgraph(n);
                checker
                    .graph()
                    .port_index(node, p.pg_offset())
                    .expect("valid port")
            })
            .collect_vec();
        checker.get_intervals_from_boundary_ports(ports)
    }

    /// Return the nodes that are within the given line intervals.
    pub fn nodes_in_intervals<'a>(
        &'a self,
        intervals: &'a LineIntervals,
    ) -> impl Iterator<Item = Base::Node> + 'a {
        let (checker, node_map) = self.init_checker();
        checker
            .nodes_in_intervals(intervals)
            .map(|pg_node| node_map.from_portgraph(pg_node))
    }

    /// Get the lines passing through the given port.
    pub fn lines_at_port(&self, node: Base::Node, port: impl Into<Port>) -> &[LineIndex] {
        let (checker, node_map) = self.init_checker();
        let port = checker
            .graph()
            .port_index(node_map.to_portgraph(node), port.into().pg_offset())
            .expect("valid port");
        checker.lines_at_port(port)
    }

    /// Extend the given intervals to include the given node.
    ///
    /// Return whether the interval was successfully extended to contain `node`,
    /// i.e. whether adding `node` to the subgraph represented by the intervals
    /// results in another subgraph that can be expressed as line intervals.
    ///
    /// If `false` is returned, the `intervals` are left unchanged.
    pub fn try_extend_intervals(&self, intervals: &mut LineIntervals, node: Base::Node) -> bool {
        let (checker, node_map) = self.init_checker();
        let node = node_map.to_portgraph(node);
        checker.try_extend_intervals(intervals, node)
    }

    /// Get the position of a node on its lines.
    pub fn get_position(&self, node: Base::Node) -> Position {
        let (checker, node_map) = self.init_checker();
        let node = node_map.to_portgraph(node);
        checker.get_position(node)
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
/// Verifies that input and output ports are valid subgraph boundaries, i.e.
/// they belong to nodes within the subgraph and are linked to at least one node
/// outside of the subgraph. This does NOT check convexity proper, i.e. whether
/// the set of nodes form a convex induced graph.
fn validate_subgraph<H: HugrView>(
    hugr: &H,
    nodes: &[H::Node],
    inputs: &IncomingPorts<H::Node>,
    outputs: &OutgoingPorts<H::Node>,
    function_calls: &IncomingPorts<H::Node>,
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
    if let Some((n, p)) = iter_io(inputs, outputs).find(|&(n, p)| is_non_value_edge(hugr, n, p)) {
        return Err(InvalidSubgraph::UnsupportedEdgeKind(n, p));
    }

    let boundary_ports = iter_io(inputs, outputs).collect_vec();
    // Check that the boundary ports are all in the subgraph.
    if let Some(&(n, p)) = boundary_ports.iter().find(|(n, _)| !node_set.contains(n)) {
        Err(InvalidSubgraphBoundary::PortNodeNotInSet(n, p))?;
    }
    // Check that every inside port has at least one linked port outside.
    if let Some(&(n, p)) = boundary_ports.iter().find(|&&(n, p)| {
        hugr.linked_ports(n, p)
            .all(|(n1, _)| node_set.contains(&n1))
    }) {
        Err(InvalidSubgraphBoundary::DisconnectedBoundaryPort(n, p))?;
    }

    // Check that every incoming port of a node in the subgraph whose source is not
    // in the subgraph belongs to inputs.
    let mut must_be_inputs = nodes
        .iter()
        .flat_map(|&n| hugr.node_inputs(n).map(move |p| (n, p)))
        .filter(|&(n, p)| {
            hugr.linked_ports(n, p)
                .any(|(n1, _)| !node_set.contains(&n1))
        });
    if !must_be_inputs.all(|(n, p)| {
        let mut all_inputs = inputs.iter().chain(function_calls);
        all_inputs.any(|nps| nps.contains(&(n, p)))
    }) {
        return Err(InvalidSubgraph::NotConvex);
    }
    // Check that every outgoing port of a node in the subgraph whose target is not
    // in the subgraph belongs to outputs.
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

    // Check
    //  - no incoming partition is empty
    //  - all inputs within a partition are linked to the same outgoing port
    for inp in inputs {
        let &(in_node, in_port) = inp.first().ok_or(InvalidSubgraphBoundary::EmptyPartition)?;
        let exp_output_node_port = hugr
            .single_linked_output(in_node, in_port)
            .expect("valid dfg wire");
        if let Some(output_node_port) = inp
            .iter()
            .map(|&(in_node, in_port)| {
                hugr.single_linked_output(in_node, in_port)
                    .expect("valid dfg wire")
            })
            .find(|&p| p != exp_output_node_port)
        {
            return Err(InvalidSubgraphBoundary::MismatchedOutputPort(
                (in_node, in_port),
                exp_output_node_port,
                output_node_port,
            )
            .into());
        }
    }

    // Check edge types are equal within partition and copyable if partition size >
    // 1
    if let Some((i, _)) = inputs.iter().enumerate().find(|(_, ports)| {
        let Some(edge_t) = get_edge_type(hugr, ports) else {
            return true;
        };
        let require_copy = ports.len() > 1;
        require_copy && !edge_t.copyable()
    }) {
        Err(InvalidSubgraphBoundary::MismatchedTypes(i))?;
    }

    // Check that all function calls are other ports of edge kind
    // [EdgeKind::Function].
    for calls in function_calls {
        if !calls
            .iter()
            .map(|&(n, p)| hugr.single_linked_output(n, p))
            .all_equal()
        {
            let (n, p) = calls[0];
            return Err(InvalidSubgraph::UnsupportedEdgeKind(n, p.into()));
        }
        for &(n, p) in calls {
            let op = hugr.get_optype(n);
            if op.static_input_port() != Some(p) {
                return Err(InvalidSubgraph::UnsupportedEdgeKind(n, p.into()));
            }
        }
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
fn get_input_output_ports<H: HugrView>(
    hugr: &H,
) -> Result<(IncomingPorts<H::Node>, OutgoingPorts<H::Node>), InvalidSubgraph<H::Node>> {
    let [inp, out] = hugr
        .get_io(HugrView::entrypoint(&hugr))
        .expect("invalid DFG");
    if let Some(p) = hugr
        .node_outputs(inp)
        .find(|&p| is_non_value_edge(hugr, inp, p.into()))
    {
        return Err(InvalidSubgraph::UnsupportedEdgeKind(inp, p.into()));
    }
    if let Some(p) = hugr
        .node_inputs(out)
        .find(|&p| is_non_value_edge(hugr, out, p.into()))
    {
        return Err(InvalidSubgraph::UnsupportedEdgeKind(out, p.into()));
    }

    let dfg_inputs = HugrView::get_optype(&hugr, inp)
        .as_input()
        .unwrap()
        .signature()
        .output_ports();
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
    Ok((inputs, outputs))
}

/// Whether a port is linked to a static or other edge (e.g. order edge).
fn is_non_value_edge<H: HugrView>(hugr: &H, node: H::Node, port: Port) -> bool {
    let op = hugr.get_optype(node);
    let is_other = op.other_port(port.direction()) == Some(port) && hugr.is_linked(node, port);
    let is_static = op.static_port(port.direction()) == Some(port) && hugr.is_linked(node, port);
    is_other || is_static
}

impl<'a, 'c, G: HugrView, Checker: Clone> From<&'a ConvexChecker<'c, G, Checker>>
    for std::borrow::Cow<'a, ConvexChecker<'c, G, Checker>>
{
    fn from(value: &'a ConvexChecker<'c, G, Checker>) -> Self {
        Self::Borrowed(value)
    }
}

/// Errors that can occur while constructing a [`SimpleReplacement`].
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum InvalidReplacement {
    /// No `DataflowParent` root in replacement graph.
    #[error("The root of the replacement {node} is a {}, but only dataflow parents are supported.", op.name())]
    InvalidDataflowGraph {
        /// The node ID of the root node.
        node: Node,
        /// The op type of the root node.
        op: Box<OpType>,
    },
    /// Replacement graph type mismatch.
    #[error(
        "Replacement graph type mismatch. Expected {expected}, got {}.",
        actual.clone().map_or("none".to_string(), |t| t.to_string()))
    ]
    InvalidSignature {
        /// The expected signature.
        expected: Box<Signature>,
        /// The actual signature.
        actual: Option<Box<Signature>>,
    },
    /// `SiblingSubgraph` is not convex.
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
        "Not a sibling subgraph. {first_node} has parent {first_parent}, but {other_node} has parent {other_parent}."
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
    /// The subgraphs induced by the nodes and the boundary do not match.
    #[error("The subgraphs induced by the nodes and the boundary do not match.")]
    InvalidNodeSet,
    /// An outgoing non-local edge was found.
    #[error("Unsupported edge kind at ({_0}, {_1:?}).")]
    UnsupportedEdgeKind(N, Port),
}

/// Errors that can occur while constructing a [`SiblingSubgraph`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum InvalidSubgraphBoundary<N: HugrNode = Node> {
    /// A boundary port's node is not in the set of nodes.
    #[error("(node {0}, port {1}) is in the boundary, but node {0} is not in the set.")]
    PortNodeNotInSet(N, Port),
    /// A boundary port has no connections outside the subgraph.
    #[error(
        "(node {0}, port {1}) is in the boundary, but the port is not connected to a node outside the subgraph."
    )]
    DisconnectedBoundaryPort(N, Port),
    /// There's a non-unique input-boundary port.
    #[error("A port in the input boundary is used multiple times.")]
    NonUniqueInput,
    /// There's an empty partition in the input boundary.
    #[error("A partition in the input boundary is empty.")]
    EmptyPartition,
    /// A partition in the input boundary has ports linked to different output
    /// ports.
    #[error("expected port {0:?} to be linked to {1:?}, but is linked to {2:?}.")]
    MismatchedOutputPort((N, IncomingPort), (N, OutgoingPort), (N, OutgoingPort)),
    /// Different types in a partition of the input boundary.
    #[error("The partition {0} in the input boundary has ports with different types.")]
    MismatchedTypes(usize),
}

/// Error returned when trying to set an invalid output boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("Invalid output ports: {0:?}")]
pub enum InvalidOutputPorts<N: HugrNode = Node> {
    /// Some ports weren't in the original output boundary.
    #[error("{port} in {node} was not part of the original boundary.")]
    UnknownOutput {
        /// The unknown port.
        port: OutgoingPort,
        /// The port's node
        node: N,
    },
    /// Linear ports must appear exactly once.
    #[error("Linear ports must appear exactly once.")]
    NonUniqueLinear,
}

/// Returns true if all linear ports in the set are unique.
fn has_unique_linear_ports<H: HugrView>(host: &H, ports: &OutgoingPorts<H::Node>) -> bool {
    let linear_ports: Vec<_> = ports
        .iter()
        .filter(|&&(n, p)| {
            host.get_optype(n)
                .port_kind(p)
                .is_some_and(|pk| pk.is_linear())
        })
        .collect();
    let unique_ports: HashSet<_> = linear_ports.iter().collect();
    linear_ports.len() == unique_ports.len()
}

#[cfg(test)]
mod tests {
    use cool_asserts::assert_matches;
    use rstest::{fixture, rstest};

    use crate::builder::{endo_sig, inout_sig};
    use crate::extension::prelude::{MakeTuple, UnpackTuple};
    use crate::hugr::Patch;
    use crate::ops::Const;
    use crate::ops::handle::DataflowParentID;
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
                    function_calls: Vec::new(),
                })
            }
        }
    }

    /// A Module with a single function from three qubits to three qubits.
    /// The function applies a CX gate to the first two qubits and a Rz gate
    /// (with a constant angle) to the last qubit.
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
        assert!(sub.validate(&func, Default::default()).is_ok());

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
        assert!(sub.validate(&func, Default::default()).is_ok());
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
        );
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
                vec![
                    hugr.linked_ports(inp, first_cx_edge)
                        .map(|(n, p)| (n, p.as_incoming().unwrap()))
                        .collect()
                ],
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

    /// A subgraphs mixed with multiports caused a `NonConvex` error.
    /// <https://github.com/CQCL/hugr/issues/1294>
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
        assert_eq!(func_defn.signature(), &func.signature(&func_graph).into());
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

    /// Test the behaviour of the sibling subgraph when built from a single
    /// node.
    #[test]
    fn single_node_subgraph() {
        // A hugr with a single NOT operation, with disconnected output.
        let mut b = DFGBuilder::new(Signature::new(bool_t(), type_row![])).unwrap();
        let inw = b.input_wires().exactly_one().unwrap();
        let not_n = b.add_dataflow_op(LogicOp::Not, [inw]).unwrap();
        // Unconnected output, discarded
        let h = b.finish_hugr_with_outputs([]).unwrap();

        // When built with `from_node`, the subgraph's signature is the same as the
        // node's. (bool input, bool output)
        let subg = SiblingSubgraph::from_node(not_n.node(), &h);
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(vec![bool_t()], vec![bool_t()]).io()
        );

        // `from_nodes` is different, is it only uses incoming and outgoing edges to
        // compute the signature. In this case, the output is disconnected, so
        // it is not part of the subgraph signature.
        let subg = SiblingSubgraph::try_from_nodes([not_n.node()], &h).unwrap();
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(vec![bool_t()], vec![]).io()
        );
    }

    /// Test the behaviour of the sibling subgraph when built from a single
    /// node with no inputs or outputs.
    #[test]
    fn singleton_disconnected_subgraph() {
        // A hugr with some empty MakeTuple operations.
        let op = MakeTuple::new(type_row![]);

        let mut b = DFGBuilder::new(Signature::new_endo(type_row![])).unwrap();
        let _mk_tuple_1 = b.add_dataflow_op(op.clone(), []).unwrap();
        let mk_tuple_2 = b.add_dataflow_op(op.clone(), []).unwrap();
        let _mk_tuple_3 = b.add_dataflow_op(op, []).unwrap();
        // Unconnected output, discarded
        let h = b.finish_hugr_with_outputs([]).unwrap();

        // When built with `try_from_nodes`, the subgraph's signature is the same as the
        // node's. (empty input, tuple output)
        let subg = SiblingSubgraph::from_node(mk_tuple_2.node(), &h);
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(type_row![], vec![Type::new_tuple(type_row![])]).io()
        );

        // `from_nodes` is different, is it only uses incoming and outgoing edges to
        // compute the signature. In this case, the output is disconnected, so
        // it is not part of the subgraph signature.
        let subg = SiblingSubgraph::try_from_nodes([mk_tuple_2.node()], &h).unwrap();
        assert_eq!(subg.nodes().len(), 1);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new_endo(type_row![]).io()
        );
    }

    /// Run `try_from_nodes` including some complete graph components.
    #[test]
    fn partially_connected_subgraph() {
        // A hugr with some empty MakeTuple operations.
        let tuple_op = MakeTuple::new(type_row![]);
        let untuple_op = UnpackTuple::new(type_row![]);
        let tuple_t = Type::new_tuple(type_row![]);

        let mut b = DFGBuilder::new(Signature::new(type_row![], vec![tuple_t.clone()])).unwrap();
        let mk_tuple_1 = b.add_dataflow_op(tuple_op.clone(), []).unwrap();
        let untuple_1 = b
            .add_dataflow_op(untuple_op.clone(), [mk_tuple_1.out_wire(0)])
            .unwrap();
        let mk_tuple_2 = b.add_dataflow_op(tuple_op.clone(), []).unwrap();
        let _mk_tuple_3 = b.add_dataflow_op(tuple_op, []).unwrap();
        // Output the 2nd tuple output
        let h = b
            .finish_hugr_with_outputs([mk_tuple_2.out_wire(0)])
            .unwrap();

        let subgraph_nodes = [mk_tuple_1.node(), mk_tuple_2.node(), untuple_1.node()];

        // `try_from_nodes` uses incoming and outgoing edges to compute the signature.
        let subg = SiblingSubgraph::try_from_nodes(subgraph_nodes, &h).unwrap();
        assert_eq!(subg.nodes().len(), 3);
        assert_eq!(
            subg.signature(&h).io(),
            Signature::new(type_row![], vec![tuple_t]).io()
        );
    }

    #[test]
    fn test_set_outgoing_ports() {
        let (hugr, func_root) = build_3not_hugr().unwrap();
        let [inp, out] = hugr.get_io(func_root).unwrap();
        let not1 = hugr.output_neighbours(inp).exactly_one().ok().unwrap();
        let not1_out = hugr.node_outputs(not1).next().unwrap();

        // Create a subgraph with just the NOT gate
        let mut subgraph = SiblingSubgraph::from_node(not1, &hugr);

        // Initially should have one output
        assert_eq!(subgraph.outgoing_ports().len(), 1);

        // Try to set two outputs by copying the existing one
        let new_outputs = vec![(not1, not1_out), (not1, not1_out)];
        assert!(subgraph.set_outgoing_ports(new_outputs, &hugr).is_ok());

        // Should now have two outputs
        assert_eq!(subgraph.outgoing_ports().len(), 2);

        // Try to set an invalid output (from a different node)
        let invalid_outputs = vec![(not1, not1_out), (out, 2.into())];
        assert!(matches!(
            subgraph.set_outgoing_ports(invalid_outputs, &hugr),
            Err(InvalidOutputPorts::UnknownOutput { .. })
        ));

        // Should still have two outputs from before
        assert_eq!(subgraph.outgoing_ports().len(), 2);
    }

    #[test]
    fn test_set_outgoing_ports_linear() {
        let (hugr, func_root) = build_hugr().unwrap();
        let [inp, _out] = hugr.get_io(func_root).unwrap();
        let rz = hugr.output_neighbours(inp).nth(2).unwrap();
        let rz_out = hugr.node_outputs(rz).next().unwrap();

        // Create a subgraph with just the CX gate
        let mut subgraph = SiblingSubgraph::from_node(rz, &hugr);

        // Initially should have one output
        assert_eq!(subgraph.outgoing_ports().len(), 1);

        // Try to set two outputs by copying the existing one (should fail for linear
        // ports)
        let new_outputs = vec![(rz, rz_out), (rz, rz_out)];
        assert!(matches!(
            subgraph.set_outgoing_ports(new_outputs, &hugr),
            Err(InvalidOutputPorts::NonUniqueLinear)
        ));

        // Should still have one output
        assert_eq!(subgraph.outgoing_ports().len(), 1);
    }

    #[test]
    fn test_try_from_nodes_with_intervals() {
        let (hugr, func_root) = build_3not_hugr().unwrap();
        let line_checker = LineConvexChecker::new(&hugr, func_root);
        let [inp, _out] = hugr.get_io(func_root).unwrap();
        let not1 = hugr.output_neighbours(inp).exactly_one().ok().unwrap();
        let not2 = hugr.output_neighbours(not1).exactly_one().ok().unwrap();

        let intervals = line_checker.get_intervals_from_nodes([not1, not2]).unwrap();
        let subgraph =
            SiblingSubgraph::try_from_nodes_with_intervals([not1, not2], &intervals, &line_checker)
                .unwrap();
        let exp_subgraph = SiblingSubgraph::try_from_nodes([not1, not2], &hugr).unwrap();

        assert_eq!(subgraph, exp_subgraph);
        assert_eq!(
            line_checker.nodes_in_intervals(&intervals).collect_vec(),
            [not1, not2]
        );

        let intervals2 = line_checker
            .get_intervals_from_boundary_ports([
                (not1, IncomingPort::from(0).into()),
                (not2, OutgoingPort::from(0).into()),
            ])
            .unwrap();
        let subgraph2 = SiblingSubgraph::try_from_nodes_with_intervals(
            [not1, not2],
            &intervals2,
            &line_checker,
        )
        .unwrap();
        assert_eq!(subgraph2, exp_subgraph);
    }

    #[test]
    fn test_validate() {
        let (hugr, func_root) = build_3not_hugr().unwrap();
        let func = hugr.with_entrypoint(func_root);
        let checker = TopoConvexChecker::new(&func, func_root);
        let [inp, _out] = hugr.get_io(func_root).unwrap();
        let not1 = hugr.output_neighbours(inp).exactly_one().ok().unwrap();
        let not2 = hugr.output_neighbours(not1).exactly_one().ok().unwrap();
        let not3 = hugr.output_neighbours(not2).exactly_one().ok().unwrap();

        // A valid boundary, and convex
        let sub = SiblingSubgraph::new_unchecked(
            vec![vec![(not1, 0.into())]],
            vec![(not2, 0.into())],
            vec![],
            vec![not1, not2],
        );
        assert_eq!(sub.validate(&func, ValidationMode::SkipConvexity), Ok(()));
        assert_eq!(sub.validate(&func, ValidationMode::CheckConvexity), Ok(()));
        assert_eq!(
            sub.validate(&func, ValidationMode::WithChecker(&checker)),
            Ok(())
        );

        // A valid boundary, but not convex
        let sub = SiblingSubgraph::new_unchecked(
            vec![vec![(not1, 0.into())], vec![(not3, 0.into())]],
            vec![(not1, 0.into()), (not3, 0.into())],
            vec![],
            vec![not1, not3],
        );
        assert_eq!(sub.validate(&func, ValidationMode::SkipConvexity), Ok(()));
        assert_eq!(
            sub.validate(&func, ValidationMode::CheckConvexity),
            Err(InvalidSubgraph::NotConvex)
        );
        assert_eq!(
            sub.validate(&func, ValidationMode::WithChecker(&checker)),
            Err(InvalidSubgraph::NotConvex)
        );

        // An invalid boundary (missing an input)
        let sub = SiblingSubgraph::new_unchecked(
            vec![vec![(not1, 0.into())]],
            vec![(not1, 0.into()), (not3, 0.into())],
            vec![],
            vec![not1, not3],
        );
        assert_eq!(
            sub.validate(&func, ValidationMode::SkipConvexity),
            Err(InvalidSubgraph::InvalidNodeSet)
        );
    }

    #[fixture]
    pub(crate) fn hugr_call_subgraph() -> Hugr {
        let mut builder = ModuleBuilder::new();
        let decl_node = builder.declare("test", endo_sig(bool_t()).into()).unwrap();
        let mut main = builder.define_function("main", endo_sig(bool_t())).unwrap();
        let [bool] = main.input_wires_arr();

        let [bool] = main
            .add_dataflow_op(LogicOp::Not, [bool])
            .unwrap()
            .outputs_arr();

        // Chain two calls to the same function
        let [bool] = main.call(&decl_node, &[], [bool]).unwrap().outputs_arr();
        let [bool] = main.call(&decl_node, &[], [bool]).unwrap().outputs_arr();

        let main_def = main.finish_with_outputs([bool]).unwrap();

        let mut hugr = builder.finish_hugr().unwrap();
        hugr.set_entrypoint(main_def.node());
        hugr
    }

    #[rstest]
    fn test_call_subgraph_from_dfg(hugr_call_subgraph: Hugr) {
        let subg =
            SiblingSubgraph::try_new_dataflow_subgraph::<_, DataflowParentID>(&hugr_call_subgraph)
                .unwrap();

        assert_eq!(subg.function_calls.len(), 1);
        assert_eq!(subg.function_calls[0].len(), 2);
    }

    #[rstest]
    fn test_call_subgraph_from_nodes(hugr_call_subgraph: Hugr) {
        let call_nodes = hugr_call_subgraph
            .children(hugr_call_subgraph.entrypoint())
            .filter(|&n| hugr_call_subgraph.get_optype(n).is_call())
            .collect_vec();

        let subg =
            SiblingSubgraph::try_from_nodes(call_nodes.clone(), &hugr_call_subgraph).unwrap();
        assert_eq!(subg.function_calls.len(), 1);
        assert_eq!(subg.function_calls[0].len(), 2);

        let subg =
            SiblingSubgraph::try_from_nodes(call_nodes[0..1].to_owned(), &hugr_call_subgraph)
                .unwrap();
        assert_eq!(subg.function_calls.len(), 1);
        assert_eq!(subg.function_calls[0].len(), 1);
    }

    #[rstest]
    fn test_call_subgraph_from_boundary(hugr_call_subgraph: Hugr) {
        let call_nodes = hugr_call_subgraph
            .children(hugr_call_subgraph.entrypoint())
            .filter(|&n| hugr_call_subgraph.get_optype(n).is_call())
            .collect_vec();
        let not_node = hugr_call_subgraph
            .children(hugr_call_subgraph.entrypoint())
            .filter(|&n| hugr_call_subgraph.get_optype(n) == &LogicOp::Not.into())
            .exactly_one()
            .ok()
            .unwrap();

        let subg = SiblingSubgraph::try_new(
            vec![
                vec![(not_node, IncomingPort::from(0))],
                call_nodes
                    .iter()
                    .map(|&n| (n, IncomingPort::from(1)))
                    .collect_vec(),
            ],
            vec![(call_nodes[1], OutgoingPort::from(0))],
            &hugr_call_subgraph,
        )
        .unwrap();

        assert_eq!(subg.function_calls.len(), 1);
        assert_eq!(subg.function_calls[0].len(), 2);
    }
}

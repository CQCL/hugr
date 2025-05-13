//! Implementation of the `SimpleReplace` operation.

use std::collections::HashMap;

use crate::core::HugrNode;
use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::views::SiblingSubgraph;
pub use crate::hugr::views::sibling_subgraph::InvalidReplacement;
use crate::hugr::{HugrMut, HugrView};
use crate::ops::{OpTag, OpTrait, OpType};
use crate::types::EdgeKind;
use crate::{Hugr, IncomingPort, Node, OutgoingPort, Port, PortIndex};

use itertools::Itertools;

use thiserror::Error;

use super::inline_dfg::InlineDFGError;
use super::{BoundaryPort, HostPort, PatchHugrMut, PatchVerification, ReplacementPort};

/// Specification of a simple replacement operation.
///
/// # Type parameters
///
/// - `N`: The type of nodes in the host hugr.
#[derive(Debug, Clone)]
pub struct SimpleReplacement<HostNode = Node> {
    /// The subgraph of the host hugr to be replaced.
    subgraph: SiblingSubgraph<HostNode>,
    /// A hugr with DFG root (consisting of replacement nodes).
    replacement: Hugr,
}

impl<HostNode: HugrNode> SimpleReplacement<HostNode> {
    /// Create a new [`SimpleReplacement`] specification without checking that
    /// the replacement has the same signature as the subgraph.
    #[inline]
    pub fn new_unchecked(subgraph: SiblingSubgraph<HostNode>, replacement: Hugr) -> Self {
        Self {
            subgraph,
            replacement,
        }
    }

    /// Create a new [`SimpleReplacement`] specification.
    ///
    /// Return a [`InvalidReplacement::InvalidSignature`] error if `subgraph`
    /// and `replacement` have different signatures.
    pub fn try_new(
        subgraph: SiblingSubgraph<HostNode>,
        host: &impl HugrView<Node = HostNode>,
        replacement: Hugr,
    ) -> Result<Self, InvalidReplacement> {
        let subgraph_sig = subgraph.signature(host);
        let repl_sig =
            replacement
                .inner_function_type()
                .ok_or(InvalidReplacement::InvalidDataflowGraph {
                    node: replacement.entrypoint(),
                    op: replacement.get_optype(replacement.entrypoint()).to_owned(),
                })?;
        if subgraph_sig != repl_sig {
            return Err(InvalidReplacement::InvalidSignature {
                expected: subgraph_sig,
                actual: Some(repl_sig.into_owned()),
            });
        }
        Ok(Self {
            subgraph,
            replacement,
        })
    }

    /// The replacement hugr.
    #[inline]
    pub fn replacement(&self) -> &Hugr {
        &self.replacement
    }

    /// Consume self and return the replacement hugr.
    #[inline]
    pub fn into_replacement(self) -> Hugr {
        self.replacement
    }

    /// Subgraph to be replaced.
    #[inline]
    pub fn subgraph(&self) -> &SiblingSubgraph<HostNode> {
        &self.subgraph
    }

    /// Check if the replacement can be applied to the given hugr.
    pub fn is_valid_rewrite(
        &self,
        h: &impl HugrView<Node = HostNode>,
    ) -> Result<(), SimpleReplacementError> {
        let parent = self.subgraph.get_parent(h);

        // 1. Check the parent node exists and is a DataflowParent.
        if !OpTag::DataflowParent.is_superset(h.get_optype(parent).tag()) {
            return Err(SimpleReplacementError::InvalidParentNode());
        }

        // 2. Check that all the to-be-removed nodes are children of it and are leaves.
        for node in self.subgraph.nodes() {
            if h.get_parent(*node) != Some(parent) || h.children(*node).next().is_some() {
                return Err(SimpleReplacementError::InvalidRemovedNode());
            }
        }

        Ok(())
    }

    /// Get the input and output nodes of the replacement hugr.
    pub fn get_replacement_io(&self) -> [Node; 2] {
        self.replacement
            .get_io(self.replacement.entrypoint())
            .expect("replacement is a DFG")
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `host` to incoming ports in `self.replacement`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `host` and the second is a port in `self.replacement`:
    ///  - The outgoing host ports are always linked to the input boundary of
    ///    `subgraph`, i.e. the ports returned by
    ///    [`SiblingSubgraph::incoming_ports`],
    ///  - The incoming replacement ports are always linked to output ports of
    ///    the [`OpTag::Input`] node of `self.replacement`.
    pub fn incoming_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            HostPort<HostNode, OutgoingPort>,
            ReplacementPort<IncomingPort>,
        ),
    > + 'a {
        // The outgoing ports at the input boundary of `subgraph`
        let subgraph_outgoing_ports = self
            .subgraph
            .incoming_ports()
            .iter()
            .map(|in_ports| *in_ports.first().expect("non-empty input partition"))
            .map(|(node, in_port)| {
                host.single_linked_output(node, in_port)
                    .expect("valid dfg wire")
            });

        // The incoming ports at the input boundary of `replacement`
        let [repl_inp, _] = self.get_replacement_io();
        let repl_incoming_ports = self
            .replacement
            .node_outputs(repl_inp)
            .filter(move |&port| is_value_port(&self.replacement, repl_inp, port))
            .map(move |repl_out_port| {
                self.replacement
                    .linked_inputs(repl_inp, repl_out_port)
                    .filter(|&(node, _)| self.replacement.get_optype(node).tag() != OpTag::Output)
            });

        // Zip the two iterators and add edges from each outgoing port to all
        // corresponding incoming ports.
        subgraph_outgoing_ports.zip(repl_incoming_ports).flat_map(
            |((subgraph_out_node, subgraph_out_port), repl_all_incoming)| {
                // add edge from outgoing port in subgraph to incoming port in
                // replacement
                repl_all_incoming.map(move |(repl_inp_node, repl_inp_port)| {
                    (
                        HostPort(subgraph_out_node, subgraph_out_port),
                        ReplacementPort(repl_inp_node, repl_inp_port),
                    )
                })
            },
        )
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `self.replacement` to incoming ports in `host`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `self.replacement` and the second is a port in `host`:
    ///  - The outgoing replacement ports are always linked to inputs of the
    ///    [`OpTag::Output`] node of `self.replacement`,
    ///  - The incoming host ports are always linked to the output boundary of
    ///    `subgraph`, i.e. the ports returned by
    ///    [`SiblingSubgraph::outgoing_ports`],
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn outgoing_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            ReplacementPort<OutgoingPort>,
            HostPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        // The incoming ports at the output boundary of `subgraph`
        let subgraph_incoming_ports =
            self.subgraph
                .outgoing_ports()
                .iter()
                .map(|&(subgraph_out_node, subgraph_out_port)| {
                    host.linked_inputs(subgraph_out_node, subgraph_out_port)
                });

        // The outgoing ports at the output boundary of `replacement`
        let [_, repl_out] = self.get_replacement_io();
        let repl_outgoing_ports = self
            .replacement
            .node_inputs(repl_out)
            .filter(move |&port| is_value_port(&self.replacement, repl_out, port))
            .map(move |repl_in_port| {
                self.replacement
                    .single_linked_output(repl_out, repl_in_port)
                    .expect("valid dfg wire")
            });

        repl_outgoing_ports.zip(subgraph_incoming_ports).flat_map(
            |((repl_out_node, repl_out_port), subgraph_all_incoming)| {
                if self.replacement.get_optype(repl_out_node).tag() != OpTag::Input {
                    Some(
                        subgraph_all_incoming.map(move |(subgraph_in_node, subgraph_in_port)| {
                            (
                                // the new output node will be updated after insertion
                                ReplacementPort(repl_out_node, repl_out_port),
                                HostPort(subgraph_in_node, subgraph_in_port),
                            )
                        }),
                    )
                    .into_iter()
                    .flatten()
                } else {
                    None.into_iter().flatten()
                }
            },
        )
    }

    /// Get all edges that the replacement would add between ports in `host`.
    ///
    /// These correspond to direct edges between the input and output nodes
    /// in the replacement graph.
    ///
    /// For each pair in the returned vector, both ports are in `host`:
    ///  - The outgoing host ports are linked to the input boundary of
    ///    `subgraph`, i.e. the ports returned by
    ///    [`SiblingSubgraph::incoming_ports`],
    ///  - The incoming host ports are linked to the output boundary of
    ///    `subgraph`, i.e. the ports returned by
    ///    [`SiblingSubgraph::outgoing_ports`].
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn host_to_host_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            HostPort<HostNode, OutgoingPort>,
            HostPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        let [repl_in, repl_out] = self.get_replacement_io();

        let empty_wires = self
            .replacement
            .node_inputs(repl_out)
            .filter(move |&port| is_value_port(&self.replacement, repl_out, port))
            .filter_map(move |repl_in_port| {
                let (repl_out_node, repl_out_port) = self
                    .replacement
                    .single_linked_output(repl_out, repl_in_port)
                    .expect("valid dfg wire");
                (repl_out_node == repl_in).then_some((repl_out_port, repl_in_port))
            });

        // The outgoing ports at the input boundary of `subgraph`
        let subgraph_input_boundary = self
            .subgraph
            .incoming_ports()
            .iter()
            .map(|node_ports| {
                let (node, port) = *node_ports.first().expect("non-empty boundary partition");
                host.single_linked_output(node, port)
                    .expect("valid dfg wire")
            })
            .collect_vec();
        // The incoming ports at the output boundary of `subgraph`
        let subgraph_output_boundary = self
            .subgraph
            .outgoing_ports()
            .iter()
            .map(|&(node, port)| host.linked_inputs(node, port).collect_vec())
            .collect_vec();

        empty_wires.flat_map(move |(repl_out_port, repl_in_port)| {
            let (host_out_node, host_out_port) = subgraph_input_boundary[repl_out_port.index()];
            subgraph_output_boundary[repl_in_port.index()]
                .clone()
                .into_iter()
                .map(move |(host_in_node, host_in_port)| {
                    (
                        HostPort(host_out_node, host_out_port),
                        HostPort(host_in_node, host_in_port),
                    )
                })
        })
    }

    /// Get the incoming port at the output node of `self.replacement`
    /// that corresponds to the given outgoing port on the subgraph output
    /// boundary.
    ///
    /// The host `port` should be a port in `self.subgraph().outgoing_ports()`.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_host_output(
        &self,
        port: impl Into<HostPort<HostNode, OutgoingPort>>,
    ) -> Option<ReplacementPort<IncomingPort>> {
        let HostPort(node, port) = port.into();
        let pos = self
            .subgraph
            .outgoing_ports()
            .iter()
            .position(|&node_port| node_port == (node, port))?;
        let incoming_port: IncomingPort = pos.into();
        let [_, rep_output] = self.get_replacement_io();
        Some(ReplacementPort(rep_output, incoming_port))
    }

    /// Get the incoming ports in the input boundary of `subgraph` that
    /// correspond to the given output port at the input node of `replacement`
    ///
    /// Return ports in `self.subgraph().incoming_ports()`.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_replacement_input(
        &self,
        port: impl Into<ReplacementPort<OutgoingPort>>,
    ) -> impl Iterator<Item = HostPort<HostNode, IncomingPort>> + '_ {
        let ReplacementPort(node, port) = port.into();
        let [repl_input, _] = self.get_replacement_io();

        let ports = if node == repl_input {
            self.subgraph.incoming_ports().get(port.index())
        } else {
            None
        };
        ports
            .into_iter()
            .flat_map(|ports| ports.iter().map(|&(n, p)| HostPort(n, p)))
    }

    /// Get all edges that the replacement would add between `host` and
    /// `self.replacement`.
    ///
    /// This is equivalent to chaining the results of
    /// [`Self::incoming_boundary`], [`Self::outgoing_boundary`], and
    /// [`Self::host_to_host_boundary`].
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn all_boundary_edges<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            BoundaryPort<HostNode, OutgoingPort>,
            BoundaryPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        let incoming_boundary = self
            .incoming_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));
        let outgoing_boundary = self
            .outgoing_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));
        let host_to_host_boundary = self
            .host_to_host_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));

        incoming_boundary
            .chain(outgoing_boundary)
            .chain(host_to_host_boundary)
    }
}

impl<HostNode: HugrNode> PatchVerification for SimpleReplacement<HostNode> {
    type Error = SimpleReplacementError;
    type Node = HostNode;

    fn verify(&self, h: &impl HugrView<Node = HostNode>) -> Result<(), SimpleReplacementError> {
        self.is_valid_rewrite(h)
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = HostNode> {
        self.subgraph.nodes().iter().copied()
    }
}

/// Result of applying a [`SimpleReplacement`].
pub struct Outcome<HostNode = Node> {
    /// Map from Node in replacement to corresponding Node in the result Hugr
    pub node_map: HashMap<Node, HostNode>,
    /// Nodes removed from the result Hugr and their weights
    pub removed_nodes: HashMap<HostNode, OpType>,
}

impl<N: HugrNode> PatchHugrMut for SimpleReplacement<N> {
    type Outcome = Outcome<N>;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<Self::Outcome, Self::Error> {
        self.is_valid_rewrite(h)?;

        let parent = self.subgraph.get_parent(h);

        // We proceed to connect the edges between the newly inserted
        // replacement and the rest of the graph.
        //
        // Existing connections to the removed subgraph will be automatically
        // removed when the nodes are removed.

        // 1. Get the boundary edges
        let boundary_edges = self.all_boundary_edges(h).collect_vec();

        let Self {
            replacement,
            subgraph,
            ..
        } = self;

        // Nodes to remove from the replacement hugr
        let repl_io = replacement
            .get_io(replacement.entrypoint())
            .expect("replacement is DFG-rooted");
        let repl_entrypoint = replacement.entrypoint();

        // 2. Insert the replacement as a whole.
        let InsertionResult {
            inserted_entrypoint: new_entrypoint,
            mut node_map,
        } = h.insert_hugr(parent, replacement);

        // remove the Input and Output from h and node_map
        for node in repl_io {
            let node_h = node_map[&node];
            h.remove_node(node_h);
            node_map.remove(&node);
        }

        // make all (remaining) replacement top level children children of the parent
        for child in h.children(new_entrypoint).collect_vec() {
            h.set_parent(child, parent);
        }

        // remove the replacement entrypoint from h and node_map
        h.remove_node(new_entrypoint);
        node_map.remove(&repl_entrypoint);

        // 3. Insert all boundary edges.
        for (src, tgt) in boundary_edges {
            let (src_node, src_port) = src.map_replacement(&node_map);
            let (tgt_node, tgt_port) = tgt.map_replacement(&node_map);
            h.connect(src_node, src_port, tgt_node, tgt_port);
        }

        // 4. Remove all nodes in subgraph and edges between them.
        let removed_nodes = subgraph
            .nodes()
            .iter()
            .map(|&node| (node, h.remove_node(node)))
            .collect();

        Ok(Outcome {
            node_map,
            removed_nodes,
        })
    }
}

/// Error from a [`SimpleReplacement`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimpleReplacementError {
    /// Invalid parent node.
    #[error("Parent node is invalid.")]
    InvalidParentNode(),
    /// Node requested for removal is invalid.
    #[error("A node requested for removal is invalid.")]
    InvalidRemovedNode(),
    /// Node in replacement graph is invalid.
    #[error("A node in the replacement graph is invalid.")]
    InvalidReplacementNode(),
    /// Inlining replacement failed.
    #[error("Inlining replacement failed: {0}")]
    InliningFailed(#[from] InlineDFGError),
}

fn is_value_port<N: HugrNode>(
    hugr: &impl HugrView<Node = N>,
    node: N,
    port: impl Into<Port>,
) -> bool {
    hugr.get_optype(node)
        .port_kind(port)
        .as_ref()
        .is_some_and(EdgeKind::is_value)
}

#[cfg(test)]
pub(in crate::hugr::patch) mod test {
    use itertools::Itertools;
    use rstest::{fixture, rstest};

    use std::collections::{HashMap, HashSet};

    use crate::Node;
    use crate::builder::test::n_identity;
    use crate::builder::{
        BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        HugrBuilder, ModuleBuilder, endo_sig, inout_sig,
    };
    use crate::extension::prelude::{bool_t, qb_t};
    use crate::hugr::patch::simple_replace::Outcome;
    use crate::hugr::patch::{HostPort, PatchVerification, ReplacementPort};
    use crate::hugr::views::{HugrView, SiblingSubgraph};
    use crate::hugr::{Hugr, HugrMut, Patch};
    use crate::ops::OpTag;
    use crate::ops::OpTrait;
    use crate::ops::handle::NodeHandle;
    use crate::std_extensions::logic::LogicOp;
    use crate::std_extensions::logic::test::and_op;
    use crate::types::{Signature, Type};
    use crate::utils::test_quantum_extension::{cx_gate, h_gate};

    use super::SimpleReplacement;

    /// Creates a hugr like the following:
    /// --   H   --
    /// -- [DFG] --
    /// where [DFG] is:
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    fn make_hugr() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let _f_id = {
            let mut func_builder = module_builder
                .define_function("main", Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]))?;

            let [qb0, qb1, qb2] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb2])?;

            let mut inner_builder =
                func_builder.dfg_builder_endo([(qb_t(), qb0), (qb_t(), qb1)])?;
            let inner_graph = {
                let [wire0, wire1] = inner_builder.input_wires_arr();
                let wire2 = inner_builder.add_dataflow_op(h_gate(), vec![wire0])?;
                let wire3 = inner_builder.add_dataflow_op(h_gate(), vec![wire1])?;
                let wire45 = inner_builder
                    .add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
                let [wire4, wire5] = wire45.outputs_arr();
                let wire6 = inner_builder.add_dataflow_op(h_gate(), vec![wire4])?;
                let wire7 = inner_builder.add_dataflow_op(h_gate(), vec![wire5])?;
                inner_builder.finish_with_outputs(wire6.outputs().chain(wire7.outputs()))
            }?;

            func_builder.finish_with_outputs(inner_graph.outputs().chain(q_out.outputs()))?
        };
        Ok(module_builder.finish_hugr()?)
    }

    #[fixture]
    pub(in crate::hugr::patch) fn simple_hugr() -> Hugr {
        make_hugr().unwrap()
    }
    /// Creates a hugr with a DFG root like the following:
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn make_dfg_hugr() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;
        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire0])?;
        let wire3 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire45 =
            dfg_builder.add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
        dfg_builder.finish_hugr_with_outputs(wire45.outputs())
    }

    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr() -> Hugr {
        make_dfg_hugr().unwrap()
    }

    /// Creates a hugr with a DFG root like the following:
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn make_dfg_hugr2() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;

        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire2out = wire2.outputs().exactly_one().unwrap();
        let wireoutvec = vec![wire0, wire2out];
        dfg_builder.finish_hugr_with_outputs(wireoutvec)
    }

    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr2() -> Hugr {
        make_dfg_hugr2().unwrap()
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (1) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (0) NOT ├───┤
    ///  └─────────┘   │    ┌─────────┐
    ///                └────┤ (2) NOT ├──
    ///                     └─────────┘
    /// This can be replaced with an empty hugr coping the input to both
    /// outputs.
    ///
    /// Returns the hugr and the nodes of the NOT gates, in order.
    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr_copy_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let not_1 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b1] = not_1.outputs_arr();

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node(), not_1.node()],
        )
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (1) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (0) NOT ├───┤
    ///  └─────────┘   │
    ///                └─────────────────
    ///
    /// This can be replaced with a single NOT op, coping the input to the first
    /// output.
    ///
    /// Returns the hugr and the nodes of the NOT ops, in order.
    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr_half_not_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let b1 = b;

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node()],
        )
    }

    #[rstest]
    /// Replace the
    ///      ┌───┐
    /// ──■──┤ H ├
    /// ┌─┴─┐├───┤
    /// ┤ X ├┤ H ├
    /// └───┘└───┘
    /// part of
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    /// with
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn test_simple_replacement(
        simple_hugr: Hugr,
        dfg_hugr: Hugr,
        #[values(apply_simple, apply_replace)] applicator: impl Fn(&mut Hugr, SimpleReplacement),
    ) {
        let mut h: Hugr = simple_hugr;
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .entry_descendants()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr;
        // 3. Construct the input and output matchings
        // 3.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .entry_descendants()
            .find(|node: &Node| *n.get_optype(*node) == cx_gate().into())
            .unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let (n_cx_out_0, _n_cx_out_1) = n.node_outputs(n_node_cx).take(2).collect_tuple().unwrap();
        let n_port_2 = n.linked_inputs(n_node_cx, n_cx_out_0).next().unwrap().1;
        // 3.3. Locate the ports we need to specify as "glue" in h
        let h_h0_out = h.node_outputs(h_node_h0).next().unwrap();
        // 4. Define the replacement
        let r = SimpleReplacement {
            subgraph: SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            replacement: n,
        };

        // Check output boundary
        assert_eq!(
            r.map_host_output((h_node_h0, h_h0_out)).unwrap(),
            ReplacementPort::from((r.get_replacement_io()[1], n_port_2))
        );

        // Check invalidation set
        assert_eq!(
            HashSet::<_>::from_iter(r.invalidation_set()),
            HashSet::<_>::from_iter([h_node_cx, h_node_h0, h_node_h1]),
        );

        applicator(&mut h, r);
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├──■──
        // ├───┤├───┤┌─┴─┐
        // ┤ H ├┤ H ├┤ X ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(), Ok(()));
    }

    #[rstest]
    /// Replace the
    ///
    /// ──■──
    /// ┌─┴─┐
    /// ┤ X ├
    /// └───┘
    /// part of
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    /// with
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn test_simple_replacement_with_empty_wires(simple_hugr: Hugr, dfg_hugr2: Hugr) {
        let mut h: Hugr = simple_hugr;

        // 1. Locate the CX in h
        let h_node_cx: Node = h
            .entry_descendants()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let s: Vec<Node> = vec![h_node_cx];
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr2;
        // 3. Construct the input and output matchings
        // 3.1. Locate the Output and its predecessor H in n
        let n_node_output = n.get_io(n.entrypoint()).unwrap()[1];
        let (_n_node_input, n_node_h) = n.input_neighbours(n_node_output).collect_tuple().unwrap();
        // 4. Define the replacement
        let r = SimpleReplacement {
            subgraph: SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            replacement: n,
        };
        let Outcome {
            node_map,
            removed_nodes,
        } = h.apply_patch(r).unwrap();

        assert_eq!(
            node_map.into_keys().collect::<HashSet<_>>(),
            [n_node_h].into_iter().collect::<HashSet<_>>(),
        );
        assert_eq!(
            removed_nodes.into_keys().collect::<HashSet<_>>(),
            [h_node_cx].into_iter().collect::<HashSet<_>>(),
        );

        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├
        // ├───┤├───┤┌───┐
        // ┤ H ├┤ H ├┤ H ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(), Ok(()));
    }

    #[test]
    fn test_replace_cx_cross() {
        let q_row: Vec<Type> = vec![qb_t(), qb_t()];
        let mut builder = DFGBuilder::new(endo_sig(q_row)).unwrap();
        let mut circ = builder.as_circuit(builder.input_wires());
        circ.append(cx_gate(), [0, 1]).unwrap();
        circ.append(cx_gate(), [1, 0]).unwrap();
        let wires = circ.finish();
        let mut h = builder.finish_hugr_with_outputs(wires).unwrap();
        let replacement = h.clone();
        let orig = h.clone();

        let removal = h
            .entry_descendants()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();
        h.apply_patch(
            SimpleReplacement::try_new(
                SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
                &h,
                replacement,
            )
            .unwrap(),
        )
        .unwrap();

        // They should be the same, up to node indices
        assert_eq!(h.num_edges(), orig.num_edges());
    }

    #[test]
    fn test_replace_after_copy() {
        let one_bit = vec![bool_t()];
        let two_bit = vec![bool_t(), bool_t()];

        let mut builder = DFGBuilder::new(endo_sig(one_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw = builder
            .add_dataflow_op(and_op(), [inw, inw])
            .unwrap()
            .outputs();
        let mut h = builder.finish_hugr_with_outputs(outw).unwrap();

        let mut builder = DFGBuilder::new(inout_sig(two_bit, one_bit)).unwrap();
        let inw = builder.input_wires();
        let outw = builder.add_dataflow_op(and_op(), inw).unwrap().outputs();
        let repl = builder.finish_hugr_with_outputs(outw).unwrap();

        let orig = h.clone();

        let removal = h
            .entry_descendants()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();

        h.apply_patch(
            SimpleReplacement::try_new(
                SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
                &h,
                repl,
            )
            .unwrap(),
        )
        .unwrap();

        // Nothing changed
        assert_eq!(h.num_nodes(), orig.num_nodes());
    }

    /// Remove all the NOT gates in [`dfg_hugr_copy_bools`] by connecting the
    /// input directly to the outputs.
    ///
    /// https://github.com/CQCL/hugr/issues/1190
    #[rstest]
    fn test_copy_inputs(dfg_hugr_copy_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_copy_bools;
        let (input_not, output_not_0, output_not_1) = nodes.into_iter().collect_tuple().unwrap();

        let replacement = {
            let b =
                DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            b.finish_hugr_with_outputs([w, w]).unwrap()
        };

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0, output_not_1], &hugr)
                .unwrap();

        let rewrite = SimpleReplacement {
            subgraph,
            replacement,
        };
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.entry_descendants().count(), 3);
    }

    /// Remove one of the NOT ops in [`dfg_hugr_half_not_bools`] by connecting
    /// the input directly to the output.
    ///
    /// https://github.com/CQCL/hugr/issues/1323
    #[rstest]
    fn test_half_nots(dfg_hugr_half_not_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_half_not_bools;
        let (input_not, output_not_0) = nodes.into_iter().collect_tuple().unwrap();

        let replacement = {
            let mut b =
                DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            let not = b.add_dataflow_op(LogicOp::Not, vec![w]).unwrap();
            let [w_not] = not.outputs_arr();
            b.finish_hugr_with_outputs([w, w_not]).unwrap()
        };

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0], &hugr).unwrap();

        let rewrite = SimpleReplacement {
            subgraph,
            replacement,
        };
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.entry_descendants().count(), 4);
    }

    #[rstest]
    fn test_nested_replace(dfg_hugr2: Hugr) {
        // replace a node with a hugr with children

        let mut h = dfg_hugr2;
        let h_node = h
            .entry_descendants()
            .find(|node: &Node| *h.get_optype(*node) == h_gate().into())
            .unwrap();

        // build a nested identity dfg
        let mut nest_build = DFGBuilder::new(Signature::new_endo(qb_t())).unwrap();
        let [input] = nest_build.input_wires_arr();
        let inner_build = nest_build.dfg_builder_endo([(qb_t(), input)]).unwrap();
        let inner_dfg = n_identity(inner_build).unwrap();
        let replacement = nest_build
            .finish_hugr_with_outputs([inner_dfg.out_wire(0)])
            .unwrap();
        let subgraph = SiblingSubgraph::try_from_nodes(vec![h_node], &h).unwrap();

        let rewrite = SimpleReplacement::try_new(subgraph, &h, replacement).unwrap();

        assert_eq!(h.entry_descendants().count(), 4);

        rewrite.apply(&mut h).unwrap_or_else(|e| panic!("{e}"));
        h.validate().unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(h.entry_descendants().count(), 6);
    }

    use crate::hugr::patch::replace::Replacement;
    fn to_replace(h: &impl HugrView<Node = Node>, s: SimpleReplacement) -> Replacement {
        use crate::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec};

        let [in_, out] = s.get_replacement_io();
        let mu_inp = s
            .incoming_boundary(h)
            .map(
                |(HostPort(src, src_port), ReplacementPort(tgt, tgt_port))| {
                    if tgt == out {
                        unimplemented!()
                    }
                    NewEdgeSpec {
                        src,
                        tgt,
                        kind: NewEdgeKind::Value {
                            src_pos: src_port,
                            tgt_pos: tgt_port,
                        },
                    }
                },
            )
            .collect();
        let mu_out = s
            .outgoing_boundary(h)
            .map(
                |(ReplacementPort(src, src_port), HostPort(tgt, tgt_port))| {
                    if src == in_ {
                        unimplemented!()
                    }
                    NewEdgeSpec {
                        src,
                        tgt,
                        kind: NewEdgeKind::Value {
                            src_pos: src_port,
                            tgt_pos: tgt_port,
                        },
                    }
                },
            )
            .collect();
        let mut replacement = s.replacement;
        replacement.remove_node(in_);
        replacement.remove_node(out);
        Replacement {
            removal: s.subgraph.nodes().to_vec(),
            replacement,
            adoptions: HashMap::new(),
            mu_inp,
            mu_out,
            mu_new: vec![],
        }
    }

    fn apply_simple(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_patch(rw).unwrap();
    }

    fn apply_replace(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_patch(to_replace(h, rw)).unwrap();
    }
}

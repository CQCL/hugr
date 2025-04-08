//! Implementation of the `SimpleReplace` operation.

use std::collections::HashMap;

use crate::core::HugrNode;
use crate::hugr::hugrmut::InsertionResult;
pub use crate::hugr::internal::HugrMutInternals;
use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrMut, HugrView};
use crate::ops::{OpTag, OpTrait, OpType};
use crate::{Hugr, IncomingPort, Node, OutgoingPort};

use itertools::Itertools;

use thiserror::Error;

use super::inline_dfg::InlineDFGError;
use super::{ApplyPatchHugrMut, BoundaryPort, HostPort, ReplacementPort, VerifyPatch};

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
    /// A map from (target ports of edges from the Input node of `replacement`)
    /// to (target ports of edges from nodes not in `subgraph` to nodes in `subgraph`).
    nu_inp: HashMap<(Node, IncomingPort), (HostNode, IncomingPort)>,
    /// A map from (target ports of edges from nodes in `subgraph` to nodes not
    /// in `subgraph`) to (input ports of the Output node of `replacement`).
    nu_out: HashMap<(HostNode, IncomingPort), IncomingPort>,
}

impl<HostNode: HugrNode> SimpleReplacement<HostNode> {
    /// Create a new [`SimpleReplacement`] specification.
    #[inline]
    pub fn new(
        subgraph: SiblingSubgraph<HostNode>,
        replacement: Hugr,
        nu_inp: HashMap<(Node, IncomingPort), (HostNode, IncomingPort)>,
        nu_out: HashMap<(HostNode, IncomingPort), IncomingPort>,
    ) -> Self {
        Self {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        }
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
    pub fn get_replacement_io(&self) -> Result<[Node; 2], SimpleReplacementError> {
        self.replacement
            .get_io(self.replacement.root())
            .ok_or(SimpleReplacementError::InvalidParentNode())
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `host` to incoming ports in `self.replacement`.
    ///
    /// The incoming ports returned are always connected to outputs of
    /// the [`OpTag::Input`] node of `self.replacement`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `host` and the second is a port in `self.replacement`.
    pub fn incoming_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            HostPort<HostNode, OutgoingPort>,
            ReplacementPort<IncomingPort>,
        ),
    > + 'a {
        // For each p = self.nu_inp[q] such that q is not an Output port,
        // there will be an edge from the predecessor of p to (the new copy of) q.
        self.nu_inp
            .iter()
            .filter(|&((rep_inp_node, _), _)| {
                self.replacement.get_optype(*rep_inp_node).tag() != OpTag::Output
            })
            .map(
                |(&(rep_inp_node, rep_inp_port), (rem_inp_node, rem_inp_port))| {
                    // add edge from predecessor of (s_inp_node, s_inp_port) to (new_inp_node, n_inp_port)
                    let (rem_inp_pred_node, rem_inp_pred_port) = host
                        .single_linked_output(*rem_inp_node, *rem_inp_port)
                        .unwrap();
                    (
                        HostPort(rem_inp_pred_node, rem_inp_pred_port),
                        ReplacementPort(rep_inp_node, rep_inp_port),
                    )
                },
            )
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `self.replacement` to incoming ports in `host`.
    ///
    /// The outgoing ports returned are always connected to inputs of
    /// the [`OpTag::Output`] node of `self.replacement`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `self.replacement` and the second is a port in `host`.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn outgoing_boundary<'a>(
        &'a self,
        _host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            ReplacementPort<OutgoingPort>,
            HostPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        let [_, replacement_output_node] = self.get_replacement_io().expect("replacement is a DFG");

        // For each q = self.nu_out[p] such that the predecessor of q is not an Input port,
        // there will be an edge from (the new copy of) the predecessor of q to p.
        self.nu_out
            .iter()
            .filter_map(move |(&(rem_out_node, rem_out_port), rep_out_port)| {
                let (rep_out_pred_node, rep_out_pred_port) = self
                    .replacement
                    .single_linked_output(replacement_output_node, *rep_out_port)
                    .unwrap();
                (self.replacement.get_optype(rep_out_pred_node).tag() != OpTag::Input).then_some({
                    (
                        // the new output node will be updated after insertion
                        ReplacementPort(rep_out_pred_node, rep_out_pred_port),
                        HostPort(rem_out_node, rem_out_port),
                    )
                })
            })
    }

    /// Get all edges that the replacement would add between ports in `host`.
    ///
    /// These correspond to direct edges between the input and output nodes
    /// in the replacement graph.
    ///
    /// For each pair in the returned vector, the both ports are in `host`.
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
        let [_, replacement_output_node] = self.get_replacement_io().expect("replacement is a DFG");

        // For each q = self.nu_out[p1], p0 = self.nu_inp[q], add an edge from the predecessor of p0
        // to p1.
        self.nu_out
            .iter()
            .filter_map(move |(&(rem_out_node, rem_out_port), &rep_out_port)| {
                self.nu_inp
                    .get(&(replacement_output_node, rep_out_port))
                    .map(|&(rem_inp_node, rem_inp_port)| {
                        let (rem_inp_pred_node, rem_inp_pred_port) = host
                            .single_linked_output(rem_inp_node, rem_inp_port)
                            .unwrap();
                        (
                            HostPort(rem_inp_pred_node, rem_inp_pred_port),
                            HostPort(rem_out_node, rem_out_port),
                        )
                    })
            })
    }

    /// Get the incoming port at the output node of `self.replacement` that
    /// corresponds to the given host output port.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_host_output(
        &self,
        port: impl Into<HostPort<HostNode, IncomingPort>>,
    ) -> Option<ReplacementPort<IncomingPort>> {
        let HostPort(node, port) = port.into();
        let [_, rep_output] = self.get_replacement_io().expect("replacement is a DFG");
        self.nu_out
            .get(&(node, port))
            .map(|&rep_out_port| ReplacementPort(rep_output, rep_out_port))
    }

    /// Get the incoming port in `subgraph` that corresponds to the given
    /// replacement input port.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_replacement_input(
        &self,
        port: impl Into<ReplacementPort<IncomingPort>>,
    ) -> Option<HostPort<HostNode, IncomingPort>> {
        let ReplacementPort(node, port) = port.into();
        self.nu_inp.get(&(node, port)).copied().map(Into::into)
    }

    /// Get all edges that the replacement would add between `host` and
    /// `self.replacement`.
    ///
    /// This is equivalent to chaining the results of [`Self::incoming_boundary`],
    /// [`Self::outgoing_boundary`], and [`Self::host_to_host_boundary`].
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

impl<HostNode: HugrNode> VerifyPatch for SimpleReplacement<HostNode> {
    type Error = SimpleReplacementError;
    type Node = HostNode;

    fn verify(&self, h: &impl HugrView<Node = HostNode>) -> Result<(), SimpleReplacementError> {
        self.is_valid_rewrite(h)
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = HostNode> {
        let subcirc = self.subgraph.nodes().iter().copied();
        let out_neighs = self.nu_out.keys().map(|key| key.0);
        subcirc.chain(out_neighs)
    }
}

impl ApplyPatchHugrMut for SimpleReplacement<Node> {
    type Outcome = Vec<(Node, OpType)>;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(self, h: &mut impl HugrMut) -> Result<Self::Outcome, Self::Error> {
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

        // 2. Insert the replacement as a whole.
        let InsertionResult {
            new_root,
            node_map: index_map,
        } = h.insert_hugr(parent, replacement);

        // remove the Input and Output nodes from the replacement graph
        let replace_children = h.children(new_root).collect::<Vec<Node>>();
        for &io in &replace_children[..2] {
            h.remove_node(io);
        }
        // make all replacement top level children children of the parent
        for &child in &replace_children[2..] {
            h.set_parent(child, parent);
        }
        // remove the replacement root (which now has no children and no edges)
        h.remove_node(new_root);

        // 3. Insert all boundary edges.
        for (src, tgt) in boundary_edges {
            let (src_node, src_port) = src.map_replacement(&index_map);
            let (tgt_node, tgt_port) = tgt.map_replacement(&index_map);
            h.connect(src_node, src_port, tgt_node, tgt_port);
        }

        // 4. Remove all nodes in subgraph and edges between them.
        Ok(subgraph
            .nodes()
            .iter()
            .map(|&node| (node, h.remove_node(node)))
            .collect())
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

#[cfg(test)]
pub(in crate::hugr::patch) mod test {
    use itertools::Itertools;
    use rstest::{fixture, rstest};
    use std::collections::{HashMap, HashSet};

    use crate::builder::test::n_identity;
    use crate::builder::{
        endo_sig, inout_sig, BuildError, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::{bool_t, qb_t};
    use crate::extension::ExtensionSet;
    use crate::hugr::views::{HugrView, SiblingSubgraph};
    use crate::hugr::{ApplyPatch, Hugr, HugrMut};
    use crate::ops::dataflow::DataflowOpTrait;
    use crate::ops::handle::NodeHandle;
    use crate::ops::OpTag;
    use crate::ops::OpTrait;
    use crate::std_extensions::logic::test::and_op;
    use crate::std_extensions::logic::LogicOp;
    use crate::types::{Signature, Type};
    use crate::utils::test_quantum_extension::{cx_gate, h_gate, EXTENSION_ID};
    use crate::{IncomingPort, Node};

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
            let just_q: ExtensionSet = EXTENSION_ID.into();
            let mut func_builder = module_builder.define_function(
                "main",
                Signature::new_endo(vec![qb_t(), qb_t(), qb_t()])
                    .with_extension_delta(just_q.clone()),
            )?;

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
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]).with_prelude())?;
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
    /// This can be replaced with an empty hugr coping the input to both outputs.
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
    /// This can be replaced with a single NOT op, coping the input to the first output.
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
        use crate::hugr::patch::VerifyPatch;

        let mut h: Hugr = simple_hugr;
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr;
        // 3. Construct the input and output matchings
        // 3.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .nodes()
            .find(|node: &Node| *n.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (n_node_h0, n_node_h1) = n.input_neighbours(n_node_cx).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let n_port_0 = n.node_inputs(n_node_h0).next().unwrap();
        let n_port_1 = n.node_inputs(n_node_h1).next().unwrap();
        let (n_cx_out_0, n_cx_out_1) = n.node_outputs(n_node_cx).take(2).collect_tuple().unwrap();
        let n_port_2 = n.linked_inputs(n_node_cx, n_cx_out_0).next().unwrap().1;
        let n_port_3 = n.linked_inputs(n_node_cx, n_cx_out_1).next().unwrap().1;
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let h_h0_out = h.node_outputs(h_node_h0).next().unwrap();
        let h_h1_out = h.node_outputs(h_node_h1).next().unwrap();
        let (h_outp_node, h_port_2) = h.linked_inputs(h_node_h0, h_h0_out).next().unwrap();
        let h_port_3 = h.linked_inputs(h_node_h1, h_h1_out).next().unwrap().1;
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_h0, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h1, n_port_1), (h_node_cx, h_port_1));
        nu_out.insert((h_outp_node, h_port_2), n_port_2);
        nu_out.insert((h_outp_node, h_port_3), n_port_3);
        // 4. Define the replacement
        let r = SimpleReplacement {
            subgraph: SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            replacement: n,
            nu_inp,
            nu_out,
        };
        assert_eq!(
            HashSet::<_>::from_iter(r.invalidation_set()),
            HashSet::<_>::from_iter([h_node_cx, h_node_h0, h_node_h1, h_outp_node]),
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
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let s: Vec<Node> = vec![h_node_cx].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr2;
        // 3. Construct the input and output matchings
        // 3.1. Locate the Output and its predecessor H in n
        let n_node_output = n
            .nodes()
            .find(|node: &Node| n.get_optype(*node).tag() == OpTag::Output)
            .unwrap();
        let (_n_node_input, n_node_h) = n.input_neighbours(n_node_output).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let (n_port_0, n_port_1) = n
            .node_inputs(n_node_output)
            .take(2)
            .collect_tuple()
            .unwrap();
        let n_port_2 = n.node_inputs(n_node_h).next().unwrap();
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let h_port_2 = h.node_inputs(h_node_h0).next().unwrap();
        let h_port_3 = h.node_inputs(h_node_h1).next().unwrap();
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_output, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h, n_port_2), (h_node_cx, h_port_1));
        nu_out.insert((h_node_h0, h_port_2), n_port_0);
        nu_out.insert((h_node_h1, h_port_3), n_port_1);
        // 4. Define the replacement
        let r = SimpleReplacement {
            subgraph: SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            replacement: n,
            nu_inp,
            nu_out,
        };
        h.apply_rewrite(r).unwrap();
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
        let [input, output] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(wires).unwrap();
        let replacement = h.clone();
        let orig = h.clone();

        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();
        let inputs = h
            .node_outputs(input)
            .filter(|&p| {
                h.get_optype(input)
                    .as_input()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| {
                let link = h.linked_inputs(input, p).next().unwrap();
                (link, link)
            })
            .collect();
        let outputs = h
            .node_inputs(output)
            .filter(|&p| {
                h.get_optype(output)
                    .as_output()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| ((output, p), p))
            .collect();
        h.apply_rewrite(SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
            replacement,
            inputs,
            outputs,
        ))
        .unwrap();

        // They should be the same, up to node indices
        assert_eq!(h.edge_count(), orig.edge_count());
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
        let [input, _] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(outw).unwrap();

        let mut builder = DFGBuilder::new(inout_sig(two_bit, one_bit)).unwrap();
        let inw = builder.input_wires();
        let outw = builder.add_dataflow_op(and_op(), inw).unwrap().outputs();
        let [repl_input, repl_output] = builder.io();
        let repl = builder.finish_hugr_with_outputs(outw).unwrap();

        let orig = h.clone();

        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();

        let first_out_p = h.node_outputs(input).next().unwrap();
        let embedded_inputs = h.linked_inputs(input, first_out_p);
        let repl_inputs = repl
            .node_outputs(repl_input)
            .map(|p| repl.linked_inputs(repl_input, p).next().unwrap());
        let inputs = embedded_inputs.zip(repl_inputs).collect();

        let outputs = repl
            .node_inputs(repl_output)
            .filter(|&p| repl.signature(repl_output).unwrap().port_type(p).is_some())
            .map(|p| ((repl_output, p), p))
            .collect();

        h.apply_rewrite(SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
            repl,
            inputs,
            outputs,
        ))
        .unwrap();

        // Nothing changed
        assert_eq!(h.node_count(), orig.node_count());
    }

    /// Remove all the NOT gates in [`dfg_hugr_copy_bools`] by connecting the input
    /// directly to the outputs.
    ///
    /// https://github.com/CQCL/hugr/issues/1190
    #[rstest]
    fn test_copy_inputs(dfg_hugr_copy_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_copy_bools;
        let (input_not, output_not_0, output_not_1) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let replacement = {
            let b =
                DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            b.finish_hugr_with_outputs([w, w]).unwrap()
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0, output_not_1], &hugr)
                .unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
        // edges from nodes not in `removal` to nodes in `removal`).
        let nu_inp = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_output, IncomingPort::from(1)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
        // (input ports of the Output node of `replacement`).
        let nu_out = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let rewrite = SimpleReplacement {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        };
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.node_count(), 3);
    }

    /// Remove one of the NOT ops in [`dfg_hugr_half_not_bools`] by connecting the input
    /// directly to the output.
    ///
    /// https://github.com/CQCL/hugr/issues/1323
    #[rstest]
    fn test_half_nots(dfg_hugr_half_not_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_half_not_bools;
        let (input_not, output_not_0) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let (replacement, repl_not) = {
            let mut b =
                DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            let not = b.add_dataflow_op(LogicOp::Not, vec![w]).unwrap();
            let [w_not] = not.outputs_arr();
            (b.finish_hugr_with_outputs([w, w_not]).unwrap(), not.node())
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0], &hugr).unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
        // edges from nodes not in `removal` to nodes in `removal`).
        let nu_inp = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_not, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
        // (input ports of the Output node of `replacement`).
        let nu_out = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let rewrite = SimpleReplacement {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        };
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.node_count(), 4);
    }

    #[rstest]
    fn test_nested_replace(dfg_hugr2: Hugr) {
        // replace a node with a hugr with children

        let mut h = dfg_hugr2;
        let h_node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == h_gate().into())
            .unwrap();

        // build a nested identity dfg
        let mut nest_build = DFGBuilder::new(Signature::new_endo(qb_t())).unwrap();
        let [input] = nest_build.input_wires_arr();
        let inner_build = nest_build.dfg_builder_endo([(qb_t(), input)]).unwrap();
        let inner_dfg = n_identity(inner_build).unwrap();
        let inner_dfg_node = inner_dfg.node();
        let replacement = nest_build
            .finish_hugr_with_outputs([inner_dfg.out_wire(0)])
            .unwrap();
        let subgraph = SiblingSubgraph::try_from_nodes(vec![h_node], &h).unwrap();
        let nu_inp = vec![(
            (inner_dfg_node, IncomingPort::from(0)),
            (h_node, IncomingPort::from(0)),
        )]
        .into_iter()
        .collect();

        let nu_out = vec![(
            (h.get_io(h.root()).unwrap()[1], IncomingPort::from(1)),
            IncomingPort::from(0),
        )]
        .into_iter()
        .collect();

        let rewrite = SimpleReplacement::new(subgraph, replacement, nu_inp, nu_out);

        assert_eq!(h.node_count(), 4);

        rewrite.apply(&mut h).unwrap_or_else(|e| panic!("{e}"));
        h.validate().unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(h.node_count(), 6);
    }

    use crate::hugr::patch::replace::Replacement;
    fn to_replace(h: &impl HugrView<Node = Node>, s: SimpleReplacement) -> Replacement {
        use crate::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec};

        let mut replacement = s.replacement;
        let (in_, out) = replacement
            .children(replacement.root())
            .take(2)
            .collect_tuple()
            .unwrap();
        let mu_inp = s
            .nu_inp
            .iter()
            .map(|((tgt, tgt_port), (r_n, r_p))| {
                if *tgt == out {
                    unimplemented!()
                };
                let (src, src_port) = h.single_linked_output(*r_n, *r_p).unwrap();
                NewEdgeSpec {
                    src,
                    tgt: *tgt,
                    kind: NewEdgeKind::Value {
                        src_pos: src_port,
                        tgt_pos: *tgt_port,
                    },
                }
            })
            .collect();
        let mu_out = s
            .nu_out
            .iter()
            .map(|((tgt, tgt_port), out_port)| {
                let (src, src_port) = replacement.single_linked_output(out, *out_port).unwrap();
                if src == in_ {
                    unimplemented!()
                };
                NewEdgeSpec {
                    src,
                    tgt: *tgt,
                    kind: NewEdgeKind::Value {
                        src_pos: src_port,
                        tgt_pos: *tgt_port,
                    },
                }
            })
            .collect();
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
        h.apply_rewrite(rw).unwrap();
    }

    fn apply_replace(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_rewrite(to_replace(h, rw)).unwrap();
    }
}

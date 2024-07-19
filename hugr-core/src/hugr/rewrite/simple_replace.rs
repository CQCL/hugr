//! Implementation of the `SimpleReplace` operation.

use std::collections::{HashMap, HashSet};

use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrMut, HugrView, NodeMetadataMap, Rewrite};
use crate::ops::{OpTag, OpTrait, OpType};
use crate::{Hugr, IncomingPort, Node, OutgoingPort};
use thiserror::Error;

/// Specification of a simple replacement operation.
#[derive(Debug, Clone)]
pub struct SimpleReplacement {
    /// The subgraph of the hugr to be replaced.
    subgraph: SiblingSubgraph,
    /// A hugr with DFG root (consisting of replacement nodes).
    replacement: Hugr,
    /// A map from (target ports of edges from the Input node of `replacement`) to (target ports of
    /// edges from nodes not in `removal` to nodes in `removal`).
    nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)>,
    /// A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
    /// (input ports of the Output node of `replacement`).
    nu_out: HashMap<(Node, IncomingPort), IncomingPort>,
}

impl SimpleReplacement {
    /// Create a new [`SimpleReplacement`] specification.
    #[inline]
    pub fn new(
        subgraph: SiblingSubgraph,
        replacement: Hugr,
        nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)>,
        nu_out: HashMap<(Node, IncomingPort), IncomingPort>,
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

    /// Subgraph to be replaced.
    #[inline]
    pub fn subgraph(&self) -> &SiblingSubgraph {
        &self.subgraph
    }
}

impl Rewrite for SimpleReplacement {
    type Error = SimpleReplacementError;
    type ApplyResult = ();
    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, _h: &impl HugrView) -> Result<(), SimpleReplacementError> {
        unimplemented!()
    }

    fn apply(mut self, h: &mut impl HugrMut) -> Result<(), SimpleReplacementError> {
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
        // 3. Do the replacement.
        // 3.1. Add copies of all replacement nodes and edges to h. Exclude Input/Output nodes.
        // Create map from old NodeIndex (in self.replacement) to new NodeIndex (in self).
        let mut index_map: HashMap<Node, Node> = HashMap::new();
        let replacement_nodes = self
            .replacement
            .children(self.replacement.root())
            .collect::<Vec<Node>>();
        // slice of nodes omitting Input and Output:
        let replacement_inner_nodes = &replacement_nodes[2..];
        let self_output_node = h.children(parent).nth(1).unwrap();
        let replacement_output_node = *replacement_nodes.get(1).unwrap();
        for &node in replacement_inner_nodes {
            // Add the nodes.
            let op: &OpType = self.replacement.get_optype(node);
            let new_node = h.add_node_after(self_output_node, op.clone());
            index_map.insert(node, new_node);

            // Move the metadata
            let meta: Option<NodeMetadataMap> = self.replacement.take_node_metadata(node);
            h.overwrite_node_metadata(new_node, meta);
        }
        // Add edges between all newly added nodes matching those in replacement.
        for &node in replacement_inner_nodes {
            let new_node = index_map.get(&node).unwrap();
            for outport in self.replacement.node_outputs(node) {
                for target in self.replacement.linked_inputs(node, outport) {
                    if self.replacement.get_optype(target.0).tag() != OpTag::Output {
                        let new_target = index_map.get(&target.0).unwrap();
                        h.connect(*new_node, outport, *new_target, target.1);
                    }
                }
            }
        }

        // Now we proceed to connect the edges between the newly inserted
        // replacement and the rest of the graph.
        //
        // We delay creating these connections to avoid them getting mixed with
        // the pre-existing ones in the following logic.
        //
        // Existing connections to the removed subgraph will be automatically
        // removed when the nodes are removed.
        let mut connect: HashSet<(Node, OutgoingPort, Node, IncomingPort)> = HashSet::new();

        // 3.2. For each p = self.nu_inp[q] such that q is not an Output port, add an edge from the
        // predecessor of p to (the new copy of) q.
        for ((rep_inp_node, rep_inp_port), (rem_inp_node, rem_inp_port)) in &self.nu_inp {
            if self.replacement.get_optype(*rep_inp_node).tag() != OpTag::Output {
                // add edge from predecessor of (s_inp_node, s_inp_port) to (new_inp_node, n_inp_port)
                let (rem_inp_pred_node, rem_inp_pred_port) = h
                    .single_linked_output(*rem_inp_node, *rem_inp_port)
                    .unwrap();
                let new_inp_node = index_map.get(rep_inp_node).unwrap();
                connect.insert((
                    rem_inp_pred_node,
                    rem_inp_pred_port,
                    *new_inp_node,
                    *rep_inp_port,
                ));
            }
        }
        // 3.3. For each q = self.nu_out[p] such that the predecessor of q is not an Input port, add an
        // edge from (the new copy of) the predecessor of q to p.
        for ((rem_out_node, rem_out_port), rep_out_port) in &self.nu_out {
            let (rep_out_pred_node, rep_out_pred_port) = self
                .replacement
                .single_linked_output(replacement_output_node, *rep_out_port)
                .unwrap();
            if self.replacement.get_optype(rep_out_pred_node).tag() != OpTag::Input {
                let new_out_node = index_map.get(&rep_out_pred_node).unwrap();
                connect.insert((
                    *new_out_node,
                    rep_out_pred_port,
                    *rem_out_node,
                    *rem_out_port,
                ));
            }
        }
        // 3.4. For each q = self.nu_out[p1], p0 = self.nu_inp[q], add an edge from the predecessor of p0
        // to p1.
        //
        // i.e. the replacement graph has direct edges between the input and output nodes.
        for ((rem_out_node, rem_out_port), &rep_out_port) in &self.nu_out {
            let rem_inp_nodeport = self.nu_inp.get(&(replacement_output_node, rep_out_port));
            if let Some((rem_inp_node, rem_inp_port)) = rem_inp_nodeport {
                // add edge from predecessor of (rem_inp_node, rem_inp_port) to (rem_out_node, rem_out_port):
                let (rem_inp_pred_node, rem_inp_pred_port) = h
                    .single_linked_output(*rem_inp_node, *rem_inp_port)
                    .unwrap();
                // Delay connecting the nodes until after processing all nu_out
                // entries.
                //
                // Otherwise, we might disconnect other wires in `rem_inp_node`
                // that are needed for the following iterations.
                connect.insert((
                    rem_inp_pred_node,
                    rem_inp_pred_port,
                    *rem_out_node,
                    *rem_out_port,
                ));
            }
        }
        connect
            .into_iter()
            .for_each(|(src_node, src_port, tgt_node, tgt_port)| {
                h.connect(src_node, src_port, tgt_node, tgt_port);
            });

        // 3.5. Remove all nodes in self.removal and edges between them.
        for &node in self.subgraph.nodes() {
            h.remove_node(node);
        }
        Ok(())
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        let subcirc = self.subgraph.nodes().iter().copied();
        let out_neighs = self.nu_out.keys().map(|key| key.0);
        subcirc.chain(out_neighs)
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
}

#[cfg(test)]
pub(in crate::hugr::rewrite) mod test {
    use itertools::Itertools;
    use rstest::{fixture, rstest};
    use std::collections::{HashMap, HashSet};

    use crate::builder::{
        endo_sig, inout_sig, BuildError, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::BOOL_T;
    use crate::extension::{ExtensionSet, EMPTY_REG, PRELUDE_REGISTRY};
    use crate::hugr::views::{HugrView, SiblingSubgraph};
    use crate::hugr::{Hugr, HugrMut, Rewrite};
    use crate::ops::dataflow::DataflowOpTrait;
    use crate::ops::handle::NodeHandle;
    use crate::ops::OpTag;
    use crate::ops::OpTrait;
    use crate::std_extensions::logic::test::and_op;
    use crate::std_extensions::logic::NotOp;
    use crate::type_row;
    use crate::types::{Signature, Type};
    use crate::utils::test_quantum_extension::{cx_gate, h_gate, EXTENSION_ID};
    use crate::{IncomingPort, Node};

    use super::SimpleReplacement;

    const QB: Type = crate::extension::prelude::QB_T;

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
                Signature::new_endo(type_row![QB, QB, QB]).with_extension_delta(just_q.clone()),
            )?;

            let [qb0, qb1, qb2] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb2])?;

            let mut inner_builder = func_builder.dfg_builder_endo([(QB, qb0), (QB, qb1)])?;
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
        Ok(module_builder.finish_prelude_hugr()?)
    }

    #[fixture]
    pub(in crate::hugr::rewrite) fn simple_hugr() -> Hugr {
        make_hugr().unwrap()
    }
    /// Creates a hugr with a DFG root like the following:
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn make_dfg_hugr() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(type_row![QB, QB]))?;
        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire0])?;
        let wire3 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire45 =
            dfg_builder.add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
        dfg_builder.finish_prelude_hugr_with_outputs(wire45.outputs())
    }

    #[fixture]
    pub(in crate::hugr::rewrite) fn dfg_hugr() -> Hugr {
        make_dfg_hugr().unwrap()
    }

    /// Creates a hugr with a DFG root like the following:
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn make_dfg_hugr2() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(type_row![QB, QB]))?;

        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire2out = wire2.outputs().exactly_one().unwrap();
        let wireoutvec = vec![wire0, wire2out];
        dfg_builder.finish_prelude_hugr_with_outputs(wireoutvec)
    }

    #[fixture]
    pub(in crate::hugr::rewrite) fn dfg_hugr2() -> Hugr {
        make_dfg_hugr2().unwrap()
    }

    /// A hugr with a DFG root mapping BOOL_T to (BOOL_T, BOOL_T)
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
    pub(in crate::hugr::rewrite) fn dfg_hugr_copy_bools() -> (Hugr, Vec<Node>) {
        fn build() -> Result<(Hugr, Vec<Node>), BuildError> {
            let mut dfg_builder =
                DFGBuilder::new(inout_sig(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]))?;
            let [b] = dfg_builder.input_wires_arr();

            let not_inp = dfg_builder.add_dataflow_op(NotOp, vec![b])?;
            let [b] = not_inp.outputs_arr();

            let not_0 = dfg_builder.add_dataflow_op(NotOp, vec![b])?;
            let [b0] = not_0.outputs_arr();
            let not_1 = dfg_builder.add_dataflow_op(NotOp, vec![b])?;
            let [b1] = not_1.outputs_arr();

            Ok((
                dfg_builder.finish_prelude_hugr_with_outputs([b0, b1])?,
                vec![not_inp.node(), not_0.node(), not_1.node()],
            ))
        }

        build().unwrap()
    }

    /// A hugr with a DFG root mapping BOOL_T to (BOOL_T, BOOL_T)
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
    pub(in crate::hugr::rewrite) fn dfg_hugr_half_not_bools() -> (Hugr, Vec<Node>) {
        fn build() -> Result<(Hugr, Vec<Node>), BuildError> {
            let mut dfg_builder =
                DFGBuilder::new(inout_sig(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]))?;
            let [b] = dfg_builder.input_wires_arr();

            let not_inp = dfg_builder.add_dataflow_op(NotOp, vec![b])?;
            let [b] = not_inp.outputs_arr();

            let not_0 = dfg_builder.add_dataflow_op(NotOp, vec![b])?;
            let [b0] = not_0.outputs_arr();
            let b1 = b;

            Ok((
                dfg_builder.finish_prelude_hugr_with_outputs([b0, b1])?,
                vec![not_inp.node(), not_0.node()],
            ))
        }

        build().unwrap()
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
        assert_eq!(h.update_validate(&PRELUDE_REGISTRY), Ok(()));
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
        assert_eq!(h.update_validate(&PRELUDE_REGISTRY), Ok(()));
    }

    #[test]
    fn test_replace_cx_cross() {
        let q_row: Vec<Type> = vec![QB, QB];
        let mut builder = DFGBuilder::new(endo_sig(q_row)).unwrap();
        let mut circ = builder.as_circuit(builder.input_wires());
        circ.append(cx_gate(), [0, 1]).unwrap();
        circ.append(cx_gate(), [1, 0]).unwrap();
        let wires = circ.finish();
        let [input, output] = builder.io();
        let mut h = builder.finish_prelude_hugr_with_outputs(wires).unwrap();
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
        let one_bit = type_row![BOOL_T];
        let two_bit = type_row![BOOL_T, BOOL_T];

        let mut builder = DFGBuilder::new(endo_sig(one_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw = builder
            .add_dataflow_op(and_op(), [inw, inw])
            .unwrap()
            .outputs();
        let [input, _] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(outw, &EMPTY_REG).unwrap();

        let mut builder = DFGBuilder::new(inout_sig(two_bit, one_bit)).unwrap();
        let inw = builder.input_wires();
        let outw = builder.add_dataflow_op(and_op(), inw).unwrap().outputs();
        let [repl_input, repl_output] = builder.io();
        let repl = builder.finish_hugr_with_outputs(outw, &EMPTY_REG).unwrap();

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
    fn test_copy_inputs(
        dfg_hugr_copy_bools: (Hugr, Vec<Node>),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut hugr, nodes) = dfg_hugr_copy_bools;
        let (input_not, output_not_0, output_not_1) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let replacement = {
            let b = DFGBuilder::new(Signature::new(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]))?;
            let [w] = b.input_wires_arr();
            b.finish_prelude_hugr_with_outputs([w, w])?
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0, output_not_1], &hugr)?;
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

        assert_eq!(hugr.update_validate(&PRELUDE_REGISTRY), Ok(()));
        assert_eq!(hugr.node_count(), 3);

        Ok(())
    }

    /// Remove one of the NOT ops in [`dfg_hugr_half_not_bools`] by connecting the input
    /// directly to the output.
    ///
    /// https://github.com/CQCL/hugr/issues/1323
    #[rstest]
    fn test_half_nots(
        dfg_hugr_half_not_bools: (Hugr, Vec<Node>),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut hugr, nodes) = dfg_hugr_half_not_bools;
        let (input_not, output_not_0) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let (replacement, repl_not) = {
            let mut b = DFGBuilder::new(inout_sig(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]))?;
            let [w] = b.input_wires_arr();
            let not = b.add_dataflow_op(NotOp, vec![w])?;
            let [w_not] = not.outputs_arr();
            (b.finish_prelude_hugr_with_outputs([w, w_not])?, not.node())
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph = SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0], &hugr)?;
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

        println!("Graph:\n{}", hugr.mermaid_string());
        println!("Subgraph: {subgraph:?}");
        println!("Replacement:\n{}", replacement.mermaid_string());
        println!("nu_inp: {nu_inp:?}");
        println!("nu_out: {nu_out:?}");

        let rewrite = SimpleReplacement {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        };
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.update_validate(&PRELUDE_REGISTRY), Ok(()));
        assert_eq!(hugr.node_count(), 4);

        Ok(())
    }

    use crate::hugr::rewrite::replace::Replacement;
    fn to_replace(h: &impl HugrView, s: SimpleReplacement) -> Replacement {
        use crate::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec};

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
        h.apply_rewrite(rw).unwrap()
    }

    fn apply_replace(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_rewrite(to_replace(h, rw)).unwrap();
    }
}

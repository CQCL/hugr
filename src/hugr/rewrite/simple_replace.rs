//! Implementation of the `SimpleReplace` operation.

use std::collections::HashMap;

use itertools::Itertools;

use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrMut, HugrView, NodeMetadata};
use crate::{
    hugr::{Node, Rewrite},
    ops::{OpTag, OpTrait, OpType},
    Hugr, Port,
};
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
    nu_inp: HashMap<(Node, Port), (Node, Port)>,
    /// A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
    /// (input ports of the Output node of `replacement`).
    nu_out: HashMap<(Node, Port), Port>,
}

impl SimpleReplacement {
    /// Create a new [`SimpleReplacement`] specification.
    pub fn new(
        subgraph: SiblingSubgraph,
        replacement: Hugr,
        nu_inp: HashMap<(Node, Port), (Node, Port)>,
        nu_out: HashMap<(Node, Port), Port>,
    ) -> Self {
        Self {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        }
    }

    /// The replacement hugr.
    pub fn replacement(&self) -> &Hugr {
        &self.replacement
    }

    /// Subgraph to be replaced.
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

    fn apply(self, h: &mut impl HugrMut) -> Result<(), SimpleReplacementError> {
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
            let new_node = h.add_op_after(self_output_node, op.clone()).unwrap();
            index_map.insert(node, new_node);

            // Move the metadata
            let meta: &NodeMetadata = self.replacement.get_metadata(node);
            h.set_metadata(node, meta.clone());
        }
        // Add edges between all newly added nodes matching those in replacement.
        // TODO This will probably change when implicit copies are implemented.
        for &node in replacement_inner_nodes {
            let new_node = index_map.get(&node).unwrap();
            for outport in self.replacement.node_outputs(node) {
                for target in self.replacement.linked_ports(node, outport) {
                    if self.replacement.get_optype(target.0).tag() != OpTag::Output {
                        let new_target = index_map.get(&target.0).unwrap();
                        h.connect(*new_node, outport.index(), *new_target, target.1.index())
                            .unwrap();
                    }
                }
            }
        }
        // 3.2. For each p = self.nu_inp[q] such that q is not an Output port, add an edge from the
        // predecessor of p to (the new copy of) q.
        for ((rep_inp_node, rep_inp_port), (rem_inp_node, rem_inp_port)) in &self.nu_inp {
            if self.replacement.get_optype(*rep_inp_node).tag() != OpTag::Output {
                // add edge from predecessor of (s_inp_node, s_inp_port) to (new_inp_node, n_inp_port)
                let (rem_inp_pred_node, rem_inp_pred_port) = h
                    .linked_ports(*rem_inp_node, *rem_inp_port)
                    .exactly_one()
                    .ok() // PortLinks does not implement Debug
                    .unwrap();
                h.disconnect(*rem_inp_node, *rem_inp_port).unwrap();
                let new_inp_node = index_map.get(rep_inp_node).unwrap();
                h.connect(
                    rem_inp_pred_node,
                    rem_inp_pred_port.index(),
                    *new_inp_node,
                    rep_inp_port.offset.index(),
                )
                .unwrap();
            }
        }
        // 3.3. For each q = self.nu_out[p] such that the predecessor of q is not an Input port, add an
        // edge from (the new copy of) the predecessor of q to p.
        for ((rem_out_node, rem_out_port), rep_out_port) in &self.nu_out {
            let (rep_out_pred_node, rep_out_pred_port) = self
                .replacement
                .linked_ports(replacement_output_node, *rep_out_port)
                .exactly_one()
                .unwrap();
            if self.replacement.get_optype(rep_out_pred_node).tag() != OpTag::Input {
                let new_out_node = index_map.get(&rep_out_pred_node).unwrap();
                h.disconnect(*rem_out_node, *rem_out_port).unwrap();
                h.connect(
                    *new_out_node,
                    rep_out_pred_port.index(),
                    *rem_out_node,
                    rem_out_port.index(),
                )
                .unwrap();
            }
        }
        // 3.4. For each q = self.nu_out[p1], p0 = self.nu_inp[q], add an edge from the predecessor of p0
        // to p1.
        for ((rem_out_node, rem_out_port), &rep_out_port) in &self.nu_out {
            let rem_inp_nodeport = self.nu_inp.get(&(replacement_output_node, rep_out_port));
            if let Some((rem_inp_node, rem_inp_port)) = rem_inp_nodeport {
                // add edge from predecessor of (rem_inp_node, rem_inp_port) to (rem_out_node, rem_out_port):
                let (rem_inp_pred_node, rem_inp_pred_port) = h
                    .linked_ports(*rem_inp_node, *rem_inp_port)
                    .exactly_one()
                    .ok() // PortLinks does not implement Debug
                    .unwrap();
                h.disconnect(*rem_inp_node, *rem_inp_port).unwrap();
                h.disconnect(*rem_out_node, *rem_out_port).unwrap();
                h.connect(
                    rem_inp_pred_node,
                    rem_inp_pred_port.index(),
                    *rem_out_node,
                    rem_out_port.index(),
                )
                .unwrap();
            }
        }
        // 3.5. Remove all nodes in self.removal and edges between them.
        for &node in self.subgraph.nodes() {
            h.remove_node(node).unwrap();
        }
        Ok(())
    }
}

/// Error from a [`SimpleReplacement`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
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
    use portgraph::Direction;
    use rstest::{fixture, rstest};
    use std::collections::HashMap;

    use crate::builder::{
        BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::BOOL_T;
    use crate::extension::{EMPTY_REG, PRELUDE_REGISTRY};
    use crate::hugr::views::{HugrView, SiblingSubgraph};
    use crate::hugr::{Hugr, HugrMut, Node};
    use crate::ops::OpTag;
    use crate::ops::{OpTrait, OpType};
    use crate::std_extensions::logic::test::and_op;
    use crate::std_extensions::quantum::test::{cx_gate, h_gate};
    use crate::types::{FunctionType, Type};
    use crate::{type_row, Port};

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
            let mut func_builder = module_builder.define_function(
                "main",
                FunctionType::new(type_row![QB, QB, QB], type_row![QB, QB, QB]).pure(),
            )?;

            let [qb0, qb1, qb2] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb2])?;

            let mut inner_builder = func_builder.dfg_builder(
                FunctionType::new(type_row![QB, QB], type_row![QB, QB]),
                None,
                [qb0, qb1],
            )?;
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
        let mut dfg_builder =
            DFGBuilder::new(FunctionType::new(type_row![QB, QB], type_row![QB, QB]))?;
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
        let mut dfg_builder =
            DFGBuilder::new(FunctionType::new(type_row![QB, QB], type_row![QB, QB]))?;
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
    fn test_simple_replacement(simple_hugr: Hugr, dfg_hugr: Hugr) {
        let mut h: Hugr = simple_hugr;
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == OpType::LeafOp(cx_gate()))
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr;
        // 3. Construct the input and output matchings
        // 3.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .nodes()
            .find(|node: &Node| *n.get_optype(*node) == OpType::LeafOp(cx_gate()))
            .unwrap();
        let (n_node_h0, n_node_h1) = n.input_neighbours(n_node_cx).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let n_port_0 = n.node_ports(n_node_h0, Direction::Incoming).next().unwrap();
        let n_port_1 = n.node_ports(n_node_h1, Direction::Incoming).next().unwrap();
        let (n_cx_out_0, n_cx_out_1) = n
            .node_ports(n_node_cx, Direction::Outgoing)
            .take(2)
            .collect_tuple()
            .unwrap();
        let n_port_2 = n.linked_ports(n_node_cx, n_cx_out_0).next().unwrap().1;
        let n_port_3 = n.linked_ports(n_node_cx, n_cx_out_1).next().unwrap().1;
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h
            .node_ports(h_node_cx, Direction::Incoming)
            .take(2)
            .collect_tuple()
            .unwrap();
        let h_h0_out = h.node_ports(h_node_h0, Direction::Outgoing).next().unwrap();
        let h_h1_out = h.node_ports(h_node_h1, Direction::Outgoing).next().unwrap();
        let (h_outp_node, h_port_2) = h.linked_ports(h_node_h0, h_h0_out).next().unwrap();
        let h_port_3 = h.linked_ports(h_node_h1, h_h1_out).next().unwrap().1;
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, Port), (Node, Port)> = HashMap::new();
        let mut nu_out: HashMap<(Node, Port), Port> = HashMap::new();
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
        h.apply_rewrite(r).unwrap();
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├──■──
        // ├───┤├───┤┌─┴─┐
        // ┤ H ├┤ H ├┤ X ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(&PRELUDE_REGISTRY), Ok(()));
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
            .find(|node: &Node| *h.get_optype(*node) == OpType::LeafOp(cx_gate()))
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
        let h_port_2 = h.node_ports(h_node_h0, Direction::Incoming).next().unwrap();
        let h_port_3 = h.node_ports(h_node_h1, Direction::Incoming).next().unwrap();
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, Port), (Node, Port)> = HashMap::new();
        let mut nu_out: HashMap<(Node, Port), Port> = HashMap::new();
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
        assert_eq!(h.validate(&PRELUDE_REGISTRY), Ok(()));
    }

    #[test]
    fn test_replace_cx_cross() {
        let q_row: Vec<Type> = vec![QB, QB];
        let mut builder = DFGBuilder::new(FunctionType::new(q_row.clone(), q_row)).unwrap();
        let mut circ = builder.as_circuit(builder.input_wires().collect());
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
            .filter(|&p| h.get_optype(input).signature().get(p).is_some())
            .map(|p| {
                let link = h.linked_ports(input, p).next().unwrap();
                (link, link)
            })
            .collect();
        let outputs = h
            .node_inputs(output)
            .filter(|&p| h.get_optype(output).signature().get(p).is_some())
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

        let mut builder =
            DFGBuilder::new(FunctionType::new(one_bit.clone(), one_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw = builder
            .add_dataflow_op(and_op(), [inw, inw])
            .unwrap()
            .outputs();
        let [input, _] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(outw, &EMPTY_REG).unwrap();

        let mut builder = DFGBuilder::new(FunctionType::new(two_bit, one_bit)).unwrap();
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
        let embedded_inputs = h.linked_ports(input, first_out_p);
        let repl_inputs = repl
            .node_outputs(repl_input)
            .map(|p| repl.linked_ports(repl_input, p).next().unwrap());
        let inputs = embedded_inputs.zip(repl_inputs).collect();

        let outputs = repl
            .node_inputs(repl_output)
            .filter(|&p| repl.get_optype(repl_output).signature().get(p).is_some())
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
}

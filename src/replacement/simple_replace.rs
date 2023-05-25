use std::collections::{HashMap, HashSet};

use crate::{hugr::Node, Hugr, Port};
use thiserror::Error;

/// Specification of a simple replacement operation.
#[derive(Debug, Clone)]
pub struct SimpleReplacement {
    /// The common DFG parent of all nodes to be replaced.
    pub p: Node,
    /// The set of nodes to remove (a convex set of leaf children of `p`).
    pub s: HashSet<Node>,
    /// A hugr with DFG root (consisting of replacement nodes).
    pub n: Hugr,
    /// A map from (target ports of edges from the Input node of n) to (target ports of edges from
    /// non-s nodes to s nodes).
    pub nu_inp: HashMap<(Node, Port), (Node, Port)>,
    /// A map from (target ports of edges from s nodes to non-s nodes) to (input ports of the Output
    /// node of n).
    pub nu_out: HashMap<(Node, Port), Port>,
}

impl SimpleReplacement {
    /// Create a new [`SimpleReplacement`] specification.
    pub fn new(
        p: Node,
        s: HashSet<Node>,
        n: Hugr,
        nu_inp: HashMap<(Node, Port), (Node, Port)>,
        nu_out: HashMap<(Node, Port), Port>,
    ) -> Self {
        Self {
            p,
            s,
            n,
            nu_inp,
            nu_out,
        }
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
mod test {
    use std::collections::{HashMap, HashSet};

    use itertools::Itertools;
    use portgraph::Direction;

    use crate::builder::{BuildError, Container, Dataflow, ModuleBuilder};
    use crate::hugr::view::HugrView;
    use crate::hugr::{Hugr, Node};
    use crate::ops::tag::OpTag;
    use crate::ops::{DataflowOp, LeafOp, OpType};
    use crate::types::{LinearType, Signature, SimpleType};
    use crate::{type_row, Port};

    use super::SimpleReplacement;

    const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

    /// Creates a hugr with a DFG.
    fn make_hugr() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let _f_id = {
            let mut func_builder = module_builder.declare_and_def(
                "main",
                Signature::new_df(type_row![QB, QB, QB], type_row![QB, QB, QB]),
            )?;

            let [qb0, qb1, qb2] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(
                OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                vec![qb2],
            )?;

            let mut inner_builder =
                func_builder.dfg_builder(vec![(QB, qb0), (QB, qb1)], type_row![QB, QB])?;
            let inner_graph = {
                let [wire0, wire1] = inner_builder.input_wires_arr();
                let wire2 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire0],
                )?;
                let wire3 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire1],
                )?;
                let wire45 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::CX }),
                    wire2.outputs().chain(wire3.outputs()),
                )?;
                let [wire4, wire5] = wire45.outputs_arr();
                let wire6 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire4],
                )?;
                let wire7 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire5],
                )?;
                inner_builder.finish_with_outputs(wire6.outputs().chain(wire7.outputs()))
            }?;

            func_builder.finish_with_outputs(inner_graph.outputs().chain(q_out.outputs()))?
        };
        module_builder.finish()
    }

    /// Creates a hugr with a DFG with which to replace a subgraph.
    fn make_dfg_hugr() -> Result<Hugr, BuildError> {
        // TODO This will change when we have DFG-rooted HUGRs implemented. For now, this is a
        // module HUGR with one DFG node inside it.
        let mut module_builder = ModuleBuilder::new();
        let _f_id = {
            let mut func_builder = module_builder.declare_and_def(
                "main",
                Signature::new_df(type_row![QB, QB], type_row![QB, QB]),
            )?;

            let [qb0, qb1] = func_builder.input_wires_arr();

            let mut inner_builder =
                func_builder.dfg_builder(vec![(QB, qb0), (QB, qb1)], type_row![QB, QB])?;
            let inner_graph = {
                let [wire0, wire1] = inner_builder.input_wires_arr();
                let wire2 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire0],
                )?;
                let wire3 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![wire1],
                )?;
                let wire45 = inner_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::CX }),
                    wire2.outputs().chain(wire3.outputs()),
                )?;
                inner_builder.finish_with_outputs(wire45.outputs())
            }?;

            func_builder.finish_with_outputs(inner_graph.outputs())?
        };
        module_builder.finish()
    }

    #[test]
    fn test_simple_replacement() {
        let mut h: Hugr = make_hugr().ok().unwrap();
        // crate::utils::test::viz_dotstr(&h.dot_string());
        // 1. Find the DFG node for the inner circuit
        let p: Node = h
            .nodes()
            .find(|node: &Node| h.get_optype(*node).tag() == OpTag::Dfg)
            .unwrap();
        // 2. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| {
                *h.get_optype(*node) == OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::CX })
            })
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: HashSet<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 3. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = make_dfg_hugr().ok().unwrap();
        // crate::utils::test::viz_dotstr(&n.dot_string());
        // 4. Construct the input and output matchings
        // 4.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .nodes()
            .find(|node: &Node| {
                *n.get_optype(*node) == OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::CX })
            })
            .unwrap();
        let (n_node_h0, n_node_h1) = n.input_neighbours(n_node_cx).collect_tuple().unwrap();
        // 4.2. Locate the ports we need to specify as "glue" in n
        let n_port_0 = n.node_ports(n_node_h0, Direction::Incoming).next().unwrap();
        let n_port_1 = n.node_ports(n_node_h1, Direction::Incoming).next().unwrap();
        let (n_cx_out_0, n_cx_out_1) = n
            .node_ports(n_node_cx, Direction::Outgoing)
            .collect_tuple()
            .unwrap();
        let n_port_2 = n.linked_port(n_node_cx, n_cx_out_0).unwrap().1;
        let n_port_3 = n.linked_port(n_node_cx, n_cx_out_1).unwrap().1;
        // 4.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h
            .node_ports(h_node_cx, Direction::Incoming)
            .collect_tuple()
            .unwrap();
        let h_h0_out = h.node_ports(h_node_h0, Direction::Outgoing).next().unwrap();
        let h_h1_out = h.node_ports(h_node_h1, Direction::Outgoing).next().unwrap();
        let (h_outp_node, h_port_2) = h.linked_port(h_node_h0, h_h0_out).unwrap();
        let h_port_3 = h.linked_port(h_node_h1, h_h1_out).unwrap().1;
        // 4.4. Construct the maps
        let mut nu_inp: HashMap<(Node, Port), (Node, Port)> = HashMap::new();
        let mut nu_out: HashMap<(Node, Port), Port> = HashMap::new();
        nu_inp.insert((n_node_h0, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h1, n_port_1), (h_node_cx, h_port_1));
        nu_out.insert((h_outp_node, h_port_2), n_port_2);
        nu_out.insert((h_outp_node, h_port_3), n_port_3);
        // 5. Define the replacement
        let r = SimpleReplacement {
            p,
            s,
            n,
            nu_inp,
            nu_out,
        };
        h.apply_simple_replacement(r).ok();
        // crate::utils::test::viz_dotstr(&h.dot_string());
        assert_eq!(h.validate(), Ok(()));
    }
}

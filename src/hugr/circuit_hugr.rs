//! A simple Hugr for circuit-like computations
use std::collections::HashMap;
use std::iter;

use bitvec::bitvec;
use itertools::Itertools;
use portgraph::algorithms::toposort;
use portgraph::{Direction, NodeIndex, PortGraph, UnmanagedDenseMap};
use portmatching::pattern::InvalidPattern;

use crate::convex::ConvexChecker;
use crate::ops::dataflow::IOTrait;
use crate::ops::{Const, ConstValue, Input, Output, DFG};
use crate::types::Signature;
use crate::{
    ops::{tag::OpTag, LeafOp, OpTrait, OpType},
    pattern::HugrPattern,
    Port, SimpleReplacement,
};

use bitvec::prelude::BitVec;

use super::{Hugr, HugrMut, HugrView, Node};

/// A simple DFG rooted Hugr with a single layer of hierarchy
///
/// Everything is a sibling graph. The kind of graph that we
/// can use SimpleReplacement with.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CircuitHugr(Hugr);

impl CircuitHugr {
    /// Create a new CircuitHugr from a Hugr
    pub fn new(hugr: Hugr) -> Self {
        // Root must be DFG
        if hugr.get_optype(hugr.root()).tag() != OpTag::Dfg {
            panic!("Root must be DFG");
        }
        // All other nodes must be children of root
        for node in hugr.nodes() {
            if node != hugr.root() && hugr.get_parent(node) != Some(hugr.root()) {
                panic!("All nodes must be children of root");
            }
        }
        // There must be at least an input and output node as children of DFG
        if hugr.children(hugr.root()).count() < 2 {
            panic!("There must be at least an input and output node");
        }
        Self(hugr)
    }

    /// The underlying Hugr
    pub fn hugr(&self) -> &Hugr {
        &self.0
    }

    /// The underlying Hugr
    pub fn hugr_mut(&mut self) -> &mut Hugr {
        &mut self.0
    }

    fn graph(&self) -> &PortGraph {
        self.hugr().graph.as_portgraph()
    }

    /// The input node of the circuit
    pub fn input_node(&self) -> Node {
        let root = self.hugr().root();
        self.hugr().children(root).next().expect("No input node")
    }

    /// The output node of the circuit
    pub fn output_node(&self) -> Node {
        let root = self.hugr().root();
        self.hugr()
            .children(root)
            .skip(1)
            .next()
            .expect("No input node")
    }

    /// All the outgoing wires of the input node
    pub fn input_ports(&self) -> impl Iterator<Item = Port> + '_ {
        let input = self.input_node();
        self.hugr()
            .node_outputs(input)
            .filter(move |&p| self.hugr().get_optype(input).signature().get(p).is_some())
    }

    /// All the incoming wires of the output node
    pub fn output_ports(&self) -> impl Iterator<Item = Port> + '_ {
        let output = self.output_node();
        self.hugr()
            .node_inputs(output)
            .filter(move |&p| self.hugr().get_optype(output).signature().get(p).is_some())
    }

    /// The number of nodes in the circuit
    pub fn node_count(&self) -> usize {
        self.hugr().node_count() - 1
    }

    /// The number of nodes in the circuit
    pub fn leaves(&self) -> impl Iterator<Item = OpType> + '_ {
        self.hugr()
            .nodes()
            .map(|n| self.hugr().get_optype(n).clone())
            .filter(|op| op.tag() == OpTag::Leaf)
    }

    /// A convexity checker for the circuit
    pub fn convex_checker(&self, roots: impl IntoIterator<Item = Node>) -> ConvexChecker<'_> {
        ConvexChecker::new(roots.into_iter().map(|n| n.index), self.graph())
    }

    /// Wires in circuit with no gates
    pub fn blank_wires(&self) -> BitVec {
        let mut blanks = bitvec![0; self.input_ports().count()];
        let input = self.input_node();
        for (i, p) in self.input_ports().enumerate() {
            let op = self.hugr().get_optype(input);
            if op.signature().get(p).is_none() {
                continue;
            }
            let linked_nodes = self
                .hugr()
                .linked_ports(input, p)
                .map(|(n, _)| n)
                .collect_vec();
            if linked_nodes.is_empty() || linked_nodes.iter().all(|n| n == &self.output_node()) {
                blanks.set(i, true);
            }
        }
        blanks
    }

    /// Remove wires at input indices `remove`.
    pub fn remove_wires(&mut self, remove: &BitVec) {
        let mut remove_output = bitvec![0; self.output_ports().count()];
        if remove.is_empty() {
            return;
        }
        let old_input = self.input_node();
        let old_output = self.output_node();

        // Create new input
        let new_input_sig = {
            let old_sig = &self.hugr().get_optype(old_input).signature();
            let mut new_out = Vec::new();
            for i in 0..old_sig.output_count() {
                if remove[i] {
                    let output = self
                        .hugr()
                        .linked_ports(old_input, Port::new_outgoing(i))
                        .next();
                    if let Some((n, p)) = output {
                        assert_eq!(n, self.output_node());
                        remove_output.set(p.index(), true);
                    }
                    continue;
                } else {
                    let t = old_sig.get_df(Port::new_outgoing(i)).unwrap();
                    new_out.push(t.clone());
                }
            }
            new_out
        };
        let new_input = self.hugr_mut().add_op(Input::new(new_input_sig.clone()));

        // Wire up new input
        let mut next_input = 0;
        for (i, p) in self.input_ports().enumerate().collect_vec() {
            let Some(link) = self.hugr().linked_ports(old_input, p).next() else {
                next_input += !remove[i] as usize;
                continue
            };
            self.hugr_mut().disconnect(link.0, link.1).unwrap();
            self.hugr_mut().disconnect(old_input, p).unwrap();
            if !remove[i] {
                self.hugr_mut()
                    .connect(new_input, next_input, link.0, link.1.index())
                    .unwrap();
                next_input += 1;
            }
        }

        // Remove old input
        self.hugr_mut()
            .move_before_sibling(new_input, old_input)
            .unwrap();
        self.hugr_mut().remove_op(old_input).unwrap();

        if remove_output.is_empty() {
            let OpType::DFG(old_dfg) = self.hugr().get_optype(self.hugr().root()) else {
                panic!("expect DFG root");
            };
            let signature = Signature::new_df(new_input_sig, old_dfg.signature.output.clone());
            let new_dfg = DFG { signature };
            let dfg_root = self.hugr().root();
            self.hugr_mut().replace_op(dfg_root, new_dfg);
            return;
        }

        // Create new output
        let new_output_sig = {
            let old_sig = &self.hugr().get_optype(old_output).signature();
            let mut new_in = Vec::new();
            for i in 0..old_sig.input_count() {
                if remove_output[i] {
                    continue;
                } else {
                    let t = old_sig.get_df(Port::new_incoming(i)).unwrap();
                    new_in.push(t.clone());
                }
            }
            new_in
        };
        let new_output = self.hugr_mut().add_op(Output::new(new_output_sig.clone()));

        // Wire up new output
        let mut next_output = 0;
        for (i, p) in self.output_ports().enumerate().collect_vec() {
            let Some(link) = self.hugr().linked_ports(old_output, p).next() else {
                    next_output += !remove_output[i] as usize;
                    continue
                };
            self.hugr_mut().disconnect(link.0, link.1).unwrap();
            self.hugr_mut().disconnect(old_output, p).unwrap();
            if !remove_output[i] {
                self.hugr_mut()
                    .connect(link.0, link.1.index(), new_output, next_output)
                    .unwrap();
                next_output += 1;
            }
        }

        // Remove old output
        self.hugr_mut()
            .move_before_sibling(new_output, old_output)
            .unwrap();
        self.hugr_mut().remove_op(old_output).unwrap();

        // Replace DFG signature
        let signature = Signature::new_df(new_input_sig, new_output_sig);
        let new_dfg = DFG { signature };
        let dfg_root = self.hugr().root();
        self.hugr_mut().replace_op(dfg_root, new_dfg);
    }

    /// Consume Hugr into a pattern for matching.
    ///
    /// Currently ignores hierarchy and any non-dataflow ops.
    pub fn into_pattern(self) -> Result<HugrPattern, InvalidPattern> {
        use portmatching::WeightedPattern;

        if self.input_ports().any(|p| {
            let (node, _) = self
                .hugr()
                .linked_ports(self.input_node(), p)
                .next()
                .unwrap();
            node == self.output_node()
        }) {
            return Err(InvalidPattern::DisconnectedPattern);
        }

        let to_remove = [self.hugr().root(), self.input_node(), self.output_node()];

        // TODO: support MultiPortGraph
        let Hugr {
            ref graph,
            ref op_types,
            ..
        } = self.0;

        let mut graph = graph.as_portgraph().clone();
        let mut leaf_ops = UnmanagedDenseMap::new();

        // Remove root and input/output
        for Node { index: n } in to_remove {
            graph.remove_node(n);
        }

        // Copy over the LeafOps
        for n in graph.nodes_iter() {
            if let OpType::LeafOp(leaf_op) = &op_types[n] {
                leaf_ops[n] = leaf_op.clone().into();
            }
        }

        let pattern = WeightedPattern::from_weighted_graph(graph, leaf_ops)?;
        Ok(HugrPattern::new(pattern))
    }

    /// A SimpleReplacement replacing `this` with `new` in `within`
    pub fn simple_replacement(
        &self,
        newc: CircuitHugr,
        within: &Hugr,
        pattern_root: Node,
        within_root: Node,
    ) -> Option<SimpleReplacement> {
        let mut embedding: HashMap<_, _> = [(pattern_root.index, within_root.index)].into();
        complete_embedding(
            &mut embedding,
            self.graph(),
            within.graph.as_portgraph(),
            |n| {
                !matches!(
                    self.hugr().get_optype(n.into()),
                    OpType::Input(_) | OpType::Output(_)
                )
            },
        );
        let linked_port = |hugr: &Hugr, n, p| hugr.linked_ports(n, p).next().unwrap();
        let embedded_inputs = self
            .input_ports()
            .map(|p| linked_port(self.hugr(), self.input_node(), p))
            .map(|(n, p)| (embedding[&n.index].into(), p));
        let newc_inputs = newc
            .input_ports()
            // There might be more than one linked port, in which case they
            // all map to the same embedded port
            .map(|p| newc.hugr().linked_ports(newc.input_node(), p));
        let inputs = newc_inputs
            .zip(embedded_inputs)
            .flat_map(|(new, old)| new.zip(iter::repeat(old)))
            .collect();
        let embedded_outputs = self
            .output_ports()
            .map(|p| linked_port(self.hugr(), self.output_node(), p))
            .map(|(n, p)| (embedding[&n.index].into(), p))
            // TODO: can this lead to the wrong port?
            .map(|(n, p)| linked_port(within, n, p));
        let outputs = embedded_outputs.zip(newc.output_ports()).collect();
        let parent = within.get_parent(within_root)?;
        for &n in embedding.values() {
            if within.get_parent(n.into()) != Some(parent) {
                return None;
            }
        }
        let removal = embedding.into_values().map_into().collect();
        Some(SimpleReplacement::new(
            parent, removal, newc.0, inputs, outputs,
        ))
    }
}

fn complete_embedding<P: FnMut(NodeIndex) -> bool>(
    embedding: &mut HashMap<NodeIndex, NodeIndex>,
    from: &PortGraph,
    to: &PortGraph,
    mut node_filter: P,
) {
    while let Some(edge) = {
        from.nodes_iter()
            .filter(|&n| node_filter(n))
            .filter(|n| !embedding.contains_key(n))
            .filter_map(|unknown| {
                from.all_links(unknown)
                    .flatten()
                    .find(|&p| embedding.contains_key(&from.port_node(p).unwrap()))
            })
            .next()
    } {
        let embedded_edge = to
            .port_index(
                embedding[&from.port_node(edge).unwrap()],
                from.port_offset(edge).unwrap(),
            )
            .expect("invalid pattern");
        let unknown = from.port_node(from.port_link(edge).unwrap()).unwrap();
        let embedded_unknown = to
            .port_node(to.port_link(embedded_edge).expect("invalid pattern"))
            .unwrap();
        embedding.insert(unknown, embedded_unknown);
    }
}

/// Compute hash of a circuit
pub fn circuit_hash(circ: &CircuitHugr) -> usize {
    // adapted from Quartz (Apache 2.0)
    // https://github.com/quantum-compiler/quartz/blob/2e13eb7ffb3c5c5fe96cf5b4246f4fd7512e111e/src/quartz/tasograph/tasograph.cpp#L410
    let mut total: usize = 0;

    let mut hash_vals = HashMap::new();

    let _ophash = |o| 17 * 13 + op_hash(o).expect(&format!("unhashable op: {o:?}"));

    let initial_nodes = circ
        .graph()
        .nodes_iter()
        .filter(|&n| circ.graph().num_inputs(n) == 0 && n != circ.hugr().root().index);

    for nid in toposort::<BitVec>(circ.graph(), initial_nodes, Direction::Outgoing) {
        let mut myhash = _ophash(circ.hugr().get_optype(nid.into()));

        for incoming in circ.graph().inputs(nid) {
            let Some(outgoing) = circ.graph().port_link(incoming) else { continue };
            let src = circ.graph().port_node(outgoing).expect("invalid port");
            debug_assert!(hash_vals.contains_key(&src));

            let mut edgehash = hash_vals[&src];

            // TODO check if overflow arithmetic is intended

            edgehash = edgehash.wrapping_mul(31).wrapping_add(outgoing.index());
            edgehash = edgehash.wrapping_mul(31).wrapping_add(incoming.index());

            myhash = myhash.wrapping_add(edgehash);
        }
        hash_vals.insert(nid, myhash);
        total = total.wrapping_add(myhash);
    }

    total
}

fn op_hash(op: &OpType) -> Option<usize> {
    Some(match op {
        OpType::Input(_) => 5,
        OpType::Output(_) => 6,
        OpType::LeafOp(op) => match op {
            LeafOp::H => 1,
            LeafOp::CX => 2,
            LeafOp::ZZMax => 3,
            LeafOp::Reset => 4,
            LeafOp::Noop(_) => 5,
            LeafOp::Measure => 6,
            LeafOp::AddF64 => 7,
            LeafOp::RxF64 => 8,
            LeafOp::RzF64 => 9,
            LeafOp::X => 10,
            _ => return None,
        },
        OpType::LoadConstant(_) => 11,
        // copy nodes show up as module
        OpType::Module(_) => 0,
        &OpType::Const(Const(ConstValue::F64(_))) => 12,
        _ => return None,
    })
}

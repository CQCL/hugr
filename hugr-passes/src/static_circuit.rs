use std::{
    collections::{BTreeSet, HashMap, HashSet},
    hash::{Hash, Hasher},
    ops::Deref,
};

use ascent::Lattice;
use hugr_core::{
    extension::{prelude::QB_T, ExtensionRegistry, ExtensionSet},
    ops::{ExtensionOp, NamedOp as _},
    HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire,
};
use itertools::{zip_eq, Itertools};
use petgraph::algo::toposort;

use crate::{
    dataflow::{
        AbstractValue, ConstLoader, DFContext, Machine, NoDefaultConversionToSum, PartialValue, Sum,
    },
    validation::ValidationLevel,
};

#[derive(Eq, PartialEq, Debug, Clone, PartialOrd, Ord)]
pub struct Gate {
    // The nodes that execute this gate. They must form a "co-cycle" i.e. every
    // path from input to output contains at most one of these nodes
    nodes: BTreeSet<Node>,
    gate: String,
    gating_nodes: BTreeSet<Node>,
    hash: u64,
}

impl Gate {
    pub fn show(&self) -> String {
        format!(
            "{} [{}] ({})",
            &self.gate,
            self.gating_nodes.iter().join(","),
            self.nodes.iter().join(",")
        )
    }

    fn calc_hash(&mut self) {
        let mut hasher = std::hash::DefaultHasher::default();
        self.nodes.hash(&mut hasher);
        self.gate.hash(&mut hasher);
        self.gating_nodes.hash(&mut hasher);
        self.hash = hasher.finish()
    }

    pub fn new(
        node: Node,
        gate: impl Into<String>,
        gating_nodes: impl IntoIterator<Item = Node>,
    ) -> Self {
        let mut g = Self {
            nodes: BTreeSet::from_iter([node]),
            gate: gate.into(),
            gating_nodes: BTreeSet::from_iter(gating_nodes),
            hash: 0,
        };
        g.calc_hash();
        g
    }
    pub fn join(mut self, other: Self) -> Option<Self> {
        if self.gate != other.gate || self.gating_nodes != other.gating_nodes {
            None?;
        }

        self.nodes.extend(other.nodes);
        self.calc_hash();
        Some(self)
    }

    pub fn commutes_lt(&self, other: &Self) -> bool {
        if self.gating_nodes.intersection(&other.gating_nodes).count() != 0 || self <= other {
            return false;
        };
        true
    }
}

impl Hash for Gate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state)
    }
}

#[derive(Eq, PartialEq, Debug, Clone, PartialOrd, Ord, Hash)]
struct QbHistory(Node, Vec<Gate>);

impl QbHistory {
    pub fn init(qb: Node) -> Self {
        Self(qb, vec![])
    }

    pub fn push(&mut self, node: Node, g: impl Into<String>, qbs: impl IntoIterator<Item = Node>) {
        self.1.push(Gate::new(node, g.into(), qbs))
    }

    fn qb_index(&self) -> Node {
        self.0
    }
}

impl TryFrom<Sum<QbHistory>> for QbHistory {
    type Error = NoDefaultConversionToSum<QbHistory>;

    fn try_from(_: Sum<QbHistory>) -> Result<Self, Self::Error> {
        Err(NoDefaultConversionToSum::default())
    }
}

impl AbstractValue for QbHistory {
    fn try_join(self, other: Self) -> Option<Self> {
        eprintln!("join: {:?} ^ {:?} ", &self, &other);
        let r = if self.0 == other.0 && self.1.len() == other.1.len() {
            Some(QbHistory(
                self.0,
                zip_eq(self.1, other.1)
                    .map(|(a, b)| a.join(b))
                    .collect::<Option<Vec<_>>>()?,
            ))
        } else {
            None
        };
        panic!("dougrulz");
        eprintln!("join:  => {:?}", &r);
        r
    }

    fn try_meet(self, other: Self) -> Option<Self> {
        (self.0 == other.0).then_some(self)
    }
}

pub struct StaticCircuitContext<'a, H> {
    pub hugr: &'a H,
    pub extensions: ExtensionSet,
    pub alloc_ops: HashSet<Node>,
    pub free_ops: HashSet<Node>,
}

impl<H> Clone for StaticCircuitContext<'_, H> {
    fn clone(&self) -> Self {
        Self {
            hugr: self.hugr,
            extensions: self.extensions.clone(),
            alloc_ops: self.alloc_ops.clone(),
            free_ops: self.free_ops.clone(),
        }
    }
}

impl<'a, H: HugrView> StaticCircuitContext<'a, H> {
    fn assert_invariants(&self) {
        debug_assert!(self.alloc_ops.is_disjoint(&self.free_ops));
        for &n in self.alloc_ops.iter().chain(&self.free_ops) {
            debug_assert!(self.hugr.valid_node(n));
            let Some(e) = self.hugr.get_optype(n).as_extension_op() else {
                panic!("node {n} is not in extensions")
            };
            debug_assert!(self.extensions.contains(e.def().extension()));
        }
    }
}

impl<'a, H: HugrView> ConstLoader<QbHistory> for StaticCircuitContext<'a, H> {}

impl<'a, H> Deref for StaticCircuitContext<'a, H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        self.hugr
    }
}

impl<'a, H: HugrView> DFContext<QbHistory> for StaticCircuitContext<'a, H> {
    type View = H;

    fn interpret_leaf_op(
        &self,
        node: Node,
        e: &ExtensionOp,
        ins: &[PartialValue<QbHistory>],
        outs: &mut [PartialValue<QbHistory>],
    ) {
        self.assert_invariants();
        let extension = e.def().extension();
        if !self.extensions.contains(extension) {
            return;
        }

        if ins.iter().any(|x| x == &PartialValue::Bottom) {
            return;
        }

        // assert_eq!(ins.len(), self.hugr.num_inputs(node));
        // assert_eq!(outs.len(), self.hugr.num_outputs(node));

        let qb_ins = qubits_in(self.hugr, node).collect::<Vec<_>>();
        let qb_outs = qubits_out(self.hugr, node).collect::<Vec<_>>();

        if qb_ins.is_empty() && qb_outs.is_empty() {
            return;
        }

        if let Some(&qb_i) = self.alloc_ops.get(&node) {
            assert_eq!(0, qb_ins.len());
            assert_eq!(1, qb_outs.len());
            for (i, out) in outs.iter_mut().enumerate() {
                if qb_outs[0] == i.into() {
                    out.join_mut(QbHistory::init(qb_i).into());
                } else {
                    out.join_mut(PartialValue::Top);
                }
            }
        } else if self.free_ops.contains(&node) {
            assert_eq!(0, qb_outs.len());
            for out in outs.iter_mut() {
                out.join_mut(PartialValue::Top);
            }
        } else {
            assert_eq!(qb_outs.len(), qb_ins.len(), "{e:?}");
            let mb_in_qbs = qb_ins
                .iter()
                .map(|&i| match &ins[i.index()] {
                    PartialValue::Value(qbh) => Some((i, qbh)),
                    _ => None,
                })
                .collect::<Option<HashMap<_, _>>>();
            let gating_qbs = mb_in_qbs
                .iter()
                .flat_map(|hash_map| hash_map.iter().map(|(_, qbh)| qbh.qb_index()))
                .collect_vec();
            let qb_out_to_in: HashMap<_, _> = zip_eq(qb_outs, qb_ins).collect();
            for (i, v) in outs.iter_mut().enumerate() {
                let mb_join_val = (|| {
                    let in_p = qb_out_to_in.get(&OutgoingPort::from(i))?;
                    let hash_map = mb_in_qbs.as_ref()?;
                    let qbh: &QbHistory = hash_map.get(in_p)?;
                    let mut qbh = qbh.clone();
                    qbh.push(node, e.name(), gating_qbs.clone());
                    Some(qbh.into())
                })();

                v.join_mut(mb_join_val.unwrap_or(PartialValue::Top));
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
/// TODO docs
pub struct StaticCircuitPass {
    validation: ValidationLevel,
}

#[derive(Debug)]
struct StaticCircuitPassError(Box<dyn std::error::Error>);

impl StaticCircuitPass {
    /// Sets the validation level used before and after the pass is run
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    pub fn run(
        &self,
        scc: StaticCircuitContext<'_, impl HugrView>,
        registry: &ExtensionRegistry,
    ) -> Result<StaticCircuitResult, Box<dyn std::error::Error>> {
        // eprintln!(
        //     "StaticCircuitPass::run alloc_nodes: {:?} free_nodes: {:?}",
        //     &scc.alloc_ops, &scc.free_ops
        // );
        scc.assert_invariants();
        self.validation
            .run_validated_pass(scc.hugr, registry, |hugr, _| {
                let results = Machine::default().run(scc.clone(), []);

                Ok(StaticCircuitResult {
                    qubit_history: scc
                        .free_ops
                        .iter()
                        .filter_map(|&free_node| {
                            let wire = {
                                let qbs_in = qubits_in(hugr, free_node).collect::<Vec<_>>();
                                assert_eq!(1, qbs_in.len());
                                let (n, p) = hugr.single_linked_output(free_node, qbs_in[0])?;
                                Wire::new(n, p)
                            };

                            let QbHistory(alloc_node, gates) =
                                results.try_read_wire_value(wire).ok()?;
                            Some((free_node, (alloc_node, gates)))
                        })
                        .collect::<HashMap<_, _>>(),
                })
            })
    }
}

pub struct StaticCircuitResult {
    qubit_history: HashMap<Node, (Node, Vec<Gate>)>,
}

impl StaticCircuitResult {
    pub fn static_circuit<H>(self, scc: StaticCircuitContext<'_, H>) -> Option<Vec<Gate>> {
        let mut frees = scc.free_ops.clone();
        let mut allocs = scc.alloc_ops.clone();
        let mut gate_graph = petgraph::graph::Graph::<Gate, ()>::new();
        let mut gate_to_node = HashMap::new();
        for (free_node, (alloc_node, qb_gates)) in self.qubit_history.into_iter() {
            assert!(frees.remove(&free_node));
            assert!(allocs.remove(&alloc_node));

            let gate_indices = qb_gates
                .into_iter()
                .map(|gate| {
                    *gate_to_node
                        .entry(gate)
                        .or_insert_with_key(|gate| gate_graph.add_node(gate.clone()))
                })
                .collect_vec();

            for (from, to) in gate_indices.into_iter().tuple_windows() {
                gate_graph.add_edge(from, to, ());
            }
        }

        if !frees.is_empty() || !allocs.is_empty() {
            None?
        }

        let mut gates = toposort(&gate_graph, None)
            .ok()?
            .into_iter()
            .map(|x| gate_graph[x].clone())
            .collect_vec();

        let mut changes = true;
        while changes {
            changes = false;

            for (g1, g2) in (0..gates.len()).tuple_windows() {
                if gates[g1].commutes_lt(&gates[g2]) {
                    changes = true;
                    gates.swap(g1, g2)
                }
            }
        }
        Some(gates)
    }
}

fn qubits_in(hugr: &impl HugrView, node: Node) -> impl Iterator<Item = IncomingPort> + '_ {
    hugr.in_value_types(node)
        .filter_map(|(p, t)| (t == QB_T).then_some(p))
}

fn qubits_out(hugr: &impl HugrView, node: Node) -> impl Iterator<Item = OutgoingPort> + '_ {
    hugr.out_value_types(node)
        .filter_map(|(p, t)| (t == QB_T).then_some(p))
}

// pub fn check_results(
//     results: impl IntoIterator<Item = (Node, Option<(Node, Vec<Gate>)>)>,
// ) -> Result<Option<Vec<Gate>>, Box<dyn std::error::Error>> {
//     let mut g = petgraph::graph::Graph::<Gate, ()>::new();
//     for (_, mb_gates) in results.into_iter() {
//         let Some((_, gates)) = mb_gates else {
//             return Ok(None);
//         };
//         let node_to_index = gates
//             .iter()
//             .enumerate()
//             .map(|(i, gate)| (i, g.add_node(gate.clone())))
//             .collect::<HashMap<_, _>>();

//         for (from, to) in gates
//             .iter()
//             .enumerate()
//             .map(|(i, x)| node_to_index[&i])
//             .tuple_windows()
//         {
//             g.add_edge(from, to, ());
//         }
//     }
//     Ok(Some(
//         petgraph::algo::toposort(&g, None)
//             .map_err(|x| format!("check_results: graph is cyclic: {x:?}"))?
//             .into_iter()
//             .map(|i| g.node_weight(i).unwrap().clone())
//             .collect(),
//     ))
// }

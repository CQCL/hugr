use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
};

use ascent::Lattice;
use hugr_core::{
    extension::{prelude::QB_T, ExtensionRegistry, ExtensionSet},
    ops::{ExtensionOp, NamedOp as _},
    HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire,
};
use itertools::{zip_eq, Itertools};

use crate::{
    dataflow::{
        AbstractValue, ConstLoader, DFContext, Machine, NoDefaultConversionToSum, PartialValue, Sum,
    },
    validation::ValidationLevel,
};

#[derive(Eq, PartialEq, Debug, Clone, PartialOrd, Ord, Hash)]
pub enum Gate {
    Gate(Node, String, Vec<Node>),
}

#[derive(Eq, PartialEq, Debug, Clone, PartialOrd, Ord, Hash)]
enum QbHistory {
    Bottom,
    H(Node, Vec<Gate>),
    Top,
}

impl QbHistory {
    pub fn init(qb: Node) -> Self {
        Self::H(qb, vec![])
    }

    pub fn push(&mut self, node: Node, g: impl Into<String>, qbs: impl Into<Vec<Node>>) {
        if let Self::H(_, v) = self {
            v.push(Gate::Gate(node, g.into(), qbs.into()));
        }
    }

    fn qb_index(&self) -> Option<Node> {
        if let Self::H(qb_i, _) = self {
            Some(*qb_i)
        } else {
            None
        }
    }
}

impl TryFrom<Sum<QbHistory>> for QbHistory {
    type Error = NoDefaultConversionToSum<QbHistory>;

    fn try_from(_: Sum<QbHistory>) -> Result<Self, Self::Error> {
        Err(NoDefaultConversionToSum::default())
    }
}

impl AbstractValue for QbHistory {}

pub struct StaticCircuitContext<'a, H> {
    pub hugr: &'a H,
    pub extensions: &'a ExtensionSet,
    pub alloc_ops: &'a HashSet<Node>,
    pub free_ops: &'a HashSet<Node>,
}

impl<H> Clone for StaticCircuitContext<'_, H> {
    fn clone(&self) -> Self {
        Self {
            hugr: self.hugr,
            extensions: self.extensions,
            alloc_ops: self.alloc_ops,
            free_ops: self.free_ops,
        }
    }
}

impl<'a, H: HugrView> StaticCircuitContext<'a, H> {
    fn assert_invariants(&self) {
        debug_assert!(self.alloc_ops.is_disjoint(self.free_ops));
        for &n in self.alloc_ops.iter().chain(self.free_ops) {
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

        assert_eq!(ins.len(), self.hugr.num_inputs(node));
        assert_eq!(outs.len(), self.hugr.num_outputs(node));

        let qb_ins = qubits_in(self.hugr, node).collect::<Vec<_>>();
        let qb_outs = qubits_out(self.hugr, node).collect::<Vec<_>>();

        if qb_ins.is_empty() && qb_outs.is_empty() {
            return;
        }

        if let Some(&qb_i) = self.alloc_ops.get(&node) {
            assert_eq!(0, qb_ins.len());
            assert_eq!(1, qb_outs.len());
            outs[0].join_mut(QbHistory::init(qb_i).into());
        } else if self.free_ops.contains(&node) {
            assert_eq!(0, qb_outs.len());
        } else {
            assert_eq!(qb_outs.len(), qb_ins.len());
            if let Some(in_hs) = qb_ins
                .iter()
                .map(|&i| match &ins[i.index()] {
                    PartialValue::Value(qbh) => qbh.qb_index().map(|qb_i| (qb_i, qbh)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
            {
                let qbs = in_hs.iter().map(|(qb_i, _)| *qb_i).collect_vec();
                for ((_, qbh), o) in zip_eq(in_hs, qb_outs) {
                    let mut qbh = qbh.clone();
                    qbh.push(node, e.name(), qbs.clone());
                    outs[o.index()].join_mut(qbh.into());
                }
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
    ) -> Result<HashMap<Node, Option<(Node, Vec<Gate>)>>, Box<dyn std::error::Error>> {
        scc.assert_invariants();
        self.validation
            .run_validated_pass(scc.hugr, registry, |hugr, _| {
                let results = Machine::default().run(scc.clone(), []);

                Ok(scc
                    .free_ops
                    .iter()
                    .map(|&free_node| {
                        let gates = (|| {
                            let wire = {
                                let qbs_in = qubits_in(hugr, free_node).collect::<Vec<_>>();
                                assert_eq!(1, qbs_in.len());
                                let (n, p) = hugr.single_linked_output(free_node, qbs_in[0])?;
                                Wire::new(n, p)
                            };
                            if let Ok(QbHistory::H(i, gates)) = results.try_read_wire_value(wire)
                            {
                                Some((i, gates.clone()))
                            } else {
                                None
                            }
                        })();
                        (free_node, gates)
                    })
                    .collect())
            })
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

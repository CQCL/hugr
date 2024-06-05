use std::hash::{Hash, Hasher};

use ascent::{ascent_run, Lattice};
use hugr_core::hugr::views::{DescendantsGraph, HierarchyView};
use hugr_core::ops::{OpTag, OpTrait, Value};
use hugr_core::types::{SumType, Type, TypeRow};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};
use itertools::{zip_eq, Itertools};
use std::collections::HashMap;

#[derive(PartialEq, Clone, Eq)]
struct HashableHashMap<K: Hash + std::cmp::Eq, V>(HashMap<K, V>);

impl<K: Hash + std::cmp::Eq, V: Hash> Hash for HashableHashMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.keys().for_each(|k| k.hash(state));
        self.0.values().for_each(|v| v.hash(state));
    }
}

#[derive(PartialEq, Clone, Eq, Hash)]
enum PartialValue {
    Bottom(Type),
    Value(Node, Type),
    PartialSum(HashableHashMap<usize, Vec<PartialValue>>, SumType),
    Top(Type),
}

impl PartialValue {
    fn get_type(&self) -> Type {
        match self {
            PartialValue::Bottom(t) => t.clone(),
            PartialValue::Value(_, t) => t.clone(),
            PartialValue::PartialSum(_, t) => t.clone().into(),
            PartialValue::Top(t) => t.clone(),
        }
    }

    fn top_from_hugr(hugr: &impl HugrView, node: Node, port: OutgoingPort) -> Self {
        Self::Top(
            hugr.signature(node)
                .unwrap()
                .out_port_type(port)
                .unwrap()
                .clone(),
        )
    }

    fn from_load_constant(hugr: &impl HugrView, node: Node) -> Self {
        let load_op = hugr.get_optype(node).as_load_constant().unwrap();
        let const_node = hugr
            .single_linked_output(node, load_op.constant_port())
            .unwrap()
            .0;
        let const_op = hugr.get_optype(const_node).as_const().unwrap();
        Self::Value(const_node, const_op.get_type())
    }

    fn tuple_from_value_row(r: &ValueRow) -> Self {
        unimplemented!()
    }

}

impl PartialOrd for PartialValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // TODO we can do better
        (self == other).then_some(std::cmp::Ordering::Equal)
    }
}

impl Lattice for PartialValue {
    fn meet(self, _other: Self) -> Self {
        // should not be required
        todo!()
    }

    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn join_mut(&mut self, other: Self) -> bool {
        debug_assert_eq!(self.get_type(), other.get_type());
        match (self, other) {
            (Self::Bottom(_), _) => false,
            (s, rhs @ Self::Bottom(_)) => {
                *s = rhs;
                true
            }
            (_, Self::Top(_)) => false,
            (s @ Self::Top(_), x) => {
                *s = x;
                true
            }
            (Self::Value(n1, t), Self::Value(n2, _)) if n1 == &n2 => false,
            (
                Self::PartialSum(HashableHashMap(hm1), t),
                Self::PartialSum(HashableHashMap(hm2), _),
            ) => {
                let mut changed = false;
                for (k, v) in hm2 {
                    let row = hm1.entry(k).or_insert_with(|| {
                        changed = true;
                        t.get_variant(k)
                            .unwrap()
                            .iter()
                            .cloned()
                            .map(Self::Top)
                            .collect_vec()
                    });
                    for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                        changed |= lhs.join_mut(rhs);
                    }
                }
                changed
            }
            (s, _) => {
                *s = Self::Bottom(s.get_type());
                true
            }
        }
    }
}

// fn input_row<'a>(inp: impl Iterator<Item = (&'a Wire, &'a PartialValue)>) -> impl Iterator<Item=ValueRow> {
//     todo!()
// }

#[derive(PartialEq, Clone, Eq, Hash, PartialOrd)]
enum ValueRow {
    Values(Vec<PartialValue>),
    Bottom,
}

impl ValueRow {
    fn into_partial_value(self) -> PartialValue {
        todo!()
    }

    fn new(tr: &TypeRow) -> Self {
        Self::Values(tr.iter().cloned().map(PartialValue::Top).collect_vec())
    }

    fn singleton(tr: &TypeRow, idx: usize, v: PartialValue) -> Self {
        let mut r = Self::new(tr);
        if let Self::Values(vec) = &mut r {
            vec[idx] = v;
        }
        r
    }

    fn iter(&self) -> impl Iterator<Item=(IncomingPort,PartialValue)> {
        std::iter::empty()
    }
}

impl Lattice for ValueRow {
    fn meet(self, other: Self) -> Self {
        todo!()
    }

    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn join_mut(&mut self, other: Self) -> bool {
        match (self, other) {
            (Self::Bottom, _) => false,
            (s, o @ Self::Bottom) => {
                *s = o;
                true
            }
            (s, Self::Values(vs2)) => {
                let (b, r) = if let Self::Values(vs1) = s {
                    if vs1.len() == vs2.len() {
                        let mut changed = false;
                        for (v1, v2) in zip_eq(vs1.iter_mut(), vs2.into_iter()) {
                            changed |= v1.join_mut(v2);
                        }
                        (false, changed)
                    } else {
                        (true, true)
                    }
                } else {
                    panic!("impossible")
                };
                if b {
                    *s = Self::Bottom;
                }
                r
            }
        }
    }
}

fn node_in_value_row<'a>(
    ins: impl Iterator<Item = (&'a Node, &'a IncomingPort, &'a PartialValue)>,
) -> impl Iterator<Item = ValueRow> {
    std::iter::empty()
}

fn tc(hugr: &impl HugrView, node: Node) {
    assert!(OpTag::DataflowParent.is_superset(hugr.get_optype(node).tag()));
    let d = DescendantsGraph::<'_, Node>::try_new(hugr, node).unwrap();
    ascent_run! {
        relation node(Node) = d.nodes().map(|x| (x,)).collect_vec();

        relation in_wire(Node, IncomingPort);
        in_wire(n,p) <-- node(n), for p in d.node_inputs(*n);

        relation out_wire(Node, OutgoingPort);
        out_wire(n,p) <-- node(n), for p in d.node_outputs(*n);

        lattice node_in_value_row(Node, ValueRow);
        node_in_value_row(n, ValueRow::new(&hugr.signature(*n).unwrap().input)) <-- node(n);

        lattice out_wire_value(Node, OutgoingPort, PartialValue);
        out_wire_value(n,p, PartialValue::top_from_hugr(hugr,*n,*p)) <-- out_wire(n,p);

        node_in_value_row(n,ValueRow::singleton(&hugr.signature(*n).unwrap().input, ip.index(), v.clone())) <-- in_wire(n, ip),
            if let Some((m,op)) = hugr.single_linked_output(*n, *ip), out_wire_value(m, op, ?v);

        lattice in_wire_value(Node, IncomingPort, PartialValue);
        in_wire_value(n,p,v) <-- node_in_value_row(n, ?vr), for (p,v) in vr.iter();

        relation load_constant_node(Node);
        load_constant_node(n) <-- node(n), if hugr.get_optype(*n).is_load_constant();
        out_wire_value(n, 0.into(), PartialValue::from_load_constant(hugr, *n)) <-- load_constant_node(n);

        relation make_tuple_node(Node);
        make_tuple_node(n) <-- node(n), if hugr.get_optype(*n).is_make_tuple();

        out_wire_value(n,0.into(), PartialValue::tuple_from_value_row(vs)) <-- make_tuple_node(n), node_in_value_row(n, ?vs);
    };
}

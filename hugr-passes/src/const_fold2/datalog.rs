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

struct ValueCache(HashMap<Node,Value>);

impl ValueCache {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&mut self, node: Node, value: &Value) -> ValueHandle {
        self.0.entry(node).or_insert_with(|| value.clone());
        ValueHandle(node)
    }
}

#[derive(PartialEq,Eq,Clone,Hash)]
struct ValueHandle(Node);

impl ValueHandle {
    fn new(node: Node) -> Self {
        Self(node)
    }
}

#[derive(PartialEq, Clone, Eq, Hash)]
enum PartialValue {
    Bottom,
    Value(ValueHandle),
    PartialSum(HashableHashMap<usize, Vec<PartialValue>>),
    Top,
}

impl PartialValue {
    const BOTTOM: Self = Self::Bottom;
    const BOTTOM_REF: &'static Self = &Self::BOTTOM;
    fn from_load_constant(cache: &mut ValueCache, hugr: &impl HugrView, node: Node) -> Self {
        let load_op = hugr.get_optype(node).as_load_constant().unwrap();
        let const_node = hugr
            .single_linked_output(node, load_op.constant_port())
            .unwrap()
            .0;
        let const_op = hugr.get_optype(const_node).as_const().unwrap();
        Self::Value(cache.get(const_node, const_op.value()))
    }

    fn tuple_from_value_row(r: &ValueRow) -> Self {
        if !r.initialised() {
            return Self::Top
        }
        match r {
           ValueRow::Bottom  => Self::Bottom,
           ValueRow::Values(vs) => {
            PartialValue::PartialSum(HashableHashMap([(0usize, vs.clone())].into_iter().collect()))
           }
        }
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
        match (self, other) {
            (Self::Bottom, _) => false,
            (s, Self::Bottom) => {
                *s = Self::Bottom;
                true
            }
            (_, Self::Top) => false,
            (s @ Self::Top, x) => {
                *s = Self::Top;
                true
            }
            (Self::Value(h1), Self::Value(h2)) if h1 == &h2 => false,
            (
                Self::PartialSum(HashableHashMap(hm1)),
                Self::PartialSum(HashableHashMap(hm2))
            ) => {
                let mut changed = false;
                for (k, v) in hm2 {
                    if let Some(row) = hm1.get_mut(&k) {
                        for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                            changed |= lhs.join_mut(rhs);
                        }
                    } else {
                        hm1.insert(k, v);
                        changed = true;
                    }
                }
                changed
            }
            (s, _) => {
                *s = Self::Bottom;
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

    fn new(len: usize) -> Self {
        Self::Values(vec![PartialValue::Top; len])
    }

    fn singleton(len: usize, idx: usize, v: PartialValue) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        if let Self::Values(vec) = &mut r {
            vec[idx] = v;
        }
        r
    }

    fn singleton_from_row(r: &TypeRow, idx: usize, v: PartialValue) -> Self {
        Self::singleton(r.len(),idx,v)
    }

    fn top_from_row(r: &TypeRow) -> Self {
        Self::new(r.len())
    }

    fn iter<'a>(&'a self, h: &'a impl HugrView, n: Node) -> impl Iterator<Item=(IncomingPort,&PartialValue)> + 'a {
        match self {
            Self::Values(v) => {
                either::Either::Left(zip_eq(h.node_inputs(n), v.iter()))
            }
            Self::Bottom => either::Either::Right(h.node_inputs(n).map(|x| (x,PartialValue::BOTTOM_REF)))
        }
    }

    fn initialised(&self) -> bool {
        if let Self::Values(v) = self {
            v.iter().all(|x| x != &PartialValue::Top)
        } else {
            true
        }
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

fn tc(hugr: &impl HugrView, node: Node) -> Vec<(Node, OutgoingPort, PartialValue)> {
    assert!(OpTag::DataflowParent.is_superset(hugr.get_optype(node).tag()));
    let d = DescendantsGraph::<'_, Node>::try_new(hugr, node).unwrap();
    let mut cache = ValueCache::new();

    let singleton_in_row = |n: &Node, ip: &IncomingPort, v: &PartialValue| -> ValueRow {
        ValueRow::singleton_from_row(&hugr.signature(*n).unwrap().input, ip.index(), v.clone())
    };

    let top_row = |n: &Node| -> ValueRow {
        ValueRow::top_from_row(&hugr.signature(*n).unwrap().input)
    };
    ascent_run! {
        relation node(Node) = d.nodes().map(|x| (x,)).collect_vec();

        relation in_wire(Node, IncomingPort);
        in_wire(n,p) <-- node(n), for p in d.node_inputs(*n);

        relation out_wire(Node, OutgoingPort);
        out_wire(n,p) <-- node(n), for p in d.node_outputs(*n);

        lattice out_wire_value(Node, OutgoingPort, PartialValue);
        out_wire_value(n,p, PartialValue::Top) <-- out_wire(n,p);

        lattice node_in_value_row(Node, ValueRow);
        node_in_value_row(n, top_row(n)) <-- node(n);
        node_in_value_row(n, singleton_in_row(n,ip,v)) <-- in_wire(n, ip),
            if let Some((m,op)) = hugr.single_linked_output(*n, *ip),
            out_wire_value(m, op, v);

        lattice in_wire_value(Node, IncomingPort, PartialValue);
        in_wire_value(n,p,v) <-- node_in_value_row(n, vr), for (p,v) in vr.iter(hugr,*n);

        relation load_constant_node(Node);
        load_constant_node(n) <-- node(n), if hugr.get_optype(*n).is_load_constant();

        out_wire_value(n, 0.into(), PartialValue::from_load_constant(&mut cache, hugr, *n)) <--
            load_constant_node(n);

        relation make_tuple_node(Node);
        make_tuple_node(n) <-- node(n), if hugr.get_optype(*n).is_make_tuple();

        out_wire_value(n,0.into(), PartialValue::tuple_from_value_row(vs)) <--
            make_tuple_node(n), node_in_value_row(n, vs);

    }.out_wire_value
}

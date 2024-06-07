use std::hash::{Hash, Hasher};

use ascent::{ascent_run, Lattice};
use either::Either;
use hugr_core::hugr::views::{DescendantsGraph, HierarchyView};
use hugr_core::ops::{OpTag, OpTrait, Value};
use hugr_core::types::{FunctionType, SumType, Type, TypeEnum, TypeRow};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};
use itertools::{zip_eq, Itertools};
use std::collections::HashMap;
use std::sync::Arc;

mod context;

use context::DataflowContext;

use self::context::ValueHandle;

#[derive(PartialEq, Clone, Eq, Debug)]
struct HashableHashMap<K: Hash + std::cmp::Eq, V>(HashMap<K, V>);

impl<K: Hash + std::cmp::Eq, V: Hash> Hash for HashableHashMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.keys().for_each(|k| k.hash(state));
        self.0.values().for_each(|v| v.hash(state));
    }
}

#[derive(PartialEq, Clone, Eq, Hash, Debug)]
enum PartialValue {
    Bottom,
    Value(context::ValueHandle),
    PartialSum(HashableHashMap<usize, Vec<PartialValue>>),
    Top,
}

impl From<ValueHandle> for PartialValue {
    fn from(v: ValueHandle) -> Self {
        match v.value() {
            Value::Tuple { vs } => {
                let vec = (0..vs.len()).map(|i| PartialValue::from(v.index(i)).into()).collect();
                Self::PartialSum(HashableHashMap([(0, vec)].into_iter().collect()))
            }
            Value::Sum { tag, values, .. } => {
                let vec = (0..values.len()).map(|i| PartialValue::from(v.index(i)).into()).collect();
                Self::PartialSum(HashableHashMap([(*tag, vec)].into_iter().collect()))
            }
            _ => Self::Value(v)
        }
    }
}

impl PartialValue {
    const BOTTOM: Self = Self::Bottom;
    const BOTTOM_REF: &'static Self = &Self::BOTTOM;

    fn initialised(&self) -> bool {
        !self.is_top()
    }

    fn is_top(&self) -> bool { self == &PartialValue::Top }

    fn from_load_constant<'a, H: HugrView>(
        context: &context::DataflowContext<'a, H>,
        node: Node,
    ) -> Self {
        let load_op = context.hugr().get_optype(node).as_load_constant().unwrap();
        let const_node = context
            .hugr()
            .single_linked_output(node, load_op.constant_port())
            .unwrap()
            .0;
        let const_op = context.hugr().get_optype(const_node).as_const().unwrap();
        context.value_handle(const_node, const_op.value()).into()
    }

    fn tuple_from_value_row(r: &ValueRow) -> Self {
        // if !r.initialised() {
        //     return Self::Top;
        // }
        PartialValue::PartialSum(HashableHashMap([(0usize, r.0.clone())].into_iter().collect()))
    }

    fn tuple_field_value(&self, idx: usize) -> Self {
        match self {
            Self::Top => Self::Top,
            Self::PartialSum(HashableHashMap(hm)) => {
                if let Ok((0, row)) = hm.iter().exactly_one() {
                    assert!(row.len() > idx);
                    row[idx].clone()
                } else {
                    Self::Bottom
                }
            },
            Self::Value(v) => Self::Value(v.index(idx)),
            _ => Self::Bottom
        }
    }

    fn variant_field_value(&self, variant: usize, idx: usize) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::PartialSum(HashableHashMap(hm)) => {
                if let Some(row) = hm.get(&variant) {
                    assert!(row.len() > idx);
                    row[idx].clone()
                } else {
                    // We must return top. if self were to gain this variant, we would return the element of that variant.
                    // We must ensure that the value return now is <= that future value
                    Self::Top
                }
            },
            Self::Value(v) if v.tag() == variant => {
                Self::Value(v.index(idx))
            },
            _ => Self::Top
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<Value,Self> {
        let r = match self {
            Self::Value(v) => v.value().clone(),
            Self::PartialSum(HashableHashMap(hm)) => {
                let err = |hm| Err(Self::PartialSum(HashableHashMap(hm)));
                let Ok((k,v)) = hm.iter().exactly_one() else {
                    return err(hm);
                };
                let TypeEnum::Sum(st) = typ.as_type_enum() else {
                    return err(hm);
                };
                let Some(r) = st.get_variant(*k) else {
                    return err(hm);
                };
                if v.len() != r.len() {
                    return err(hm);
                }

                let Ok(vs) = zip_eq(v.into_iter(), r.into_iter()).map(|(v,t)| v.clone().try_into_value(t)).collect::<Result<Vec<_>, _>>() else {
                    return err(hm);
                };

                Value::sum(*k, vs, st.clone()).map_err(|_| Self::PartialSum(HashableHashMap(hm)))?
            },
            x => Err(x)?

        };
        assert_eq!(typ, &r.get_type());
        Ok(r)
    }

    fn join_value_handle(&mut self, vh: ValueHandle) -> bool {
        let mut new_self = self;
        match &mut new_self {
            Self::Bottom => {
                false
            },
            s@Self::Value(_) => {
                let Self::Value(v) = *s else { unreachable!() };
                if v == &vh {
                    false
                } else {
                    **s = Self::Bottom;
                    true
                }
            },
            s@Self::PartialSum(_) => {
                match vh.into() {
                    Self::Value(_) => {
                        **s = Self::Bottom;
                        true
                    }
                    other => s.join_mut(other)
                }
            },
            s@Self::Top => {
                **s = vh.into();
                true
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
        // println!("join {self:?}\n{:?}", &other);
        let mut s = self;
        let changed = match (&mut s, other) {
            (Self::Bottom, _) => false,
            (s, other@Self::Bottom) => {
                **s = other;
                true
            },
            (_, Self::Top) => false,
            (s@Self::Top, other) => {
                **s = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) if h1 == &h2 || h1.value() == h2.value() => {
                false
            }
            (s@Self::PartialSum(_), Self::PartialSum(HashableHashMap(hm2))) => {
                let mut changed = false;
                let Self::PartialSum(HashableHashMap(hm1)) = *s else { unreachable!() };
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
            (s@Self::Value(_), other@Self::PartialSum(_)) => {
                let mut old_self = other;
                std::mem::swap(*s, &mut old_self);
                let Self::Value(h) = old_self else { unreachable!() };
                s.join_value_handle(h)
            }
            (s@Self::PartialSum(_), Self::Value(h)) => {
                s.join_value_handle(h)
            }
            (s,_) => {
                **s = Self::Bottom;
                false
            }
        };
        // if changed {
            // println!("join new self: {:?}", s);
        // }
        changed
    }
}

// fn input_row<'a>(inp: impl Iterator<Item = (&'a Wire, &'a PartialValue)>) -> impl Iterator<Item=ValueRow> {
//     todo!()
// }

#[derive(PartialEq, Clone, Eq, Hash, PartialOrd)]
struct ValueRow(Vec<PartialValue>);

impl ValueRow {
    // fn into_partial_value(self) -> PartialValue {
    //     todo!()
    // }

    fn new(len: usize) -> Self {
        Self(vec![PartialValue::Top; len])
    }

    fn singleton(len: usize, idx: usize, v: PartialValue) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
    }

    fn singleton_from_row(r: &TypeRow, idx: usize, v: PartialValue) -> Self {
        Self::singleton(r.len(), idx, v)
    }

    fn top_from_row(r: &TypeRow) -> Self {
        Self::new(r.len())
    }

    fn iter<'b, 'a, H: HugrView>(
        &'b self,
        context: &'b Ctx<'a,H>,
        n: Node,
    ) -> impl Iterator<Item = (IncomingPort, &PartialValue)> + 'b {
        zip_eq(value_inputs(context, n), self.0.iter())
    }

    fn initialised(&self) -> bool {
        self.0.iter().all(|x| x != &PartialValue::Top)
    }
}

impl Lattice for ValueRow {
    fn meet(self, _other: Self) -> Self {
        todo!()
    }

    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn join_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.join_mut(v2);
        }
        changed
    }
}


type ArcCtx<'a, H: HugrView> = Arc<DataflowContext<'a,H>>;
type Ctx<'a, H: HugrView> = DataflowContext<'a,H>;

fn top_row<'a, H: HugrView>(context: &Ctx<'a, H>, n: Node) -> ValueRow {
    if let Some(sig) = context.hugr().signature(n) {
        ValueRow::new(sig.input_count())
    } else {
        ValueRow::new(0)
    }
}

fn singleton_in_row<'a, H: HugrView>(
    context: &Ctx<'a, H>,
    n: &Node,
    ip: &IncomingPort,
    v: &PartialValue,
) -> ValueRow {
    let Some(sig) = context.hugr().signature(*n) else {
        panic!("dougrulz");
    };
    if sig.input_count() <= ip.index() {
        panic!("bad port index: {} >= {}: {}", ip.index(), sig.input_count(), context.hugr().get_optype(*n).description());
    }
    ValueRow::singleton_from_row(
        &context.hugr().signature(*n).unwrap().input,
        ip.index(),
        v.clone(),
    )
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
enum IO { Input, Output }

fn value_inputs<'a,H: HugrView>(context: &Ctx<'a, H>, n: Node) -> impl Iterator<Item = IncomingPort> + 'a {
    context.hugr().in_value_types(n).map(|x| x.0)
}

fn value_outputs<'a,H: HugrView>(context: &Ctx<'a, H>, n: Node) -> impl Iterator<Item = OutgoingPort> + 'a {
    context.hugr().out_value_types(n).map(|x| x.0)
}

fn tail_loop_worker<'b, 'a,H: HugrView>(context: &Ctx<'a, H>, n: Node, output_p: IncomingPort, control_variant: usize, v: &'b PartialValue) -> impl Iterator<Item = (OutgoingPort, PartialValue)> + 'b {
    let tail_loop_op = context.get_optype(n).as_tail_loop().unwrap();
    let num_variant_vals = if control_variant == 0 {
        tail_loop_op.just_inputs.len()
    } else {
        tail_loop_op.just_outputs.len()
    };
    if output_p.index() == 0 {
        Either::Left((0..num_variant_vals).map(move |i| (i.into(), v.variant_field_value(control_variant, i))))
    } else {
        Either::Right(std::iter::once(((num_variant_vals + output_p.index()).into(), v.clone())))
    }
}

ascent::ascent! {
    struct Dataflow<'a, H: HugrView>;
    relation context(ArcCtx<'a, H>);
    relation node(ArcCtx<'a, H>, Node);
    relation in_wire(ArcCtx<'a,H>, Node, IncomingPort);
    relation out_wire(ArcCtx<'a,H>, Node, OutgoingPort);
    lattice out_wire_value(ArcCtx<'a,H>, Node, OutgoingPort, PartialValue);
    lattice node_in_value_row(ArcCtx<'a,H>, Node, ValueRow);
    lattice in_wire_value(ArcCtx<'a,H>, Node, IncomingPort, PartialValue);

    node(c, n) <-- context(c), for n in c.nodes();

    in_wire(c, n,p) <-- node(c, n), for p in value_inputs(c, *n);

    out_wire(c, n,p) <-- node(c, n), for p in value_outputs(c, *n);

    // All out wire values are initialised to Top. If any value is Top after
    // running we can infer that execution never reaches that value.
    out_wire_value(c, n,p, PartialValue::Top) <-- out_wire(c, n,p);

    in_wire_value(c, n, ip, v) <-- in_wire(c, n, ip),
        if let Some((m,op)) = c.single_linked_output(*n, *ip),
        out_wire_value(c, m, op, v);


    node_in_value_row(c, n, top_row(c, *n)) <-- node(c, n);
    node_in_value_row(c, n, singleton_in_row(c, n, p, v)) <-- in_wire_value(c, n, p, v);

    // LoadConstant
    relation load_constant_node(ArcCtx<'a, H>, Node);
    load_constant_node(c, n) <-- node(c, n), if c.get_optype(*n).is_load_constant();

    out_wire_value(c, n, 0.into(), PartialValue::from_load_constant(c, *n)) <--
        load_constant_node(c, n);

    // MakeTuple
    relation make_tuple_node(ArcCtx<'a,H>, Node);
    make_tuple_node(c, n) <-- node(c, n), if c.get_optype(*n).is_make_tuple();

    out_wire_value(c, n, 0.into(), PartialValue::tuple_from_value_row(vs)) <--
        make_tuple_node(c, n), node_in_value_row(c, n, vs);

    // UnpackTuple
    relation unpack_tuple_node(ArcCtx<'a, H>, Node);
    unpack_tuple_node(c,n) <-- node(c, n), if c.get_optype(*n).is_unpack_tuple();

    out_wire_value(c, n, p, v.tuple_field_value(p.index())) <-- unpack_tuple_node(c, n), in_wire_value(c, n, IncomingPort::from(0), v), out_wire(c, n, p);

    // DFG
    relation dfg_node(ArcCtx<'a, H>, Node);
    dfg_node(c,n) <-- node(c, n), if c.get_optype(*n).is_dfg();
    relation dfg_io_node(ArcCtx<'a, H>, Node, Node, IO);
    dfg_io_node(c,dfg,n,io) <-- dfg_node(c,dfg),
        if let Some([i,o]) = c.get_io(*dfg),
        for (n, io) in [(i, IO::Input), (o, IO::Output)];

    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <--
        dfg_io_node(c,dfg,i, IO::Input), in_wire_value(c, dfg, p, v);
    out_wire_value(c, dfg, OutgoingPort::from(p.index()), v) <--
        dfg_io_node(c,dfg,o, IO::Output), in_wire_value(c, o, p, v);


    // TailLoop
    relation tail_loop_node(ArcCtx<'a, H>, Node);
    tail_loop_node(c,n) <-- node(c, n), if c.get_optype(*n).is_tail_loop();
    relation tail_loop_io_node(ArcCtx<'a, H>, Node, Node, IO);
    tail_loop_io_node(c,tl,n, io) <-- tail_loop_node(c,tl),
        if let Some([i,o]) = c.get_io(*tl),
        for (n,io) in [(i,IO::Input), (o, IO::Output)];

    // inputs of tail loop propagate to Input node of child region
    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <--
        tail_loop_io_node(c,tl,i, IO::Input), in_wire_value(c, tl, p, v);
    // Output node of child region propagate to Input node of child region
    out_wire_value(c, i, input_p, v) <--
        tail_loop_io_node(c,tl,i, IO::Input),
        tail_loop_io_node(c,tl,o, IO::Output),
        in_wire_value(c, o, output_p, output_v),
        for (input_p, v) in tail_loop_worker(c, *tl, *output_p, 0, output_v);
    // Output node of child region propagate to outputs of tail loop
    out_wire_value(c, tl, p, v) <--
        tail_loop_io_node(c,tl,o, IO::Output),
        in_wire_value(c, o, output_p, output_v),
        for (p, v) in tail_loop_worker(c, *tl, *output_p, 1, output_v);


}

impl<'a, H: HugrView> Dataflow<'a, H> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run_hugr(&mut self, hugr: &'a H) -> ArcCtx<'a,H> {
        let context = context::DataflowContext::new(hugr);
        self.context.push((context.clone(),));
        self.run();
        context
    }

    pub fn read_out_wire_partial_value(&self, context: &Ctx<'a,H>, w: Wire) -> Option<PartialValue> {
        self.out_wire_value.iter().find_map(|(c,n,p,v)| (c.as_ref() == context && &w.node() == n && &w.source() == p).then_some(v.clone()))
    }

    pub fn read_out_wire_value(&self, context: &Ctx<'a,H>, w: Wire) -> Option<Value> {
        // dbg!(&w);
        let pv = self.read_out_wire_partial_value(context, w)?;
        // dbg!(&pv);
        let (_, typ) = context.hugr().out_value_types(w.node()).find(|(p,_)| *p == w.source()).unwrap();
        pv.try_into_value(&typ).ok()
    }
}

#[cfg(test)]
mod test {
    use hugr_core::{builder::{DFGBuilder, Dataflow, HugrBuilder, SubContainer}, extension::{prelude::BOOL_T, EMPTY_REG}, ops::{UnpackTuple, Value}, type_row, types::{FunctionType, SumType}};

    use crate::const_fold2::datalog::PartialValue;


    #[test]
    fn test_make_tuple() {
        let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
        let v1 = builder.add_load_value(Value::false_val());
        let v2 = builder.add_load_value(Value::true_val());
        let v3 = builder.make_tuple([v1,v2]).unwrap();
        let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

        let mut machine = super::Dataflow::new();
        let c = machine.run_hugr(&hugr);

        let x = machine.read_out_wire_value(&c, v3).unwrap();
        assert_eq!(x, Value::tuple([Value::false_val(), Value::true_val()]));
    }

    #[test]
    fn test_unpack_tuple() {
        let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
        let v1 = builder.add_load_value(Value::false_val());
        let v2 = builder.add_load_value(Value::true_val());
        let v3 = builder.make_tuple([v1,v2]).unwrap();
        let [o1,o2] = builder.add_dataflow_op(UnpackTuple::new(type_row![BOOL_T,BOOL_T]), [v3]).unwrap().outputs_arr();
        let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

        let mut machine = super::Dataflow::new();
        let c = machine.run_hugr(&hugr);

        let o1_r = machine.read_out_wire_value(&c, o1).unwrap();
        assert_eq!(o1_r, Value::false_val());
        let o2_r = machine.read_out_wire_value(&c, o2).unwrap();
        assert_eq!(o2_r, Value::true_val());
    }

    #[test]
    fn test_unpack_const() {
        let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
        let v1 = builder.add_load_value(Value::tuple([Value::true_val()]));
        let [o] = builder.add_dataflow_op(UnpackTuple::new(type_row![BOOL_T]), [v1]).unwrap().outputs_arr();
        let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

        let mut machine = super::Dataflow::new();
        let c = machine.run_hugr(&hugr);

        let o_r = machine.read_out_wire_value(&c, o).unwrap();
        assert_eq!(o_r, Value::true_val());
    }

    #[test]
    fn test_tail_loop_never_iterates() {
        let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
        let r_v = Value::unit_sum(3,6).unwrap();
        let r_w = builder.add_load_value(Value::sum(1, [r_v.clone()], SumType::new([type_row![], r_v.get_type().into()])).unwrap());
        let tlb = builder.tail_loop_builder([], [], vec![r_v.get_type()].into()).unwrap();
        let [tl_o] = tlb.finish_with_outputs(r_w, []).unwrap().outputs_arr();
        let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

        let mut machine = super::Dataflow::new();
        let c = machine.run_hugr(&hugr);
        // dbg!(&machine.tail_loop_io_node);
        // dbg!(&machine.out_wire_value);

        let o_r = machine.read_out_wire_value(&c, tl_o).unwrap();
        assert_eq!(o_r, r_v);
    }

    #[test]
    fn test_tail_loop_always_iterates() {
        let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
        let r_w = builder.add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
        let tlb = builder.tail_loop_builder([], [], vec![BOOL_T].into()).unwrap();
        let [tl_o] = tlb.finish_with_outputs(r_w, []).unwrap().outputs_arr();
        let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

        let mut machine = super::Dataflow::new();
        let c = machine.run_hugr(&hugr);
        // dbg!(&machine.tail_loop_io_node);
        // dbg!(&machine.out_wire_value);

        let o_r = machine.read_out_wire_partial_value(&c, tl_o).unwrap();
        assert_eq!(o_r, PartialValue::Top);
    }
}

// fn tc(hugr: &impl HugrView, node: Node) -> Vec<(Node, OutgoingPort, PartialValue)> {
//     assert!(OpTag::DataflowParent.is_superset(hugr.get_optype(node).tag()));
//     let d = DescendantsGraph::<'_, Node>::try_new(hugr, node).unwrap();
//     let mut cache = ValueCache::new();

//     let singleton_in_row = |n: &Node, ip: &IncomingPort, v: &PartialValue| -> ValueRow {
//         ValueRow::singleton_from_row(&hugr.signature(*n).unwrap().input, ip.index(), v.clone())
//     };

//     let top_row = |n: &Node| -> ValueRow {
//         ValueRow::top_from_row(&hugr.signature(*n).unwrap().input)
//     };
//     // ascent! {
//     //     relation node(Node) = d.nodes().map(|x| (x,)).collect_vec();

//     //     relation in_wire(Node, IncomingPort);
//     //     in_wire(n,p) <-- node(n), for p in d.node_inputs(*n);

//     //     relation out_wire(Node, OutgoingPort);
//     //     out_wire(n,p) <-- node(n), for p in d.node_outputs(*n);

//     //     lattice out_wire_value(Node, OutgoingPort, PartialValue);
//     //     out_wire_value(n,p, PartialValue::Top) <-- out_wire(n,p);

//     //     lattice node_in_value_row(Node, ValueRow);
//     //     node_in_value_row(n, top_row(n)) <-- node(n);
//     //     node_in_value_row(n, singleton_in_row(n,ip,v)) <-- in_wire(n, ip),
//     //         if let Some((m,op)) = hugr.single_linked_output(*n, *ip),
//     //         out_wire_value(m, op, v);

//     //     lattice in_wire_value(Node, IncomingPort, PartialValue);
//     //     in_wire_value(n,p,v) <-- node_in_value_row(n, vr), for (p,v) in vr.iter(hugr,*n);

//     //     relation load_constant_node(Node);
//     //     load_constant_node(n) <-- node(n), if hugr.get_optype(*n).is_load_constant();

//     //     out_wire_value(n, 0.into(), PartialValue::from_load_constant(&mut cache, hugr, *n)) <--
//     //         load_constant_node(n);

//     //     relation make_tuple_node(Node);
//     //     make_tuple_node(n) <-- node(n), if hugr.get_optype(*n).is_make_tuple();

//     //     out_wire_value(n,0.into(), PartialValue::tuple_from_value_row(vs)) <--
//     //         make_tuple_node(n), node_in_value_row(n, vs);

//     // }.out_wire_value
// }

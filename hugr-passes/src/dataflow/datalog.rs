//! [ascent] datalog implementation of analysis.
//! Since ascent-(macro-)generated code generates a bunch of warnings,
//! keep code in here to a minimum.
#![allow(
    clippy::clone_on_copy,
    clippy::unused_enumerate_index,
    clippy::collapsible_if
)]

use ascent::lattice::{BoundedLattice, Lattice};
use itertools::zip_eq;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use hugr_core::extension::prelude::{MakeTuple, UnpackTuple};
use hugr_core::ops::{OpTrait, OpType};
use hugr_core::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _};

use super::{AbstractValue, DFContext, PartialValue};

type PV<V> = PartialValue<V>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IO {
    Input,
    Output,
}

ascent::ascent! {
    pub(super) struct AscentProgram<V: AbstractValue, C: DFContext<V>>;
    relation context(C);
    relation out_wire_value_proto(Node, OutgoingPort, PV<V>);

    relation node(C, Node);
    relation in_wire(C, Node, IncomingPort);
    relation out_wire(C, Node, OutgoingPort);
    relation parent_of_node(C, Node, Node);
    relation io_node(C, Node, Node, IO);
    lattice out_wire_value(C, Node, OutgoingPort, PV<V>);
    lattice node_in_value_row(C, Node, ValueRow<V>);
    lattice in_wire_value(C, Node, IncomingPort, PV<V>);

    node(c, n) <-- context(c), for n in c.nodes();

    in_wire(c, n,p) <-- node(c, n), for p in value_inputs(c.as_ref(), *n);

    out_wire(c, n,p) <-- node(c, n), for p in value_outputs(c.as_ref(), *n);

    parent_of_node(c, parent, child) <--
        node(c, child), if let Some(parent) = c.get_parent(*child);

    io_node(c, parent, child, io) <-- node(c, parent),
      if let Some([i,o]) = c.get_io(*parent),
      for (child,io) in [(i,IO::Input),(o,IO::Output)];
    // We support prepopulating out_wire_value via out_wire_value_proto.
    //
    // out wires that do not have prepopulation values are initialised to bottom.
    out_wire_value(c, n,p, PV::bottom()) <-- out_wire(c, n,p);
    out_wire_value(c, n, p, v) <-- out_wire(c,n,p) , out_wire_value_proto(n, p, v);

    in_wire_value(c, n, ip, v) <-- in_wire(c, n, ip),
        if let Some((m,op)) = c.single_linked_output(*n, *ip),
        out_wire_value(c, m, op, v);


    node_in_value_row(c, n, ValueRow::new(sig.input_count())) <-- node(c, n), if let Some(sig) = c.signature(*n);
    node_in_value_row(c, n, ValueRow::single_known(c.signature(*n).unwrap().input_count(), p.index(), v.clone())) <-- in_wire_value(c, n, p, v);

    out_wire_value(c, n, p, v) <--
       node(c, n),
       let op_t = c.get_optype(*n),
       if !op_t.is_container(),
       if let Some(sig) = op_t.dataflow_signature(),
       node_in_value_row(c, n, vs),
       if let Some(outs) = propagate_leaf_op(c, *n, &vs[..], sig.output_count()),
       for (p,v) in (0..).map(OutgoingPort::from).zip(outs);

    // DFG
    relation dfg_node(C, Node);
    dfg_node(c,n) <-- node(c, n), if c.get_optype(*n).is_dfg();

    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
      io_node(c, dfg, i, IO::Input), in_wire_value(c, dfg, p, v);

    out_wire_value(c, dfg, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
        io_node(c,dfg,o, IO::Output), in_wire_value(c, o, p, v);


    // TailLoop

    // inputs of tail loop propagate to Input node of child region
    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- node(c, tl),
        if c.get_optype(*tl).is_tail_loop(),
        io_node(c,tl,i, IO::Input),
        in_wire_value(c, tl, p, v);

    // Output node of child region propagate to Input node of child region
    out_wire_value(c, in_n, OutgoingPort::from(out_p), v) <-- node(c, tl_n),
        if let Some(tailloop) = c.get_optype(*tl_n).as_tail_loop(),
        io_node(c,tl_n,in_n, IO::Input),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node

        if let Some(fields) = out_in_row.unpack_first(0, tailloop.just_inputs.len()), // if it is possible for tag to be 0
        for (out_p, v) in fields.enumerate();

    // Output node of child region propagate to outputs of tail loop
    out_wire_value(c, tl_n, OutgoingPort::from(out_p), v) <-- node(c, tl_n),
        if let Some(tailloop) = c.get_optype(*tl_n).as_tail_loop(),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node
        if let Some(fields) = out_in_row.unpack_first(1, tailloop.just_outputs.len()), // if it is possible for the tag to be 1
        for (out_p, v) in fields.enumerate();

    // Conditional
    relation conditional_node(C, Node);
    relation case_node(C,Node,usize, Node);

    conditional_node (c,n)<-- node(c, n), if c.get_optype(*n).is_conditional();
    case_node(c,cond,i, case) <-- conditional_node(c,cond),
      for (i, case) in c.children(*cond).enumerate(),
      if c.get_optype(case).is_case();

    // inputs of conditional propagate into case nodes
    out_wire_value(c, i_node, OutgoingPort::from(out_p), v) <--
      case_node(c, cond, case_index, case),
      io_node(c, case, i_node, IO::Input),
      node_in_value_row(c, cond, in_row),
      let conditional = c.get_optype(*cond).as_conditional().unwrap(),
      if let Some(fields) = in_row.unpack_first(*case_index, conditional.sum_rows[*case_index].len()),
      for (out_p, v) in fields.enumerate();

    // outputs of case nodes propagate to outputs of conditional *if* case reachable
    out_wire_value(c, cond, OutgoingPort::from(o_p.index()), v) <--
      case_node(c, cond, i, case),
      in_wire_value(c, cond, IncomingPort::from(0), control),
      if control.supports_tag(*i),
      io_node(c, case, o, IO::Output),
      in_wire_value(c, o, o_p, v);

    lattice case_reachable(C, Node, Node, bool);
    case_reachable(c, cond, case, reachable) <-- case_node(c,cond,i,case),
        in_wire_value(c, cond, IncomingPort::from(0), v),
        let reachable = v.supports_tag(*i);

    // CFG
    relation cfg_node(C, Node);
    relation dfb_block(C, Node, Node);
    cfg_node(c,n) <-- node(c, n), if c.get_optype(*n).is_cfg();
    dfb_block(c,cfg,blk) <-- cfg_node(c, cfg), for blk in c.children(*cfg), if c.get_optype(blk).is_dataflow_block();

    // Where do the values "fed" along a control-flow edge come out?
    relation _cfg_succ_dest(C, Node, Node, Node);
    _cfg_succ_dest(c, cfg, blk, inp) <-- dfb_block(c, cfg, blk), io_node(c, blk, inp, IO::Input);
    _cfg_succ_dest(c, cfg, exit, cfg) <-- cfg_node(c, cfg), if let Some(exit) = c.children(*cfg).nth(1);

    // Inputs of CFG propagate to entry block
    out_wire_value(c, i_node, OutgoingPort::from(p.index()), v) <--
        cfg_node(c, cfg),
        if let Some(entry) = c.children(*cfg).next(),
        io_node(c, entry, i_node, IO::Input),
        in_wire_value(c, cfg, p, v);

    // Outputs of each block propagated to successor blocks or (if exit block) then CFG itself
    out_wire_value(c, dest, OutgoingPort::from(out_p), v) <--
        dfb_block(c, cfg, pred),
        let df_block = c.get_optype(*pred).as_dataflow_block().unwrap(),
        for (succ_n, succ) in c.output_neighbours(*pred).enumerate(),
        io_node(c, pred, out_n, IO::Output),
        _cfg_succ_dest(c, cfg, succ, dest),
        node_in_value_row(c, out_n, out_in_row),
        if let Some(fields) = out_in_row.unpack_first(succ_n, df_block.sum_rows.get(succ_n).unwrap().len()),
        for (out_p, v) in fields.enumerate();
}

fn propagate_leaf_op<V: AbstractValue>(
    c: &impl DFContext<V>,
    n: Node,
    ins: &[PV<V>],
    num_outs: usize,
) -> Option<ValueRow<V>> {
    match c.get_optype(n) {
        // Handle basics here. I guess (given the current interface) we could allow
        // DFContext to handle these but at the least we'd want these impls to be
        // easily available for reuse.
        op if op.cast::<MakeTuple>().is_some() => Some(ValueRow::from_iter([PV::new_variant(
            0,
            ins.iter().cloned(),
        )])),
        op if op.cast::<UnpackTuple>().is_some() => {
            let [tup] = ins.iter().collect::<Vec<_>>().try_into().unwrap();
            tup.variant_values(0, value_outputs(c.as_ref(), n).count())
                .map(ValueRow::from_iter)
        }
        OpType::Tag(t) => Some(ValueRow::from_iter([PV::new_variant(
            t.tag,
            ins.iter().cloned(),
        )])),
        OpType::Input(_) | OpType::Output(_) | OpType::ExitBlock(_) => None, // handled by parent
        OpType::Const(_) => None, // handled by LoadConstant:
        OpType::LoadConstant(load_op) => {
            assert!(ins.is_empty()); // static edge, so need to find constant
            let const_node = c
                .single_linked_output(n, load_op.constant_port())
                .unwrap()
                .0;
            let const_val = c.get_optype(const_node).as_const().unwrap().value();
            Some(ValueRow::single_known(
                1,
                0,
                c.value_from_const(n, const_val),
            ))
        }
        OpType::ExtensionOp(e) => {
            // Interpret op. Default is we know nothing about the outputs (they still happen!)
            let mut outs = vec![PartialValue::Top; num_outs];
            // It'd be nice to convert these to [(IncomingPort, Value)] to pass to the context,
            // thus keeping PartialValue hidden, but AbstractValues
            // are not necessarily convertible to Value.
            c.interpret_leaf_op(n, e, ins, &mut outs[..]);
            Some(ValueRow::from_iter(outs))
        }
        o => todo!("Unhandled: {:?}", o), // At least CallIndirect, and OpType is "non-exhaustive"
    }
}

fn value_inputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = IncomingPort> + '_ {
    h.in_value_types(n).map(|x| x.0)
}

fn value_outputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = OutgoingPort> + '_ {
    h.out_value_types(n).map(|x| x.0)
}

// Wrap a (known-length) row of values into a lattice. Perhaps could be part of partial_value.rs?

#[derive(PartialEq, Clone, Eq, Hash)]
struct ValueRow<V>(Vec<PartialValue<V>>);

impl<V: AbstractValue> ValueRow<V> {
    fn new(len: usize) -> Self {
        Self(vec![PartialValue::bottom(); len])
    }

    fn single_known(len: usize, idx: usize, v: PartialValue<V>) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
    }

    pub fn unpack_first(
        &self,
        variant: usize,
        len: usize,
    ) -> Option<impl Iterator<Item = PartialValue<V>>> {
        let vals = self[0].variant_values(variant, len)?;
        Some(vals.into_iter().chain(self.0[1..].to_owned()))
    }
}

impl<V> FromIterator<PartialValue<V>> for ValueRow<V> {
    fn from_iter<T: IntoIterator<Item = PartialValue<V>>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<V: PartialEq> PartialOrd for ValueRow<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<V: AbstractValue> Lattice for ValueRow<V> {
    fn join_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.join_mut(v2);
        }
        changed
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.meet_mut(v2);
        }
        changed
    }
}

impl<V> IntoIterator for ValueRow<V> {
    type Item = PartialValue<V>;

    type IntoIter = <Vec<PartialValue<V>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<V, Idx> Index<Idx> for ValueRow<V>
where
    Vec<PartialValue<V>>: Index<Idx>,
{
    type Output = <Vec<PartialValue<V>> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
    }
}

impl<V, Idx> IndexMut<Idx> for ValueRow<V>
where
    Vec<PartialValue<V>>: IndexMut<Idx>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

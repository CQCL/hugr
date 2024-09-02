#![allow(
    clippy::clone_on_copy,
    clippy::unused_enumerate_index,
    clippy::collapsible_if
)]

use ascent::lattice::BoundedLattice;
use hugr_core::extension::prelude::{MakeTuple, UnpackTuple};
use std::collections::HashMap;
use std::hash::Hash;

use hugr_core::ops::{OpType, Value};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

mod utils;

// TODO separate this into its own analysis?
use utils::TailLoopTermination;

use super::partial_value::AbstractValue;
use super::value_row::ValueRow;
type PV<V> = super::partial_value::PartialValue<V>;

pub trait DFContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    fn interpret_leaf_op(&self, node: Node, ins: &[PV<V>]) -> Option<ValueRow<V>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IO {
    Input,
    Output,
}

ascent::ascent! {
    struct AscentProgram<V: AbstractValue, C: DFContext<V>>;
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

    in_wire(c, n,p) <-- node(c, n), for p in utils::value_inputs(c.as_ref(), *n);

    out_wire(c, n,p) <-- node(c, n), for p in utils::value_outputs(c.as_ref(), *n);

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


    node_in_value_row(c, n, ValueRow::new(utils::input_count(c.as_ref(), *n))) <-- node(c, n);
    node_in_value_row(c, n, ValueRow::single_known(c.signature(*n).unwrap().input.len(), p.index(), v.clone())) <-- in_wire_value(c, n, p, v);

    out_wire_value(c, n, p, v) <--
       node(c, n),
       if !c.get_optype(*n).is_container(),
       node_in_value_row(c, n, vs),
       if let Some(outs) = propagate_leaf_op(c, *n, &vs[..]),
       for (p,v) in (0..).map(OutgoingPort::from).zip(outs);

    // DFG
    relation dfg_node(C, Node);
    dfg_node(c,n) <-- node(c, n), if c.get_optype(*n).is_dfg();

    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
      io_node(c, dfg, i, IO::Input), in_wire_value(c, dfg, p, v);

    out_wire_value(c, dfg, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
        io_node(c,dfg,o, IO::Output), in_wire_value(c, o, p, v);


    // TailLoop
    relation tail_loop_node(C, Node);
    tail_loop_node(c,n) <-- node(c, n), if c.get_optype(*n).is_tail_loop();

    // inputs of tail loop propagate to Input node of child region
    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- tail_loop_node(c, tl),
        io_node(c,tl,i, IO::Input), in_wire_value(c, tl, p, v);

    // Output node of child region propagate to Input node of child region
    out_wire_value(c, in_n, out_p, v) <-- tail_loop_node(c, tl_n),
        io_node(c,tl_n,in_n, IO::Input),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node
        if let Some(tailloop) = c.get_optype(*tl_n).as_tail_loop(),
        if let Some(fields) = out_in_row.unpack_first(0, tailloop.just_inputs.len()), // if it is possible for tag to be 0
        for (out_p, v) in (0..).map(OutgoingPort::from).zip(fields);

    // Output node of child region propagate to outputs of tail loop
    out_wire_value(c, tl_n, out_p, v) <-- tail_loop_node(c, tl_n),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node
        if let Some(tailloop) = c.get_optype(*tl_n).as_tail_loop(),
        if let Some(fields) = out_in_row.unpack_first(1, tailloop.just_outputs.len()), // if it is possible for the tag to be 1
        for (out_p, v) in (0..).map(OutgoingPort::from).zip(fields);

    // Conditional
    relation conditional_node(C, Node);
    relation case_node(C,Node,usize, Node);

    conditional_node (c,n)<-- node(c, n), if c.get_optype(*n).is_conditional();
    case_node(c,cond,i, case) <-- conditional_node(c,cond),
      for (i, case) in c.children(*cond).enumerate(),
      if c.get_optype(case).is_case();

    // inputs of conditional propagate into case nodes
    out_wire_value(c, i_node, i_p, v) <--
      case_node(c, cond, case_index, case),
      io_node(c, case, i_node, IO::Input),
      node_in_value_row(c, cond, in_row),
      //in_wire_value(c, cond, cond_in_p, cond_in_v),
      if let Some(conditional) = c.get_optype(*cond).as_conditional(),
      if let Some(fields) = in_row.unpack_first(*case_index, conditional.sum_rows[*case_index].len()),
      for (i_p, v) in (0..).map(OutgoingPort::from).zip(fields);

    // outputs of case nodes propagate to outputs of conditional
    out_wire_value(c, cond, OutgoingPort::from(o_p.index()), v) <--
      case_node(c, cond, _, case),
      io_node(c, case, o, IO::Output),
      in_wire_value(c, o, o_p, v);

    lattice case_reachable(C, Node, Node, bool);
    case_reachable(c, cond, case, reachable) <-- case_node(c,cond,i,case),
        in_wire_value(c, cond, IncomingPort::from(0), v),
        let reachable = v.supports_tag(*i);

}

fn propagate_leaf_op<V: AbstractValue>(
    c: &impl DFContext<V>,
    n: Node,
    ins: &[PV<V>],
) -> Option<ValueRow<V>> {
    match c.get_optype(n) {
        // Handle basics here. I guess (given the current interface) we could allow
        // DFContext to handle these but at the least we'd want these impls to be
        // easily available for reuse.
        op if op.cast::<MakeTuple>().is_some() => {
            Some(ValueRow::from_iter([PV::variant(0, ins.iter().cloned())]))
        }
        op if op.cast::<UnpackTuple>().is_some() => {
            let [tup] = ins.iter().collect::<Vec<_>>().try_into().unwrap();
            tup.variant_values(0, utils::value_outputs(c.as_ref(), n).count())
                .map(ValueRow::from_iter)
        }
        OpType::Tag(t) => Some(ValueRow::from_iter([PV::variant(
            t.tag,
            ins.iter().cloned(),
        )])),
        OpType::Input(_) | OpType::Output(_) => None, // handled by parent
        // It'd be nice to convert these to [(IncomingPort, Value)] to pass to the context,
        // thus keeping PartialValue hidden, but AbstractValues
        // are not necessarily convertible to Value!
        _ => c.interpret_leaf_op(n, ins).map(ValueRow::from_iter),
    }
}

pub struct Machine<V: AbstractValue, C: DFContext<V>>(
    AscentProgram<V, C>,
    Option<HashMap<Wire, PV<V>>>,
);

/// derived-Default requires the context to be Defaultable, which is unnecessary
impl<V: AbstractValue, C: DFContext<V>> Default for Machine<V, C> {
    fn default() -> Self {
        Self(Default::default(), None)
    }
}

/// Usage:
/// 1. [Self::new()]
/// 2. Zero or more [Self::propolutate_out_wires] with initial values
/// 3. Exactly one [Self::run] to do the analysis
/// 4. Results then available via [Self::read_out_wire_partial_value] and [Self::read_out_wire_value]
impl<V: AbstractValue, C: DFContext<V>> Machine<V, C> {
    pub fn propolutate_out_wires(&mut self, wires: impl IntoIterator<Item = (Wire, PV<V>)>) {
        assert!(self.1.is_none());
        self.0
            .out_wire_value_proto
            .extend(wires.into_iter().map(|(w, v)| (w.node(), w.source(), v)));
    }

    pub fn run(&mut self, context: C) {
        assert!(self.1.is_none());
        self.0.context.push((context,));
        self.0.run();
        self.1 = Some(
            self.0
                .out_wire_value
                .iter()
                .map(|(_, n, p, v)| (Wire::new(*n, *p), v.clone()))
                .collect(),
        )
    }

    pub fn read_out_wire_partial_value(&self, w: Wire) -> Option<PV<V>> {
        self.1.as_ref().unwrap().get(&w).cloned()
    }

    pub fn tail_loop_terminates(&self, hugr: impl HugrView, node: Node) -> TailLoopTermination {
        assert!(hugr.get_optype(node).is_tail_loop());
        let [_, out] = hugr.get_io(node).unwrap();
        TailLoopTermination::from_control_value(
            self.0
                .in_wire_value
                .iter()
                .find_map(|(_, n, p, v)| (*n == out && p.index() == 0).then_some(v))
                .unwrap(),
        )
    }

    pub fn case_reachable(&self, hugr: impl HugrView, case: Node) -> bool {
        assert!(hugr.get_optype(case).is_case());
        let cond = hugr.get_parent(case).unwrap();
        assert!(hugr.get_optype(cond).is_conditional());
        self.0
            .case_reachable
            .iter()
            .find_map(|(_, cond2, case2, i)| (&cond == cond2 && &case == case2).then_some(*i))
            .unwrap()
    }
}

impl<V: AbstractValue, C: DFContext<V>> Machine<V, C>
where
    Value: From<V>,
{
    pub fn read_out_wire_value(&self, hugr: impl HugrView, w: Wire) -> Option<Value> {
        // dbg!(&w);
        let pv = self.read_out_wire_partial_value(w)?;
        // dbg!(&pv);
        let (_, typ) = hugr
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        pv.try_into_value(&typ).ok()
    }
}

#[cfg(test)]
mod test;

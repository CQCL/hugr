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
use hugr_core::ops::OpType;
use hugr_core::types::Signature;
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IO {
    Input,
    Output,
}

/// Clients of the dataflow framework (particular analyses, such as constant folding)
/// must implement this trait (including providing an appropriate domain type `PV`).
pub trait DFContext<PV: AbstractValue>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    /// Given lattice values for each input, produce lattice values for (what we know of)
    /// the outputs. Returning `None` indicates nothing can be deduced.
    fn interpret_leaf_op(&self, node: Node, ins: &[PV]) -> Option<Vec<PV>>;
}

/// Values which can be the domain for dataflow analysis. Must be able to deconstructed
/// into (and constructed from) Sums as these determine control flow.
pub trait AbstractValue: BoundedLattice + Clone + Eq + Hash + std::fmt::Debug {
    /// Create a new instance representing a Sum with a single known tag
    /// and (recursive) representations of the elements within that tag.
    fn new_variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self;

    /// New instance of unit type (i.e. the only possible value, with no contents)
    fn new_unit() -> Self {
        Self::new_variant(0, [])
    }

    /// Test whether this value *might* be a Sum with the specified tag.
    fn supports_tag(&self, tag: usize) -> bool;

    /// If this value might be a Sum with the specified tag, return values
    /// describing the elements of the Sum, otherwise `None`.
    ///
    /// Implementations must hold the invariant that for all `x`, `tag` and `len`:
    ///  `x.variant_values(tag, len).is_some() == x.supports_tag(tag)`
    fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<Self>>;
}

ascent::ascent! {
    pub(super) struct AscentProgram<PV: AbstractValue, C: DFContext<PV>>;
    relation context(C);
    relation out_wire_value_proto(Node, OutgoingPort, PV);

    relation node(C, Node);
    relation in_wire(C, Node, IncomingPort);
    relation out_wire(C, Node, OutgoingPort);
    relation parent_of_node(C, Node, Node);
    relation io_node(C, Node, Node, IO);
    lattice out_wire_value(C, Node, OutgoingPort, PV);
    lattice node_in_value_row(C, Node, ValueRow<PV>);
    lattice in_wire_value(C, Node, IncomingPort, PV);

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


    node_in_value_row(c, n, ValueRow::new(input_count(c.as_ref(), *n))) <-- node(c, n);
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

fn propagate_leaf_op<PV: AbstractValue>(
    c: &impl DFContext<PV>,
    n: Node,
    ins: &[PV],
) -> Option<ValueRow<PV>> {
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
        OpType::Input(_) | OpType::Output(_) => None, // handled by parent
        // It'd be nice to convert these to [(IncomingPort, Value)] to pass to the context,
        // thus keeping PartialValue hidden, but AbstractValues
        // are not necessarily convertible to Value!
        _ => c.interpret_leaf_op(n, ins).map(ValueRow::from_iter),
    }
}

fn input_count(h: &impl HugrView, n: Node) -> usize {
    h.signature(n)
        .as_ref()
        .map(Signature::input_count)
        .unwrap_or(0)
}

fn value_inputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = IncomingPort> + '_ {
    h.in_value_types(n).map(|x| x.0)
}

fn value_outputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = OutgoingPort> + '_ {
    h.out_value_types(n).map(|x| x.0)
}

// Wrap a (known-length) row of values into a lattice. Perhaps could be part of partial_value.rs?

#[derive(PartialEq, Clone, Eq, Hash)]
struct ValueRow<PV>(Vec<PV>);

impl<PV: AbstractValue> ValueRow<PV> {
    pub fn new(len: usize) -> Self {
        Self(vec![PV::bottom(); len])
    }

    pub fn single_known(len: usize, idx: usize, v: PV) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
    }

    pub fn iter(&self) -> impl Iterator<Item = &PV> {
        self.0.iter()
    }

    pub fn unpack_first(
        &self,
        variant: usize,
        len: usize,
    ) -> Option<impl Iterator<Item = PV> + '_> {
        self[0]
            .variant_values(variant, len)
            .map(|vals| vals.into_iter().chain(self.iter().skip(1).cloned()))
    }

    // fn initialised(&self) -> bool {
    //     self.0.iter().all(|x| x != &PV::top())
    // }
}

impl<PV> FromIterator<PV> for ValueRow<PV> {
    fn from_iter<T: IntoIterator<Item = PV>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<V: PartialEq + PartialOrd> PartialOrd for ValueRow<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<V: AbstractValue> Lattice for ValueRow<V> {
    fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
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

    fn meet_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.meet_mut(v2);
        }
        changed
    }
}

impl<PV> IntoIterator for ValueRow<PV> {
    type Item = PV;

    type IntoIter = <Vec<PV> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<PV, Idx> Index<Idx> for ValueRow<PV>
where
    Vec<PV>: Index<Idx>,
{
    type Output = <Vec<PV> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
    }
}

impl<PV, Idx> IndexMut<Idx> for ValueRow<PV>
where
    Vec<PV>: IndexMut<Idx>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

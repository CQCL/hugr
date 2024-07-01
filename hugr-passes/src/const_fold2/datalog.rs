use ascent::lattice::{ord_lattice::OrdLattice, BoundedLattice, Dual, Lattice};
use delegate::delegate;
use itertools::{zip_eq, Itertools};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use either::Either;
use hugr_core::ops::{OpTag, OpTrait, Value};
use hugr_core::partial_value::{PartialValue, ValueHandle};
use hugr_core::types::{EdgeKind, FunctionType, SumType, Type, TypeEnum, TypeRow};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

mod context;
mod utils;

pub use context::{ArcDataflowContext, DFContext, ValueCache};
pub use utils::{TailLoopTermination, ValueRow, IO, PV};

ascent::ascent! {
    struct AscentProgram<C: DFContext>;
    relation context(C);
    relation out_wire_value_proto(Node, OutgoingPort, PV);

    relation node(C, Node);
    relation in_wire(C, Node, IncomingPort);
    relation out_wire(C, Node, OutgoingPort);
    relation parent_of_node(C, Node, Node);
    relation io_node(C, Node, Node, IO);
    lattice out_wire_value(C, Node, OutgoingPort, PV);
    lattice node_in_value_row(C, Node, ValueRow);
    lattice in_wire_value(C, Node, IncomingPort, PV);

    node(c, n) <-- context(c), for n in c.hugr().nodes();

    in_wire(c, n,p) <-- node(c, n), for p in utils::value_inputs(c, *n);

    out_wire(c, n,p) <-- node(c, n), for p in utils::value_outputs(c, *n);

    parent_of_node(c, parent, child) <--
        node(c, child), if let Some(parent) = c.hugr().get_parent(*child);

    io_node(c, parent, child, io) <-- node(c, parent),
      if let Some([i,o]) = c.hugr().get_io(*parent),
      for (child,io) in [(i,IO::Input),(o,IO::Output)];
    // We support prepopulating out_wire_value via out_wire_value_proto.
    //
    // out wires that do not have prepopulation values are initialised to bottom.
    out_wire_value(c, n,p, PV::bottom()) <-- out_wire(c, n,p);
    out_wire_value(c, n, p, v) <-- out_wire(c,n,p) , out_wire_value_proto(n, p, v);

    in_wire_value(c, n, ip, v) <-- in_wire(c, n, ip),
        if let Some((m,op)) = c.hugr().single_linked_output(*n, *ip),
        out_wire_value(c, m, op, v);


    node_in_value_row(c, n, utils::bottom_row(c, *n)) <-- node(c, n);
    node_in_value_row(c, n, utils::singleton_in_row(c, n, p, v.clone())) <-- in_wire_value(c, n, p, v);


    // Per node-type rules
    // TODO do all leaf ops with a rule
    // define `fn propagate_leaf_op(Context, Node, ValueRow) -> ValueRow

    // LoadConstant
    relation load_constant_node(C, Node);
    load_constant_node(c, n) <-- node(c, n), if c.hugr().get_optype(*n).is_load_constant();

    out_wire_value(c, n, 0.into(), utils::partial_value_from_load_constant(c, *n)) <--
        load_constant_node(c, n);


    // MakeTuple
    relation make_tuple_node(C, Node);
    make_tuple_node(c, n) <-- node(c, n), if c.hugr().get_optype(*n).is_make_tuple();

    out_wire_value(c, n, 0.into(), utils::partial_value_tuple_from_value_row(vs.clone())) <--
        make_tuple_node(c, n), node_in_value_row(c, n, vs);


    // UnpackTuple
    relation unpack_tuple_node(C, Node);
    unpack_tuple_node(c,n) <-- node(c, n), if c.hugr().get_optype(*n).is_unpack_tuple();

    out_wire_value(c, n, p, v.tuple_field_value(p.index())) <--
        unpack_tuple_node(c, n),
        in_wire_value(c, n, IncomingPort::from(0), v),
        out_wire(c, n, p);


    // DFG
    relation dfg_node(C, Node);
    dfg_node(c,n) <-- node(c, n), if c.hugr().get_optype(*n).is_dfg();

    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
      io_node(c, dfg, i, IO::Input), in_wire_value(c, dfg, p, v);

    out_wire_value(c, dfg, OutgoingPort::from(p.index()), v) <-- dfg_node(c,dfg),
        io_node(c,dfg,o, IO::Output), in_wire_value(c, o, p, v);


    // TailLoop
    relation tail_loop_node(C, Node);
    tail_loop_node(c,n) <-- node(c, n), if c.hugr().get_optype(*n).is_tail_loop();

    // inputs of tail loop propagate to Input node of child region
    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <-- tail_loop_node(c, tl),
        io_node(c,tl,i, IO::Input), in_wire_value(c, tl, p, v);

    // Output node of child region propagate to Input node of child region
    out_wire_value(c, in_n, out_p, v) <-- tail_loop_node(c, tl_n),
        io_node(c,tl_n,in_n, IO::Input),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node
        if out_in_row[0].supports_tag(0), // if it is possible for tag to be 0
        if let Some(tailloop) = c.hugr().get_optype(*tl_n).as_tail_loop(),
        let variant_len = tailloop.just_inputs.len(),
        for (out_p, v) in out_in_row.iter(c, *out_n).flat_map(
            |(input_p, v)| utils::outputs_for_variant(input_p, 0, variant_len, v)
        );

    // Output node of child region propagate to outputs of tail loop
    out_wire_value(c, tl_n, out_p, v) <-- tail_loop_node(c, tl_n),
        io_node(c,tl_n,out_n, IO::Output),
        node_in_value_row(c, out_n, out_in_row), // get the whole input row for the output node
        if out_in_row[0].supports_tag(1), // if it is possible for the tag to be 1
        if let Some(tailloop) = c.hugr().get_optype(*tl_n).as_tail_loop(),
        let variant_len = tailloop.just_outputs.len(),
        for (out_p, v) in out_in_row.iter(c, *out_n).flat_map(
            |(input_p, v)| utils::outputs_for_variant(input_p, 1, variant_len, v)
        );

    lattice tail_loop_termination(C,Node,TailLoopTermination);
    tail_loop_termination(c,tl_n,TailLoopTermination::bottom()) <--
        tail_loop_node(c,tl_n);
    tail_loop_termination(c,tl_n,TailLoopTermination::from_control_value(v)) <--
        tail_loop_node(c,tl_n),
        io_node(c,tl,out_n, IO::Output),
        in_wire_value(c, out_n, IncomingPort::from(0), v);


    // Conditional
    relation conditional_node(C, Node);
    relation case_node(C,Node,usize, Node);

    conditional_node (c,n)<-- node(c, n), if c.hugr().get_optype(*n).is_conditional();
    case_node(c,cond,i, case) <-- conditional_node(c,cond),
      for (i, case) in c.hugr().children(*cond).enumerate(),
      if c.hugr().get_optype(case).is_case();

    // inputs of conditional propagate into case nodes
    out_wire_value(c, i_node, i_p, v) <--
      case_node(c, cond, case_index, case),
      io_node(c, case, i_node, IO::Input),
      in_wire_value(c, cond, cond_in_p, cond_in_v),
      if let Some(conditional) = c.hugr().get_optype(*cond).as_conditional(),
      let variant_len = conditional.sum_rows[*case_index].len(),
      for (i_p, v) in utils::outputs_for_variant(*cond_in_p, *case_index, variant_len, cond_in_v);

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

struct Machine<'a, H: HugrView> {
    program: AscentProgram<ArcDataflowContext<'a, H>>,
    cache: Arc<Mutex<ValueCache>>,
}

impl<'a, H: HugrView> Machine<'a, H> {
    pub fn new() -> Self {
        Self {
            program: Default::default(),
            cache: ValueCache::new(),
        }
    }

    pub fn propolutate_out_wires(&mut self, wires: impl IntoIterator<Item=(Wire,PartialValue)>) {
        self.program.out_wire_value_proto.extend(wires.into_iter().map(|(w,v)| (w.node(), w.source(), v.into())));
    }

    pub fn run_hugr(&mut self, hugr: &'a H) -> ArcDataflowContext<'a, H> {
        let context = ArcDataflowContext::new(hugr, self.cache.clone());
        self.program.context.push((context.clone(),));
        self.program.run();
        context
    }

    pub fn read_out_wire_partial_value(
        &self,
        context: &ArcDataflowContext<'a, H>,
        w: Wire,
    ) -> Option<PartialValue> {
        self.program.out_wire_value.iter().find_map(|(c, n, p, v)| {
            (c == context && &w.node() == n && &w.source() == p).then(|| v.clone().into())
        })
    }

    pub fn read_out_wire_value(
        &self,
        context: &ArcDataflowContext<'a, H>,
        w: Wire,
    ) -> Option<Value> {
        // dbg!(&w);
        let pv = self.read_out_wire_partial_value(context, w)?;
        // dbg!(&pv);
        let (_, typ) = context
            .hugr()
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        pv.try_into_value(&typ).ok()
    }

    pub fn tail_loop_terminates(
        &self,
        context: &ArcDataflowContext<'a, H>,
        node: Node,
    ) -> TailLoopTermination {
        assert!(context.get_optype(node).is_tail_loop());
        self.program
            .tail_loop_termination
            .iter()
            .find_map(|(c, n, v)| (c == context && n == &node).then_some(*v))
            .unwrap()
    }

    pub fn case_reachable(&self,
        context: &ArcDataflowContext<'a, H>,
        case: Node,) -> bool {
        assert!(context.get_optype(case).is_case());
        let cond = context.hugr().get_parent(case).unwrap();
        assert!(context.get_optype(cond).is_conditional());
        self.program.case_reachable.iter().find_map(|(c,cond2,case2,i)| (c == context && &cond == cond2 && &case == case2).then_some(*i)).unwrap()
    }
}

#[cfg(test)]
mod test;

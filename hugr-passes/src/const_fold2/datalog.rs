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
    relation node(C, Node);
    relation in_wire(C, Node, IncomingPort);
    relation out_wire(C, Node, OutgoingPort);
    relation parent_of_node(C, Node, Node);
    lattice out_wire_value(C, Node, OutgoingPort, PV);
    lattice node_in_value_row(C, Node, ValueRow);
    lattice in_wire_value(C, Node, IncomingPort, PV);

    node(c, n) <-- context(c), for n in c.hugr().nodes();

    in_wire(c, n,p) <-- node(c, n), for p in utils::value_inputs(c, *n);

    out_wire(c, n,p) <-- node(c, n), for p in utils::value_outputs(c, *n);

    parent_of_node(c, parent, child) <--
        node(c, child), if let Some(parent) = c.hugr().get_parent(*child);

    // All out wire values are initialised to Bottom. If any value is Bottom after
    // running we can infer that execution never reaches that value.
    out_wire_value(c, n,p, PV::bottom()) <-- out_wire(c, n,p);

    in_wire_value(c, n, ip, v) <-- in_wire(c, n, ip),
        if let Some((m,op)) = c.hugr().single_linked_output(*n, *ip),
        out_wire_value(c, m, op, v);


    node_in_value_row(c, n, utils::bottom_row(c, *n)) <-- node(c, n);
    node_in_value_row(c, n, utils::singleton_in_row(c, n, p, v.clone())) <-- in_wire_value(c, n, p, v);

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

    out_wire_value(c, n, p, v.tuple_field_value(p.index())) <-- unpack_tuple_node(c, n), in_wire_value(c, n, IncomingPort::from(0), v), out_wire(c, n, p);

    // DFG
    relation dfg_node(C, Node);
    dfg_node(c,n) <-- node(c, n), if c.hugr().get_optype(*n).is_dfg();
    relation dfg_io_node(C, Node, Node, IO);
    dfg_io_node(c,dfg,n,io) <-- dfg_node(c,dfg),
        if let Some([i,o]) = c.hugr().get_io(*dfg),
        for (n, io) in [(i, IO::Input), (o, IO::Output)];

    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <--
        dfg_io_node(c,dfg,i, IO::Input), in_wire_value(c, dfg, p, v);
    out_wire_value(c, dfg, OutgoingPort::from(p.index()), v) <--
        dfg_io_node(c,dfg,o, IO::Output), in_wire_value(c, o, p, v);


    // TailLoop
    relation tail_loop_node(C, Node);
    tail_loop_node(c,n) <-- node(c, n), if c.hugr().get_optype(*n).is_tail_loop();
    relation tail_loop_io_node(C, Node, Node, IO);
    tail_loop_io_node(c,tl,n, io) <-- tail_loop_node(c,tl),
        if let Some([i,o]) = c.hugr().get_io(*tl),
        for (n,io) in [(i,IO::Input), (o, IO::Output)];

    // inputs of tail loop propagate to Input node of child region
    out_wire_value(c, i, OutgoingPort::from(p.index()), v) <--
        tail_loop_io_node(c,tl,i, IO::Input), in_wire_value(c, tl, p, v);


    // Output node of child region propagate to Input node of child region
    out_wire_value(c, i, input_p, v) <--
        tail_loop_io_node(c,tl,i, IO::Input),
        tail_loop_io_node(c,tl,o, IO::Output),
        in_wire_value(c, o, output_p, output_v),
        for (input_p, v) in utils::tail_loop_worker(c, *tl, *output_p, 0, output_v);
    // Output node of child region propagate to outputs of tail loop
    out_wire_value(c, tl, p, v) <--
        tail_loop_io_node(c,tl,o, IO::Output),
        in_wire_value(c, o, output_p, output_v),
        for (p, v) in utils::tail_loop_worker(c, *tl, *output_p, 1, output_v);

    lattice tail_loop_termination(C,Node,OrdLattice<TailLoopTermination>);
    tail_loop_termination(c,tl,TailLoopTermination::NeverTerminates.into()) <-- tail_loop_node(c,tl);
    tail_loop_termination(c,tl,TailLoopTermination::from_control_value(v).into()) <-- tail_loop_node(c,tl),
        tail_loop_io_node(c,tl,o, IO::Output),
        in_wire_value(c, o, Into::<IncomingPort>::into(0usize), v);
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
            .find_map(|(c, n, v)| (c == context && n == &node).then_some(v.0.clone()))
            .unwrap()
    }
}

#[cfg(test)]
mod test;

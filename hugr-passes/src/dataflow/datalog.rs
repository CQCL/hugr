//! [ascent] datalog implementation of analysis.

use ascent::lattice::BoundedLattice;
use itertools::Itertools;

use hugr_core::extension::prelude::{MakeTuple, UnpackTuple};
use hugr_core::ops::{OpTrait, OpType};
use hugr_core::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

use super::value_row::ValueRow;
use super::{AbstractValue, AnalysisResults, DFContext, PartialValue};

type PV<V> = PartialValue<V>;

/// Basic structure for performing an analysis. Usage:
/// 1. Get a new instance via [Self::default()]
/// 2. (Optionally / for tests) zero or more [Self::prepopulate_wire] with initial values
/// 3. Call [Self::run] to produce [AnalysisResults]
pub struct Machine<V: AbstractValue>(Vec<(Node, IncomingPort, PartialValue<V>)>);

/// derived-Default requires the context to be Defaultable, which is unnecessary
impl<V: AbstractValue> Default for Machine<V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<V: AbstractValue> Machine<V> {
    /// Provide initial values for some wires.
    // Likely for test purposes only - should we make non-pub or #[cfg(test)] ?
    pub fn prepopulate_wire(&mut self, h: &impl HugrView, wire: Wire, value: PartialValue<V>) {
        self.0.extend(
            h.linked_inputs(wire.node(), wire.source())
                .map(|(n, inp)| (n, inp, value.clone())),
        );
    }

    /// Run the analysis (iterate until a lattice fixpoint is reached),
    /// given initial values for some of the root node inputs.
    /// (Note that `in_values` will not be useful for `Case` or `DFB`-rooted Hugrs,
    /// but should handle other containers.)
    /// The context passed in allows interpretation of leaf operations.
    pub fn run<H: HugrView>(
        mut self,
        context: &impl DFContext<V>,
        hugr: H,
        in_values: impl IntoIterator<Item = (IncomingPort, PartialValue<V>)>,
    ) -> AnalysisResults<V, H> {
        let root = hugr.root();
        self.0
            .extend(in_values.into_iter().map(|(p, v)| (root, p, v)));
        run_datalog(self.0, context, hugr)
    }
}

pub(super) fn run_datalog<V: AbstractValue, H: HugrView>(
    in_wire_value_proto: Vec<(Node, IncomingPort, PV<V>)>,
    c: &impl DFContext<V>,
    hugr: H,
) -> AnalysisResults<V, H> {
    // ascent-(macro-)generated code generates a bunch of warnings,
    // keep code in here to a minimum.
    #![allow(
        clippy::clone_on_copy,
        clippy::unused_enumerate_index,
        clippy::collapsible_if
    )]
    let all_results = ascent::ascent_run! {
        pub(super) struct AscentProgram<V: AbstractValue>;
        relation node(Node);
        relation in_wire(Node, IncomingPort);
        relation out_wire(Node, OutgoingPort);
        relation parent_of_node(Node, Node);
        relation input_child(Node, Node);
        relation output_child(Node, Node);
        lattice out_wire_value(Node, OutgoingPort, PV<V>);
        lattice in_wire_value(Node, IncomingPort, PV<V>);
        lattice node_in_value_row(Node, ValueRow<V>);

        node(n) <-- for n in hugr.nodes();

        in_wire(n, p) <-- node(n), for (p,_) in hugr.in_value_types(*n); // Note, gets connected inports only
        out_wire(n, p) <-- node(n), for (p,_) in hugr.out_value_types(*n); // (and likewise)

        parent_of_node(parent, child) <--
            node(child), if let Some(parent) = hugr.get_parent(*child);

        input_child(parent, input) <-- node(parent), if let Some([input, _output]) = hugr.get_io(*parent);
        output_child(parent, output) <-- node(parent), if let Some([_input, output]) = hugr.get_io(*parent);

        // Initialize all wires to bottom
        out_wire_value(n, p, PV::bottom()) <-- out_wire(n, p);

        in_wire_value(n, ip, v) <-- in_wire(n, ip),
            if let Some((m, op)) = hugr.single_linked_output(*n, *ip),
            out_wire_value(m, op, v);

        // We support prepopulating in_wire_value via in_wire_value_proto.
        in_wire_value(n, p, PV::bottom()) <-- in_wire(n, p);
        in_wire_value(n, p, v) <-- for (n, p, v) in in_wire_value_proto.iter(),
          node(n),
          if let Some(sig) = hugr.signature(*n),
          if sig.input_ports().contains(p);

        node_in_value_row(n, ValueRow::new(sig.input_count())) <-- node(n), if let Some(sig) = hugr.signature(*n);
        node_in_value_row(n, ValueRow::single_known(hugr.signature(*n).unwrap().input_count(), p.index(), v.clone())) <-- in_wire_value(n, p, v);

        out_wire_value(n, p, v) <--
           node(n),
           let op_t = hugr.get_optype(*n),
           if !op_t.is_container(),
           if let Some(sig) = op_t.dataflow_signature(),
           node_in_value_row(n, vs),
           if let Some(outs) = propagate_leaf_op(c, &hugr, *n, &vs[..], sig.output_count()),
           for (p, v) in (0..).map(OutgoingPort::from).zip(outs);

        // DFG
        relation dfg_node(Node);
        dfg_node(n) <-- node(n), if hugr.get_optype(*n).is_dfg();

        out_wire_value(i, OutgoingPort::from(p.index()), v) <-- dfg_node(dfg),
          input_child(dfg, i), in_wire_value(dfg, p, v);

        out_wire_value(dfg, OutgoingPort::from(p.index()), v) <-- dfg_node(dfg),
            output_child(dfg, o), in_wire_value(o, p, v);


        // TailLoop

        // inputs of tail loop propagate to Input node of child region
        out_wire_value(i, OutgoingPort::from(p.index()), v) <-- node(tl),
            if hugr.get_optype(*tl).is_tail_loop(),
            input_child(tl, i),
            in_wire_value(tl, p, v);

        // Output node of child region propagate to Input node of child region
        out_wire_value(in_n, OutgoingPort::from(out_p), v) <-- node(tl),
            if let Some(tailloop) = hugr.get_optype(*tl).as_tail_loop(),
            input_child(tl, in_n),
            output_child(tl, out_n),
            node_in_value_row(out_n, out_in_row), // get the whole input row for the output node
            if let Some(fields) = out_in_row.unpack_first(0, tailloop.just_inputs.len()), // if it is possible for tag to be 0
            for (out_p, v) in fields.enumerate();

        // Output node of child region propagate to outputs of tail loop
        out_wire_value(tl, OutgoingPort::from(out_p), v) <-- node(tl),
            if let Some(tailloop) = hugr.get_optype(*tl).as_tail_loop(),
            output_child(tl, out_n),
            node_in_value_row(out_n, out_in_row), // get the whole input row for the output node
            if let Some(fields) = out_in_row.unpack_first(1, tailloop.just_outputs.len()), // if it is possible for the tag to be 1
            for (out_p, v) in fields.enumerate();

        // Conditional
        relation conditional_node(Node);
        relation case_node(Node, usize, Node);

        conditional_node(n)<-- node(n), if hugr.get_optype(*n).is_conditional();
        case_node(cond, i, case) <-- conditional_node(cond),
          for (i, case) in hugr.children(*cond).enumerate(),
          if hugr.get_optype(case).is_case();

        // inputs of conditional propagate into case nodes
        out_wire_value(i_node, OutgoingPort::from(out_p), v) <--
          case_node(cond, case_index, case),
          input_child(case, i_node),
          node_in_value_row(cond, in_row),
          let conditional = hugr.get_optype(*cond).as_conditional().unwrap(),
          if let Some(fields) = in_row.unpack_first(*case_index, conditional.sum_rows[*case_index].len()),
          for (out_p, v) in fields.enumerate();

        // outputs of case nodes propagate to outputs of conditional *if* case reachable
        out_wire_value(cond, OutgoingPort::from(o_p.index()), v) <--
          case_node(cond, _i, case),
          case_reachable(cond, case),
          output_child(case, o),
          in_wire_value(o, o_p, v);

        relation case_reachable(Node, Node);
        case_reachable(cond, case) <-- case_node(cond, i, case),
            in_wire_value(cond, IncomingPort::from(0), v),
            if v.supports_tag(*i);

        // CFG
        relation cfg_node(Node);
        cfg_node(n) <-- node(n), if hugr.get_optype(*n).is_cfg();

        // Reachability
        relation bb_reachable(Node, Node);
        bb_reachable(cfg, entry) <-- cfg_node(cfg), if let Some(entry) = hugr.children(*cfg).next();
        bb_reachable(cfg, bb) <-- cfg_node(cfg),
            bb_reachable(cfg, pred),
            output_child(pred, pred_out),
            in_wire_value(pred_out, IncomingPort::from(0), predicate),
            for (tag, bb) in hugr.output_neighbours(*pred).enumerate(),
            if predicate.supports_tag(tag);

        // Relation: in `CFG` <Node>, values fed along a control-flow edge to <Node>
        //     come out of Value outports of <Node>.
        relation _cfg_succ_dest(Node, Node, Node);
        _cfg_succ_dest(cfg, blk, inp) <-- cfg_node(cfg),
            for blk in hugr.children(*cfg),
            if hugr.get_optype(blk).is_dataflow_block(),
            input_child(blk, inp);
        _cfg_succ_dest(cfg, exit, cfg) <-- cfg_node(cfg), if let Some(exit) = hugr.children(*cfg).nth(1);

        // Inputs of CFG propagate to entry block
        out_wire_value(i_node, OutgoingPort::from(p.index()), v) <--
            cfg_node(cfg),
            if let Some(entry) = hugr.children(*cfg).next(),
            input_child(entry, i_node),
            in_wire_value(cfg, p, v);

        // Outputs of each reachable block propagated to successor block or (if exit block) then CFG itself
        out_wire_value(dest, OutgoingPort::from(out_p), v) <--
            bb_reachable(cfg, pred),
            if let Some(df_block) = hugr.get_optype(*pred).as_dataflow_block(),
            for (succ_n, succ) in hugr.output_neighbours(*pred).enumerate(),
            output_child(pred, out_n),
            _cfg_succ_dest(cfg, succ, dest),
            node_in_value_row(out_n, out_in_row),
            if let Some(fields) = out_in_row.unpack_first(succ_n, df_block.sum_rows.get(succ_n).unwrap().len()),
            for (out_p, v) in fields.enumerate();

        // Call
        relation func_call(Node, Node);
        func_call(call, func_defn) <--
            node(call),
            if hugr.get_optype(*call).is_call(),
            if let Some(func_defn) = hugr.static_source(*call);

        out_wire_value(inp, OutgoingPort::from(p.index()), v) <--
            func_call(call, func),
            input_child(func, inp),
            in_wire_value(call, p, v);

        out_wire_value(call, OutgoingPort::from(p.index()), v) <--
            func_call(call, func),
            output_child(func, outp),
            in_wire_value(outp, p, v);
    };
    let out_wire_values = all_results
        .out_wire_value
        .iter()
        .map(|(n, p, v)| (Wire::new(*n, *p), v.clone()))
        .collect();
    AnalysisResults {
        hugr,
        out_wire_values,
        in_wire_value: all_results.in_wire_value,
        case_reachable: all_results.case_reachable,
        bb_reachable: all_results.bb_reachable,
    }
}

fn propagate_leaf_op<V: AbstractValue>(
    c: &impl DFContext<V>,
    hugr: &impl HugrView,
    n: Node,
    ins: &[PV<V>],
    num_outs: usize,
) -> Option<ValueRow<V>> {
    match hugr.get_optype(n) {
        // Handle basics here. I guess (given the current interface) we could allow
        // DFContext to handle these but at the least we'd want these impls to be
        // easily available for reuse.
        op if op.cast::<MakeTuple>().is_some() => Some(ValueRow::from_iter([PV::new_variant(
            0,
            ins.iter().cloned(),
        )])),
        op if op.cast::<UnpackTuple>().is_some() => {
            let elem_tys = op.cast::<UnpackTuple>().unwrap().0;
            let [tup] = ins.iter().collect::<Vec<_>>().try_into().unwrap();
            tup.variant_values(0, elem_tys.len())
                .map(ValueRow::from_iter)
        }
        OpType::Tag(t) => Some(ValueRow::from_iter([PV::new_variant(
            t.tag,
            ins.iter().cloned(),
        )])),
        OpType::Input(_) | OpType::Output(_) | OpType::ExitBlock(_) => None, // handled by parent
        OpType::Call(_) => None,  // handled via Input/Output of FuncDefn
        OpType::Const(_) => None, // handled by LoadConstant:
        OpType::LoadConstant(load_op) => {
            assert!(ins.is_empty()); // static edge, so need to find constant
            let const_node = hugr
                .single_linked_output(n, load_op.constant_port())
                .unwrap()
                .0;
            let const_val = hugr.get_optype(const_node).as_const().unwrap().value();
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

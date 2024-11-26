//! [ascent] datalog implementation of analysis.

use std::collections::hash_map::RandomState;
use std::collections::HashSet; // Moves to std::hash in Rust 1.76

use ascent::lattice::BoundedLattice;
use itertools::Itertools;

use hugr_core::extension::prelude::{MakeTuple, UnpackTuple};
use hugr_core::ops::{OpTrait, OpType, TailLoop};
use hugr_core::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

use super::value_row::ValueRow;
use super::{
    partial_from_const, row_contains_bottom, AbstractValue, AnalysisResults, DFContext,
    PartialValue,
};

type PV<V> = PartialValue<V>;

/// Basic structure for performing an analysis. Usage:
/// 1. Get a new instance via [Self::default()]
/// 2. (Optionally) zero or more calls to [Self::prepopulate_wire] and/or
///    [Self::prepopulate_df_inputs] with initial values.
///    For example, to analyse a [Module](OpType::Module)-rooted Hugr as a library,
///    [Self::prepopulate_df_inputs] can be used on each externally-callable
///    [FuncDefn](OpType::FuncDefn) to set all inputs to [PartialValue::Top].
/// 3. Call [Self::run] to produce [AnalysisResults]
pub struct Machine<V: AbstractValue>(Vec<(Node, IncomingPort, PartialValue<V>)>);

/// derived-Default requires the context to be Defaultable, which is unnecessary
impl<V: AbstractValue> Default for Machine<V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<V: AbstractValue> Machine<V> {
    /// Provide initial values for a wire - these will be `join`d with any computed.
    pub fn prepopulate_wire(&mut self, h: &impl HugrView, w: Wire, v: PartialValue<V>) {
        self.0.extend(
            h.linked_inputs(w.node(), w.source())
                .map(|(n, inp)| (n, inp, v.clone())),
        );
    }

    /// Provide initial values for the inputs to a [DataflowParent](hugr_core::ops::OpTag::DataflowParent)
    /// (that is, values on the wires leaving the [Input](OpType::Input) child thereof).
    /// Any out-ports of said same `Input` node, not given values by `in_values`, are set to [PartialValue::Top].
    pub fn prepopulate_df_inputs(
        &mut self,
        h: &impl HugrView,
        parent: Node,
        in_values: impl IntoIterator<Item = (OutgoingPort, PartialValue<V>)>,
    ) {
        // Put values onto out-wires of Input node
        let [inp, _] = h.get_io(parent).unwrap();
        let mut vals = vec![PartialValue::Top; h.signature(inp).unwrap().output_types().len()];
        for (ip, v) in in_values {
            vals[ip.index()] = v;
        }
        for (i, v) in vals.into_iter().enumerate() {
            self.prepopulate_wire(h, Wire::new(inp, i), v);
        }
    }

    /// Run the analysis (iterate until a lattice fixpoint is reached),
    /// given initial values for some of the root node inputs. For a
    /// [Module](OpType::Module)-rooted Hugr, these are input to the function `"main"`.
    /// The context passed in allows interpretation of leaf operations.
    ///
    /// # Panics
    /// May panic in various ways if the Hugr is invalid;
    /// or if any `in_values` are provided for a module-rooted Hugr without a function `"main"`.
    pub fn run<H: HugrView>(
        mut self,
        context: &impl DFContext<V>,
        hugr: H,
        in_values: impl IntoIterator<Item = (IncomingPort, PartialValue<V>)>,
    ) -> AnalysisResults<V, H> {
        let mut in_values = in_values.into_iter();
        let root = hugr.root();
        // Some nodes do not accept values as dataflow inputs - for these
        // we must find the corresponding Input node.
        let input_node_parent = match hugr.get_optype(root) {
            OpType::Module(_) => {
                let main = hugr.children(root).find(|n| {
                    hugr.get_optype(*n)
                        .as_func_defn()
                        .is_some_and(|f| f.name == "main")
                });
                if main.is_none() && in_values.next().is_some() {
                    panic!("Cannot give inputs to module with no 'main'");
                }
                main
            }
            OpType::DataflowBlock(_) | OpType::Case(_) | OpType::FuncDefn(_) => Some(root),
            // Could also do Dfg above, but ok here too:
            _ => None, // Just feed into node inputs
        };
        // Any inputs we don't have values for, we must assume `Top` to ensure safety of analysis
        // (Consider: for a conditional that selects *either* the unknown input *or* value V,
        // analysis must produce Top == we-know-nothing, not `V` !)
        if let Some(p) = input_node_parent {
            self.prepopulate_df_inputs(
                &hugr,
                p,
                in_values.map(|(p, v)| (OutgoingPort::from(p.index()), v)),
            );
        } else {
            // Put values onto in-wires of root node, datalog will do the rest
            self.0.extend(in_values.map(|(p, v)| (root, p, v)));
            let got_inputs: HashSet<_, RandomState> = self
                .0
                .iter()
                .filter_map(|(n, p, _)| (n == &root).then_some(*p))
                .collect();
            for p in hugr.signature(root).unwrap_or_default().input_ports() {
                if !got_inputs.contains(&p) {
                    self.0.push((root, p, PartialValue::Top));
                }
            }
        }
        // Note/TODO, if analysis is running on a subregion then we should do similar
        // for any nonlocal edges providing values from outside the region.
        run_datalog(context, hugr, self.0)
    }
}

pub(super) fn run_datalog<V: AbstractValue, H: HugrView>(
    ctx: &impl DFContext<V>,
    hugr: H,
    in_wire_value_proto: Vec<(Node, IncomingPort, PV<V>)>,
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
        relation node(Node); // <Node> exists in the hugr
        relation in_wire(Node, IncomingPort); // <Node> has an <IncomingPort> of `EdgeKind::Value`
        relation out_wire(Node, OutgoingPort); // <Node> has an <OutgoingPort> of `EdgeKind::Value`
        relation parent_of_node(Node, Node); // <Node> is parent of <Node>
        relation input_child(Node, Node); // <Node> has 1st child <Node> that is its `Input`
        relation output_child(Node, Node); // <Node> has 2nd child <Node> that is its `Output`
        lattice out_wire_value(Node, OutgoingPort, PV<V>); // <Node> produces, on <OutgoingPort>, the value <PV>
        lattice in_wire_value(Node, IncomingPort, PV<V>); // <Node> receives, on <IncomingPort>, the value <PV>
        lattice node_in_value_row(Node, ValueRow<V>); // <Node>'s inputs are <ValueRow>

        node(n) <-- for n in hugr.nodes();

        in_wire(n, p) <-- node(n), for (p,_) in hugr.in_value_types(*n); // Note, gets connected inports only
        out_wire(n, p) <-- node(n), for (p,_) in hugr.out_value_types(*n); // (and likewise)

        parent_of_node(parent, child) <--
            node(child), if let Some(parent) = hugr.get_parent(*child);

        input_child(parent, input) <-- node(parent), if let Some([input, _output]) = hugr.get_io(*parent);
        output_child(parent, output) <-- node(parent), if let Some([_input, output]) = hugr.get_io(*parent);

        // Initialize all wires to bottom
        out_wire_value(n, p, PV::bottom()) <-- out_wire(n, p);

        // Outputs to inputs
        in_wire_value(n, ip, v) <-- in_wire(n, ip),
            if let Some((m, op)) = hugr.single_linked_output(*n, *ip),
            out_wire_value(m, op, v);

        // Prepopulate in_wire_value from in_wire_value_proto.
        in_wire_value(n, p, PV::bottom()) <-- in_wire(n, p);
        in_wire_value(n, p, v) <-- for (n, p, v) in in_wire_value_proto.iter(),
          node(n),
          if let Some(sig) = hugr.signature(*n),
          if sig.input_ports().contains(p);

        // Assemble node_in_value_row from in_wire_value's
        node_in_value_row(n, ValueRow::new(sig.input_count())) <-- node(n), if let Some(sig) = hugr.signature(*n);
        node_in_value_row(n, ValueRow::new(hugr.signature(*n).unwrap().input_count()).set(p.index(), v.clone())) <-- in_wire_value(n, p, v);

        // Interpret leaf ops
        out_wire_value(n, p, v) <--
           node(n),
           let op_t = hugr.get_optype(*n),
           if !op_t.is_container(),
           if let Some(sig) = op_t.dataflow_signature(),
           node_in_value_row(n, vs),
           if let Some(outs) = propagate_leaf_op(ctx, &hugr, *n, &vs[..], sig.output_count()),
           for (p, v) in (0..).map(OutgoingPort::from).zip(outs);

        // DFG --------------------
        relation dfg_node(Node); // <Node> is a `DFG`
        dfg_node(n) <-- node(n), if hugr.get_optype(*n).is_dfg();

        out_wire_value(i, OutgoingPort::from(p.index()), v) <-- dfg_node(dfg),
          input_child(dfg, i), in_wire_value(dfg, p, v);

        out_wire_value(dfg, OutgoingPort::from(p.index()), v) <-- dfg_node(dfg),
            output_child(dfg, o), in_wire_value(o, p, v);

        // TailLoop --------------------
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
            node_in_value_row(out_n, out_in_row), // get the whole input row for the output node...
            // ...and select just what's possible for CONTINUE_TAG, if anything
            if let Some(fields) = out_in_row.unpack_first(TailLoop::CONTINUE_TAG, tailloop.just_inputs.len()),
            for (out_p, v) in fields.enumerate();

        // Output node of child region propagate to outputs of tail loop
        out_wire_value(tl, OutgoingPort::from(out_p), v) <-- node(tl),
            if let Some(tailloop) = hugr.get_optype(*tl).as_tail_loop(),
            output_child(tl, out_n),
            node_in_value_row(out_n, out_in_row), // get the whole input row for the output node...
            // ... and select just what's possible for BREAK_TAG, if anything
            if let Some(fields) = out_in_row.unpack_first(TailLoop::BREAK_TAG, tailloop.just_outputs.len()),
            for (out_p, v) in fields.enumerate();

        // Conditional --------------------
        // <Node> is a `Conditional` and its <usize>'th child (a `Case`) is <Node>:
        relation case_node(Node, usize, Node);
        case_node(cond, i, case) <-- node(cond),
          if hugr.get_optype(*cond).is_conditional(),
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

        // In `Conditional` <Node>, child `Case` <Node> is reachable given our knowledge of predicate:
        relation case_reachable(Node, Node);
        case_reachable(cond, case) <-- case_node(cond, i, case),
            in_wire_value(cond, IncomingPort::from(0), v),
            if v.supports_tag(*i);

        // CFG --------------------
        relation cfg_node(Node); // <Node> is a `CFG`
        cfg_node(n) <-- node(n), if hugr.get_optype(*n).is_cfg();

        // In `CFG` <Node>, basic block <Node> is reachable given our knowledge of predicates:
        relation bb_reachable(Node, Node);
        bb_reachable(cfg, entry) <-- cfg_node(cfg), if let Some(entry) = hugr.children(*cfg).next();
        bb_reachable(cfg, bb) <-- cfg_node(cfg),
            bb_reachable(cfg, pred),
            output_child(pred, pred_out),
            in_wire_value(pred_out, IncomingPort::from(0), predicate),
            for (tag, bb) in hugr.output_neighbours(*pred).enumerate(),
            if predicate.supports_tag(tag);

        // Inputs of CFG propagate to entry block
        out_wire_value(i_node, OutgoingPort::from(p.index()), v) <--
            cfg_node(cfg),
            if let Some(entry) = hugr.children(*cfg).next(),
            input_child(entry, i_node),
            in_wire_value(cfg, p, v);

        // In `CFG` <Node>, values fed along a control-flow edge to <Node>
        //     come out of Value outports of <Node>:
        relation _cfg_succ_dest(Node, Node, Node);
        _cfg_succ_dest(cfg, exit, cfg) <-- cfg_node(cfg), if let Some(exit) = hugr.children(*cfg).nth(1);
        _cfg_succ_dest(cfg, blk, inp) <-- cfg_node(cfg),
            for blk in hugr.children(*cfg),
            if hugr.get_optype(blk).is_dataflow_block(),
            input_child(blk, inp);

        // Outputs of each reachable block propagated to successor block or CFG itself
        out_wire_value(dest, OutgoingPort::from(out_p), v) <--
            bb_reachable(cfg, pred),
            if let Some(df_block) = hugr.get_optype(*pred).as_dataflow_block(),
            for (succ_n, succ) in hugr.output_neighbours(*pred).enumerate(),
            output_child(pred, out_n),
            _cfg_succ_dest(cfg, succ, dest),
            node_in_value_row(out_n, out_in_row),
            if let Some(fields) = out_in_row.unpack_first(succ_n, df_block.sum_rows.get(succ_n).unwrap().len()),
            for (out_p, v) in fields.enumerate();

        // Call --------------------
        relation func_call(Node, Node); // <Node> is a `Call` to `FuncDefn` <Node>
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
    ctx: &impl DFContext<V>,
    hugr: &impl HugrView,
    n: Node,
    ins: &[PV<V>],
    num_outs: usize,
) -> Option<ValueRow<V>> {
    match hugr.get_optype(n) {
        // Handle basics here. We could instead leave these to DFContext,
        // but at least we'd want these impls to be easily reusable.
        op if op.cast::<MakeTuple>().is_some() => Some(ValueRow::from_iter([PV::new_variant(
            0,
            ins.iter().cloned(),
        )])),
        op if op.cast::<UnpackTuple>().is_some() => {
            let elem_tys = op.cast::<UnpackTuple>().unwrap().0;
            let tup = ins.iter().exactly_one().unwrap();
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
            Some(ValueRow::singleton(partial_from_const(ctx, n, const_val)))
        }
        OpType::LoadFunction(load_op) => {
            assert!(ins.is_empty()); // static edge
            let func_node = hugr
                .single_linked_output(n, load_op.function_port())
                .unwrap()
                .0;
            // Node could be a FuncDefn or a FuncDecl, so do not pass the node itself
            Some(ValueRow::singleton(
                ctx.value_from_function(func_node, &load_op.type_args)
                    .map_or(PV::Top, PV::Value),
            ))
        }
        OpType::ExtensionOp(e) => {
            Some(ValueRow::from_iter(if row_contains_bottom(ins) {
                // So far we think one or more inputs can't happen.
                // So, don't pollute outputs with Top, and wait for better knowledge of inputs.
                vec![PartialValue::Bottom; num_outs]
            } else {
                // Interpret op using DFContext
                // Default to Top i.e.  can't figure out anything about the outputs
                let mut outs = vec![PartialValue::Top; num_outs];
                // It might be nice to convert `ins` to [(IncomingPort, Value)], or some
                // other concrete value, for the context, but PV contains more information,
                // and try_into_concrete may fail.
                ctx.interpret_leaf_op(n, e, ins, &mut outs[..]);
                outs
            }))
        }
        o => todo!("Unhandled: {:?}", o), // At least CallIndirect, and OpType is "non-exhaustive"
    }
}

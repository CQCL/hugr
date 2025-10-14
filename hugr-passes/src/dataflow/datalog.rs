//! [ascent] datalog implementation of analysis.

use std::collections::{HashMap, HashSet};

use ascent::Lattice;
use ascent::lattice::BoundedLattice;
use itertools::Itertools;

use hugr_core::extension::prelude::{MakeTuple, UnpackTuple};
use hugr_core::ops::{DataflowOpTrait, OpTrait, OpType, TailLoop};
use hugr_core::{HugrView, IncomingPort, OutgoingPort, PortIndex as _, Wire};

use super::value_row::ValueRow;
use super::{
    AbstractValue, AnalysisResults, DFContext, LoadedFunction, PartialValue, partial_from_const,
    row_contains_bottom,
};

type PV<V, N> = PartialValue<V, N>;

type NodeInputs<V, N> = Vec<(IncomingPort, PV<V, N>)>;

/// Basic structure for performing an analysis. Usage:
/// 1. Make a new instance via [`Self::new()`]
/// 2. (Optionally) zero or more calls to [`Self::prepopulate_wire`] and/or
///    [`Self::prepopulate_inputs`] with initial values.
///    For example, to analyse a [Module](OpType::Module)-rooted Hugr,
///    [`Self::prepopulate_inputs`] can be used on each externally-callable
///    [`FuncDefn`](OpType::FuncDefn) to set all inputs to [`PartialValue::Top`].
/// 3. Call [`Self::run`] to produce [`AnalysisResults`]
pub struct Machine<H: HugrView, V: AbstractValue>(H, HashMap<H::Node, NodeInputs<V, H::Node>>);

impl<H: HugrView, V: AbstractValue> Machine<H, V> {
    /// Create a new Machine to analyse the given Hugr(View)
    pub fn new(hugr: H) -> Self {
        Self(hugr, Default::default())
    }
}

impl<H: HugrView, V: AbstractValue> Machine<H, V> {
    /// Provide initial values for a wire - these will be `join`d with any computed
    /// or any value previously prepopulated for the same Wire.
    pub fn prepopulate_wire(&mut self, w: Wire<H::Node>, v: PartialValue<V, H::Node>) {
        for (n, inp) in self.0.linked_inputs(w.node(), w.source()) {
            self.1.entry(n).or_default().push((inp, v.clone()));
        }
    }

    /// Provide initial values for the inputs to a container node
    /// (a [`DataflowParent`](hugr_core::ops::OpTag::DataflowParent), [CFG](hugr_core::ops::CFG)
    /// or [Conditional](hugr_core::ops::Conditional)).
    /// Any inputs not given values by `in_values`, are set to [`PartialValue::Top`].
    /// Multiple calls for the same `parent` will `join` values for corresponding ports.
    #[expect(
        clippy::result_large_err,
        reason = "Not called recursively and not a performance bottleneck"
    )]
    #[inline]
    pub fn prepopulate_inputs(
        &mut self,
        parent: H::Node,
        in_values: impl IntoIterator<Item = (IncomingPort, PartialValue<V, H::Node>)>,
    ) -> Result<(), OpType> {
        if !self.0.contains_node(parent) {
            return Ok(());
        }
        match self.0.get_optype(parent) {
            OpType::DataflowBlock(_) | OpType::Case(_) | OpType::FuncDefn(_) => {
                // Put values onto out-wires of Input node
                let [inp, _] = self.0.get_io(parent).unwrap();
                let mut vals =
                    vec![PartialValue::Top; self.0.signature(inp).unwrap().output_types().len()];
                for (ip, v) in in_values {
                    vals[ip.index()] = v;
                }
                for (i, v) in vals.into_iter().enumerate() {
                    self.prepopulate_wire(Wire::new(inp, i), v);
                }
            }
            OpType::DFG(_) | OpType::TailLoop(_) | OpType::CFG(_) | OpType::Conditional(_) => {
                // dataflow will handle this and propagate to the correct Input node(s)
                let mut vals =
                    vec![PartialValue::Top; self.0.signature(parent).unwrap().input_types().len()];
                for (ip, v) in in_values {
                    vals[ip.index()] = v;
                }
                self.1
                    .entry(parent)
                    .or_default()
                    .extend(vals.into_iter().enumerate().map(|(i, v)| (i.into(), v)));
            }
            op => return Err(op.clone()),
        }
        Ok(())
    }

    /// Run the analysis (iterate until a lattice fixpoint is reached).
    /// As a shortcut, for Hugrs whose [HugrView::entrypoint] is a
    /// [`FuncDefn`](OpType::FuncDefn), [CFG](OpType::CFG), [DFG](OpType::DFG),
    /// [Conditional](OpType::Conditional) or [`TailLoop`](OpType::TailLoop) only
    /// (that is: *not* [Module](OpType::Module),
    /// [`DataflowBlock`](OpType::DataflowBlock) or [Case](OpType::Case)),
    /// `in_values` may provide initial values for the entrypoint-node inputs,
    ///  equivalent to calling `prepopulate_inputs` with the entrypoint node.
    ///
    /// The context passed in allows interpretation of leaf operations.
    ///
    /// # Panics
    /// May panic in various ways if the Hugr is invalid;
    /// or if any `in_values` are provided for a module-rooted Hugr.
    pub fn run(
        mut self,
        context: impl DFContext<V, Node = H::Node>,
        in_values: impl IntoIterator<Item = (IncomingPort, PartialValue<V, H::Node>)>,
    ) -> AnalysisResults<V, H> {
        if self.0.entrypoint_optype().is_module() {
            assert!(
                in_values.into_iter().next().is_none(),
                "No inputs possible for Module"
            );
        } else {
            let ep = self.0.entrypoint();
            let mut p = in_values.into_iter().peekable();
            // We must provide some inputs to the root so that they are Top rather than Bottom.
            // (However, this test will fail for DataflowBlock or Case roots, i.e. if no
            // inputs have been provided they will still see Bottom. We could store the "input"
            // values for even these nodes in self.1 and then convert to actual Wire values
            // (outputs from the Input node) before we run_datalog, but we would need to have
            // a separate store of output-wire values in self to keep prepopulate_wire working.)
            if p.peek().is_some() || !self.1.contains_key(&ep) {
                self.prepopulate_inputs(ep, p).unwrap();
            }
        }
        run_datalog(
            context,
            self.0,
            self.1
                .into_iter()
                .flat_map(|(n, vals)| vals.into_iter().map(move |(ip, v)| (n, ip, v)))
                .collect(),
        )
    }
}

pub(super) type InWire<V, N> = (N, IncomingPort, PartialValue<V, N>);

pub(super) fn run_datalog<V: AbstractValue, H: HugrView>(
    mut ctx: impl DFContext<V, Node = H::Node>,
    hugr: H,
    in_wire_value_proto: Vec<InWire<V, H::Node>>,
) -> AnalysisResults<V, H> {
    // ascent-(macro-)generated code generates a bunch of warnings,
    // keep code in here to a minimum.
    #![allow(
        clippy::clone_on_copy,
        clippy::unused_enumerate_index,
        clippy::collapsible_if
    )]
    let all_results = ascent::ascent_run! {
        pub(super) struct AscentProgram<V: AbstractValue, H: HugrView>;
        relation node(H::Node); // <Node> exists in the hugr
        relation in_wire(H::Node, IncomingPort); // <Node> has an <IncomingPort> of `EdgeKind::Value`
        relation out_wire(H::Node, OutgoingPort); // <Node> has an <OutgoingPort> of `EdgeKind::Value`
        relation parent_of_node(H::Node, H::Node); // <Node> is parent of <Node>
        relation input_child(H::Node, H::Node); // <Node> has 1st child <Node> that is its `Input`
        relation output_child(H::Node, H::Node); // <Node> has 2nd child <Node> that is its `Output`
        lattice out_wire_value(H::Node, OutgoingPort, PV<V, H::Node>); // <Node> produces, on <OutgoingPort>, the value <PV>
        lattice in_wire_value(H::Node, IncomingPort, PV<V, H::Node>); // <Node> receives, on <IncomingPort>, the value <PV>
        lattice node_in_value_row(H::Node, ValueRow<V, H::Node>); // <Node>'s inputs are <ValueRow>

        // Analyse all nodes as this will compute the most accurate results for the desired nodes
        // (i.e. the entry_descendants). Moreover, this is the only sound policy until we correctly
        // mark incoming edges as `Top`, see https://github.com/CQCL/hugr/issues/2254), so is a
        // workaround for that.
        // When that issue is solved, we can consider a flag to restrict analysis to the subregion
        // (for efficiency - will still decrease accuracy of solutions, but will at least be safe).
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
        in_wire_value(n, p, v) <-- for (n, p, v) in &in_wire_value_proto,
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
           if let Some(outs) = propagate_leaf_op(&mut ctx, &hugr, *n, &vs[..], sig.output_count()),
           for (p, v) in (0..).map(OutgoingPort::from).zip(outs);

        // DFG --------------------
        relation dfg_node(H::Node); // <Node> is a `DFG`
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
        relation case_node(H::Node, usize, H::Node);
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
        relation case_reachable(H::Node, H::Node);
        case_reachable(cond, case) <-- case_node(cond, i, case),
            in_wire_value(cond, IncomingPort::from(0), v),
            if v.supports_tag(*i);

        // CFG --------------------
        relation cfg_node(H::Node); // <Node> is a `CFG`
        cfg_node(n) <-- node(n), if hugr.get_optype(*n).is_cfg();

        // In `CFG` <Node>, basic block <Node> is reachable given our knowledge of predicates:
        relation bb_reachable(H::Node, H::Node);
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
        relation _cfg_succ_dest(H::Node, H::Node, H::Node);
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
        relation func_call(H::Node, H::Node); // <Node> is a `Call` to `FuncDefn` <Node>
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

        // CallIndirect --------------------
        lattice indirect_call(H::Node, LatticeWrapper<H::Node>); // <Node> is an `IndirectCall` to `FuncDefn` <Node>
        indirect_call(call, tgt) <--
            node(call),
            if let OpType::CallIndirect(_) = hugr.get_optype(*call),
            in_wire_value(call, IncomingPort::from(0), v),
            let tgt = load_func(v);

        out_wire_value(inp, OutgoingPort::from(p.index()-1), v) <--
            indirect_call(call, lv),
            if let LatticeWrapper::Value(func) = lv,
            input_child(func, inp),
            in_wire_value(call, p, v)
            if p.index() > 0;

        out_wire_value(call, OutgoingPort::from(p.index()), v) <--
            indirect_call(call, lv),
            if let LatticeWrapper::Value(func) = lv,
            output_child(func, outp),
            in_wire_value(outp, p, v);

        // Default out-value is Bottom, but if we can't determine the called function,
        // assign everything to Top
        out_wire_value(call, p, PV::Top) <--
            node(call),
            if let OpType::CallIndirect(ci) = hugr.get_optype(*call),
            in_wire_value(call, IncomingPort::from(0), v),
            // Second alternative below addresses function::Value's:
            if matches!(v, PartialValue::Top | PartialValue::Value(_)),
            for p in ci.signature().output_ports();
    };
    let entry_descs = hugr.entry_descendants().collect::<HashSet<_>>();
    let out_wire_values = all_results
        .out_wire_value
        .iter()
        .filter(|(n, _, _)| entry_descs.contains(n))
        .map(|(n, p, v)| (Wire::new(*n, *p), v.clone()))
        .collect();
    AnalysisResults {
        hugr,
        out_wire_values,
        in_wire_value: all_results
            .in_wire_value
            .into_iter()
            .filter(|(n, _, _)| entry_descs.contains(n))
            .collect(),
        case_reachable: all_results
            .case_reachable
            .into_iter()
            .filter(|(_, n)| entry_descs.contains(n))
            .collect(),
        bb_reachable: all_results
            .bb_reachable
            .into_iter()
            .filter(|(_, n)| entry_descs.contains(n))
            .collect(),
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd)]
enum LatticeWrapper<T> {
    Bottom,
    Value(T),
    Top,
}

impl<N: PartialEq + PartialOrd> Lattice for LatticeWrapper<N> {
    fn meet_mut(&mut self, other: Self) -> bool {
        if *self == other || *self == LatticeWrapper::Bottom || other == LatticeWrapper::Top {
            return false;
        }
        if *self == LatticeWrapper::Top || other == LatticeWrapper::Bottom {
            *self = other;
            return true;
        }
        // Both are `Value`s and not equal
        *self = LatticeWrapper::Bottom;
        true
    }

    fn join_mut(&mut self, other: Self) -> bool {
        if *self == other || *self == LatticeWrapper::Top || other == LatticeWrapper::Bottom {
            return false;
        }
        if *self == LatticeWrapper::Bottom || other == LatticeWrapper::Top {
            *self = other;
            return true;
        }
        // Both are `Value`s and are not equal
        *self = LatticeWrapper::Top;
        true
    }
}

fn load_func<V, N: Copy>(v: &PV<V, N>) -> LatticeWrapper<N> {
    match v {
        PartialValue::Bottom | PartialValue::PartialSum(_) => LatticeWrapper::Bottom,
        PartialValue::LoadedFunction(LoadedFunction { func_node, .. }) => {
            LatticeWrapper::Value(*func_node)
        }
        PartialValue::Value(_) | PartialValue::Top => LatticeWrapper::Top,
    }
}

fn propagate_leaf_op<V: AbstractValue, H: HugrView>(
    ctx: &mut impl DFContext<V, Node = H::Node>,
    hugr: &H,
    n: H::Node,
    ins: &[PV<V, H::Node>],
    num_outs: usize,
) -> Option<ValueRow<V, H::Node>> {
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
        OpType::Call(_) | OpType::CallIndirect(_) => None, // handled via Input/Output of FuncDefn
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
            Some(ValueRow::singleton(PartialValue::new_load(
                func_node,
                load_op.type_args.clone(),
            )))
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
        // We only call propagate_leaf_op for dataflow op non-containers,
        o => todo!("Unhandled: {:?}", o), // and OpType is non-exhaustive
    }
}

#[cfg(test)]
mod test {
    use ascent::Lattice;

    use super::LatticeWrapper;

    #[test]
    fn latwrap_join() {
        for lv in [
            LatticeWrapper::Value(3),
            LatticeWrapper::Value(5),
            LatticeWrapper::Top,
        ] {
            let mut subject = LatticeWrapper::Bottom;
            assert!(subject.join_mut(lv.clone()));
            assert_eq!(subject, lv);
            assert!(!subject.join_mut(lv.clone()));
            assert_eq!(subject, lv);
            assert_eq!(
                subject.join_mut(LatticeWrapper::Value(11)),
                lv != LatticeWrapper::Top
            );
            assert_eq!(subject, LatticeWrapper::Top);
        }
    }

    #[test]
    fn latwrap_meet() {
        for lv in [
            LatticeWrapper::Bottom,
            LatticeWrapper::Value(3),
            LatticeWrapper::Value(5),
        ] {
            let mut subject = LatticeWrapper::Top;
            assert!(subject.meet_mut(lv.clone()));
            assert_eq!(subject, lv);
            assert!(!subject.meet_mut(lv.clone()));
            assert_eq!(subject, lv);
            assert_eq!(
                subject.meet_mut(LatticeWrapper::Value(11)),
                lv != LatticeWrapper::Bottom
            );
            assert_eq!(subject, LatticeWrapper::Bottom);
        }
    }
}

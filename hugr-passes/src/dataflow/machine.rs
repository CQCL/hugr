use std::collections::HashMap;

use hugr_core::{ops::Value, types::ConstTypeError, HugrView, IncomingPort, Node, PortIndex, Wire};
use itertools::Itertools;

use super::{datalog::AscentProgram, AbstractValue, DFContext, PartialValue};

/// Basic structure for performing an analysis. Usage:
/// 1. Get a new instance via [Self::default()]
/// 2. (Optionally / for tests) zero or more [Self::prepopulate_wire] with initial values
/// 3. Call [Self::run] to produce [AnalysisResults] which can be inspected via
/// [read_out_wire](AnalysisResults::read_out_wire)
pub struct Machine<V: AbstractValue, C: DFContext<V>>(AscentProgram<V, C>);

/// Results of a dataflow analysis.
pub struct AnalysisResults<V: AbstractValue, C: DFContext<V>>(
    AscentProgram<V, C>, // Already run - kept for tests/debug
    HashMap<Wire, PartialValue<V>>,
);

/// derived-Default requires the context to be Defaultable, which is unnecessary
impl<V: AbstractValue, C: DFContext<V>> Default for Machine<V, C> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<V: AbstractValue, C: DFContext<V>> Machine<V, C> {
    /// Provide initial values for some wires.
    // Likely for test purposes only - should we make non-pub or #[cfg(test)] ?
    pub fn prepopulate_wire(&mut self, h: &impl HugrView, wire: Wire, value: PartialValue<V>) {
        self.0.in_wire_value_proto.extend(
            h.linked_inputs(wire.node(), wire.source())
                .map(|(n, inp)| (n, inp, value.clone())),
        );
    }

    /// Run the analysis (iterate until a lattice fixpoint is reached),
    /// given initial values for some of the root node inputs.
    /// (Note that `in_values` will not be useful for `Case` or `DFB`-rooted Hugrs,
    /// but should handle other containers.)
    /// The context passed in allows interpretation of leaf operations.
    pub fn run(
        mut self,
        context: C,
        in_values: impl IntoIterator<Item = (IncomingPort, PartialValue<V>)>,
    ) -> AnalysisResults<V, C> {
        let root = context.root();
        self.0
            .in_wire_value_proto
            .extend(in_values.into_iter().map(|(p, v)| (root, p, v)));
        self.0.context.push((context,));
        self.0.run();
        let results = self
            .0
            .out_wire_value
            .iter()
            .map(|(_, n, p, v)| (Wire::new(*n, *p), v.clone()))
            .collect();
        AnalysisResults(self.0, results)
    }
}

impl<V: AbstractValue, C: DFContext<V>> AnalysisResults<V, C> {
    fn context(&self) -> &C {
        let (c,) = self.0.context.iter().exactly_one().ok().unwrap();
        c
    }

    /// Gets the lattice value computed for the given wire
    pub fn read_out_wire(&self, w: Wire) -> Option<PartialValue<V>> {
        self.1.get(&w).cloned()
    }

    /// Tells whether a [TailLoop] node can terminate, i.e. whether
    /// `Break` and/or `Continue` tags may be returned by the nested DFG.
    /// Returns `None` if the specified `node` is not a [TailLoop].
    ///
    /// [TailLoop]: hugr_core::ops::TailLoop
    pub fn tail_loop_terminates(&self, node: Node) -> Option<TailLoopTermination> {
        let hugr = self.context();
        hugr.get_optype(node).as_tail_loop()?;
        let [_, out] = hugr.get_io(node).unwrap();
        Some(TailLoopTermination::from_control_value(
            self.0
                .in_wire_value
                .iter()
                .find_map(|(_, n, p, v)| (*n == out && p.index() == 0).then_some(v))
                .unwrap(),
        ))
    }

    /// Tells whether a [Case] node is reachable, i.e. whether the predicate
    /// to its parent [Conditional] may possibly have the tag corresponding to the [Case].
    /// Returns `None` if the specified `case` is not a [Case], or is not within a [Conditional]
    /// (e.g. a [Case]-rooted Hugr).
    ///
    /// [Case]: hugr_core::ops::Case
    /// [Conditional]: hugr_core::ops::Conditional
    pub fn case_reachable(&self, case: Node) -> Option<bool> {
        let hugr = self.context();
        hugr.get_optype(case).as_case()?;
        let cond = hugr.get_parent(case)?;
        hugr.get_optype(cond).as_conditional()?;
        Some(
            self.0
                .case_reachable
                .iter()
                .any(|(_, cond2, case2)| &cond == cond2 && &case == case2),
        )
    }

    /// Tells us if a block ([DataflowBlock] or [ExitBlock]) in a [CFG] is known
    /// to be reachable. (Returns `None` if argument is not a child of a CFG.)
    ///
    /// [CFG]: hugr_core::ops::CFG
    /// [DataflowBlock]: hugr_core::ops::DataflowBlock
    /// [ExitBlock]: hugr_core::ops::ExitBlock
    pub fn bb_reachable(&self, bb: Node) -> Option<bool> {
        let hugr = self.context();
        let cfg = hugr.get_parent(bb)?; // Not really required...??
        hugr.get_optype(cfg).as_cfg()?;
        let t = hugr.get_optype(bb);
        if !t.is_dataflow_block() && !t.is_exit_block() {
            return None;
        };
        Some(
            self.0
                .bb_reachable
                .iter()
                .any(|(_, cfg2, bb2)| *cfg2 == cfg && *bb2 == bb),
        )
    }
}

impl<V: AbstractValue, C: DFContext<V>> AnalysisResults<V, C>
where
    Value: From<V>,
{
    /// Reads a [Value] from an output wire, if the lattice value computed for it can be turned
    /// into one. (The lattice value must be either a single [Value](Self::Value) or
    /// a [Sum](PartialValue::PartialSum with a single known tag.)
    ///
    /// # Errors
    /// `None` if the analysis did not result in a single value on that wire
    /// `Some(e)` if conversion to a [Value] produced a [ConstTypeError]
    ///
    /// # Panics
    ///
    /// If a [Type] for the specified wire could not be extracted from the Hugr
    pub fn try_read_wire_value(&self, w: Wire) -> Result<Value, Option<ConstTypeError>> {
        let v = self.read_out_wire(w).ok_or(None)?;
        let (_, typ) = self
            .context()
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        v.try_into_value(&typ)
    }
}

/// Tells whether a loop iterates (never, always, sometimes)
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum TailLoopTermination {
    /// The loop never exits (is an infinite loop); no value is ever
    /// returned out of the loop. (aka, Bottom.)
    // TODO what about a loop that never exits OR continues because of a nested infinite loop?
    NeverBreaks,
    /// The loop never iterates (so is equivalent to a [DFG](hugr_core::ops::DFG),
    /// modulo untupling of the control value)
    NeverContinues,
    /// The loop might iterate and/or exit. (aka, Top)
    BreaksAndContinues,
}

impl TailLoopTermination {
    fn from_control_value<V: AbstractValue>(v: &PartialValue<V>) -> Self {
        let (may_continue, may_break) = (v.supports_tag(0), v.supports_tag(1));
        if may_break {
            if may_continue {
                Self::BreaksAndContinues
            } else {
                Self::NeverContinues
            }
        } else {
            Self::NeverBreaks
        }
    }
}

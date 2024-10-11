use std::collections::HashMap;

use hugr_core::{HugrView, Node, PortIndex, Wire};

use super::{datalog::AscentProgram, AbstractValue, DFContext, PartialValue};

/// Basic structure for performing an analysis. Usage:
/// 1. Get a new instance via [Self::default()]
/// 2. Zero or more [Self::propolutate_out_wires] with initial values
/// 3. Exactly one [Self::run] to do the analysis
/// 4. Results then available via [Self::read_out_wire]
pub struct Machine<V: AbstractValue, C: DFContext<V>>(
    AscentProgram<V, C>,
    Option<HashMap<Wire, PartialValue<V>>>,
);

/// derived-Default requires the context to be Defaultable, which is unnecessary
impl<V: AbstractValue, C: DFContext<V>> Default for Machine<V, C> {
    fn default() -> Self {
        Self(Default::default(), None)
    }
}

impl<V: AbstractValue, C: DFContext<V>> Machine<V, C> {
    /// Provide initial values for some wires.
    /// (For example, if some properties of the Hugr's inputs are known.)
    pub fn propolutate_out_wires(
        &mut self,
        wires: impl IntoIterator<Item = (Wire, PartialValue<V>)>,
    ) {
        assert!(self.1.is_none());
        self.0
            .out_wire_value_proto
            .extend(wires.into_iter().map(|(w, v)| (w.node(), w.source(), v)));
    }

    /// Run the analysis (iterate until a lattice fixpoint is reached).
    /// The context passed in allows interpretation of leaf operations.
    ///
    /// # Panics
    ///
    /// If this Machine has been run already.
    ///
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

    /// Gets the lattice value computed by [Self::run] for the given wire
    pub fn read_out_wire(&self, w: Wire) -> Option<PartialValue<V>> {
        self.1.as_ref().unwrap().get(&w).cloned()
    }

    /// Tells whether a [TailLoop] node can terminate, i.e. whether
    /// `Break` and/or `Continue` tags may be returned by the nested DFG.
    /// Returns `None` if the specified `node` is not a [TailLoop].
    ///
    /// [TailLoop]: hugr_core::ops::TailLoop
    pub fn tail_loop_terminates(
        &self,
        hugr: impl HugrView,
        node: Node,
    ) -> Option<TailLoopTermination> {
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
    pub fn case_reachable(&self, hugr: impl HugrView, case: Node) -> Option<bool> {
        hugr.get_optype(case).as_case()?;
        let cond = hugr.get_parent(case)?;
        hugr.get_optype(cond).as_conditional()?;
        Some(
            self.0
                .case_reachable
                .iter()
                .find(|(_, cond2, case2)| &cond == cond2 && &case == case2)
                .is_some(),
        )
    }

    /// Tells us if a block ([DataflowBlock] or [ExitBlock]) in a [CFG] is known
    /// to be reachable. (Returns `None` if argument is not a child of a CFG.)
    pub fn bb_reachable(&self, hugr: impl HugrView, bb: Node) -> Option<bool> {
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
                .find(|(_, cfg2, bb2)| *cfg2 == cfg && *bb2 == bb)
                .is_some(),
        )
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

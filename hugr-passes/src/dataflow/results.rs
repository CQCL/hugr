use std::collections::HashMap;

use hugr_core::{ops::Value, types::ConstTypeError, HugrView, IncomingPort, Node, PortIndex, Wire};

use super::{AbstractValue, DFContext, PartialValue};

/// Results of a dataflow analysis, packaged with the Hugr for easy inspection.
/// Methods allow inspection, specifically [read_out_wire](Self::read_out_wire).
pub struct AnalysisResults<V: AbstractValue, H> {
    pub(super) hugr: H,
    pub(super) in_wire_value: Vec<(Node, IncomingPort, PartialValue<V>)>,
    pub(super) case_reachable: Vec<(Node, Node)>,
    pub(super) bb_reachable: Vec<(Node, Node)>,
    pub(super) out_wire_values: HashMap<Wire, PartialValue<V>>,
}

impl<V: AbstractValue, C: DFContext<V>> AnalysisResults<V, C> {
    /// Gets the lattice value computed for the given wire
    pub fn read_out_wire(&self, w: Wire) -> Option<PartialValue<V>> {
        self.out_wire_values.get(&w).cloned()
    }

    /// Tells whether a [TailLoop] node can terminate, i.e. whether
    /// `Break` and/or `Continue` tags may be returned by the nested DFG.
    /// Returns `None` if the specified `node` is not a [TailLoop].
    ///
    /// [TailLoop]: hugr_core::ops::TailLoop
    pub fn tail_loop_terminates(&self, node: Node) -> Option<TailLoopTermination> {
        self.hugr.get_optype(node).as_tail_loop()?;
        let [_, out] = self.hugr.get_io(node).unwrap();
        Some(TailLoopTermination::from_control_value(
            self.in_wire_value
                .iter()
                .find_map(|(n, p, v)| (*n == out && p.index() == 0).then_some(v))
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
        self.hugr.get_optype(case).as_case()?;
        let cond = self.hugr.get_parent(case)?;
        self.hugr.get_optype(cond).as_conditional()?;
        Some(
            self.case_reachable
                .iter()
                .any(|(cond2, case2)| &cond == cond2 && &case == case2),
        )
    }

    /// Tells us if a block ([DataflowBlock] or [ExitBlock]) in a [CFG] is known
    /// to be reachable. (Returns `None` if argument is not a child of a CFG.)
    ///
    /// [CFG]: hugr_core::ops::CFG
    /// [DataflowBlock]: hugr_core::ops::DataflowBlock
    /// [ExitBlock]: hugr_core::ops::ExitBlock
    pub fn bb_reachable(&self, bb: Node) -> Option<bool> {
        let cfg = self.hugr.get_parent(bb)?; // Not really required...??
        self.hugr.get_optype(cfg).as_cfg()?;
        let t = self.hugr.get_optype(bb);
        if !t.is_dataflow_block() && !t.is_exit_block() {
            return None;
        };
        Some(
            self.bb_reachable
                .iter()
                .any(|(cfg2, bb2)| *cfg2 == cfg && *bb2 == bb),
        )
    }
}

impl<V: AbstractValue, C: DFContext<V>> AnalysisResults<V, C>
where
    Value: From<V>,
{
    /// Reads a [Value] from an output wire, if the lattice value computed for it can be turned
    /// into one. (The lattice value must be either a single [Value](PartialValue::Value) or
    /// a [Sum](PartialValue::PartialSum) with a single known tag.)
    ///
    /// # Errors
    /// `None` if the analysis did not result in a single value on that wire
    /// `Some(e)` if conversion to a [Value] produced a [ConstTypeError]
    ///
    /// # Panics
    ///
    /// If a [Type](hugr_core::types::Type) for the specified wire could not be extracted from the Hugr
    pub fn try_read_wire_value(&self, w: Wire) -> Result<Value, Option<ConstTypeError>> {
        let v = self.read_out_wire(w).ok_or(None)?;
        let (_, typ) = self
            .hugr
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

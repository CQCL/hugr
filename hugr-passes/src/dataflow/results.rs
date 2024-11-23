use std::collections::HashMap;

use hugr_core::{HugrView, IncomingPort, Node, PortIndex, Wire};

use super::{partial_value::ExtractValueError, AbstractValue, DFContext, PartialValue, Sum};

/// Results of a dataflow analysis, packaged with the Hugr for easy inspection.
/// Methods allow inspection, specifically [read_out_wire](Self::read_out_wire).
pub struct AnalysisResults<V: AbstractValue, C: DFContext<V>> {
    pub(super) ctx: C,
    pub(super) in_wire_value: Vec<(Node, IncomingPort, PartialValue<V>)>,
    pub(super) case_reachable: Vec<(Node, Node)>,
    pub(super) bb_reachable: Vec<(Node, Node)>,
    pub(super) out_wire_values: HashMap<Wire, PartialValue<V>>,
}

impl<V: AbstractValue, C: DFContext<V>> AnalysisResults<V, C> {
    /// Allows to use the [HugrView] contained within
    pub fn hugr(&self) -> &C::View {
        &self.ctx
    }

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
        self.hugr().get_optype(node).as_tail_loop()?;
        let [_, out] = self.hugr().get_io(node).unwrap();
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
        self.hugr().get_optype(case).as_case()?;
        let cond = self.hugr().get_parent(case)?;
        self.hugr().get_optype(cond).as_conditional()?;
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
        let cfg = self.hugr().get_parent(bb)?; // Not really required...??
        self.hugr().get_optype(cfg).as_cfg()?;
        let t = self.hugr().get_optype(bb);
        (t.is_dataflow_block() || t.is_exit_block()).then(|| {
            self.bb_reachable
                .iter()
                .any(|(cfg2, bb2)| *cfg2 == cfg && *bb2 == bb)
        })
    }

    /// Reads a concrete representation of the value on an output wire, if the lattice value
    /// computed for the wire can be turned into such. (The lattice value must be either a
    /// [PartialValue::Value] or a [PartialValue::PartialSum] with a single possible tag.)
    ///
    /// # Errors
    /// `None` if the analysis did not produce a result for that wire
    /// `Some(e)` if [conversion to a concrete value](PartialValue::try_into_concrete) failed with error `e`
    ///
    /// # Panics
    ///
    /// If a [Type](hugr_core::types::Type) for the specified wire could not be extracted from the Hugr
    pub fn try_read_wire_value<V2, VE, SE>(
        &self,
        w: Wire,
    ) -> Result<V2, Option<ExtractValueError<V, VE, SE>>>
    where
        V2: TryFrom<V, Error = VE> + TryFrom<Sum<V2>, Error = SE>,
    {
        let v = self.read_out_wire(w).ok_or(None)?;
        let (_, typ) = self
            .hugr()
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        v.try_into_concrete(&typ).map_err(Some)
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

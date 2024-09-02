use std::collections::HashMap;

use hugr_core::{ops::Value, HugrView, Node, PortIndex, Wire};

use super::{datalog::AscentProgram, AbstractValue, DFContext, PartialValue};

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

/// Usage:
/// 1. Get a new instance via [Self::default()]
/// 2. Zero or more [Self::propolutate_out_wires] with initial values
/// 3. Exactly one [Self::run] to do the analysis
/// 4. Results then available via [Self::read_out_wire_partial_value] and [Self::read_out_wire_value]
impl<V: AbstractValue, C: DFContext<V>> Machine<V, C> {
    pub fn propolutate_out_wires(
        &mut self,
        wires: impl IntoIterator<Item = (Wire, PartialValue<V>)>,
    ) {
        assert!(self.1.is_none());
        self.0
            .out_wire_value_proto
            .extend(wires.into_iter().map(|(w, v)| (w.node(), w.source(), v)));
    }

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

    pub fn read_out_wire_partial_value(&self, w: Wire) -> Option<PartialValue<V>> {
        self.1.as_ref().unwrap().get(&w).cloned()
    }

    pub fn tail_loop_terminates(&self, hugr: impl HugrView, node: Node) -> TailLoopTermination {
        assert!(hugr.get_optype(node).is_tail_loop());
        let [_, out] = hugr.get_io(node).unwrap();
        TailLoopTermination::from_control_value(
            self.0
                .in_wire_value
                .iter()
                .find_map(|(_, n, p, v)| (*n == out && p.index() == 0).then_some(v))
                .unwrap(),
        )
    }

    pub fn case_reachable(&self, hugr: impl HugrView, case: Node) -> bool {
        assert!(hugr.get_optype(case).is_case());
        let cond = hugr.get_parent(case).unwrap();
        assert!(hugr.get_optype(cond).is_conditional());
        self.0
            .case_reachable
            .iter()
            .find_map(|(_, cond2, case2, i)| (&cond == cond2 && &case == case2).then_some(*i))
            .unwrap()
    }
}

impl<V: AbstractValue, C: DFContext<V>> Machine<V, C>
where
    Value: From<V>,
{
    pub fn read_out_wire_value(&self, hugr: impl HugrView, w: Wire) -> Option<Value> {
        // dbg!(&w);
        let pv = self.read_out_wire_partial_value(w)?;
        // dbg!(&pv);
        let (_, typ) = hugr
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        pv.try_into_value(&typ).ok()
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum TailLoopTermination {
    Bottom,
    ExactlyZeroContinues,
    Top,
}

impl TailLoopTermination {
    pub fn from_control_value<V: AbstractValue>(v: &PartialValue<V>) -> Self {
        let (may_continue, may_break) = (v.supports_tag(0), v.supports_tag(1));
        if may_break && !may_continue {
            Self::ExactlyZeroContinues
        } else if may_break && may_continue {
            Self::Top
        } else {
            Self::Bottom
        }
    }
}

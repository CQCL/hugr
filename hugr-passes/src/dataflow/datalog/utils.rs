use super::super::partial_value::{AbstractValue, PartialValue};
use hugr_core::{types::Signature, HugrView, IncomingPort, Node, OutgoingPort};

pub(super) fn input_count(h: &impl HugrView, n: Node) -> usize {
    h.signature(n)
        .as_ref()
        .map(Signature::input_count)
        .unwrap_or(0)
}

pub(super) fn value_inputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = IncomingPort> + '_ {
    h.in_value_types(n).map(|x| x.0)
}

pub(super) fn value_outputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = OutgoingPort> + '_ {
    h.out_value_types(n).map(|x| x.0)
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

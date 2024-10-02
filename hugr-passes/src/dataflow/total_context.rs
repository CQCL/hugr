use std::hash::Hash;

use hugr_core::{ops::OpTrait, Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex};

use super::partial_value::{AbstractValue, PartialValue, ValueOrSum};
use super::DFContext;

/// A simpler interface like [DFContext] but where the context only cares about
/// values that are completely known (in the lattice `V`)
/// rather than e.g. Sums potentially of two variants each of known values.
pub trait TotalContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    type InterpretableVal: TryFrom<ValueOrSum<V>>;
    fn interpret_leaf_op(
        &self,
        node: Node,
        ins: &[(IncomingPort, Self::InterpretableVal)],
    ) -> Vec<(OutgoingPort, V)>;
}

impl<V: AbstractValue, T: TotalContext<V>> DFContext<V> for T {
    fn interpret_leaf_op(
        &self,
        node: Node,
        ins: &[PartialValue<V>],
    ) -> Option<Vec<PartialValue<V>>> {
        let op = self.get_optype(node);
        let sig = op.dataflow_signature()?;
        let known_ins = sig
            .input_types()
            .iter()
            .enumerate()
            .zip(ins.iter())
            .filter_map(|((i, ty), pv)| {
                pv.clone()
                    .try_into_value(ty)
                    // Discard PVs which don't produce ValueOrSum, i.e. Bottom/Top :-)
                    .ok()
                    // And discard any ValueOrSum that don't produce V - this is a bit silent :-(
                    .and_then(|v_s| T::InterpretableVal::try_from(v_s).ok())
                    .map(|v| (IncomingPort::from(i), v))
            })
            .collect::<Vec<_>>();
        let known_outs = self.interpret_leaf_op(node, &known_ins);
        (!known_outs.is_empty()).then(|| {
            let mut res = vec![PartialValue::Bottom; sig.output_count()];
            for (p, v) in known_outs {
                res[p.index()] = v.into();
            }
            res
        })
    }
}

use std::hash::Hash;

use hugr_core::{ops::OpTrait, Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex};

use super::datalog::{AbstractValue, DFContext, FromSum, PartialValue, ValueRow};

/// A simpler interface like [DFContext] but where the context only cares about
/// values that are completely known (in the lattice `V`)
/// rather than e.g. Sums potentially of two variants each of known values.
pub trait TotalContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    type InterpretableVal: FromSum + From<V>;
    fn interpret_leaf_op(
        &self,
        node: Node,
        ins: &[(IncomingPort, Self::InterpretableVal)],
    ) -> Vec<(OutgoingPort, V)>;
}

impl<V: AbstractValue, T: TotalContext<V>> DFContext<V> for T {
    fn interpret_leaf_op(&self, node: Node, ins: &[PartialValue<V>]) -> Option<ValueRow<V>> {
        let op = self.get_optype(node);
        let sig = op.dataflow_signature()?;
        let known_ins = sig
            .input_types()
            .into_iter()
            .enumerate()
            .zip(ins.iter())
            .filter_map(|((i, ty), pv)| {
                pv.clone()
                    .try_into_value(ty)
                    .ok()
                    .map(|v| (IncomingPort::from(i), v))
            })
            .collect::<Vec<_>>();
        let known_outs = self.interpret_leaf_op(node, &known_ins);
        (!known_outs.is_empty()).then(|| {
            let mut res = ValueRow::new(sig.output_count());
            for (p, v) in known_outs {
                res[p.index()] = v.into();
            }
            res
        })
    }

    fn hugr(&self) -> &impl HugrView {
        // Adding `fn hugr(&self) -> &impl HugrView` to trait TotalContext
        // and calling that here requires a lifetime bound on V, so avoid that
        self.as_ref()
    }
}

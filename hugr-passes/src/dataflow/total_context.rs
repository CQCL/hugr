use std::hash::Hash;

use ascent::lattice::BoundedLattice;
use hugr_core::{ops::OpTrait, Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex};

use super::{AbstractValue, DFContext, PartialValue, Sum};

/// A simpler interface like [DFContext] but where the context only cares about
/// values that are completely known (as `V`s), i.e. not `Bottom`, `Top`, or
/// Sums of potentially multiple variants.
pub trait TotalContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    /// The representation of values on which [Self::interpret_leaf_op] operates
    type InterpretableVal: From<V> + TryFrom<Sum<Self::InterpretableVal>>;
    /// Interpret a leaf op.
    /// `ins` gives the input ports for which we know (interpretable) values, and will be non-empty.
    /// Returns a list of output ports for which we know (abstract) values (may be empty).
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
                    .try_into_value::<<Self as TotalContext<V>>::InterpretableVal>(ty)
                    .ok()
                    .map(|v| (IncomingPort::from(i), v))
            })
            .collect::<Vec<_>>();
        let known_outs = self.interpret_leaf_op(node, &known_ins);
        (!known_outs.is_empty()).then(|| {
            let mut res = vec![PartialValue::bottom(); sig.output_count()];
            for (p, v) in known_outs {
                res[p.index()] = v.into();
            }
            res
        })
    }
}

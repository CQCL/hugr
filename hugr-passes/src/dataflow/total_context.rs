use hugr_core::ops::{DataflowOpTrait, ExtensionOp};
use hugr_core::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex};

use super::partial_value::{AbstractValue, PartialValue, Sum};
use super::{ConstLoader, DFContext};

/// A simpler interface like [DFContext] but where the context only cares about
/// values that are completely known (in the lattice `V`) rather than partially
/// (e.g. no [PartialSum]s of more than one variant, no top/bottom)
pub trait TotalContext<V>: ConstLoader<V> {
    /// Representation of a (single, non-partial) value usable for interpretation
    type InterpretableVal: From<V> + TryFrom<Sum<Self::InterpretableVal>>;

    /// Interpret an (extension) operation given total values for some of the in-ports
    /// `ins` will be a non-empty slice with distinct [IncomingPort]s.
    fn interpret_leaf_op(
        &self,
        node: Node,
        e: &ExtensionOp,
        ins: &[(IncomingPort, Self::InterpretableVal)],
    ) -> Vec<(OutgoingPort, PartialValue<V>)>;
}

impl<V: AbstractValue, T: TotalContext<V> + ConstLoader<V> + HugrView> DFContext<V> for T {
    fn interpret_leaf_op(
        &self,
        node: Node,
        op: &ExtensionOp,
        ins: &[PartialValue<V>],
        outs: &mut [PartialValue<V>],
    ) {
        let sig = op.signature();
        let known_ins = sig
            .input_types()
            .iter()
            .enumerate()
            .zip(ins.iter())
            .filter_map(|((i, ty), pv)| {
                let v = match pv {
                    PartialValue::Bottom | PartialValue::Top => None,
                    PartialValue::Value(v) => Some(v.clone().into()),
                    PartialValue::PartialSum(ps) => T::InterpretableVal::try_from(
                        ps.clone().try_into_value::<T::InterpretableVal>(ty).ok()?,
                    )
                    .ok(),
                }?;
                Some((IncomingPort::from(i), v))
            })
            .collect::<Vec<_>>();
        for (p, v) in self.interpret_leaf_op(node, op, &known_ins) {
            outs[p.index()] = v;
        }
    }
}
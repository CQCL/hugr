use std::hash::Hash;

use hugr_core::ops::Value;
use hugr_core::types::{ConstTypeError, SumType, Type, TypeEnum, TypeRow};
use hugr_core::{ops::OpTrait, Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex};
use itertools::{zip_eq, Itertools};

use super::datalog::DFContext;
use super::partial_value::{AbstractValue, PartialSum, PartialValue};
use super::ValueRow;

pub trait FromSum: Sized {
    type Err: std::error::Error;
    fn try_new_sum(
        tag: usize,
        items: impl IntoIterator<Item = Self>,
        st: &SumType,
    ) -> Result<Self, Self::Err>;
    fn debug_check_is_type(&self, _ty: &Type) {}
}

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

impl FromSum for Value {
    type Err = ConstTypeError;
    fn try_new_sum(
        tag: usize,
        items: impl IntoIterator<Item = Self>,
        st: &hugr_core::types::SumType,
    ) -> Result<Self, ConstTypeError> {
        Self::sum(tag, items, st.clone())
    }
}

// These are here because they rely on FromSum, that they are `impl PartialSum/Value`
// is merely a nice syntax.
impl<V: AbstractValue> PartialValue<V> {
    pub fn try_into_value<V2: FromSum + From<V>>(self, typ: &Type) -> Result<V2, Self> {
        let r: V2 = match self {
            Self::Value(v) => Ok(v.clone().into()),
            Self::PartialSum(ps) => ps.try_into_value(typ).map_err(Self::PartialSum),
            x => Err(x),
        }?;
        r.debug_check_is_type(typ);
        Ok(r)
    }
}

impl<V: AbstractValue> PartialSum<V> {
    pub fn try_into_value<V2: FromSum + From<V>>(self, typ: &Type) -> Result<V2, Self> {
        let Ok((k, v)) = self.0.iter().exactly_one() else {
            Err(self)?
        };

        let TypeEnum::Sum(st) = typ.as_type_enum() else {
            Err(self)?
        };
        let Some(r) = st.get_variant(*k) else {
            Err(self)?
        };
        let Ok(r): Result<TypeRow, _> = r.clone().try_into() else {
            Err(self)?
        };
        if v.len() != r.len() {
            return Err(self);
        }
        match zip_eq(v, r.iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(vs) => V2::try_new_sum(*k, vs, st).map_err(|_| self),
            Err(_) => Err(self),
        }
    }
}

impl<V: AbstractValue, T: TotalContext<V>> DFContext<V> for T {
    fn interpret_leaf_op(&self, node: Node, ins: &[PartialValue<V>]) -> Option<ValueRow<V>> {
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
}

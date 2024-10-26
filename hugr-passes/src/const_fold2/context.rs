use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::{ExtensionOp, Value};
use hugr_core::{IncomingPort, Node, OutgoingPort};

use super::value_handle::ValueHandle;
use crate::dataflow::{ConstLoader, PartialValue, TotalContext};

/// A [context](crate::dataflow::DFContext) that uses [ValueHandle]s
/// and performs [ExtensionOp::constant_fold] (using [Value]s for extension-op inputs).
///
/// Just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation.
#[derive(Debug)]
pub struct ConstFoldContext;

impl ConstLoader<ValueHandle> for ConstFoldContext {
    fn value_from_opaque(
        &self,
        node: Node,
        fields: &[usize],
        val: &OpaqueValue,
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_opaque(node, fields, val.clone()))
    }

    fn value_from_const_hugr(
        &self,
        node: Node,
        fields: &[usize],
        h: &hugr_core::Hugr,
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_const_hugr(
            node,
            fields,
            Box::new(h.clone()),
        ))
    }

    // Do not handle (Load)Function/value_from_function yet.
}

impl TotalContext<ValueHandle> for ConstFoldContext {
    type InterpretableVal = Value;

    fn interpret_leaf_op(
        &self,
        n: Node,
        op: &ExtensionOp,
        ins: &[(IncomingPort, Value)],
    ) -> Vec<(OutgoingPort, PartialValue<ValueHandle>)> {
        let ins = ins.iter().map(|(p, v)| (*p, v.clone())).collect::<Vec<_>>();
        op.constant_fold(&ins).map_or(Vec::new(), |outs| {
            outs.into_iter()
                .map(|(p, v)| {
                    (
                        p,
                        self.value_from_const(n, &v), // Hmmm, should (at least) also key by p
                    )
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod test {
    use hugr_core::ops::{constant::CustomConst, Value};
    use hugr_core::std_extensions::arithmetic::{float_types::ConstF64, int_types::ConstInt};
    use hugr_core::{types::SumType, Node};
    use itertools::Itertools;
    use rstest::rstest;

    use crate::{
        const_fold2::ConstFoldContext,
        dataflow::{ConstLoader, PartialValue},
    };

    #[rstest]
    #[case(ConstInt::new_u(4, 2).unwrap(), true)]
    #[case(ConstF64::new(std::f64::consts::PI), false)]
    fn value_handling(#[case] k: impl CustomConst + Clone, #[case] eq: bool) {
        let n = Node::from(portgraph::NodeIndex::new(7));
        let st = SumType::new([vec![k.get_type()], vec![]]);
        let subject_val = Value::sum(0, [k.clone().into()], st).unwrap();
        let v1 = ConstFoldContext.value_from_const(n, &subject_val);

        let v1_subfield = {
            let PartialValue::PartialSum(ps1) = v1 else {
                panic!()
            };
            ps1.0
                .into_iter()
                .exactly_one()
                .unwrap()
                .1
                .into_iter()
                .exactly_one()
                .unwrap()
        };

        let v2 = ConstFoldContext.value_from_const(n, &k.into());
        if eq {
            assert_eq!(v1_subfield, v2);
        } else {
            assert_ne!(v1_subfield, v2);
        }
    }
}

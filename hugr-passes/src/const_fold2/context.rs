use hugr_core::hugr::views::{DescendantsGraph, ExtractHugr, HierarchyView};
use hugr_core::ops::{constant::OpaqueValue, handle::FuncID, ExtensionOp, Value};
use hugr_core::types::TypeArg;
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort};

use super::value_handle::ValueHandle;
use crate::dataflow::{ConstLoader, PartialValue, TotalContext};

/// A [context](crate::dataflow::DFContext) that uses [ValueHandle]s
/// and performs [ExtensionOp::constant_fold] (using [Value]s for extension-op inputs).
///
/// Just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation.
#[derive(Debug)]
pub struct ConstFoldContext<H>(pub H);

impl<H:HugrView> AsRef<Hugr> for ConstFoldContext<H> {
    fn as_ref(&self) -> &Hugr {
        self.0.base_hugr()
    }
}

impl<H: HugrView> ConstLoader<ValueHandle> for ConstFoldContext<H> {
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

    fn value_from_function(&self, node: Node, type_args: &[TypeArg]) -> Option<ValueHandle> {
        if type_args.len() > 0 {
            // TODO: substitution across Hugr (https://github.com/CQCL/hugr/issues/709)
            return None;
        };
        // Returning the function body as a value, here, would be sufficient for inlining IndirectCall
        // but not for transforming to a direct Call.
        let func = DescendantsGraph::<FuncID<true>>::try_new(&self.0, node).ok()?;
        Some(ValueHandle::new_const_hugr(
            node,
            &[],
            Box::new(func.extract_hugr()),
        ))
    }
}

impl<H: HugrView> TotalContext<ValueHandle> for ConstFoldContext<H> {
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
    use hugr_core::{types::SumType, Hugr, Node};
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
        let ctx: ConstFoldContext<Hugr> = ConstFoldContext(Hugr::default());
        let v1 = ctx.value_from_const(n, &subject_val);

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

        let v2 = ctx.value_from_const(n, &k.into());
        if eq {
            assert_eq!(v1_subfield, v2);
        } else {
            assert_ne!(v1_subfield, v2);
        }
    }
}

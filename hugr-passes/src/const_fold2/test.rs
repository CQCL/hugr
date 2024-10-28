#![cfg(test)]

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
    let mut temp = Hugr::default();
    let ctx: ConstFoldContext<Hugr> = ConstFoldContext(&mut temp);
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

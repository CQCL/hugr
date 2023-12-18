//! Constant folding routines.

use crate::{
    extension::ConstFoldResult,
    ops::{Const, LeafOp, OpType},
    types::{Type, TypeEnum},
    values::Value,
    IncomingPort,
};

fn out_row(consts: impl IntoIterator<Item = Const>) -> ConstFoldResult {
    let vec = consts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i.into(), c))
        .collect();

    Some(vec)
}

fn sort_by_in_port(consts: &[(IncomingPort, Const)]) -> Vec<&(IncomingPort, Const)> {
    let mut v: Vec<_> = consts.iter().collect();
    v.sort_by_key(|(i, _)| i);
    v
}

fn sorted_consts(consts: &[(IncomingPort, Const)]) -> Vec<&Const> {
    sort_by_in_port(consts)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}
/// For a given op and consts, attempt to evaluate the op.
pub fn fold_const(op: &OpType, consts: &[(IncomingPort, Const)]) -> ConstFoldResult {
    let op = op.as_leaf_op()?;

    match op {
        LeafOp::Noop { .. } => out_row([consts.first()?.1.clone()]),
        LeafOp::MakeTuple { .. } => {
            out_row([Const::new_tuple(sorted_consts(consts).into_iter().cloned())])
        }
        LeafOp::UnpackTuple { .. } => {
            let c = &consts.first()?.1;

            if let Value::Tuple { vs } = c.value() {
                if let TypeEnum::Tuple(tys) = c.const_type().as_type_enum() {
                    return out_row(tys.iter().zip(vs.iter()).map(|(t, v)| {
                        Const::new(v.clone(), t.clone())
                            .expect("types should already have been checked")
                    }));
                }
            }
            None
        }

        LeafOp::Tag { tag, variants } => out_row([Const::new(
            Value::sum(*tag, consts.first()?.1.value().clone()),
            Type::new_sum(variants.clone()),
        )
        .unwrap()]),
        LeafOp::CustomOp(_) => {
            let ext_op = op.as_extension_op()?;

            ext_op.constant_fold(consts)
        }
        _ => None,
    }
}

#[cfg(test)]
mod test {
    use crate::{
        extension::{ExtensionRegistry, PRELUDE},
        ops::LeafOp,
        std_extensions::arithmetic::int_types::{ConstIntU, INT_TYPES},
        types::TypeArg,
    };
    use rstest::rstest;

    use super::*;

    fn i2c(b: u64) -> Const {
        Const::new(
            ConstIntU::new(5, b).unwrap().into(),
            INT_TYPES[5].to_owned(),
        )
        .unwrap()
    }

    fn u64_add() -> LeafOp {
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            crate::std_extensions::arithmetic::int_ops::EXTENSION.to_owned(),
            crate::std_extensions::arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        crate::std_extensions::arithmetic::int_ops::EXTENSION
            .instantiate_extension_op("iadd", [TypeArg::BoundedNat { n: 5 }], &reg)
            .unwrap()
            .into()
    }
    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 1, 1)]
    #[case(23, 435, 458)]
    // c = a + b
    fn test_add(#[case] a: u64, #[case] b: u64, #[case] c: u64) {
        let consts = vec![(0.into(), i2c(a)), (1.into(), i2c(b))];
        let add_op: OpType = u64_add().into();
        let out = fold_const(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), i2c(c))]);
    }
}

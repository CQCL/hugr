//! Constant folding routines.

use crate::{
    ops::{Const, OpType},
    values::Value,
    IncomingPort, OutgoingPort,
};

/// For a given op and consts, attempt to evaluate the op.
pub fn fold_const(
    op: &OpType,
    consts: &[(IncomingPort, Const)],
) -> Option<Vec<(OutgoingPort, Const)>> {
    consts.iter().find_map(|(_, cnst)| match cnst.value() {
        Value::Extension { c: (c,) } => c.fold(op, consts),
        Value::Tuple { .. } => todo!(),
        Value::Sum { .. } => todo!(),
        Value::Function { .. } => None,
    })
}

#[cfg(test)]
mod test {
    use crate::{
        extension::PRELUDE_REGISTRY,
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
        crate::std_extensions::arithmetic::int_types::EXTENSION
            .instantiate_extension_op("iadd", [TypeArg::BoundedNat { n: 5 }], &PRELUDE_REGISTRY)
            .unwrap()
            .into()
    }
    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 1, 1)]
    #[case(23, 435, 458)]
    // c = a && b
    fn test_and(#[case] a: u64, #[case] b: u64, #[case] c: u64) {
        let consts = vec![(0.into(), i2c(a)), (1.into(), i2c(b))];
        let add_op: OpType = u64_add().into();
        let out = fold_const(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), i2c(c))]);
    }
}

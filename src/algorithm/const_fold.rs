//! Constant folding routines.

use crate::{
    extension::ConstFoldResult,
    ops::{custom::ExternalOp, Const, LeafOp, OpType},
    values::Value,
    IncomingPort, OutgoingPort,
};

/// For a given op and consts, attempt to evaluate the op.
pub fn fold_const(op: &OpType, consts: &[(IncomingPort, Const)]) -> ConstFoldResult {
    let op = op.as_leaf_op()?;
    let ext_op = op.as_extension_op()?;

    ext_op.constant_fold(consts)
}

#[cfg(test)]
mod test {
    use crate::{
        extension::{ExtensionRegistry, FoldOutput, PRELUDE, PRELUDE_REGISTRY},
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

        assert_eq!(&out[..], &[(0.into(), FoldOutput::Value(Box::new(i2c(c))))]);
    }

    #[test]
    // a = a + 0
    fn test_zero_add() {
        for in_port in [0, 1] {
            let other_in = 1 - in_port;
            let consts = vec![(in_port.into(), i2c(0))];
            let add_op: OpType = u64_add().into();
            let out = fold_const(&add_op, &consts).unwrap();
            assert_eq!(&out[..], &[(0.into(), FoldOutput::Input(other_in.into()))]);
        }
    }
}

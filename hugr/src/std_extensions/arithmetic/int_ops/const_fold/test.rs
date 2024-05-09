#[cfg(test)]
mod test {

    use crate::algorithm::const_fold::constant_fold_pass;
    use crate::builder::handle::Outputs;
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::{sum_with_error, ConstError, ConstString, BOOL_T, STRING_TYPE};
    use crate::extension::{ExtensionRegistry, PRELUDE};
    use crate::ops::Value;
    use crate::std_extensions::arithmetic;
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use crate::std_extensions::logic::{self, NaryLogic};
    use crate::types::{FunctionType, Type, TypeRow};
    use crate::utils::test::assert_fully_folded;
    use crate::{type_row, Hugr};

    use lazy_static::lazy_static;
    use rstest::rstest;

    lazy_static! {
        static ref TEST_REG: ExtensionRegistry = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
    }

    fn const_fold_test(
        expected_value: impl Into<Value>,
        reg: &ExtensionRegistry,
        f: impl FnOnce(&mut DFGBuilder<Hugr>) -> Outputs,
    ) {
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let out_wires = f(&mut build);
        let mut h = build.finish_hugr_with_outputs(out_wires, reg).unwrap();
        constant_fold_pass(&mut h, reg);
        assert_fully_folded(&h, &expected_value.into());
    }

    #[test]
    fn test_fold_iwiden_u() {
        // pseudocode:
        //
        // x0 := int_u<4>(13);
        // x1 := iwiden_u<4, 5>(x0);
        // output x1 == int_u<5>(13);
        const_fold_test(ConstInt::new_u(5, 13).unwrap(), &TEST_REG, |build| {
            let x0 = build.add_load_const(Value::extension(ConstInt::new_u(4, 13).unwrap()));
            let x1 = build
                .add_dataflow_op(IntOpDef::iwiden_u.with_two_log_widths(4, 5), [x0])
                .unwrap();
            x1.outputs()
        })
    }

    #[test]
    fn test_fold_iwiden_s() {
        // pseudocode:
        //
        // x0 := int_u<4>(-3);
        // x1 := iwiden_u<4, 5>(x0);
        // output x1 == int_s<5>(-3);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(4, -3).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::iwiden_s.with_two_log_widths(4, 5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -3).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_inarrow_u() {
        // pseudocode:
        //
        // x0 := int_u<5>(13);
        // x1 := inarrow_u<5, 4>(x0);
        // output x1 == int_u<4>(13);
        let sum_type = sum_with_error(INT_TYPES[4].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 13).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::inarrow_u.with_two_log_widths(5, 4), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(4, 13).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_inarrow_s() {
        // pseudocode:
        //
        // x0 := int_s<5>(-3);
        // x1 := inarrow_s<5, 4>(x0);
        // output x1 == int_s<4>(-3);
        let sum_type = sum_with_error(INT_TYPES[4].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -3).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::inarrow_s.with_two_log_widths(5, 4), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(4, -3).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_itobool() {
        // pseudocode:
        //
        // x0 := int_u<0>(1);
        // x1 := itobool(x0);
        // output x1 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(0, 1).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::itobool.without_log_width(), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ifrombool() {
        // pseudocode:
        //
        // x0 := false
        // x1 := ifrombool(x0);
        // output x1 == int_u<0>(0);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[0].clone()])).unwrap();
        let x0 = build.add_load_const(Value::false_val());
        let x1 = build
            .add_dataflow_op(IntOpDef::ifrombool.without_log_width(), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(0, 0).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ieq() {
        // pseudocode:
        // x0, x1 := int_s<3>(-1), int_u<3>(255)
        // x2 := ieq(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(3, -1).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 255).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ieq.with_log_width(3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ine() {
        // pseudocode:
        // x0, x1 := int_u<5>(3), int_u<5>(4)
        // x2 := ine(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ine.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ilt_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(3), int_u<5>(4)
        // x2 := ilt_u(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ilt_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ilt_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(3), int_s<5>(-4)
        // x2 := ilt_s(x0, x1)
        // output x2 == false;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, -4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ilt_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::false_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_igt_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(3), int_u<5>(4)
        // x2 := ilt_u(x0, x1)
        // output x2 == false;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::igt_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::false_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_igt_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(3), int_s<5>(-4)
        // x2 := ilt_s(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, -4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::igt_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ile_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(3), int_u<5>(3)
        // x2 := ile_u(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ile_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ile_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-4), int_s<5>(-4)
        // x2 := ile_s(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -4).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, -4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ile_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ige_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(3), int_u<5>(4)
        // x2 := ilt_u(x0, x1)
        // output x2 == false;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ige_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::false_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ige_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(3), int_s<5>(-4)
        // x2 := ilt_s(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, -4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ige_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imax_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(7), int_u<5>(11);
        // x2 := imax_u(x0, x1);
        // output x2 == int_u<5>(11);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 7).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 11).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imax_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 11).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imax_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-2), int_s<5>(1);
        // x2 := imax_u(x0, x1);
        // output x2 == int_s<5>(1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, 1).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imax_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, 1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imin_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(7), int_u<5>(11);
        // x2 := imin_u(x0, x1);
        // output x2 == int_u<5>(7);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 7).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 11).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imin_u.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 7).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imin_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-2), int_s<5>(1);
        // x2 := imin_u(x0, x1);
        // output x2 == int_s<5>(-2);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, 1).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imin_s.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -2).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_iadd() {
        // pseudocode:
        // x0, x1 := int_s<5>(-2), int_s<5>(1);
        // x2 := iadd(x0, x1);
        // output x2 == int_s<5>(-1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, 1).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::iadd.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_isub() {
        // pseudocode:
        // x0, x1 := int_s<5>(-2), int_s<5>(1);
        // x2 := isub(x0, x1);
        // output x2 == int_s<5>(-3);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, 1).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::isub.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -3).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ineg() {
        // pseudocode:
        // x0 := int_s<5>(-2);
        // x1 := ineg(x0);
        // output x1 == int_s<5>(2);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ineg.with_log_width(5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, 2).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imul() {
        // pseudocode:
        // x0, x1 := int_s<5>(-2), int_s<5>(7);
        // x2 := imul(x0, x1);
        // output x2 == int_s<5>(-14);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_s(5, 7).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imul.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -14).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idivmod_checked_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(0)
        // x2 := idivmod_checked_u(x0, x1)
        // output x2 == error
        let intpair: TypeRow = vec![INT_TYPES[5].clone(), INT_TYPES[3].clone()].into();
        let sum_type = sum_with_error(Type::new_tuple(intpair));
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(
                IntOpDef::idivmod_checked_u.with_two_log_widths(5, 3),
                [x0, x1],
            )
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idivmod_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(3);
        // x2, x3 := idivmod_u(x0, x1); // 6, 2
        // x4 := iwiden_u<3,5>(x3); // 2
        // x5 := iadd<5>(x2, x4); // 8
        // output x5 == int_u<5>(8);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let [x2, x3] = build
            .add_dataflow_op(IntOpDef::idivmod_u.with_two_log_widths(5, 3), [x0, x1])
            .unwrap()
            .outputs_arr();
        let [x4] = build
            .add_dataflow_op(IntOpDef::iwiden_u.with_two_log_widths(3, 5), [x3])
            .unwrap()
            .outputs_arr();
        let x5 = build
            .add_dataflow_op(IntOpDef::iadd.with_log_width(5), [x2, x4])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x5.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 8).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idivmod_checked_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(0)
        // x2 := idivmod_checked_s(x0, x1)
        // output x2 == error
        let intpair: TypeRow = vec![INT_TYPES[5].clone(), INT_TYPES[3].clone()].into();
        let sum_type = sum_with_error(Type::new_tuple(intpair));
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(
                IntOpDef::idivmod_checked_s.with_two_log_widths(5, 3),
                [x0, x1],
            )
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[rstest]
    #[case(20, 3, 8)]
    #[case(-20, 3, -6)]
    #[case(-20, 4, -5)]
    #[case(i64::MIN, 1, i64::MIN)]
    #[case(i64::MIN, 2, -(1i64 << 62))]
    #[case(i64::MIN, 1u64 << 63, -1)]
    // c = a/b + a%b
    fn test_fold_idivmod_s(#[case] a: i64, #[case] b: u64, #[case] c: i64) {
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[6].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(6, a).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(6, b).unwrap()));
        let [x2, x3] = build
            .add_dataflow_op(IntOpDef::idivmod_s.with_two_log_widths(6, 6), [x0, x1])
            .unwrap()
            .outputs_arr();
        let x4 = build
            .add_dataflow_op(IntOpDef::iadd.with_log_width(6), [x2, x3])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x4.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(6, c).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idiv_checked_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(0)
        // x2 := idiv_checked_u(x0, x1)
        // output x2 == error
        let sum_type = sum_with_error(INT_TYPES[5].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::idiv_checked_u.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idiv_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(3);
        // x2 := idiv_u(x0, x1);
        // output x2 == int_u<5>(6);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::idiv_u.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 6).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imod_checked_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(0)
        // x2 := imod_checked_u(x0, x1)
        // output x2 == error
        let sum_type = sum_with_error(INT_TYPES[3].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imod_checked_u.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imod_u() {
        // pseudocode:
        // x0, x1 := int_u<5>(20), int_u<3>(3);
        // x2 := imod_u(x0, x1);
        // output x2 == int_u<3>(2);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[3].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imod_u.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(3, 2).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idiv_checked_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(0)
        // x2 := idiv_checked_s(x0, x1)
        // output x2 == error
        let sum_type = sum_with_error(INT_TYPES[5].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::idiv_checked_s.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_idiv_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(3);
        // x2 := idiv_s(x0, x1);
        // output x2 == int_s<5>(-7);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::idiv_s.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_s(5, -7).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imod_checked_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(0)
        // x2 := imod_checked_u(x0, x1)
        // output x2 == error
        let sum_type = sum_with_error(INT_TYPES[3].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 0).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imod_checked_s.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::sum(
            1,
            [ConstError {
                signal: 0,
                message: "Division by zero".to_string(),
            }
            .into()],
            sum_type.clone(),
        )
        .unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_imod_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(3);
        // x2 := imod_s(x0, x1);
        // output x2 == int_u<3>(1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[3].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::imod_s.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(3, 1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_iabs() {
        // pseudocode:
        // x0 := int_s<5>(-2);
        // x1 := iabs(x0);
        // output x1 == int_s<5>(2);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -2).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::iabs.with_log_width(5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 2).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_iand() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<5>(20);
        // x2 := iand(x0, x1);
        // output x2 == int_u<5>(4);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::iand.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 4).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ior() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<5>(20);
        // x2 := ior(x0, x1);
        // output x2 == int_u<5>(30);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ior.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 30).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ixor() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<5>(20);
        // x2 := ixor(x0, x1);
        // output x2 == int_u<5>(26);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 20).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ixor.with_log_width(5), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 26).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_inot() {
        // pseudocode:
        // x0 := int_u<5>(14);
        // x1 := inot(x0);
        // output x1 == int_u<5>(17);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 14).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::inot.with_log_width(5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, (1u64 << 32) - 15).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ishl() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<3>(3);
        // x2 := ishl(x0, x1);
        // output x2 == int_u<5>(112);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ishl.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 112).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_ishr() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<3>(3);
        // x2 := ishr(x0, x1);
        // output x2 == int_u<5>(1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ishr.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_irotl() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<3>(61);
        // x2 := irotl(x0, x1);
        // output x2 == int_u<5>(2^30 + 2^31 + 1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 61).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::irotl.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 3 * (1u64 << 30) + 1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_irotr() {
        // pseudocode:
        // x0, x1 := int_u<5>(14), int_u<3>(3);
        // x2 := irotr(x0, x1);
        // output x2 == int_u<5>(2^30 + 2^31 + 1);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, 14).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::irotr.with_two_log_widths(5, 3), [x0, x1])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstInt::new_u(5, 3 * (1u64 << 30) + 1).unwrap());
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_itostring_u() {
        // pseudocode:
        // x0 := int_u<5>(17);
        // x1 := itostring_u(x0);
        // output x2 := "17";
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![STRING_TYPE])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 17).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::itostring_u.with_log_width(5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstString::new("17".into()));
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_itostring_s() {
        // pseudocode:
        // x0 := int_s<5>(-17);
        // x1 := itostring_s(x0);
        // output x2 := "-17";
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![STRING_TYPE])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -17).unwrap()));
        let x1 = build
            .add_dataflow_op(IntOpDef::itostring_s.with_log_width(5), [x0])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::extension(ConstString::new("-17".into()));
        assert_fully_folded(&h, &expected);
    }

    #[test]
    #[should_panic]
    // FIXME: https://github.com/CQCL/hugr/issues/996
    fn test_fold_int_ops() {
        // pseudocode:
        //
        // x0 := int_u<5>(3); // 3
        // x1 := int_u<5>(4); // 4
        // x2 := ine(x0, x1); // true
        // x3 := ilt_u(x0, x1); // true
        // x4 := and(x2, x3); // true
        // x5 := int_s<5>(-10) // -10
        // x6 := ilt_s(x0, x5) // false
        // x7 := or(x4, x6) // true
        // output x7
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_u(5, 3).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(5, 4).unwrap()));
        let x2 = build
            .add_dataflow_op(IntOpDef::ine.with_log_width(5), [x0, x1])
            .unwrap();
        let x3 = build
            .add_dataflow_op(IntOpDef::ilt_u.with_log_width(5), [x0, x1])
            .unwrap();
        let x4 = build
            .add_dataflow_op(
                NaryLogic::And.with_n_inputs(2),
                x2.outputs().chain(x3.outputs()),
            )
            .unwrap();
        let x5 = build.add_load_const(Value::extension(ConstInt::new_s(5, -10).unwrap()));
        let x6 = build
            .add_dataflow_op(IntOpDef::ilt_s.with_log_width(5), [x0, x5])
            .unwrap();
        let x7 = build
            .add_dataflow_op(
                NaryLogic::Or.with_n_inputs(2),
                x4.outputs().chain(x6.outputs()),
            )
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            logic::EXTENSION.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(x7.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }
}

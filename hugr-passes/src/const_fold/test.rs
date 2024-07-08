use crate::const_fold::constant_fold_pass;
use hugr_core::builder::{DFGBuilder, Dataflow, DataflowHugr};
use hugr_core::extension::prelude::{sum_with_error, ConstError, ConstString, BOOL_T, STRING_TYPE};
use hugr_core::extension::{ExtensionRegistry, PRELUDE};
use hugr_core::ops::Value;
use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
use hugr_core::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
use hugr_core::std_extensions::arithmetic::{self, float_types, int_types};
use hugr_core::std_extensions::logic::{self, NaryLogic, NotOp};
use hugr_core::type_row;
use hugr_core::types::{FunctionType, Type, TypeRow};

use rstest::rstest;

use lazy_static::lazy_static;

use super::*;
use hugr_core::builder::Container;
use hugr_core::ops::{OpType, UnpackTuple};
use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr_core::std_extensions::arithmetic::float_ops::FloatOps;
use hugr_core::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};

/// Check that a hugr just loads and returns a single expected constant.
pub fn assert_fully_folded(h: &Hugr, expected_value: &Value) {
    assert_fully_folded_with(h, |v| v == expected_value)
}

/// Check that a hugr just loads and returns a single constant, and validate
/// that constant using `check_value`.
///
/// [CustomConst::equals_const] is not required to be implemented. Use this
/// function for Values containing such a `CustomConst`.
fn assert_fully_folded_with(h: &Hugr, check_value: impl Fn(&Value) -> bool) {
    let mut node_count = 0;

    for node in h.children(h.root()) {
        let op = h.get_optype(node);
        match op {
            OpType::Input(_) | OpType::Output(_) | OpType::LoadConstant(_) => node_count += 1,
            OpType::Const(c) if check_value(c.value()) => node_count += 1,
            _ => panic!("unexpected op: {:?}\n{}", op, h.mermaid_string()),
        }
    }

    assert_eq!(node_count, 4);
}

/// int to constant
fn i2c(b: u64) -> Value {
    Value::extension(ConstInt::new_u(5, b).unwrap())
}

/// float to constant
fn f2c(f: f64) -> Value {
    ConstF64::new(f).into()
}

#[rstest]
#[case(0.0, 0.0, 0.0)]
#[case(0.0, 1.0, 1.0)]
#[case(23.5, 435.5, 459.0)]
// c = a + b
fn test_add(#[case] a: f64, #[case] b: f64, #[case] c: f64) {
    let consts = vec![(0.into(), f2c(a)), (1.into(), f2c(b))];
    let add_op: OpType = FloatOps::fadd.into();
    let outs = fold_leaf_op(&add_op, &consts)
        .unwrap()
        .into_iter()
        .map(|(p, v)| (p, v.get_custom_value::<ConstF64>().unwrap().value()))
        .collect_vec();

    assert_eq!(outs.as_slice(), &[(0.into(), c)]);
}

fn float_fn(outputs: impl Into<TypeRow>) -> FunctionType {
    FunctionType::new(type_row![], outputs).with_extension_delta(float_types::EXTENSION_ID)
}

#[test]
fn test_big() {
    /*
       Test approximately calculates
       let x = (5.6, 3.2);
       int(x.0 - x.1) == 2
    */
    let sum_type = sum_with_error(INT_TYPES[5].to_owned());
    let mut build = DFGBuilder::new(float_fn(vec![sum_type.clone().into()])).unwrap();

    let tup = build.add_load_const(Value::tuple([f2c(5.6), f2c(3.2)]));

    let unpack = build
        .add_dataflow_op(
            UnpackTuple::new(type_row![FLOAT64_TYPE, FLOAT64_TYPE]),
            [tup],
        )
        .unwrap();

    let sub = build
        .add_dataflow_op(FloatOps::fsub, unpack.outputs())
        .unwrap();
    let to_int = build
        .add_dataflow_op(ConvertOpDef::trunc_u.with_log_width(5), sub.outputs())
        .unwrap();

    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        arithmetic::int_types::EXTENSION.to_owned(),
        arithmetic::float_types::EXTENSION.to_owned(),
        arithmetic::float_ops::EXTENSION.to_owned(),
        arithmetic::conversions::EXTENSION.to_owned(),
    ])
    .unwrap();
    let mut h = build
        .finish_hugr_with_outputs(to_int.outputs(), &reg)
        .unwrap();
    assert_eq!(h.node_count(), 8);

    constant_fold_pass(&mut h, &reg);

    let expected = Value::sum(0, [i2c(2).clone()], sum_type).unwrap();
    assert_fully_folded(&h, &expected);
}

#[test]
#[cfg_attr(
    feature = "extension_inference",
    ignore = "inference fails for test graph, it shouldn't"
)]
fn test_list_ops() -> Result<(), Box<dyn std::error::Error>> {
    use hugr_core::std_extensions::collections::{self, ListOp, ListValue};

    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        logic::EXTENSION.to_owned(),
        collections::EXTENSION.to_owned(),
    ])
    .unwrap();
    let list: Value = ListValue::new(BOOL_T, [Value::false_val()]).into();
    let mut build = DFGBuilder::new(FunctionType::new(
        type_row![],
        vec![list.get_type().clone()],
    ))
    .unwrap();

    let list_wire = build.add_load_const(list.clone());

    let pop = build.add_dataflow_op(
        ListOp::Pop.with_type(BOOL_T).to_extension_op(&reg).unwrap(),
        [list_wire],
    )?;

    let push = build.add_dataflow_op(
        ListOp::Push
            .with_type(BOOL_T)
            .to_extension_op(&reg)
            .unwrap(),
        pop.outputs(),
    )?;
    let mut h = build.finish_hugr_with_outputs(push.outputs(), &reg)?;
    constant_fold_pass(&mut h, &reg);

    assert_fully_folded(&h, &list);
    Ok(())
}

#[test]
fn test_fold_and() {
    // pseudocode:
    // x0, x1 := bool(true), bool(true)
    // x2 := and(x0, x1)
    // output x2 == true;
    let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
    let x0 = build.add_load_const(Value::true_val());
    let x1 = build.add_load_const(Value::true_val());
    let x2 = build
        .add_dataflow_op(NaryLogic::And.with_n_inputs(2), [x0, x1])
        .unwrap();
    let reg =
        ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
    let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    let expected = Value::true_val();
    assert_fully_folded(&h, &expected);
}

#[test]
fn test_fold_or() {
    // pseudocode:
    // x0, x1 := bool(true), bool(false)
    // x2 := or(x0, x1)
    // output x2 == true;
    let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
    let x0 = build.add_load_const(Value::true_val());
    let x1 = build.add_load_const(Value::false_val());
    let x2 = build
        .add_dataflow_op(NaryLogic::Or.with_n_inputs(2), [x0, x1])
        .unwrap();
    let reg =
        ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
    let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    let expected = Value::true_val();
    assert_fully_folded(&h, &expected);
}

#[test]
fn test_fold_not() {
    // pseudocode:
    // x0 := bool(true)
    // x1 := not(x0)
    // output x1 == false;
    let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
    let x0 = build.add_load_const(Value::true_val());
    let x1 = build.add_dataflow_op(NotOp, [x0]).unwrap();
    let reg =
        ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
    let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    let expected = Value::false_val();
    assert_fully_folded(&h, &expected);
}

#[test]
fn orphan_output() {
    // pseudocode:
    // x0 := bool(true)
    // x1 := not(x0)
    // x2 := or(x0,x1)
    // output x2 == true;
    //
    // We arrange things so that the `or` folds away first, leaving the not
    // with no outputs.
    use hugr_core::ops::handle::NodeHandle;

    let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
    let true_wire = build.add_load_value(Value::true_val());
    // this Not will be manually replaced
    let orig_not = build.add_dataflow_op(NotOp, [true_wire]).unwrap();
    let r = build
        .add_dataflow_op(
            NaryLogic::Or.with_n_inputs(2),
            [true_wire, orig_not.out_wire(0)],
        )
        .unwrap();
    let or_node = r.node();
    let parent = build.container_node();
    let reg =
        ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
    let mut h = build.finish_hugr_with_outputs(r.outputs(), &reg).unwrap();

    // we delete the original Not and create a new One. This means it will be
    // traversed by `constant_fold_pass` after the Or.
    let new_not = h.add_node_with_parent(parent, NotOp);
    h.connect(true_wire.node(), true_wire.source(), new_not, 0);
    h.disconnect(or_node, IncomingPort::from(1));
    h.connect(new_not, 0, or_node, 1);
    h.remove_node(orig_not.node());
    constant_fold_pass(&mut h, &reg);
    assert_fully_folded(&h, &Value::true_val())
}

#[test]
fn test_folding_pass_issue_996() {
    // pseudocode:
    //
    // x0 := 3.0
    // x1 := 4.0
    // x2 := fne(x0, x1); // true
    // x3 := flt(x0, x1); // true
    // x4 := and(x2, x3); // true
    // x5 := -10.0
    // x6 := flt(x0, x5) // false
    // x7 := or(x4, x6) // true
    // output x7
    let mut build = DFGBuilder::new(float_fn(vec![BOOL_T])).unwrap();
    let x0 = build.add_load_const(Value::extension(ConstF64::new(3.0)));
    let x1 = build.add_load_const(Value::extension(ConstF64::new(4.0)));
    let x2 = build.add_dataflow_op(FloatOps::fne, [x0, x1]).unwrap();
    let x3 = build.add_dataflow_op(FloatOps::flt, [x0, x1]).unwrap();
    let x4 = build
        .add_dataflow_op(
            NaryLogic::And.with_n_inputs(2),
            x2.outputs().chain(x3.outputs()),
        )
        .unwrap();
    let x5 = build.add_load_const(Value::extension(ConstF64::new(-10.0)));
    let x6 = build.add_dataflow_op(FloatOps::flt, [x0, x5]).unwrap();
    let x7 = build
        .add_dataflow_op(
            NaryLogic::Or.with_n_inputs(2),
            x4.outputs().chain(x6.outputs()),
        )
        .unwrap();
    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        logic::EXTENSION.to_owned(),
        arithmetic::float_types::EXTENSION.to_owned(),
    ])
    .unwrap();
    let mut h = build.finish_hugr_with_outputs(x7.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    let expected = Value::true_val();
    assert_fully_folded(&h, &expected);
}

#[test]
fn test_const_fold_to_nonfinite() {
    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        arithmetic::float_types::EXTENSION.to_owned(),
    ])
    .unwrap();

    // HUGR computing 1.0 / 1.0
    let mut build = DFGBuilder::new(float_fn(vec![FLOAT64_TYPE])).unwrap();
    let x0 = build.add_load_const(Value::extension(ConstF64::new(1.0)));
    let x1 = build.add_load_const(Value::extension(ConstF64::new(1.0)));
    let x2 = build.add_dataflow_op(FloatOps::fdiv, [x0, x1]).unwrap();
    let mut h0 = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h0, &reg);
    assert_fully_folded_with(&h0, |v| {
        v.get_custom_value::<ConstF64>().unwrap().value() == 1.0
    });
    assert_eq!(h0.node_count(), 5);

    // HUGR computing 1.0 / 0.0
    let mut build = DFGBuilder::new(float_fn(vec![FLOAT64_TYPE])).unwrap();
    let x0 = build.add_load_const(Value::extension(ConstF64::new(1.0)));
    let x1 = build.add_load_const(Value::extension(ConstF64::new(0.0)));
    let x2 = build.add_dataflow_op(FloatOps::fdiv, [x0, x1]).unwrap();
    let mut h1 = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h1, &reg);
    assert_eq!(h1.node_count(), 8);
}

fn int_fn(outputs: impl Into<TypeRow>) -> FunctionType {
    FunctionType::new(type_row![], outputs).with_extension_delta(int_types::EXTENSION_ID)
}

#[test]
fn test_fold_iwiden_u() {
    // pseudocode:
    //
    // x0 := int_u<4>(13);
    // x1 := iwiden_u<4, 5>(x0);
    // output x1 == int_u<5>(13);
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
    let x0 = build.add_load_const(Value::extension(ConstInt::new_u(4, 13).unwrap()));
    let x1 = build
        .add_dataflow_op(IntOpDef::iwiden_u.with_two_log_widths(4, 5), [x0])
        .unwrap();
    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        arithmetic::int_types::EXTENSION.to_owned(),
    ])
    .unwrap();
    let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    let expected = Value::extension(ConstInt::new_u(5, 13).unwrap());
    assert_fully_folded(&h, &expected);
}

#[test]
fn test_fold_iwiden_s() {
    // pseudocode:
    //
    // x0 := int_u<4>(-3);
    // x1 := iwiden_u<4, 5>(x0);
    // output x1 == int_s<5>(-3);
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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

#[rstest]
#[case(ConstInt::new_s, IntOpDef::inarrow_s, 5, 4, -3, true)]
#[case(ConstInt::new_s, IntOpDef::inarrow_s, 5, 5, -3, true)]
#[case(ConstInt::new_s, IntOpDef::inarrow_s, 5, 1, -3, false)]
#[case(ConstInt::new_u, IntOpDef::inarrow_u, 5, 4, 13, true)]
#[case(ConstInt::new_u, IntOpDef::inarrow_u, 5, 5, 13, true)]
#[case(ConstInt::new_u, IntOpDef::inarrow_u, 5, 0, 3, false)]
fn test_fold_inarrow<I: Copy, C: Into<Value>, E: std::fmt::Debug>(
    #[case] mk_const: impl Fn(u8, I) -> Result<C, E>,
    #[case] op_def: IntOpDef,
    #[case] from_log_width: u8,
    #[case] to_log_width: u8,
    #[case] val: I,
    #[case] succeeds: bool,
) {
    // For the first case, pseudocode:
    //
    // x0 := int_s<5>(-3);
    // x1 := inarrow_s<5, 4>(x0);
    // output x1 == sum<tag=0,[int_s<4>(-3)]>;
    //
    // Other cases vary by:
    // (mk_const, op_def) => create signed or unsigned constants, create
    //   inarrow_s or inarrow_u ops;
    // (from_log_width, to_log_width) => the args to use to create the op;
    // val => the value to pass to the op
    // succeeds => whether to expect a int<to_log_width> variant or an error
    //   variant.
    let sum_type = sum_with_error(INT_TYPES[to_log_width as usize].to_owned());
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
    let x0 = build.add_load_const(mk_const(from_log_width, val).unwrap().into());
    let x1 = build
        .add_dataflow_op(
            op_def.with_two_log_widths(from_log_width, to_log_width),
            [x0],
        )
        .unwrap();
    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        arithmetic::int_types::EXTENSION.to_owned(),
    ])
    .unwrap();
    let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
    constant_fold_pass(&mut h, &reg);
    lazy_static! {
        static ref INARROW_ERROR_VALUE: Value = ConstError {
            signal: 0,
            message: "Integer too large to narrow".to_string(),
        }
        .into();
    }
    let expected = if succeeds {
        Value::sum(0, [mk_const(to_log_width, val).unwrap().into()], sum_type).unwrap()
    } else {
        Value::sum(1, [INARROW_ERROR_VALUE.clone()], sum_type).unwrap()
    };
    assert_fully_folded(&h, &expected);
}

#[test]
fn test_fold_itobool() {
    // pseudocode:
    //
    // x0 := int_u<0>(1);
    // x1 := itobool(x0);
    // output x1 == true;
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[6].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[3].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![sum_type.clone().into()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[3].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![INT_TYPES[5].clone()])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![STRING_TYPE])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![STRING_TYPE])).unwrap();
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
    let mut build = DFGBuilder::new(int_fn(vec![BOOL_T])).unwrap();
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

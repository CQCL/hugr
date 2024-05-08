//! Constant folding routines.

use std::collections::{BTreeSet, HashMap};

use itertools::Itertools;

use crate::types::SumType;
use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionRegistry},
    hugr::{
        rewrite::consts::{RemoveConst, RemoveLoadConstant},
        views::SiblingSubgraph,
        HugrMut,
    },
    ops::{OpType, Value},
    type_row,
    types::FunctionType,
    Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
};

/// Tag some output constants with [`OutgoingPort`] inferred from the ordering.
fn out_row(consts: impl IntoIterator<Item = Value>) -> ConstFoldResult {
    let vec = consts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i.into(), c))
        .collect();
    Some(vec)
}

/// Sort folding inputs with [`IncomingPort`] as key
fn sort_by_in_port(consts: &[(IncomingPort, Value)]) -> Vec<&(IncomingPort, Value)> {
    let mut v: Vec<_> = consts.iter().collect();
    v.sort_by_key(|(i, _)| i);
    v
}

/// Sort some input constants by port and just return the constants.
pub(crate) fn sorted_consts(consts: &[(IncomingPort, Value)]) -> Vec<&Value> {
    sort_by_in_port(consts)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}
/// For a given op and consts, attempt to evaluate the op.
pub fn fold_leaf_op(op: &OpType, consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
    match op {
        OpType::Noop { .. } => out_row([consts.first()?.1.clone()]),
        OpType::MakeTuple { .. } => {
            out_row([Value::tuple(sorted_consts(consts).into_iter().cloned())])
        }
        OpType::UnpackTuple { .. } => {
            let c = &consts.first()?.1;
            let Value::Tuple { vs } = c else {
                panic!("This op always takes a Tuple input.");
            };
            out_row(vs.iter().cloned())
        }

        OpType::Tag(t) => out_row([Value::sum(
            t.tag,
            consts.iter().map(|(_, konst)| konst.clone()),
            SumType::new(t.variants.clone()),
        )
        .unwrap()]),
        OpType::CustomOp(op) => {
            let ext_op = op.as_extension_op()?;
            ext_op.constant_fold(consts)
        }
        _ => None,
    }
}

/// Generate a graph that loads and outputs `consts` in order, validating
/// against `reg`.
fn const_graph(consts: Vec<Value>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Value::const_type).collect_vec();
    let mut b = DFGBuilder::new(FunctionType::new(type_row![], const_types)).unwrap();

    let outputs = consts
        .into_iter()
        .map(|c| b.add_load_const(c))
        .collect_vec();

    b.finish_hugr_with_outputs(outputs, reg).unwrap()
}

/// Given some `candidate_nodes` to search for LoadConstant operations in `hugr`,
/// return an iterator of possible constant folding rewrites. The
/// [`SimpleReplacement`] replaces an operation with constants that result from
/// evaluating it, the extension registry `reg` is used to validate the
/// replacement HUGR. The vector of [`RemoveLoadConstant`] refer to the
/// LoadConstant nodes that could be removed - they are not automatically
/// removed as they may be used by other operations.
pub fn find_consts<'a, 'r: 'a>(
    hugr: &'a impl HugrView,
    candidate_nodes: impl IntoIterator<Item = Node> + 'a,
    reg: &'r ExtensionRegistry,
) -> impl Iterator<Item = (SimpleReplacement, Vec<RemoveLoadConstant>)> + 'a {
    // track nodes for operations that have already been considered for folding
    let mut used_neighbours = BTreeSet::new();

    candidate_nodes
        .into_iter()
        .filter_map(move |n| {
            // only look at LoadConstant
            hugr.get_optype(n).is_load_constant().then_some(())?;

            let (out_p, _) = hugr.out_value_types(n).exactly_one().ok()?;
            let neighbours = hugr
                .linked_inputs(n, out_p)
                .filter(|(n, _)| used_neighbours.insert(*n))
                .collect_vec();
            if neighbours.is_empty() {
                // no uses of LoadConstant that haven't already been considered.
                return None;
            }
            let fold_iter = neighbours
                .into_iter()
                .filter_map(|(neighbour, _)| fold_op(hugr, neighbour, reg));
            Some(fold_iter)
        })
        .flatten()
}

/// Attempt to evaluate and generate rewrites for the operation at `op_node`
fn fold_op(
    hugr: &impl HugrView,
    op_node: Node,
    reg: &ExtensionRegistry,
) -> Option<(SimpleReplacement, Vec<RemoveLoadConstant>)> {
    // only support leaf folding for now.
    let neighbour_op = hugr.get_optype(op_node);
    let (in_consts, removals): (Vec<_>, Vec<_>) = hugr
        .node_inputs(op_node)
        .filter_map(|in_p| {
            let (con_op, load_n) = get_const(hugr, op_node, in_p)?;
            Some(((in_p, con_op), RemoveLoadConstant(load_n)))
        })
        .unzip();
    // attempt to evaluate op
    let folded = fold_leaf_op(neighbour_op, &in_consts)?;
    let (op_outs, consts): (Vec<_>, Vec<_>) = folded.into_iter().unzip();
    let nu_out = op_outs
        .into_iter()
        .enumerate()
        .filter_map(|(i, out)| {
            // map from the ports the op was linked to, to the output ports of
            // the replacement.
            hugr.single_linked_input(op_node, out)
                .map(|np| (np, i.into()))
        })
        .collect();
    let replacement = const_graph(consts, reg);
    let sibling_graph = SiblingSubgraph::try_from_nodes([op_node], hugr)
        .expect("Operation should form valid subgraph.");

    let simple_replace = SimpleReplacement::new(
        sibling_graph,
        replacement,
        // no inputs to replacement
        HashMap::new(),
        nu_out,
    );
    Some((simple_replace, removals))
}

/// If `op_node` is connected to a LoadConstant at `in_p`, return the constant
/// and the LoadConstant node
fn get_const(hugr: &impl HugrView, op_node: Node, in_p: IncomingPort) -> Option<(Value, Node)> {
    let (load_n, _) = hugr.single_linked_output(op_node, in_p)?;
    let load_op = hugr.get_optype(load_n).as_load_constant()?;
    let const_node = hugr
        .linked_outputs(load_n, load_op.constant_port())
        .exactly_one()
        .ok()?
        .0;

    let const_op = hugr.get_optype(const_node).as_const()?;

    // TODO avoid const clone here
    Some((const_op.as_ref().clone(), load_n))
}

/// Exhaustively apply constant folding to a HUGR.
pub fn constant_fold_pass(h: &mut impl HugrMut, reg: &ExtensionRegistry) {
    loop {
        // would be preferable if the candidates were updated to be just the
        // neighbouring nodes of those added.
        let rewrites = find_consts(h, h.nodes(), reg).collect_vec();
        if rewrites.is_empty() {
            break;
        }
        for (replace, removes) in rewrites {
            h.apply_rewrite(replace).unwrap();
            for rem in removes {
                if let Ok(const_node) = h.apply_rewrite(rem) {
                    // if the LoadConst was removed, try removing the Const too.
                    if h.apply_rewrite(RemoveConst(const_node)).is_err() {
                        // const cannot be removed - no problem
                        continue;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::extension::prelude::{sum_with_error, ConstError, ConstString, BOOL_T, STRING_TYPE};
    use crate::extension::{ExtensionRegistry, PRELUDE};
    use crate::ops::{OpType, UnpackTuple};
    use crate::std_extensions::arithmetic;
    use crate::std_extensions::arithmetic::conversions::ConvertOpDef;
    use crate::std_extensions::arithmetic::float_ops::FloatOps;
    use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use crate::std_extensions::logic::{self, NaryLogic, NotOp};
    use crate::types::{Type, TypeRow};
    use crate::utils::test::assert_fully_folded;

    use rstest::rstest;

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
        let out = fold_leaf_op(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), f2c(c))]);
    }
    #[test]
    fn test_big() {
        /*
           Test approximately calculates
           let x = (5.6, 3.2);
           int(x.0 - x.1) == 2
        */
        let sum_type = sum_with_error(INT_TYPES[5].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();

        let tup = build.add_load_const(Value::tuple([f2c(5.6), f2c(3.2)]));

        let unpack = build
            .add_dataflow_op(
                UnpackTuple {
                    tys: type_row![FLOAT64_TYPE, FLOAT64_TYPE],
                },
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

    #[rstest]
    #[case(NaryLogic::And, [true, true, true], true)]
    #[case(NaryLogic::And, [true, false, true], false)]
    #[case(NaryLogic::Or, [false, false, true], true)]
    #[case(NaryLogic::Or, [false, false, false], false)]
    fn test_logic_and(
        #[case] op: NaryLogic,
        #[case] ins: [bool; 3],
        #[case] out: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();

        let ins = ins.map(|b| build.add_load_const(Value::from_bool(b)));
        let logic_op = build.add_dataflow_op(op.with_n_inputs(ins.len() as u64), ins)?;

        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(logic_op.outputs(), &reg)?;
        constant_fold_pass(&mut h, &reg);

        assert_fully_folded(&h, &Value::from_bool(out));
        Ok(())
    }

    #[test]
    #[cfg_attr(
        feature = "extension_inference",
        ignore = "inference fails for test graph, it shouldn't"
    )]
    fn test_list_ops() -> Result<(), Box<dyn std::error::Error>> {
        use crate::std_extensions::collections::{self, ListOp, ListValue};

        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            logic::EXTENSION.to_owned(),
            collections::EXTENSION.to_owned(),
        ])
        .unwrap();
        let list: Value = ListValue::new(BOOL_T, [Value::unit_sum(0, 1).unwrap()]).into();
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![list.const_type().clone()],
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
    fn test_fold_iwiden_u() {
        // pseudocode:
        //
        // x0 := int_u<4>(13);
        // x1 := iwiden_u<4, 5>(x0);
        // output x1 == int_u<5>(13);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
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

    #[test]
    fn test_fold_idivmod_s() {
        // pseudocode:
        // x0, x1 := int_s<5>(-20), int_u<3>(3);
        // x2, x3 := idivmod_u(x0, x1); // -7, 1
        // x4 := iwiden_s<3,5>(x3); // 1
        // x5 := iadd<5>(x2, x4); // -6
        // output x5 == int_s<5>(-6);
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], vec![INT_TYPES[5].clone()])).unwrap();
        let x0 = build.add_load_const(Value::extension(ConstInt::new_s(5, -20).unwrap()));
        let x1 = build.add_load_const(Value::extension(ConstInt::new_u(3, 3).unwrap()));
        let [x2, x3] = build
            .add_dataflow_op(IntOpDef::idivmod_s.with_two_log_widths(5, 3), [x0, x1])
            .unwrap()
            .outputs_arr();
        let [x4] = build
            .add_dataflow_op(IntOpDef::iwiden_s.with_two_log_widths(3, 5), [x3])
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
        let expected = Value::extension(ConstInt::new_s(5, -6).unwrap());
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

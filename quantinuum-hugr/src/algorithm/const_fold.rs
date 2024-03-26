//! Constant folding routines.

use std::collections::{BTreeSet, HashMap};

use itertools::Itertools;

use crate::types::{Signature, SumType};
use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionRegistry},
    hugr::{
        rewrite::consts::{RemoveConst, RemoveLoadConstant},
        views::SiblingSubgraph,
        HugrMut,
    },
    ops::{Const, LeafOp},
    type_row, Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
};

/// Tag some output constants with [`OutgoingPort`] inferred from the ordering.
fn out_row(consts: impl IntoIterator<Item = Const>) -> ConstFoldResult {
    let vec = consts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i.into(), c))
        .collect();
    Some(vec)
}

/// Sort folding inputs with [`IncomingPort`] as key
fn sort_by_in_port(consts: &[(IncomingPort, Const)]) -> Vec<&(IncomingPort, Const)> {
    let mut v: Vec<_> = consts.iter().collect();
    v.sort_by_key(|(i, _)| i);
    v
}

/// Sort some input constants by port and just return the constants.
pub(crate) fn sorted_consts(consts: &[(IncomingPort, Const)]) -> Vec<&Const> {
    sort_by_in_port(consts)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}
/// For a given op and consts, attempt to evaluate the op.
pub fn fold_leaf_op(op: &LeafOp, consts: &[(IncomingPort, Const)]) -> ConstFoldResult {
    match op {
        LeafOp::Noop { .. } => out_row([consts.first()?.1.clone()]),
        LeafOp::MakeTuple { .. } => {
            out_row([Const::tuple(sorted_consts(consts).into_iter().cloned())])
        }
        LeafOp::UnpackTuple { .. } => {
            let c = &consts.first()?.1;
            let Const::Tuple { vs } = c else {
                panic!("This op always takes a Tuple input.");
            };
            out_row(vs.iter().cloned())
        }

        LeafOp::Tag { tag, variants } => out_row([Const::sum(
            *tag,
            consts.iter().map(|(_, konst)| konst.clone()),
            SumType::new(variants.clone()),
        )
        .unwrap()]),
        LeafOp::CustomOp(_) => {
            let ext_op = op.as_extension_op()?;

            ext_op.constant_fold(consts)
        }
        _ => None,
    }
}

/// Generate a graph that loads and outputs `consts` in order, validating
/// against `reg`.
fn const_graph(consts: Vec<Const>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Const::const_type).collect_vec();
    let mut b = DFGBuilder::new(Signature::new(type_row![], const_types)).unwrap();

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
    let neighbour_op = hugr.get_optype(op_node).as_leaf_op()?;
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
fn get_const(hugr: &impl HugrView, op_node: Node, in_p: IncomingPort) -> Option<(Const, Node)> {
    let (load_n, _) = hugr.single_linked_output(op_node, in_p)?;
    let load_op = hugr.get_optype(load_n).as_load_constant()?;
    let const_node = hugr
        .linked_outputs(load_n, load_op.constant_port())
        .exactly_one()
        .ok()?
        .0;

    let const_op = hugr.get_optype(const_node).as_const()?;

    // TODO avoid const clone here
    Some((const_op.clone(), load_n))
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
    use crate::extension::prelude::{sum_with_error, BOOL_T};
    use crate::extension::{ExtensionRegistry, PRELUDE};
    use crate::ops::OpType;
    use crate::std_extensions::arithmetic;
    use crate::std_extensions::arithmetic::conversions::ConvertOpDef;
    use crate::std_extensions::arithmetic::float_ops::FloatOps;
    use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
    use crate::std_extensions::arithmetic::int_types::{ConstIntU, INT_TYPES};
    use crate::std_extensions::logic::{self, NaryLogic};
    use crate::types::Signature;

    use rstest::rstest;

    /// int to constant
    fn i2c(b: u64) -> Const {
        Const::extension(ConstIntU::new(5, b).unwrap())
    }

    /// float to constant
    fn f2c(f: f64) -> Const {
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
        let out = fold_leaf_op(add_op.as_leaf_op().unwrap(), &consts).unwrap();

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
        let mut build =
            DFGBuilder::new(Signature::new(type_row![], vec![sum_type.clone().into()])).unwrap();

        let tup = build.add_load_const(Const::tuple([f2c(5.6), f2c(3.2)]));

        let unpack = build
            .add_dataflow_op(
                LeafOp::UnpackTuple {
                    tys: type_row![FLOAT64_TYPE, FLOAT64_TYPE],
                },
                [tup],
            )
            .unwrap();

        let sub = build
            .add_dataflow_op(FloatOps::fsub, unpack.outputs())
            .unwrap();
        let to_int = build
            .add_dataflow_op(ConvertOpDef::trunc_u.with_width(5), sub.outputs())
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

        let expected = Const::sum(0, [i2c(2).clone()], sum_type).unwrap();
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
        let mut build = DFGBuilder::new(Signature::new(type_row![], vec![BOOL_T])).unwrap();

        let ins = ins.map(|b| build.add_load_const(Const::from_bool(b)));
        let logic_op = build.add_dataflow_op(op.with_n_inputs(ins.len() as u64), ins)?;

        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(logic_op.outputs(), &reg)?;
        constant_fold_pass(&mut h, &reg);

        assert_fully_folded(&h, &Const::from_bool(out));
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
        let list: Const = ListValue::new(BOOL_T, [Const::unit_sum(0, 1).unwrap()]).into();
        let mut build =
            DFGBuilder::new(Signature::new(type_row![], vec![list.const_type().clone()])).unwrap();

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

    fn assert_fully_folded(h: &Hugr, expected_const: &Const) {
        // check the hugr just loads and returns a single const
        let mut node_count = 0;

        for node in h.children(h.root()) {
            let op = h.get_optype(node);
            match op {
                OpType::Input(_) | OpType::Output(_) | OpType::LoadConstant(_) => node_count += 1,
                OpType::Const(c) if c == expected_const => node_count += 1,
                _ => panic!("unexpected op: {:?}", op),
            }
        }

        assert_eq!(node_count, 4);
    }
}

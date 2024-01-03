//! Constant folding routines.

use std::collections::{BTreeSet, HashMap};

use itertools::Itertools;

use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionRegistry},
    hugr::{
        rewrite::consts::{RemoveConst, RemoveConstIgnore},
        views::SiblingSubgraph,
        HugrMut,
    },
    ops::{Const, LeafOp, OpType},
    type_row,
    types::{FunctionType, Type, TypeEnum},
    values::Value,
    Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
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
            panic!("This op always takes a Tuple input.");
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

/// Generate a graph that loads and outputs `consts` in order, validating
/// against `reg`.
fn const_graph(consts: Vec<Const>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Const::const_type).cloned().collect_vec();
    let mut b = DFGBuilder::new(FunctionType::new(type_row![], const_types)).unwrap();

    let outputs = consts
        .into_iter()
        .map(|c| b.add_load_const(c).unwrap())
        .collect_vec();

    b.finish_hugr_with_outputs(outputs, reg).unwrap()
}

/// Given some `candidate_nodes` to search for LoadConstant operations in `hugr`,
/// return an iterator of possible constant folding rewrites. The
/// [`SimpleReplacement`] replaces an operation with constants that result from
/// evaluating it, the extension registry `reg` is used to validate the
/// replacement HUGR. The vector of [`RemoveConstIgnore`] refer to the
/// LoadConstant nodes that could be removed - they are not automatically
/// removed as they may be used by other operations.
pub fn find_consts<'a, 'r: 'a>(
    hugr: &'a impl HugrView,
    candidate_nodes: impl IntoIterator<Item = Node> + 'a,
    reg: &'r ExtensionRegistry,
) -> impl Iterator<Item = (SimpleReplacement, Vec<RemoveConstIgnore>)> + 'a {
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
) -> Option<(SimpleReplacement, Vec<RemoveConstIgnore>)> {
    let (in_consts, removals): (Vec<_>, Vec<_>) = hugr
        .node_inputs(op_node)
        .filter_map(|in_p| {
            let (con_op, load_n) = get_const(hugr, op_node, in_p)?;
            Some(((in_p, con_op), RemoveConstIgnore(load_n)))
        })
        .unzip();
    let neighbour_op = hugr.get_optype(op_node);
    // attempt to evaluate op
    let folded = fold_const(neighbour_op, &in_consts)?;
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

    use crate::extension::{ExtensionRegistry, PRELUDE};
    use crate::std_extensions::arithmetic;

    use crate::std_extensions::arithmetic::float_ops::FloatOps;
    use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};

    use rstest::rstest;

    use super::*;

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
        let out = fold_const(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), f2c(c))]);
    }

    #[test]
    fn test_big() {
        /*
           Test hugr approximately calculates
           let x = (5.5, 3.25);
           x.0 - x.1 == 2.25
        */
        let mut build =
            DFGBuilder::new(FunctionType::new(type_row![], type_row![FLOAT64_TYPE])).unwrap();

        let tup = build
            .add_load_const(Const::new_tuple([f2c(5.5), f2c(3.25)]))
            .unwrap();

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

        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::float_types::EXTENSION.to_owned(),
            arithmetic::float_ops::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build.finish_hugr_with_outputs(sub.outputs(), &reg).unwrap();
        assert_eq!(h.node_count(), 7);

        constant_fold_pass(&mut h, &reg);

        assert_fully_folded(&h, &f2c(2.25));
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

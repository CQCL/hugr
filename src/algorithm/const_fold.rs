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

fn const_graph(consts: Vec<Const>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Const::const_type).cloned().collect_vec();
    let mut b = DFGBuilder::new(FunctionType::new(type_row![], const_types)).unwrap();

    let outputs = consts
        .into_iter()
        .map(|c| b.add_load_const(c).unwrap())
        .collect_vec();

    b.finish_hugr_with_outputs(outputs, reg).unwrap()
}

pub fn find_consts<'a, 'r: 'a>(
    hugr: &'a impl HugrView,
    candidate_nodes: impl IntoIterator<Item = Node> + 'a,
    reg: &'r ExtensionRegistry,
) -> impl Iterator<Item = (SimpleReplacement, Vec<RemoveConstIgnore>)> + 'a {
    let mut used_neighbours = BTreeSet::new();

    candidate_nodes
        .into_iter()
        .filter_map(move |n| {
            hugr.get_optype(n).is_load_constant().then_some(())?;

            let (out_p, _) = hugr.out_value_types(n).exactly_one().ok()?;
            let neighbours = hugr
                .linked_inputs(n, out_p)
                .filter(|(n, _)| used_neighbours.insert(*n))
                .collect_vec();
            if neighbours.is_empty() {
                return None;
            }
            let fold_iter = neighbours
                .into_iter()
                .filter_map(|(neighbour, _)| fold_op(hugr, neighbour, reg));
            Some(fold_iter)
        })
        .flatten()
}

fn fold_op(
    hugr: &impl HugrView,
    op_node: Node,
    reg: &ExtensionRegistry,
) -> Option<(SimpleReplacement, Vec<RemoveConstIgnore>)> {
    let (in_consts, removals): (Vec<_>, Vec<_>) = hugr
        .node_inputs(op_node)
        .filter_map(|in_p| get_const(hugr, op_node, in_p))
        .unzip();
    let neighbour_op = hugr.get_optype(op_node);
    let folded = fold_const(neighbour_op, &in_consts)?;
    let (op_outs, consts): (Vec<_>, Vec<_>) = folded.into_iter().unzip();
    let nu_out = op_outs
        .into_iter()
        .flat_map(|out| {
            // map from the ports the op was linked to, to the output ports of
            // the replacement.
            hugr.linked_inputs(op_node, out)
                .enumerate()
                .map(|(i, np)| (np, i.into()))
        })
        .collect();
    let replacement = const_graph(consts, reg);
    let sibling_graph = SiblingSubgraph::try_from_nodes([op_node], hugr)
        .expect("Load consts and operation should form valid subgraph.");

    let simple_replace = SimpleReplacement::new(
        sibling_graph,
        replacement,
        // no inputs to replacement
        HashMap::new(),
        nu_out,
    );
    Some((simple_replace, removals))
}

fn get_const(
    hugr: &impl HugrView,
    op_node: Node,
    in_p: IncomingPort,
) -> Option<((IncomingPort, Const), RemoveConstIgnore)> {
    let (load_n, _) = hugr.single_linked_output(op_node, in_p)?;
    let load_op = hugr.get_optype(load_n).as_load_constant()?;
    let const_node = hugr
        .linked_outputs(load_n, load_op.constant_port())
        .exactly_one()
        .ok()?
        .0;

    let const_op = hugr.get_optype(const_node).as_const()?;

    // TODO avoid const clone here
    Some(((in_p, const_op.clone()), RemoveConstIgnore(load_n)))
}

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
    use crate::hugr::rewrite::consts::RemoveConst;

    use crate::hugr::HugrMut;
    use crate::std_extensions::arithmetic;
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{ConstIntU, INT_TYPES};

    use rstest::rstest;

    use super::*;

    fn i2c(b: u64) -> Const {
        Const::new(
            ConstIntU::new(5, b).unwrap().into(),
            INT_TYPES[5].to_owned(),
        )
        .unwrap()
    }

    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 1, 1)]
    #[case(23, 435, 458)]
    // c = a + b
    fn test_add(#[case] a: u64, #[case] b: u64, #[case] c: u64) {
        let consts = vec![(0.into(), i2c(a)), (1.into(), i2c(b))];
        let add_op: OpType = IntOpDef::iadd.with_width(6).into();
        let out = fold_const(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), i2c(c))]);
    }

    #[test]
    fn test_fold() {
        let mut b = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![INT_TYPES[5].to_owned()],
        ))
        .unwrap();

        let one = b.add_load_const(i2c(1)).unwrap();
        let two = b.add_load_const(i2c(2)).unwrap();

        let add = b
            .add_dataflow_op(IntOpDef::iadd.with_width(5), [one, two])
            .unwrap();
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
            arithmetic::int_ops::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = b.finish_hugr_with_outputs(add.outputs(), &reg).unwrap();
        assert_eq!(h.node_count(), 8);

        let (repl, removes) = find_consts(&h, h.nodes(), &reg).exactly_one().ok().unwrap();
        let [remove_1, remove_2] = removes.try_into().unwrap();

        h.apply_rewrite(repl).unwrap();
        for rem in [remove_1, remove_2] {
            let const_node = h.apply_rewrite(rem).unwrap();
            h.apply_rewrite(RemoveConst(const_node)).unwrap();
        }

        assert_fully_folded(&h, &i2c(3));
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

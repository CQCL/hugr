//! Constant folding routines.

use itertools::Itertools;

use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionSet, PRELUDE_REGISTRY},
    hugr::views::SiblingSubgraph,
    ops::{Const, LeafOp, OpType},
    type_row,
    types::{FunctionType, Type, TypeEnum},
    values::Value,
    Hugr, HugrView, IncomingPort, OutgoingPort, SimpleReplacement,
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

fn const_graph(mut consts: Vec<(OutgoingPort, Const)>) -> Hugr {
    consts.sort_by_key(|(o, _)| *o);
    let consts = consts.into_iter().map(|(_, c)| c).collect_vec();
    let const_types = consts.iter().map(Const::const_type).cloned().collect_vec();
    // TODO need to get const extensions.
    let extensions: ExtensionSet = consts
        .iter()
        .map(|c| c.value().extension_reqs())
        .fold(ExtensionSet::new(), |e, e2| e.union(&e2));
    let mut b = DFGBuilder::new(FunctionType::new(type_row![], const_types)).unwrap();

    let outputs = consts
        .into_iter()
        .map(|c| b.add_load_const(c).unwrap())
        .collect_vec();

    b.finish_hugr_with_outputs(outputs, &PRELUDE_REGISTRY)
        .unwrap()
}

fn find_consts(
    hugr: &impl HugrView,
    reg: &ExtensionRegistry,
) -> impl Iterator<Item = SimpleReplacement> + '_ {
    hugr.nodes().filter_map(|n| {
        let op = hugr.get_optype(n);

        op.is_load_constant().then_some(())?;

        let neighbour = hugr.output_neighbours(n).exactly_one().ok()?;

        let mut remove_nodes = vec![neighbour];

        let all_ins = hugr
            .node_inputs(neighbour)
            .filter_map(|in_p| {
                let (in_n, _) = hugr.single_linked_output(neighbour, in_p)?;
                let op = hugr.get_optype(in_n);

                op.is_load_constant().then_some(())?;

                let const_node = hugr.input_neighbours(in_n).exactly_one().ok()?;
                let const_op = hugr.get_optype(const_node).as_const()?;

                remove_nodes.push(const_node);
                remove_nodes.push(in_n);

                // TODO avoid const clone here
                Some((in_p, const_op.clone()))
            })
            .collect_vec();

        let neighbour_op = hugr.get_optype(neighbour);

        let folded = fold_const(neighbour_op, &all_ins)?;

        let replacement = const_graph(folded);

        let sibling_graph = SiblingSubgraph::try_from_nodes(remove_nodes, hugr)
            .expect("Make unmake should be valid subgraph.");

        sibling_graph
            .create_simple_replacement(hugr, replacement)
            .ok()
    })
}

#[cfg(test)]
mod test {
    use crate::extension::{ExtensionRegistry, PRELUDE};
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
            crate::std_extensions::arithmetic::int_types::EXTENSION.to_owned(),
            crate::std_extensions::arithmetic::int_ops::EXTENSION.to_owned(),
        ])
        .unwrap();
        let h = b.finish_hugr_with_outputs(add.outputs(), &reg).unwrap();

        let consts = find_consts(&h).collect_vec();

        dbg!(consts);
    }
}

//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use hugr::builder::{CFGBuilder, HugrBuilder};
use hugr::hugr::hugrmut::HugrMut;
use itertools::Itertools;

use hugr::hugr::rewrite::inline_dfg::InlineDFG;
use hugr::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr::hugr::RootTagged;
use hugr::ops::handle::CfgID;
use hugr::ops::leaf::UnpackTuple;
use hugr::ops::{DataflowBlock, DataflowParent, Input, Output, DFG};
use hugr::{Hugr, HugrView, Node};

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
pub fn merge_basic_blocks(cfg: &mut impl HugrMut<RootHandle = CfgID>) {
    let mut worklist = cfg.nodes().collect::<Vec<_>>();
    while let Some(n) = worklist.pop() {
        // Consider merging n with its successor
        let Ok(succ) = cfg.output_neighbours(n).exactly_one() else {
            continue;
        };
        if cfg.input_neighbours(succ).count() != 1 {
            continue;
        };
        if cfg.children(cfg.root()).take(2).contains(&succ) {
            // If succ is...
            //   - the entry block, that has an implicit extra in-edge, so cannot merge with n.
            //   - the exit block, nodes in n should move *outside* the CFG - a separate pass.
            continue;
        };
        let (rep, merge_bb, dfgs) = mk_rep(cfg, n, succ);
        let node_map = cfg.hugr_mut().apply_rewrite(rep).unwrap();
        let merged_bb = *node_map.get(&merge_bb).unwrap();
        for dfg_id in dfgs {
            let n_id = *node_map.get(&dfg_id).unwrap();
            cfg.hugr_mut()
                .apply_rewrite(InlineDFG(n_id.into()))
                .unwrap();
        }
        worklist.push(merged_bb);
    }
}

fn mk_rep(
    cfg: &impl RootTagged<RootHandle = CfgID>,
    pred: Node,
    succ: Node,
) -> (Replacement, Node, [Node; 2]) {
    let pred_ty = cfg.get_optype(pred).as_dataflow_block().unwrap();
    let succ_ty = cfg.get_optype(succ).as_dataflow_block().unwrap();
    let succ_sig = succ_ty.inner_signature();

    // Make a Hugr with just a single CFG root node having the same signature.
    let mut replacement: Hugr = CFGBuilder::new(cfg.root_type().op_signature().unwrap())
        .unwrap()
        .finish_prelude_hugr()
        .unwrap();
    replacement.remove_node(replacement.children(replacement.root()).next().unwrap());

    let merged = replacement.add_node_with_parent(replacement.root(), {
        let mut merged_block = DataflowBlock {
            inputs: pred_ty.inputs.clone(),
            ..succ_ty.clone()
        };
        merged_block.extension_delta = merged_block
            .extension_delta
            .union(pred_ty.extension_delta.clone());
        merged_block
    });
    let input = replacement.add_node_with_parent(
        merged,
        Input {
            types: pred_ty.inputs.clone(),
        },
    );
    let output = replacement.add_node_with_parent(
        merged,
        Output {
            types: succ_sig.output.clone(),
        },
    );

    let dfg1 = replacement.add_node_with_parent(
        merged,
        DFG {
            signature: pred_ty.inner_signature().clone(),
        },
    );
    for (i, _) in pred_ty.inputs.iter().enumerate() {
        replacement.connect(input, i, dfg1, i)
    }

    let dfg2 = replacement.add_node_with_parent(
        merged,
        DFG {
            signature: succ_sig.clone(),
        },
    );
    for (i, _) in succ_sig.output.iter().enumerate() {
        replacement.connect(dfg2, i, output, i)
    }

    // At the junction, must unpack the first (tuple, branch predicate) output
    let tuple_elems = pred_ty.sum_rows.clone().into_iter().exactly_one().unwrap();
    let unp = replacement.add_node_with_parent(merged, UnpackTuple::new(tuple_elems.clone()));
    replacement.connect(dfg1, 0, unp, 0);
    let other_start = tuple_elems.len();
    for (i, _) in tuple_elems.iter().enumerate() {
        replacement.connect(unp, i, dfg2, i)
    }
    for (i, _) in pred_ty.other_outputs.iter().enumerate() {
        replacement.connect(dfg1, i + 1, dfg2, i + other_start)
    }
    // If there are edges from succ back to pred, we cannot do these via the mu_inp/out/new
    // edge-maps as both source and target of the new edge are in the replacement Hugr
    for (_, src_pos) in cfg.all_linked_outputs(pred).filter(|(src, _)| *src == succ) {
        replacement.connect(merged, src_pos, merged, 0);
    }
    let rep = Replacement {
        removal: vec![pred, succ],
        replacement,
        adoptions: HashMap::from([(dfg1, pred), (dfg2, succ)]),
        mu_inp: cfg
            .all_linked_outputs(pred)
            .filter(|(src, _)| *src != succ)
            .map(|(src, src_pos)| NewEdgeSpec {
                src,
                tgt: merged,
                kind: NewEdgeKind::ControlFlow { src_pos },
            })
            .collect(),
        mu_out: cfg
            .node_outputs(succ)
            .filter_map(|src_pos| {
                let tgt = cfg
                    .linked_inputs(succ, src_pos)
                    .exactly_one()
                    .ok()
                    .unwrap()
                    .0;
                if tgt == pred {
                    None
                } else {
                    Some(NewEdgeSpec {
                        src: merged,
                        tgt,
                        kind: NewEdgeKind::ControlFlow { src_pos },
                    })
                }
            })
            .collect(),
        mu_new: vec![],
    };
    (rep, merged, [dfg1, dfg2])
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use itertools::Itertools;
    use rstest::rstest;

    use hugr::builder::{CFGBuilder, DFGWrapper, Dataflow, HugrBuilder};
    use hugr::extension::prelude::{ConstUsize, PRELUDE_ID, QB_T, USIZE_T};
    use hugr::extension::{ExtensionRegistry, ExtensionSet, PRELUDE, PRELUDE_REGISTRY};
    use hugr::hugr::views::sibling::SiblingMut;
    use hugr::ops::constant::Value;
    use hugr::ops::handle::CfgID;
    use hugr::ops::{Lift, LoadConstant, Noop, OpTrait, OpType};
    use hugr::types::{FunctionType, Type, TypeRow};
    use hugr::{const_extension_ids, type_row, Extension, Hugr, HugrView, Wire};

    use super::merge_basic_blocks;

    const_extension_ids! {
        const EXT_ID: ExtensionId = "TestExt";
    }

    fn extension() -> Extension {
        let mut e = Extension::new(EXT_ID);
        e.add_op(
            "Test".into(),
            String::new(),
            FunctionType::new(
                type_row![QB_T, USIZE_T],
                TypeRow::from(vec![Type::new_sum(vec![
                    type_row![QB_T],
                    type_row![USIZE_T],
                ])]),
            ),
        )
        .unwrap();
        e
    }

    fn lifted_unary_unit_sum<B: AsMut<Hugr> + AsRef<Hugr>, T>(b: &mut DFGWrapper<B, T>) -> Wire {
        let lc = b.add_load_value(Value::unary_unit_sum());
        let lift = b
            .add_dataflow_op(Lift::new(Type::new_unit_sum(1).into(), PRELUDE_ID), [lc])
            .unwrap();
        let [w] = lift.outputs_arr();
        w
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn in_loop(#[case] self_loop: bool) -> Result<(), Box<dyn std::error::Error>> {
        /* self_loop==False:
           -> Noop1 -----> Test -> Exit       -> Noop1AndTest --> Exit
               |            |            =>     /            \
               \-<- Noop2 <-/                   \-<- Noop2 <-/
           (Noop2 -> Noop1 cannot be merged because Noop1 is the entry node)

           self_loop==True:
           -> Noop --> Test -> Exit           -> NoopAndTest --> Exit
               |        |                =>     /           \
               \--<--<--/                       \--<-----<--/
        */
        let loop_variants = type_row![QB_T];
        let exit_types = type_row![USIZE_T];
        let e = extension();
        let tst_op = e.instantiate_extension_op("Test", [], &PRELUDE_REGISTRY)?;
        let reg = ExtensionRegistry::try_new([PRELUDE.to_owned(), e])?;
        let mut h = CFGBuilder::new(
            FunctionType::new(loop_variants.clone(), exit_types.clone())
                .with_extension_delta(ExtensionSet::singleton(&PRELUDE_ID)),
        )?;
        let mut no_b1 = h.simple_entry_builder(loop_variants.clone(), 1, ExtensionSet::new())?;
        let n = no_b1.add_dataflow_op(Noop::new(QB_T), no_b1.input_wires())?;
        let br = lifted_unary_unit_sum(&mut no_b1);
        let no_b1 = no_b1.finish_with_outputs(br, n.outputs())?;
        let mut test_block = h.block_builder(
            loop_variants.clone(),
            vec![loop_variants.clone(), exit_types],
            ExtensionSet::singleton(&PRELUDE_ID),
            type_row![],
        )?;
        let [test_input] = test_block.input_wires_arr();
        let usize_cst = test_block.add_load_value(ConstUsize::new(1));
        let [tst] = test_block
            .add_dataflow_op(tst_op, [test_input, usize_cst])?
            .outputs_arr();
        let test_block = test_block.finish_with_outputs(tst, [])?;
        let loop_backedge_target = if self_loop {
            no_b1
        } else {
            let mut no_b2 = h.simple_block_builder(FunctionType::new_endo(loop_variants), 1)?;
            let n = no_b2.add_dataflow_op(Noop::new(QB_T), no_b2.input_wires())?;
            let br = lifted_unary_unit_sum(&mut no_b2);
            let nid = no_b2.finish_with_outputs(br, n.outputs())?;
            h.branch(&nid, 0, &no_b1)?;
            nid
        };
        h.branch(&no_b1, 0, &test_block)?;
        h.branch(&test_block, 0, &loop_backedge_target)?;
        h.branch(&test_block, 1, &h.exit_block())?;

        let mut h = h.finish_hugr(&reg)?;
        let r = h.root();
        merge_basic_blocks(&mut SiblingMut::<CfgID>::try_new(&mut h, r)?);
        h.update_validate(&reg).unwrap();
        assert_eq!(r, h.root());
        assert!(matches!(h.get_optype(r), OpType::CFG(_)));
        let [entry, exit] = h
            .children(r)
            .take(2)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        // Check the Noop('s) is/are in the right block(s)
        let nops = h
            .nodes()
            .filter(|n| matches!(h.get_optype(*n), OpType::Noop(_)));
        let (entry_nop, expected_backedge_target) = if self_loop {
            assert_eq!(h.children(r).len(), 2);
            (nops.exactly_one().ok().unwrap(), entry)
        } else {
            let [_, _, no_b2] = h.children(r).collect::<Vec<_>>().try_into().unwrap();
            let mut nops = nops.collect::<Vec<_>>();
            let entry_nop_idx = nops
                .iter()
                .position(|n| h.get_parent(*n) == Some(entry))
                .unwrap();
            let entry_nop = nops[entry_nop_idx];
            nops.remove(entry_nop_idx);
            let [n_op2] = nops.try_into().unwrap();
            assert_eq!(h.get_parent(n_op2), Some(no_b2));
            (entry_nop, no_b2)
        };
        assert_eq!(h.get_parent(entry_nop), Some(entry));
        assert_eq!(
            h.output_neighbours(entry).collect::<HashSet<_>>(),
            HashSet::from([expected_backedge_target, exit])
        );
        // And the Noop in the entry block is consumed by the custom Test op
        let tst = find_unique(h.nodes(), |n| {
            matches!(h.get_optype(*n), OpType::CustomOp(_))
        });
        assert_eq!(h.get_parent(tst), Some(entry));
        assert_eq!(
            h.output_neighbours(entry_nop).collect::<Vec<_>>(),
            vec![tst]
        );
        Ok(())
    }

    #[test]
    fn triple_with_permute() -> Result<(), Box<dyn std::error::Error>> {
        // Blocks are just BB1 -> BB2 -> BB3 --> Exit.
        // CFG Normalization would move everything outside the CFG and elide the CFG altogether,
        // but this is an easy-to-construct test of merge-basic-blocks only (no CFG normalization).
        let e = extension();
        let tst_op: OpType = e
            .instantiate_extension_op("Test", &[], &PRELUDE_REGISTRY)?
            .into();
        let [res_t] = tst_op
            .dataflow_signature()
            .unwrap()
            .output
            .into_owned()
            .try_into()
            .unwrap();
        let mut h = CFGBuilder::new(
            FunctionType::new(QB_T, res_t.clone())
                .with_extension_delta(ExtensionSet::singleton(&PRELUDE_ID)),
        )?;
        let mut bb1 = h.entry_builder(
            vec![type_row![]],
            type_row![USIZE_T, QB_T],
            ExtensionSet::singleton(&PRELUDE_ID),
        )?;
        let [inw] = bb1.input_wires_arr();
        let load_cst = bb1.add_load_value(ConstUsize::new(1));
        let pred = lifted_unary_unit_sum(&mut bb1);
        let bb1 = bb1.finish_with_outputs(pred, [load_cst, inw])?;

        let mut bb2 = h.block_builder(
            type_row![USIZE_T, QB_T],
            vec![type_row![]],
            ExtensionSet::new(),
            type_row![QB_T, USIZE_T],
        )?;
        let [u, q] = bb2.input_wires_arr();
        let pred = lifted_unary_unit_sum(&mut bb2);
        let bb2 = bb2.finish_with_outputs(pred, [q, u])?;

        let mut bb3 = h.block_builder(
            type_row![QB_T, USIZE_T],
            vec![type_row![]],
            ExtensionSet::new(),
            res_t.clone().into(),
        )?;
        let [q, u] = bb3.input_wires_arr();
        let tst = bb3.add_dataflow_op(tst_op, [q, u])?;
        let pred = lifted_unary_unit_sum(&mut bb3);
        let bb3 = bb3.finish_with_outputs(pred, tst.outputs())?;
        // Now add control-flow edges between basic blocks
        h.branch(&bb1, 0, &bb2)?;
        h.branch(&bb2, 0, &bb3)?;
        h.branch(&bb3, 0, &h.exit_block())?;

        let reg = ExtensionRegistry::try_new([e, PRELUDE.to_owned()])?;
        let mut h = h.finish_hugr(&reg)?;
        let root = h.root();
        merge_basic_blocks(&mut SiblingMut::try_new(&mut h, root)?);
        h.update_validate(&reg)?;

        // Should only be one BB left
        let [bb, _exit] = h.children(h.root()).collect::<Vec<_>>().try_into().unwrap();
        let tst = find_unique(h.nodes(), |n| {
            matches!(h.get_optype(*n), OpType::CustomOp(_))
        });
        assert_eq!(h.get_parent(tst), Some(bb));

        let inp = find_unique(h.nodes(), |n| matches!(h.get_optype(*n), OpType::Input(_)));
        let mut tst_inputs = h.input_neighbours(tst).collect::<Vec<_>>();
        tst_inputs.remove(tst_inputs.iter().find_position(|n| **n == inp).unwrap().0);
        let [other_input] = tst_inputs.try_into().unwrap();
        assert_eq!(
            h.get_optype(other_input),
            &(LoadConstant { datatype: USIZE_T }.into())
        );
        Ok(())
    }

    fn find_unique<T>(items: impl Iterator<Item = T>, pred: impl Fn(&T) -> bool) -> T {
        items.filter(pred).exactly_one().ok().unwrap()
    }
}

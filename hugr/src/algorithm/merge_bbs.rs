//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use itertools::Itertools;

use crate::hugr::rewrite::inline_dfg::InlineDFG;
use crate::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use crate::hugr::{HugrMut, RootTagged};
use crate::ops::handle::CfgID;
use crate::ops::leaf::UnpackTuple;
use crate::ops::{DataflowBlock, DataflowParent, Input, Output, DFG};
use crate::{Hugr, HugrView, Node};

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
pub fn merge_basic_blocks(cfg: &mut impl HugrMut<RootHandle = CfgID>) {
    for n in cfg.nodes().collect::<Vec<_>>().into_iter() {
        let Ok(succ) = cfg.output_neighbours(n).exactly_one() else {
            continue;
        };
        if cfg.input_neighbours(succ).take(2).collect::<Vec<_>>() != vec![n] {
            continue;
        };
        if cfg.nodes().take(2).contains(&succ) {
            // entry block has an additional in-edge, so cannot merge with predecessor.
            // if succ is exit block, nodes in n==p should move *outside* the CFG
            // - a separate normalization from merging BBs.
            continue;
        };
        let (rep, dfg1, dfg2) = mk_rep(cfg, n, succ);
        let node_map = cfg.hugr_mut().apply_rewrite(rep).unwrap();
        for dfg_id in [dfg1, dfg2] {
            let n_id = *node_map.get(&dfg_id).unwrap();
            cfg.hugr_mut()
                .apply_rewrite(InlineDFG(n_id.into()))
                .unwrap();
        }
    }
}

fn mk_rep(
    cfg: &impl RootTagged<RootHandle = CfgID>,
    pred: Node,
    succ: Node,
) -> (Replacement, Node, Node) {
    let pred_ty = cfg.get_optype(pred).as_dataflow_block().unwrap();
    let succ_ty = cfg.get_optype(succ).as_dataflow_block().unwrap();
    let succ_sig = succ_ty.inner_signature();
    let mut replacement: Hugr = Hugr::new(cfg.root_type().clone());
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
    let unp = replacement.add_node_with_parent(
        merged,
        UnpackTuple {
            tys: tuple_elems.clone(),
        },
    );
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
    (rep, dfg1, dfg2)
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::builder::{CFGBuilder, Dataflow, HugrBuilder};
    use crate::extension::prelude::{QB_T, USIZE_T};
    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE, PRELUDE_REGISTRY};
    use crate::hugr::views::sibling::SiblingMut;
    use crate::ops::constant::Value;
    use crate::ops::handle::CfgID;
    use crate::ops::{Noop, OpType};
    use crate::types::{FunctionType, Type, TypeRow};
    use crate::{const_extension_ids, type_row, Extension, HugrView};

    use super::merge_basic_blocks;

    const_extension_ids! {
        const EXT_ID: ExtensionId = "TestExt";
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
        let mut e = Extension::new(EXT_ID);
        e.add_op(
            "Test".into(),
            String::new(),
            FunctionType::new(
                loop_variants.clone(),
                TypeRow::from(vec![Type::new_sum(vec![
                    loop_variants.clone(),
                    exit_types.clone(),
                ])]),
            ),
        )?;
        let tst_op = e.instantiate_extension_op("Test", [], &PRELUDE_REGISTRY)?;
        let reg = ExtensionRegistry::try_new([PRELUDE.to_owned(), e])?;
        let mut h = CFGBuilder::new(FunctionType::new(loop_variants.clone(), exit_types.clone()))?;
        let mut no_b1 = h.simple_entry_builder(loop_variants.clone(), 1, ExtensionSet::new())?;
        let n = no_b1.add_dataflow_op(Noop { ty: QB_T }, no_b1.input_wires())?;
        let br = no_b1.add_load_const(Value::unary_unit_sum());
        let no_b1 = no_b1.finish_with_outputs(br, n.outputs())?;
        let mut test_block = h.block_builder(
            loop_variants.clone(),
            vec![loop_variants.clone(), exit_types],
            ExtensionSet::singleton(&EXT_ID),
            type_row![],
        )?;
        let [tst] = test_block
            .add_dataflow_op(tst_op, test_block.input_wires())?
            .outputs_arr();
        let test_block = test_block.finish_with_outputs(tst, [])?;
        let loop_backedge_target = if self_loop {
            no_b1
        } else {
            let mut no_b2 = h.simple_block_builder(FunctionType::new_endo(loop_variants), 1)?;
            let n = no_b2.add_dataflow_op(Noop { ty: QB_T }, no_b2.input_wires())?;
            let br = no_b2.add_load_const(Value::unary_unit_sum());
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
        h.validate(&reg).unwrap();
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
        let tst = h
            .nodes()
            .filter(|n| matches!(h.get_optype(*n), OpType::CustomOp(_)))
            .exactly_one()
            .ok()
            .unwrap();
        assert_eq!(h.get_parent(tst), Some(entry));
        assert_eq!(
            h.output_neighbours(entry_nop).collect::<Vec<_>>(),
            vec![tst]
        );
        Ok(())
    }
}

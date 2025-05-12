//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use hugr_core::extension::prelude::UnpackTuple;
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::views::RootCheckable;
use itertools::Itertools;

use hugr_core::hugr::patch::inline_dfg::InlineDFG;
use hugr_core::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr_core::ops::handle::CfgID;
use hugr_core::ops::{DFG, DataflowBlock, DataflowParent, Input, Output};
use hugr_core::{Hugr, HugrView, Node};

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
pub fn merge_basic_blocks<'h, H>(cfg: impl RootCheckable<&'h mut H, CfgID>)
where
    H: 'h + HugrMut<Node = Node>,
{
    let checked = cfg.try_into_checked().expect("Hugr must be a CFG region");
    let cfg = checked.into_hugr();

    let mut worklist = cfg.children(cfg.entrypoint()).collect::<Vec<_>>();
    while let Some(n) = worklist.pop() {
        // Consider merging n with its successor
        let Ok(succ) = cfg.output_neighbours(n).exactly_one() else {
            continue;
        };
        if cfg.input_neighbours(succ).count() != 1 {
            continue;
        }
        if cfg.children(cfg.entrypoint()).take(2).contains(&succ) {
            // If succ is...
            //   - the entry block, that has an implicit extra in-edge, so cannot merge with n.
            //   - the exit block, nodes in n should move *outside* the CFG - a separate pass.
            continue;
        }
        let (rep, merge_bb, dfgs) = mk_rep(cfg, n, succ);
        let node_map = cfg.apply_patch(rep).unwrap();
        let merged_bb = *node_map.get(&merge_bb).unwrap();
        for dfg_id in dfgs {
            let n_id = *node_map.get(&dfg_id).unwrap();
            cfg.apply_patch(InlineDFG(n_id.into())).unwrap();
        }
        worklist.push(merged_bb);
    }
}

fn mk_rep(
    cfg: &impl HugrView<Node = Node>,
    pred: Node,
    succ: Node,
) -> (Replacement, Node, [Node; 2]) {
    let pred_ty = cfg.get_optype(pred).as_dataflow_block().unwrap();
    let succ_ty = cfg.get_optype(succ).as_dataflow_block().unwrap();
    let succ_sig = succ_ty.inner_signature();

    // Make a Hugr with just a single CFG root node having the same signature.
    let mut replacement: Hugr = Hugr::new_with_entrypoint(cfg.entrypoint_optype().clone())
        .expect("Replacement should have a CFG entrypoint");

    let merged = replacement.add_node_with_parent(replacement.entrypoint(), {
        DataflowBlock {
            inputs: pred_ty.inputs.clone(),
            ..succ_ty.clone()
        }
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
            signature: pred_ty.inner_signature().into_owned(),
        },
    );
    for (i, _) in pred_ty.inputs.iter().enumerate() {
        replacement.connect(input, i, dfg1, i);
    }

    let dfg2 = replacement.add_node_with_parent(
        merged,
        DFG {
            signature: succ_sig.as_ref().clone(),
        },
    );
    for (i, _) in succ_sig.output.iter().enumerate() {
        replacement.connect(dfg2, i, output, i);
    }

    // At the junction, must unpack the first (tuple, branch predicate) output
    let tuple_elems = pred_ty.sum_rows.clone().into_iter().exactly_one().unwrap();
    let unp = replacement.add_node_with_parent(merged, UnpackTuple::new(tuple_elems.clone()));
    replacement.connect(dfg1, 0, unp, 0);
    let other_start = tuple_elems.len();
    for (i, _) in tuple_elems.iter().enumerate() {
        replacement.connect(unp, i, dfg2, i);
    }
    for (i, _) in pred_ty.other_outputs.iter().enumerate() {
        replacement.connect(dfg1, i + 1, dfg2, i + other_start);
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
    use std::sync::Arc;

    use hugr_core::extension::prelude::PRELUDE_ID;
    use itertools::Itertools;
    use rstest::rstest;

    use hugr_core::builder::{CFGBuilder, DFGWrapper, Dataflow, HugrBuilder, endo_sig, inout_sig};
    use hugr_core::extension::prelude::{ConstUsize, qb_t, usize_t};
    use hugr_core::ops::constant::Value;
    use hugr_core::ops::{LoadConstant, OpTrait, OpType};
    use hugr_core::types::{Signature, Type, TypeRow};
    use hugr_core::{Extension, Hugr, HugrView, Wire, const_extension_ids, type_row};

    use super::merge_basic_blocks;

    const_extension_ids! {
        const EXT_ID: ExtensionId = "TestExt";
    }

    fn extension() -> Arc<Extension> {
        Extension::new_arc(
            EXT_ID,
            hugr_core::extension::Version::new(0, 1, 0),
            |ext, extension_ref| {
                ext.add_op(
                    "Test".into(),
                    String::new(),
                    Signature::new(
                        vec![qb_t(), usize_t()],
                        TypeRow::from(vec![Type::new_sum(vec![vec![qb_t()], vec![usize_t()]])]),
                    ),
                    extension_ref,
                )
                .unwrap();
            },
        )
    }

    fn unary_unit_sum<B: AsMut<Hugr> + AsRef<Hugr>, T>(b: &mut DFGWrapper<B, T>) -> Wire {
        b.add_load_value(Value::unary_unit_sum())
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

        use hugr_core::extension::prelude::Noop;
        let loop_variants: TypeRow = vec![qb_t()].into();
        let exit_types: TypeRow = vec![usize_t()].into();
        let e = extension();
        let tst_op = e.instantiate_extension_op("Test", [])?;
        let mut h = CFGBuilder::new(inout_sig(loop_variants.clone(), exit_types.clone()))?;
        let mut no_b1 = h.simple_entry_builder(loop_variants.clone(), 1)?;
        let n = no_b1.add_dataflow_op(Noop::new(qb_t()), no_b1.input_wires())?;
        let br = unary_unit_sum(&mut no_b1);
        let no_b1 = no_b1.finish_with_outputs(br, n.outputs())?;
        let mut test_block = h.block_builder(
            loop_variants.clone(),
            vec![loop_variants.clone(), exit_types],
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
            let mut no_b2 = h.simple_block_builder(endo_sig(loop_variants), 1)?;
            let n = no_b2.add_dataflow_op(Noop::new(qb_t()), no_b2.input_wires())?;
            let br = unary_unit_sum(&mut no_b2);
            let nid = no_b2.finish_with_outputs(br, n.outputs())?;
            h.branch(&nid, 0, &no_b1)?;
            nid
        };
        h.branch(&no_b1, 0, &test_block)?;
        h.branch(&test_block, 0, &loop_backedge_target)?;
        h.branch(&test_block, 1, &h.exit_block())?;

        let mut h = h.finish_hugr()?;
        let r = h.entrypoint();
        merge_basic_blocks(&mut h);
        h.validate().unwrap();
        assert_eq!(r, h.entrypoint());
        assert!(matches!(h.get_optype(r), OpType::CFG(_)));
        let [entry, exit] = h
            .children(r)
            .take(2)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        // Check the Noop('s) is/are in the right block(s)
        let nops = h
            .entry_descendants()
            .filter(|n| h.get_optype(*n).cast::<Noop>().is_some());
        let (entry_nop, expected_backedge_target) = if self_loop {
            assert_eq!(h.children(r).count(), 2);
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
        let tst = find_unique(
            h.entry_descendants(),
            |n| matches!(h.get_optype(*n), OpType::ExtensionOp(c) if c.def().extension_id() != &PRELUDE_ID),
        );
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
        let tst_op: OpType = e.instantiate_extension_op("Test", &[])?.into();
        let [res_t] = tst_op
            .dataflow_signature()
            .unwrap()
            .into_owned()
            .output
            .into_owned()
            .try_into()
            .unwrap();
        let mut h = CFGBuilder::new(inout_sig(qb_t(), res_t.clone()))?;
        let mut bb1 = h.simple_entry_builder(vec![usize_t(), qb_t()].into(), 1)?;
        let [inw] = bb1.input_wires_arr();
        let load_cst = bb1.add_load_value(ConstUsize::new(1));
        let pred = unary_unit_sum(&mut bb1);
        let bb1 = bb1.finish_with_outputs(pred, [load_cst, inw])?;

        let mut bb2 = h.block_builder(
            vec![usize_t(), qb_t()].into(),
            vec![type_row![]],
            vec![qb_t(), usize_t()].into(),
        )?;
        let [u, q] = bb2.input_wires_arr();
        let pred = unary_unit_sum(&mut bb2);
        let bb2 = bb2.finish_with_outputs(pred, [q, u])?;

        let mut bb3 = h.block_builder(
            vec![qb_t(), usize_t()].into(),
            vec![type_row![]],
            res_t.clone().into(),
        )?;
        let [q, u] = bb3.input_wires_arr();
        let tst = bb3.add_dataflow_op(tst_op, [q, u])?;
        let pred = unary_unit_sum(&mut bb3);
        let bb3 = bb3.finish_with_outputs(pred, tst.outputs())?;
        // Now add control-flow edges between basic blocks
        h.branch(&bb1, 0, &bb2)?;
        h.branch(&bb2, 0, &bb3)?;
        h.branch(&bb3, 0, &h.exit_block())?;

        let mut h = h.finish_hugr()?;
        merge_basic_blocks(&mut h);
        h.validate()?;

        // Should only be one BB left
        let [bb, _exit] = h
            .children(h.entrypoint())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let tst = find_unique(
            h.entry_descendants(),
            |n| matches!(h.get_optype(*n), OpType::ExtensionOp(c) if c.def().extension_id() != &PRELUDE_ID),
        );
        assert_eq!(h.get_parent(tst), Some(bb));

        let inp = find_unique(h.entry_descendants(), |n| {
            matches!(h.get_optype(*n), OpType::Input(_))
        });
        let mut tst_inputs = h.input_neighbours(tst).collect::<Vec<_>>();
        tst_inputs.remove(tst_inputs.iter().find_position(|n| **n == inp).unwrap().0);
        let [other_input] = tst_inputs.try_into().unwrap();
        assert_eq!(
            h.get_optype(other_input),
            &(LoadConstant {
                datatype: usize_t()
            }
            .into())
        );
        Ok(())
    }

    fn find_unique<T>(items: impl Iterator<Item = T>, pred: impl Fn(&T) -> bool) -> T {
        items.filter(pred).exactly_one().ok().unwrap()
    }
}

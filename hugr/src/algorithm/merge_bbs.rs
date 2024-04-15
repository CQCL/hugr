//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use itertools::Itertools;

use crate::hugr::rewrite::inline_dfg::InlineDFG;
use crate::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use crate::hugr::{HugrMut, RootTagged};
use crate::ops::{handle::CfgID, DataflowBlock, DataflowParent, DFG};
use crate::{Hugr, HugrView, Node};

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
pub fn merge_basic_blocks(cfg: &mut impl HugrMut<RootHandle = CfgID>) {
    for n in cfg.nodes().collect::<Vec<_>>().into_iter() {
        let Ok(succ) = cfg.output_neighbours(n).exactly_one() else {
            continue;
        };
        let Ok(p) = cfg.input_neighbours(succ).exactly_one() else {
            continue;
        };
        assert_eq!(n, p);
        let (rep, dfg1, dfg2) = mk_rep(cfg, n, succ);
        let node_map = cfg.apply_rewrite(rep).unwrap();
        for dfg_id in [dfg1, dfg2] {
            let n_id = *node_map.get(&dfg_id).unwrap();
            cfg.apply_rewrite(InlineDFG(n_id.into())).unwrap();
        }
    }
}

fn mk_rep(
    cfg: &impl RootTagged<RootHandle = CfgID>,
    pred: Node,
    succ: Node,
) -> (Replacement, Node, Node) {
    let pred_ty = cfg.get_optype(pred).as_dataflow_block().unwrap();
    let succ_ty = cfg.get_optype(succ).as_dataflow_block().unwrap().clone();
    let signature = succ_ty.inner_signature().clone();
    let mut replacement: Hugr = Hugr::new(cfg.root_type().clone());
    let merged = replacement.add_node_with_parent(
        replacement.root(),
        DataflowBlock {
            inputs: pred_ty.inputs.clone(),
            extension_delta: pred_ty
                .extension_delta
                .clone()
                .union(succ_ty.extension_delta),
            ..succ_ty
        },
    );
    let dfg1 = replacement.add_node_with_parent(
        merged,
        DFG {
            signature: signature.clone(),
        },
    );
    let dfg2 = replacement.add_node_with_parent(merged, DFG { signature });
    for (i, _) in pred_ty.inner_signature().output().iter().enumerate() {
        replacement.connect(dfg1, i, dfg2, i)
    }
    let rep = Replacement {
        removal: vec![pred, succ],
        replacement,
        adoptions: HashMap::from([(dfg1, pred), (dfg2, succ)]),
        mu_inp: cfg
            .all_linked_outputs(pred)
            .map(|(src, src_pos)| NewEdgeSpec {
                src,
                tgt: merged,
                kind: NewEdgeKind::ControlFlow { src_pos },
            })
            .collect(),
        mu_out: cfg
            .node_outputs(succ)
            .map(|src_pos| NewEdgeSpec {
                src: merged,
                tgt: cfg
                    .linked_inputs(succ, src_pos)
                    .exactly_one()
                    .ok()
                    .unwrap()
                    .0,
                kind: NewEdgeKind::ControlFlow { src_pos },
            })
            .collect(),
        mu_new: vec![],
    };
    (rep, dfg1, dfg2)
}

#[cfg(test)]
mod test {
    use crate::builder::{CFGBuilder, Dataflow, HugrBuilder};
    use crate::extension::prelude::{QB_T, USIZE_T};
    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE};
    use crate::ops::{Const, Noop};
    use crate::types::{FunctionType, Type, TypeRow};
    use crate::{const_extension_ids, type_row, Extension};

    const_extension_ids! {
        const EXT_ID: ExtensionId = "TestExt";
    }
    #[test]
    fn in_loop() -> Result<(), Box<dyn std::error::Error>> {
        /* Noop -> Test -> Exit      NoopTest -> Exit
           |       |            =>   |    |
           \-<---<-/                 \<-<-/
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
        let reg = ExtensionRegistry::try_new([PRELUDE.to_owned(), e])?;
        let e = reg.get(&EXT_ID.to_string()).unwrap();
        let mut h = CFGBuilder::new(FunctionType::new(loop_variants.clone(), exit_types.clone()))?;
        let mut noop_block =
            h.simple_entry_builder(loop_variants.clone(), 1, ExtensionSet::new())?;
        let n = noop_block.add_dataflow_op(Noop { ty: QB_T }, noop_block.input_wires())?;
        let br = noop_block.add_load_const(Const::unary_unit_sum());
        let noop_block = noop_block.finish_with_outputs(br, n.outputs())?;
        let mut test_block = h.block_builder(
            loop_variants.clone(),
            vec![loop_variants.clone(), exit_types],
            ExtensionSet::singleton(&EXT_ID),
            type_row![],
        )?;
        let [tst] = test_block
            .add_dataflow_op(
                e.instantiate_extension_op("Test", [], &reg)?,
                test_block.input_wires(),
            )?
            .outputs_arr();
        let test_block = test_block.finish_with_outputs(tst, [])?;
        let exit_block = h.exit_block();
        h.branch(&noop_block, 0, &test_block)?;
        h.branch(&test_block, 0, &noop_block)?;
        h.branch(&test_block, 1, &exit_block)?;
        h.finish_hugr(&reg)?;
        Ok(())
    }
}

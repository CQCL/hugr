//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::views::RootCheckable;
use itertools::Itertools;

use hugr_core::hugr::patch::inline_dfg::InlineDFG;
use hugr_core::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr_core::ops::handle::CfgID;
use hugr_core::ops::{DFG, DataflowBlock, DataflowParent, Input, Output};
use hugr_core::{Hugr, HugrView, Node};

use crate::normalize_cfg::wire_unpack_first;

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
///
/// # Panics
///
/// If the [HugrView::entrypoint] of `cfg` is not an [OpType::CFG]
///
/// [OpType::CFG]: hugr_core::ops::OpType::CFG
#[deprecated(note = "Use normalize_cfg", since = "0.15.1")] // Note: as a first step, just hide this
pub fn merge_basic_blocks<'h, H>(cfg: impl RootCheckable<&'h mut H, CfgID<H::Node>>)
where
    H: 'h + HugrMut,
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
        #[expect(deprecated)] // undeprecate when hidden
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

fn mk_rep<H: HugrView>(
    cfg: &H,
    pred: H::Node,
    succ: H::Node,
) -> (Replacement<H::Node>, Node, [Node; 2]) {
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
    let dfg1_outs = replacement
        .out_value_types(dfg1)
        .enumerate()
        .map(|(i, _)| (dfg1, i.into()))
        .collect::<Vec<_>>();

    let dfg_order_out = replacement.get_optype(dfg1).other_output_port().unwrap();
    // Do not add Order edges between DFGs unless there are no value edges
    let order_srcs = (dfg1_outs.is_empty()).then_some((dfg1, dfg_order_out));

    wire_unpack_first(&mut replacement, dfg1_outs, order_srcs, dfg2);

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

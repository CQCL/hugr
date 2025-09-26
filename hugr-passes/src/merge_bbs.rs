//! Merge BBs along control-flow edges where the source BB has no other successors
//! and the target BB has no other predecessors.
use std::collections::HashMap;

use hugr_core::extension::prelude::UnpackTuple;
use hugr_core::hugr::views::{RootCheckable, RootChecked};
use hugr_core::hugr::{HugrError, hugrmut::HugrMut};
use hugr_core::types::{EdgeKind, Signature, TypeRow};
use itertools::Itertools;

use hugr_core::hugr::patch::inline_dfg::InlineDFG;
use hugr_core::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr_core::ops::handle::CfgID;
use hugr_core::ops::{DFG, DataflowBlock, DataflowParent, ExitBlock, Input, OpTag, OpType, Output};
use hugr_core::{Direction, Hugr, HugrView, Node, OutgoingPort, PortIndex};

/// Merge any basic blocks that are direct children of the specified CFG
/// i.e. where a basic block B has a single successor B' whose only predecessor
/// is B, B and B' can be combined.
///
/// # Panics
///
/// If the [HugrMut::entrypoint] of `cfg` is not an [OpType::CFG]
///
/// [OpType::CFG]: hugr_core::ops::OpType::CFG
#[deprecated(note = "Use normalize_cfg")] // Note: as a first step, just hide this
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

/// Errors from [normalize_cfg]
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum NormalizeCFGError {
    /// The requested node was not a CFG
    #[error("Requested node was not a CFG but {_0}")]
    NotCFG(OpTag),
}

/// Result from [normalize_cfg], i.e. a report of what changes were made to the Hugr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormalizeCFGResult {
    /// The entire [CFG] was converted into a [DFG]
    CFGToDFG,
    /// The CFG was preserved, but the entry or exit blocks may have changed.
    #[allow(missing_docs)]
    CFGPreserved {
        entry_changed: bool,
        exit_changed: bool,
    },
}

/// Normalize a CFG in a Hugr:
/// * Merge consecutive basic blocks i.e. where a BB has only a single successor which
///   has no predecessors
/// * If the entry block has only one successor, and no predecessors, then move its contents
///   outside/before CFG.
/// * (Similarly) if the exit block has only one predecessor, then move contents
///   outside/after CFG.
///    * If that predecessor is the entry block, then remove the CFG.
///
/// *Note that this may remove the entrypoint*; such will be reported in the result
/// ([NormalizeCFGResult::CFGRemoved])
///
/// # Errors
///
/// [NormalizeCFGError::NotCFG] If the entrypoint is not a CFG
#[allow(deprecated)] // inline/combine/refactor with merge_bbs, or just hide latter
pub fn normalize_cfg<H: HugrMut<Node = Node>>(
    mut h: &mut H,
) -> Result<NormalizeCFGResult, NormalizeCFGError> {
    let checked: RootChecked<_, CfgID<H::Node>> = RootChecked::<_, CfgID<H::Node>>::try_new(&mut h)
        .map_err(|e| match e {
            HugrError::InvalidTag { actual, .. } => NormalizeCFGError::NotCFG(actual),
            _ => unreachable!(),
        })?;
    merge_basic_blocks(checked);
    let cfg_node = h.entrypoint();

    // Further normalizations with effects outside the CFG
    let [entry, exit] = h.children(cfg_node).take(2).collect_array().unwrap();
    let entry_blk = h.get_optype(entry).as_dataflow_block().unwrap();
    let cfg_parent = h.get_parent(cfg_node).unwrap();
    // 1. If the entry block has only one successor, and no predecessors, then move contents
    // outside/before CFG. (Note if the entry block has multiple successors, but no predecessors,
    // we could move its contents outside (before) the CFG, but would need to keep an empty/identity
    // entry block - we do not do this here
    let mut entry_changed = false;
    #[allow(clippy::match_result_ok)] // let Ok(...) without .ok() borrows `cfg`
    if let Some(succ) = h.output_neighbours(entry).exactly_one().ok() {
        if succ == exit {
            // Ignore any other predecessors as they are unreachable!
            assert_eq!(
                &Signature::new(
                    entry_blk.inputs.clone(),
                    entry_blk.successor_input(0).unwrap()
                ),
                h.signature(cfg_node).unwrap().as_ref()
            );
            // Turn the CFG into a DFG containing only what was in the entry block
            // Annoying here - "while let Some(blk) = cfg.children(...).skip(1).next()" keeps iterator alive
            let children_to_remove: Vec<_> = h.children(cfg_node).skip(1).collect();
            for blk in children_to_remove {
                h.remove_subtree(blk);
            }
            while let Some(ch) = h.first_child(entry) {
                h.set_parent(ch, cfg_node);
            }
            h.remove_node(entry);
            let cfg_ty = h.optype_mut(cfg_node);
            let OpType::CFG(cfg_) = std::mem::take(cfg_ty) else {
                panic!()
            };
            *cfg_ty = OpType::DFG(DFG {
                signature: cfg_.signature,
            });
            // Unpack the first output and shuffle all the rest along
            let [_, outp] = h.get_io(cfg_node).unwrap();
            let (values, orders) = take_inputs(h, outp);
            let mut unpacked = tuple_elems(h, values[0].0, values[0].1).into_owned();
            h.add_ports(outp, Direction::Incoming, unpacked.len() as isize - 1);
            let OpType::Output(ou) = h.optype_mut(outp) else {
                panic!()
            };
            let rest = std::mem::take(&mut ou.types).into_owned();
            unpacked.extend(rest.into_iter().skip(1));
            ou.types = unpacked.into();
            add_unpack(h, values, orders, outp);
            return Ok(NormalizeCFGResult::CFGToDFG);
        } else if h.input_neighbours(entry).count() == 0 {
            // 1b. Move contents of entry block outside/before the CFG; the successor becomes the entry block.
            let [entry_input, entry_output] = h.get_io(entry).unwrap();
            let new_cfg_inputs = entry_blk.successor_input(0).unwrap();
            // Inputs to CFG go directly to consumers of the entry block's Input node
            for inp in h.node_inputs(cfg_node).collect::<Vec<_>>() {
                let srcs = h.linked_outputs(cfg_node, inp).collect::<Vec<_>>();
                h.disconnect(cfg_node, inp);
                for tgt in h
                    .linked_inputs(entry_input, inp.index())
                    .collect::<Vec<_>>()
                {
                    // Connecting all sources to all targets handles Order edges as well as Value
                    for src in srcs.iter() {
                        h.connect(src.0, src.1, tgt.0, tgt.1);
                    }
                }
            }
            h.remove_node(entry_input);

            // Update input ports
            let OpType::CFG(cfg_ty) = h.optype_mut(cfg_node) else {
                panic!()
            };
            let inputs_to_add =
                new_cfg_inputs.len() as isize - cfg_ty.signature.input.len() as isize;
            cfg_ty.signature.input = new_cfg_inputs;
            h.add_ports(cfg_node, Direction::Incoming, inputs_to_add);

            // Inputs to entry block Output node go instead to CFG
            let (entry_results, orders) = take_inputs(h, entry_output);
            h.remove_node(entry_output);
            add_unpack(h, entry_results, orders, cfg_node);

            // Transfer remaining entry children - including any used to compute the predicate
            while let Some(n) = h.first_child(entry) {
                h.set_parent(n, cfg_parent);
            }
            // old entry-node's successor is the new entry node, move into place
            h.move_before_sibling(succ, entry);
            h.remove_node(entry);
            entry_changed = true;
        }
    }
    // 2. If the exit node has a single predecessor and that predecessor has no other successors...
    let mut exit_changed = false;
    if let Some(pred) = h
        .input_neighbours(exit)
        .exactly_one()
        .ok()
        .filter(|pred| h.output_neighbours(*pred).count() == 1)
    {
        // Code in that predecessor can be moved outside (after the CFG), and the predecessor deleted
        let [_, output] = h.get_io(pred).unwrap();
        let pred_blk = h.get_optype(pred).as_dataflow_block().unwrap();
        let new_cfg_outs = pred_blk.inner_signature().into_owned().input;

        // new CFG result type and exit block
        let OpType::CFG(cfg_ty) = h.optype_mut(cfg_node) else {
            panic!()
        };
        let result_tys = std::mem::replace(&mut cfg_ty.signature.output, new_cfg_outs.clone());
        // TODO update number of CFG outports

        *h.optype_mut(pred) = ExitBlock {
            cfg_outputs: new_cfg_outs.clone(),
        }
        .into();
        h.set_num_ports(pred, h.num_ports(pred, Direction::Incoming), 0);

        h.move_before_sibling(pred, exit);
        h.remove_node(exit);
        // Move contents into new DFG
        let dfg = h.add_node_with_parent(
            cfg_parent,
            DFG {
                signature: Signature::new(new_cfg_outs, result_tys.clone()),
            },
        );
        while let Some(n) = h.first_child(pred) {
            h.set_parent(n, dfg);
        }
        // Add tuple-unpack inside the DFG
        let (values, orders) = take_inputs(h, output);
        let OpType::Output(ou) = h.optype_mut(output) else {
            panic!()
        };
        ou.types = result_tys;
        // TODO update number of input ports

        add_unpack(h, values, orders, output);
        // Move output edges.
        // TODO result_tys is almost this, but we want to move Order edges too
        for p in h.node_outputs(cfg_node).collect_vec() {
            let tgts = h.linked_inputs(cfg_node, p).collect_vec();
            h.disconnect(cfg_node, p);
            for tgt in tgts {
                h.connect(dfg, p, tgt.0, tgt.1)
            }
        }
        for p in h.node_inputs(dfg).collect_vec() {
            h.connect(cfg_node, p.index(), dfg, p);
        }
        exit_changed = true;
    }
    Ok(NormalizeCFGResult::CFGPreserved {
        entry_changed,
        exit_changed,
    })
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
    let order_srcs = (dfg1_outs.is_empty()).then_some((dfg1, dfg_order_out));
    // Do not add Order edges between DFGs unless there are no value edges
    add_unpack(&mut replacement, dfg1_outs, order_srcs, dfg2);

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

type NodePorts<N> = Vec<(N, OutgoingPort)>;
fn take_inputs<H: HugrMut>(h: &mut H, n: H::Node) -> (NodePorts<H::Node>, NodePorts<H::Node>) {
    let mut values = vec![];
    let mut orders = vec![];
    for p in h.node_inputs(n).collect_vec() {
        let srcs = h.linked_outputs(n, p).collect_vec();
        h.disconnect(n, p);
        match h.get_optype(n).port_kind(p) {
            Some(EdgeKind::Value(_)) => {
                assert_eq!(srcs.len(), 1);
                values.extend(srcs);
            }
            Some(EdgeKind::StateOrder) => {
                assert_eq!(orders, []);
                orders.extend(srcs);
            }
            k => panic!("Unexpected port kind: {:?}", k),
        }
    }
    (values, orders)
}

fn tuple_elems<H: HugrView>(h: &H, n: H::Node, p: OutgoingPort) -> TypeRow {
    match h.get_optype(n).port_kind(p) {
        Some(EdgeKind::Value(ty)) => ty.as_sum().unwrap().as_tuple().unwrap().clone(),
        p => panic!("Expected Value port not {:?}", p),
    }
    .try_into()
    .unwrap()
}

fn add_unpack<H: HugrMut>(
    h: &mut H,
    value_srcs: impl IntoIterator<Item = (H::Node, OutgoingPort)>,
    order_srcs: impl IntoIterator<Item = (H::Node, OutgoingPort)>,
    dst: H::Node,
) {
    let parent = h.get_parent(dst).unwrap();
    let mut srcs = value_srcs.into_iter();
    let src_to_unpack = srcs.next().unwrap();
    let tuple_tys = tuple_elems(h, src_to_unpack.0, src_to_unpack.1);
    let tuple_len = tuple_tys.len();
    let unp = h.add_node_with_parent(parent, UnpackTuple::new(tuple_tys));
    h.connect(src_to_unpack.0, src_to_unpack.1, unp, 0);

    for i in 0..tuple_len {
        h.connect(unp, i, dst, i);
    }
    assert_eq!(
        h.get_optype(dst).other_port_kind(Direction::Incoming),
        Some(EdgeKind::StateOrder)
    );
    let order_tgt = h.get_optype(dst).other_input_port().unwrap();
    for (i, (src, src_p)) in srcs.enumerate() {
        assert!(i + tuple_len < order_tgt.index());
        h.connect(src, src_p, dst, i + tuple_len);
    }
    for (src, src_p) in order_srcs {
        h.connect(src, src_p, dst, order_tgt);
    }
}

#[cfg(test)]
#[allow(deprecated)] // remove tests of merge_bbs, or just hide the latter
mod test {
    use std::collections::HashSet;
    use std::sync::Arc;

    use itertools::Itertools;
    use rstest::rstest;

    use hugr_core::builder::{CFGBuilder, Dataflow, HugrBuilder, endo_sig, inout_sig};
    use hugr_core::extension::prelude::{ConstUsize, Noop, PRELUDE_ID, qb_t, usize_t};
    use hugr_core::ops::constant::Value;
    use hugr_core::ops::handle::NodeHandle;
    use hugr_core::ops::{DataflowOpTrait, LoadConstant, OpTag, OpTrait, OpType};
    use hugr_core::types::{Signature, Type, TypeRow};
    use hugr_core::{Extension, HugrView, const_extension_ids, type_row};

    use crate::merge_bbs::{NormalizeCFGResult, normalize_cfg};

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

        let loop_variants: TypeRow = vec![qb_t()].into();
        let exit_types: TypeRow = vec![usize_t()].into();
        let e = extension();
        let tst_op = e.instantiate_extension_op("Test", [])?;
        let mut h = CFGBuilder::new(inout_sig(loop_variants.clone(), exit_types.clone()))?;
        let mut no_b1 = h.simple_entry_builder(loop_variants.clone(), 1)?;
        let n = no_b1.add_dataflow_op(Noop::new(qb_t()), no_b1.input_wires())?;
        let br = no_b1.add_load_value(Value::unary_unit_sum());
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
            let br = no_b2.add_load_value(Value::unary_unit_sum());
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
        let pred = bb1.add_load_value(Value::unary_unit_sum());
        let bb1 = bb1.finish_with_outputs(pred, [load_cst, inw])?;

        let mut bb2 = h.block_builder(
            vec![usize_t(), qb_t()].into(),
            vec![type_row![]],
            vec![qb_t(), usize_t()].into(),
        )?;
        let [u, q] = bb2.input_wires_arr();
        let pred = bb2.add_load_value(Value::unary_unit_sum());
        let bb2 = bb2.finish_with_outputs(pred, [q, u])?;

        let mut bb3 = h.block_builder(
            vec![qb_t(), usize_t()].into(),
            vec![type_row![]],
            res_t.clone().into(),
        )?;
        let [q, u] = bb3.input_wires_arr();
        let tst = bb3.add_dataflow_op(tst_op, [q, u])?;
        let pred = bb3.add_load_value(Value::unary_unit_sum());
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

    #[rstest]
    fn elide_cfg(#[values(false, true)] extra_blocks: bool) {
        let ext = extension();
        let op = ext.instantiate_extension_op("Test", []).unwrap();
        let out_ty = op.signature().output().clone();
        let mut cfg = CFGBuilder::new(op.signature().into_owned()).unwrap();
        let mut entry = cfg.simple_entry_builder(out_ty, 1).unwrap();
        let op_res = entry
            .add_dataflow_op(op.clone(), entry.input_wires())
            .unwrap();
        let predicate = entry.add_load_value(Value::unary_unit_sum());
        let entry = entry
            .finish_with_outputs(predicate, op_res.outputs())
            .unwrap();
        cfg.branch(&entry, 0, &cfg.exit_block()).unwrap();
        if extra_blocks {
            let Signature { input, output } = op.signature().as_ref().clone();
            for (ty, dest) in [(input, entry), (output, cfg.exit_block())] {
                let mut extra = cfg.simple_block_builder(endo_sig(ty), 1).unwrap();
                let inp = extra.input_wires();
                let branch = extra.add_load_value(Value::unary_unit_sum());
                let extra = extra.finish_with_outputs(branch, inp).unwrap();
                cfg.branch(&extra, 0, &dest).unwrap();
            }
        }
        let mut h = cfg.finish_hugr().unwrap();

        let func = h.children(h.module_root()).exactly_one().ok().unwrap();
        assert_eq!(
            h.children(func)
                .map(|n| h.get_optype(n).tag())
                .collect_vec(),
            [OpTag::Input, OpTag::Output, OpTag::Cfg]
        );
        let mut dfb_children = h
            .children(entry.node())
            .map(|n| h.get_optype(n).tag())
            .collect_vec();

        let res = normalize_cfg(&mut h);
        assert_eq!(res, Ok(NormalizeCFGResult::CFGToDFG));
        assert_eq!(h.entrypoint_optype().tag(), OpTag::Dfg);
        assert_eq!(
            h.children(func)
                .map(|n| h.get_optype(n).tag())
                .collect_vec(),
            [OpTag::Input, OpTag::Output, OpTag::Dfg,]
        );
        dfb_children.push(OpTag::Leaf);
        assert_eq!(
            h.children(h.entrypoint())
                .map(|n| h.get_optype(n).tag())
                .collect_vec(),
            dfb_children
        );
    }

    #[test]
    fn entry_before_loop() -> Result<(), Box<dyn std::error::Error>> {
        /* -> Noop --> Test -> Exit      -> Test --> Exit
                       |  |          =>     |  |
                       \<-/                 \<-/
        */
        let loop_variants: TypeRow = vec![qb_t()].into();
        let exit_types: TypeRow = vec![usize_t()].into();
        let e = extension();
        let tst_op = e.instantiate_extension_op("Test", [])?;
        let mut h = CFGBuilder::new(inout_sig(qb_t(), usize_t()))?;
        let mut nop_b = h.simple_entry_builder(loop_variants.clone(), 1)?;
        let n = nop_b.add_dataflow_op(Noop::new(qb_t()), nop_b.input_wires())?;
        let br = nop_b.add_load_value(Value::unary_unit_sum());
        let entry = nop_b.finish_with_outputs(br, n.outputs())?;

        let mut loop_b = h.block_builder(
            loop_variants.clone(),
            [loop_variants, exit_types],
            type_row![],
        )?;
        let [qb] = loop_b.input_wires_arr();
        let usz = loop_b.add_load_value(ConstUsize::new(3));
        let [tst] = loop_b.add_dataflow_op(tst_op, [qb, usz])?.outputs_arr();
        let loop_ = loop_b.finish_with_outputs(tst, [])?;
        h.branch(&entry, 0, &loop_)?;
        h.branch(&loop_, 0, &loop_)?;
        h.branch(&loop_, 1, &h.exit_block())?;

        let mut h = h.finish_hugr()?;

        let res = normalize_cfg(&mut h).unwrap();
        h.validate().unwrap();
        assert_eq!(
            res,
            NormalizeCFGResult::CFGPreserved {
                entry_changed: true,
                exit_changed: false
            }
        );
        assert_eq!(
            h.children(h.entrypoint())
                .map(|n| h.get_optype(n).tag())
                .collect_vec(),
            [OpTag::DataflowBlock, OpTag::BasicBlockExit]
        );
        let func = h.get_parent(h.entrypoint()).unwrap();
        let mut func_children = h
            .children(func)
            .map(|n| (h.get_optype(n).tag(), n))
            .into_group_map();
        let ext_ops = func_children.remove(&OpTag::Leaf).unwrap();
        assert_eq!(
            ext_ops
                .into_iter()
                .map(|n| h.get_optype(n).as_extension_op().unwrap().unqualified_id())
                .sorted()
                .collect_vec(),
            ["Noop", "UnpackTuple"]
        );

        {
            use OpTag::*;
            assert_eq!(
                func_children.keys().copied().collect::<HashSet<_>>(),
                HashSet::from([Input, Output, Cfg, Const, LoadConst])
            );
        }
        assert!(func_children.values().all(|v| v.len() == 1));

        Ok(())
    }
}

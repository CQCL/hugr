//! This module provides functions for finding non-local edges
//! in a Hugr and converting them to local edges.
use itertools::Itertools as _;

use hugr_core::{
    HugrView, IncomingPort, Wire,
    hugr::hugrmut::HugrMut,
    types::{EdgeKind, Type},
};

use crate::ComposablePass;

mod localize;
use localize::ExtraSourceReqs;

/// [ComposablePass] wrapper for [remove_nonlocal_edges]
#[derive(Clone, Debug, Hash)]
pub struct LocalizeEdges;

/// Error from [LocalizeEdges] or [remove_nonlocal_edges]
#[derive(derive_more::Error, derive_more::Display, derive_more::From, Debug, PartialEq)]
#[non_exhaustive]
pub enum LocalizeEdgesError {}

impl<H: HugrMut> ComposablePass<H> for LocalizeEdges {
    type Error = LocalizeEdgesError;

    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        remove_nonlocal_edges(hugr)
    }
}

/// Returns an iterator over all non local edges in a Hugr beneath the entrypoint.
///
/// All `(node, in_port)` pairs are returned where `in_port` is a value port connected to a
/// node whose parent is both beneath the entrypoint and different from the parent of `node`.
pub fn nonlocal_edges<H: HugrView>(hugr: &H) -> impl Iterator<Item = (H::Node, IncomingPort)> + '_ {
    hugr.entry_descendants().flat_map(move |node| {
        hugr.in_value_types(node).filter_map(move |(in_p, _)| {
            let (src, _) = hugr.single_linked_output(node, in_p)?;
            (hugr.get_parent(node) != hugr.get_parent(src)
                && ancestors(src, hugr).any(|a| a == hugr.entrypoint()))
            .then_some((node, in_p))
        })
    })
}

/// An error from [ensure_no_nonlocal_edges]
#[derive(Clone, derive_more::Error, derive_more::Display, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FindNonLocalEdgesError<N> {
    /// Some nonlocal edges were found
    #[display("Found {} nonlocal edges", _0.len())]
    #[error(ignore)] // Vec not convertible
    Edges(Vec<(N, IncomingPort)>),
}

/// Verifies that there are no non local value edges in the Hugr.
pub fn ensure_no_nonlocal_edges<H: HugrView>(
    hugr: &H,
) -> Result<(), FindNonLocalEdgesError<H::Node>> {
    let non_local_edges: Vec<_> = nonlocal_edges(hugr).collect_vec();
    if non_local_edges.is_empty() {
        Ok(())
    } else {
        Err(FindNonLocalEdgesError::Edges(non_local_edges))?
    }
}

fn just_types<'a, X: 'a>(v: impl IntoIterator<Item = &'a (X, Type)>) -> impl Iterator<Item = Type> {
    v.into_iter().map(|(_, t)| t.clone())
}

/// Converts all non-local edges in a Hugr into local ones, by inserting extra inputs
/// to container nodes and extra outports to Input nodes (and conversely to outputs of
/// [DataflowBlock]s).
///
/// [DataflowBlock]: hugr_core::ops::DataflowBlock
pub fn remove_nonlocal_edges<H: HugrMut>(hugr: &mut H) -> Result<(), LocalizeEdgesError> {
    // Group all the non-local edges in the graph by target node,
    // storing for each the source and type (well-defined as these are Value edges).
    let nonlocal_edges: Vec<_> = nonlocal_edges(hugr)
        .map(|(node, inport)| {
            // unwrap because nonlocal_edges(hugr) already skips in-ports with !=1 linked outputs.
            let (src_n, outp) = hugr.single_linked_output(node, inport).unwrap();
            debug_assert!(hugr.get_parent(src_n).unwrap() != hugr.get_parent(node).unwrap());
            let Some(EdgeKind::Value(ty)) = hugr.get_optype(src_n).port_kind(outp) else {
                panic!("impossible")
            };
            (node, (Wire::new(src_n, outp), ty))
        })
        .collect();

    if nonlocal_edges.is_empty() {
        return Ok(());
    }

    // We now compute the sources needed by each parent node.
    let needs_sources_map = {
        let mut bnsm = ExtraSourceReqs::default();
        for (target_node, (source, ty)) in nonlocal_edges.iter() {
            let parent = hugr.get_parent(*target_node).unwrap();
            debug_assert!(hugr.get_parent(parent).is_some());
            bnsm.add_edge(&*hugr, parent, *source, ty.clone());
        }
        bnsm
    };

    debug_assert!(nonlocal_edges.iter().all(|(n, (source, _))| {
        let source_parent = hugr.get_parent(source.node()).unwrap();
        let source_gp = hugr.get_parent(source_parent);
        ancestors(*n, hugr)
            .skip(1)
            .take_while(|&a| a != source_parent && source_gp.is_none_or(|gp| a != gp))
            .all(|parent| needs_sources_map.parent_needs(parent, *source))
    }));

    needs_sources_map.thread_hugr(hugr);

    Ok(())
}

fn ancestors<H: HugrView>(n: H::Node, h: &H) -> impl Iterator<Item = H::Node> {
    std::iter::successors(Some(n), |n| h.get_parent(*n))
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer},
        extension::prelude::{Noop, bool_t, either_type},
        ops::handle::{BasicBlockID, NodeHandle},
        ops::{Tag, TailLoop, Value},
        type_row,
        types::Signature,
    };
    use rstest::rstest;

    use super::*;

    #[test]
    fn ensures_no_nonlocal_edges() {
        let hugr = {
            let mut builder = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
            let [in_w] = builder.input_wires_arr();
            let [out_w] = builder
                .add_dataflow_op(Noop::new(bool_t()), [in_w])
                .unwrap()
                .outputs_arr();
            builder.finish_hugr_with_outputs([out_w]).unwrap()
        };
        ensure_no_nonlocal_edges(&hugr).unwrap();
    }

    #[test]
    fn find_nonlocal_edges() {
        let (hugr, edge) = {
            let mut builder = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
            let [in_w] = builder.input_wires_arr();
            let ([out_w], edge) = {
                let mut dfg_builder = builder
                    .dfg_builder(Signature::new(type_row![], bool_t()), [])
                    .unwrap();
                let noop = dfg_builder
                    .add_dataflow_op(Noop::new(bool_t()), [in_w])
                    .unwrap();
                let noop_edge = (noop.node(), IncomingPort::from(0));
                (
                    dfg_builder
                        .finish_with_outputs(noop.outputs())
                        .unwrap()
                        .outputs_arr(),
                    noop_edge,
                )
            };
            (builder.finish_hugr_with_outputs([out_w]).unwrap(), edge)
        };
        assert_eq!(
            ensure_no_nonlocal_edges(&hugr).unwrap_err(),
            FindNonLocalEdgesError::Edges(vec![edge])
        );
    }

    #[rstest]
    fn localize_dfg(#[values(true, false)] same_src: bool) {
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new_endo(vec![bool_t(); 2])).unwrap();
            let [w0, mut w1] = outer.input_wires_arr();
            if !same_src {
                [w1] = outer
                    .add_dataflow_op(Noop::new(bool_t()), [w1])
                    .unwrap()
                    .outputs_arr();
            }
            let inner_outs = {
                let inner = outer
                    .dfg_builder(Signature::new(vec![], vec![bool_t(); 2]), [])
                    .unwrap();
                // Note two `ext` edges to the same (Input) node here
                inner.finish_with_outputs([w0, w1]).unwrap().outputs()
            };
            outer.finish_hugr_with_outputs(inner_outs).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        remove_nonlocal_edges(&mut hugr).unwrap();
        hugr.validate().unwrap();
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn localize_tailloop() {
        let (t1, t2, t3) = (Type::UNIT, bool_t(), Type::new_unit_sum(3));
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new_endo(vec![
                t1.clone(),
                t2.clone(),
                t3.clone(),
            ]))
            .unwrap();
            let [s1, s2, s3] = outer.input_wires_arr();
            let [s2, s3] = {
                let mut inner = outer
                    .tail_loop_builder(
                        [(t1.clone(), s1)],
                        [(t3.clone(), s3)],
                        vec![t2.clone()].into(),
                    )
                    .unwrap();
                let [_s1, s3] = inner.input_wires_arr();
                let control = inner
                    .add_dataflow_op(
                        Tag::new(
                            TailLoop::BREAK_TAG,
                            vec![vec![t1.clone()].into(), vec![t2.clone()].into()],
                        ),
                        [s2],
                    )
                    .unwrap()
                    .out_wire(0);
                inner
                    .finish_with_outputs(control, [s3])
                    .unwrap()
                    .outputs_arr()
            };
            outer.finish_hugr_with_outputs([s1, s2, s3]).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        remove_nonlocal_edges(&mut hugr).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn localize_conditional() {
        let (t1, t2, t3) = (Type::UNIT, bool_t(), Type::new_unit_sum(3));
        let out_variants = vec![t1.clone().into(), t2.clone().into()];
        let out_type = Type::new_sum(out_variants.clone());
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new(
                vec![t1.clone(), t2.clone(), t3.clone()],
                out_type.clone(),
            ))
            .unwrap();
            let [s1, s2, s3] = outer.input_wires_arr();
            let [out] = {
                let mut cond = outer
                    .conditional_builder((vec![type_row![]; 3], s3), [], out_type.into())
                    .unwrap();

                {
                    let mut case = cond.case_builder(0).unwrap();
                    let [r] = case
                        .add_dataflow_op(Tag::new(0, out_variants.clone()), [s1])
                        .unwrap()
                        .outputs_arr();
                    case.finish_with_outputs([r]).unwrap();
                }
                {
                    let mut case = cond.case_builder(1).unwrap();
                    let [r] = case
                        .add_dataflow_op(Tag::new(1, out_variants.clone()), [s2])
                        .unwrap()
                        .outputs_arr();
                    case.finish_with_outputs([r]).unwrap();
                }
                {
                    let mut case = cond.case_builder(2).unwrap();
                    let u = case.add_load_value(Value::unit());
                    let [r] = case
                        .add_dataflow_op(Tag::new(0, out_variants.clone()), [u])
                        .unwrap()
                        .outputs_arr();
                    case.finish_with_outputs([r]).unwrap();
                }
                cond.finish_sub_container().unwrap().outputs_arr()
            };
            outer.finish_hugr_with_outputs([out]).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        remove_nonlocal_edges(&mut hugr).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn localize_cfg() {
        // Cfg consists of 4 dataflow blocks and an exit block
        //
        // The 4 dataflow blocks form a diamond, and the bottom block branches
        // either to the entry block or the exit block.
        //
        // The left block contains non-local uses of a value from outside the CFG (ext edge)
        // and a value from the entry block (dom edge) - the `ext` must be threaded through
        // all blocks because of the loop, the `dom` stays within (the same iter of) the loop.
        //
        // All non-trivial(i.e. more than one choice of successor) branching is
        // done on an option type to exercise both empty and occupied control
        // sums.
        //
        // All branches have an other-output.
        let branch_sum_type = either_type(Type::UNIT, Type::UNIT);
        let branch_type = Type::from(branch_sum_type.clone());
        let branch_variants = branch_sum_type
            .variants()
            .cloned()
            .map(|x| x.try_into().unwrap())
            .collect_vec();
        let ext_edge_type = bool_t();
        let dom_edge_type = Type::new_unit_sum(3);
        let other_output_type = branch_type.clone();
        let mut outer = DFGBuilder::new(Signature::new(
            vec![branch_type.clone(), ext_edge_type.clone(), Type::UNIT],
            vec![Type::UNIT, other_output_type.clone()],
        ))
        .unwrap();
        let [b, src_ext, unit] = outer.input_wires_arr();
        let mut cfg = outer
            .cfg_builder(
                [(Type::UNIT, unit), (branch_type.clone(), b)],
                vec![Type::UNIT, other_output_type.clone()].into(),
            )
            .unwrap();

        let (entry, src_dom) = {
            let mut entry = cfg
                .entry_builder(branch_variants.clone(), other_output_type.clone().into())
                .unwrap();
            let [_, b] = entry.input_wires_arr();

            let cst = entry.add_load_value(Value::unit_sum(1, 3).unwrap());

            (entry.finish_with_outputs(b, [b]).unwrap(), cst)
        };
        let exit = cfg.exit_block();

        let (bb_left, tgt_ext, tgt_dom) = {
            let mut bb = cfg
                .block_builder(
                    vec![Type::UNIT, other_output_type.clone()].into(),
                    [type_row![]],
                    other_output_type.clone().into(),
                )
                .unwrap();
            let [unit, oo] = bb.input_wires_arr();
            let tgt_ext = bb
                .add_dataflow_op(Noop::new(ext_edge_type.clone()), [src_ext])
                .unwrap();

            let tgt_dom = bb
                .add_dataflow_op(Noop::new(dom_edge_type.clone()), [src_dom])
                .unwrap();
            (
                bb.finish_with_outputs(unit, [oo]).unwrap(),
                tgt_ext,
                tgt_dom,
            )
        };

        let bb_right = {
            let mut bb = cfg
                .block_builder(
                    vec![Type::UNIT, other_output_type.clone()].into(),
                    [type_row![]],
                    other_output_type.clone().into(),
                )
                .unwrap();
            let [_b, oo] = bb.input_wires_arr();
            let unit = bb.add_load_value(Value::unit());
            bb.finish_with_outputs(unit, [oo]).unwrap()
        };

        let bb_bottom = {
            let bb = cfg
                .block_builder(
                    branch_type.clone().into(),
                    branch_variants,
                    other_output_type.clone().into(),
                )
                .unwrap();
            let [oo] = bb.input_wires_arr();
            bb.finish_with_outputs(oo, [oo]).unwrap()
        };
        cfg.branch(&entry, 0, &bb_left).unwrap();
        cfg.branch(&entry, 1, &bb_right).unwrap();
        cfg.branch(&bb_left, 0, &bb_bottom).unwrap();
        cfg.branch(&bb_right, 0, &bb_bottom).unwrap();
        cfg.branch(&bb_bottom, 0, &entry).unwrap();
        cfg.branch(&bb_bottom, 1, &exit).unwrap();
        let [unit, out] = cfg.finish_sub_container().unwrap().outputs_arr();

        let mut hugr = outer.finish_hugr_with_outputs([unit, out]).unwrap();
        let Err(FindNonLocalEdgesError::Edges(es)) = ensure_no_nonlocal_edges(&hugr) else {
            panic!()
        };
        assert_eq!(
            es,
            vec![
                (tgt_ext.node(), IncomingPort::from(0)),
                (tgt_dom.node(), IncomingPort::from(0))
            ]
        );
        remove_nonlocal_edges(&mut hugr).unwrap();
        hugr.validate().unwrap();
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
        let dfb = |bb: BasicBlockID| hugr.get_optype(bb.node()).as_dataflow_block().unwrap();
        // Entry node gets ext_edge_type added, only
        assert_eq!(
            dfb(entry).inputs[..],
            [ext_edge_type.clone(), Type::UNIT, branch_type.clone()]
        );
        // Left node gets both ext_edge_type and dom_edge_type
        assert_eq!(
            dfb(bb_left).inputs[..],
            [
                ext_edge_type.clone(),
                dom_edge_type,
                Type::UNIT,
                other_output_type
            ]
        );
        // Bottom node gets ext_edge_type added, only
        assert_eq!(dfb(bb_bottom).inputs[..], [ext_edge_type, branch_type]);
    }
}

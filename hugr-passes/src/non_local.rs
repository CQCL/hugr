//! This module provides functions for finding non-local edges
//! in a Hugr and converting them to local edges.
use std::collections::HashMap;

use hugr_core::{HugrView, IncomingPort, core::HugrNode};
use itertools::Itertools as _;

mod localize;
use localize::{BBNeedsSourcesMap, BBNeedsSourcesMapBuilder};

use hugr_core::{
    Wire,
    hugr::{HugrError, hugrmut::HugrMut},
    types::{EdgeKind, Type},
};

use crate::ComposablePass;

/// [ComposablePass] that converts all non-local edges in a Hugr
/// into local ones, by inserting extra inputs to container nodes
/// and extra outports to Input nodes.
pub struct LocalizeEdges;

#[derive(derive_more::Error, derive_more::Display, derive_more::From, Debug, PartialEq)]
#[non_exhaustive]
pub enum LocalizeEdgesError {
    HugrError(#[from] HugrError),
}

impl<H: HugrMut> ComposablePass<H> for LocalizeEdges {
    type Error = LocalizeEdgesError;

    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        remove_nonlocal_edges(hugr)
    }
}

/// Returns an iterator over all non local edges in a Hugr.
///
/// All `(node, in_port)` pairs are returned where `in_port` is a value port
/// connected to a node with a parent other than the parent of `node`.
pub fn nonlocal_edges<H: HugrView>(hugr: &H) -> impl Iterator<Item = (H::Node, IncomingPort)> + '_ {
    hugr.entry_descendants().flat_map(move |node| {
        hugr.in_value_types(node).filter_map(move |(in_p, _)| {
            let (src, _) = hugr.single_linked_output(node, in_p)?;
            (hugr.get_parent(node) != hugr.get_parent(src)).then_some((node, in_p))
        })
    })
}

// Identify all required extra inputs (for both Dom and Ext edges)
fn build_needs_sources_map<N: HugrNode>(
    hugr: impl HugrView<Node = N>,
    nonlocal_edges: &HashMap<N, WorkItem<N>>,
) -> BBNeedsSourcesMap<N> {
    let mut bnsm = BBNeedsSourcesMapBuilder::new(&hugr);
    for workitem in nonlocal_edges.values() {
        let parent = hugr.get_parent(workitem.target.0).unwrap();
        debug_assert!(hugr.get_parent(parent).is_some());
        bnsm.insert(parent, workitem.source, workitem.ty.clone());
    }
    bnsm.finish()
}

#[deprecated(note = "Use FindNonLocalEdgesError")]
pub type NonLocalEdgesError<N> = FindNonLocalEdgesError<N>;

/// An error from [ensure_no_nonlocal_edges]
#[derive(Clone, derive_more::Error, derive_more::Display, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FindNonLocalEdgesError<N> {
    #[display("Found {} nonlocal edges", _0.len())]
    #[error(ignore)]
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

#[derive(Debug, Clone)]
struct WorkItem<N: HugrNode> {
    source: Wire<N>,
    target: (N, IncomingPort),
    ty: Type,
}

fn just_types<'a, X: 'a>(v: impl IntoIterator<Item = &'a (X, Type)>) -> impl Iterator<Item = Type> {
    v.into_iter().map(|(_, t)| t.clone())
}

pub fn remove_nonlocal_edges<H: HugrMut>(hugr: &mut H) -> Result<(), LocalizeEdgesError> {
    // First we collect all the non-local edges in the graph. We associate them to a WorkItem, which tracks:
    //  * the source of the non-local edge
    //  * the target of the non-local edge
    //  * the type of the non-local edge. Note that all non-local edges are
    //    value edges, so the type is well defined.
    let nonlocal_edges_map: HashMap<_, _> = nonlocal_edges(hugr)
        .filter_map(|target @ (node, inport)| {
            let source = {
                let (n, p) = hugr.single_linked_output(node, inport)?;
                Wire::new(n, p)
            };
            debug_assert!(
                hugr.get_parent(source.node()).unwrap() != hugr.get_parent(node).unwrap()
            );
            let Some(EdgeKind::Value(ty)) =
                hugr.get_optype(source.node()).port_kind(source.source())
            else {
                panic!("impossible")
            };
            Some((node, WorkItem { source, target, ty }))
        })
        .collect();

    if nonlocal_edges_map.is_empty() {
        return Ok(());
    }

    // We now compute the sources needed by each parent node.
    // For a given non-local edge every intermediate node in the hierarchy
    // between the source's parent and the target needs that source.
    let bb_needs_sources_map = build_needs_sources_map(&hugr, &nonlocal_edges_map);

    // TODO move this out-of-line
    #[cfg(debug_assertions)]
    {
        for (&n, wi) in nonlocal_edges_map.iter() {
            let mut m = n;
            loop {
                let parent = hugr.get_parent(m).unwrap();
                if hugr.get_parent(wi.source.node()).unwrap() == parent {
                    break;
                }
                assert!(
                    bb_needs_sources_map
                        .get(parent)
                        .any(|(w, _)| *w == wi.source)
                );
                m = parent;
            }
        }

        for &bb in bb_needs_sources_map.keys() {
            assert!(hugr.get_parent(bb).is_some());
        }
    }

    bb_needs_sources_map.thread_hugr(hugr);

    Ok(())
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer},
        extension::prelude::{Noop, bool_t, either_type},
        ops::{Tag, TailLoop, Value, handle::NodeHandle},
        type_row,
        types::Signature,
    };

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

    #[test]
    fn localize_dfg() {
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
            let [w0] = outer.input_wires_arr();
            let [w1] = {
                let inner = outer
                    .dfg_builder(Signature::new_endo(bool_t()), [w0])
                    .unwrap();
                inner.finish_with_outputs([w0]).unwrap().outputs_arr()
            };
            outer.finish_hugr_with_outputs([w1]).unwrap()
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
        // Two non-local uses in the left block means that these values must
        // be threaded through all blocks, because of the loop.
        //
        // All non-trivial(i.e. more than one choice of successor) branching is
        // done on an option type to exercise both empty and occupied control
        // sums.
        //
        // All branches have an other-output.
        let mut hugr = {
            let branch_sum_type = either_type(Type::UNIT, Type::UNIT);
            let branch_type = Type::from(branch_sum_type.clone());
            let branch_variants = branch_sum_type
                .variants()
                .cloned()
                .map(|x| x.try_into().unwrap())
                .collect_vec();
            let nonlocal1_type = bool_t();
            let nonlocal2_type = Type::new_unit_sum(3);
            let other_output_type = branch_type.clone();
            let mut outer = DFGBuilder::new(Signature::new(
                vec![
                    branch_type.clone(),
                    nonlocal1_type.clone(),
                    nonlocal2_type.clone(),
                    Type::UNIT,
                ],
                vec![Type::UNIT, other_output_type.clone()],
            ))
            .unwrap();
            let [b, nl1, nl2, unit] = outer.input_wires_arr();
            let [unit, out] = {
                let mut cfg = outer
                    .cfg_builder(
                        [(Type::UNIT, unit), (branch_type.clone(), b)],
                        vec![Type::UNIT, other_output_type.clone()].into(),
                    )
                    .unwrap();

                let entry = {
                    let entry = cfg
                        .entry_builder(branch_variants.clone(), other_output_type.clone().into())
                        .unwrap();
                    let [_, b] = entry.input_wires_arr();

                    entry.finish_with_outputs(b, [b]).unwrap()
                };
                let exit = cfg.exit_block();

                let bb_left = {
                    let mut entry = cfg
                        .block_builder(
                            vec![Type::UNIT, other_output_type.clone()].into(),
                            [type_row![]],
                            other_output_type.clone().into(),
                        )
                        .unwrap();
                    let [unit, oo] = entry.input_wires_arr();
                    let [_] = entry
                        .add_dataflow_op(Noop::new(nonlocal1_type), [nl1])
                        .unwrap()
                        .outputs_arr();
                    let [_] = entry
                        .add_dataflow_op(Noop::new(nonlocal2_type), [nl2])
                        .unwrap()
                        .outputs_arr();
                    entry.finish_with_outputs(unit, [oo]).unwrap()
                };

                let bb_right = {
                    let entry = cfg
                        .block_builder(
                            vec![Type::UNIT, other_output_type.clone()].into(),
                            [type_row![]],
                            other_output_type.clone().into(),
                        )
                        .unwrap();
                    let [_b, oo] = entry.input_wires_arr();
                    entry.finish_with_outputs(unit, [oo]).unwrap()
                };

                let bb_bottom = {
                    let entry = cfg
                        .block_builder(
                            branch_type.clone().into(),
                            branch_variants,
                            other_output_type.clone().into(),
                        )
                        .unwrap();
                    let [oo] = entry.input_wires_arr();
                    entry.finish_with_outputs(oo, [oo]).unwrap()
                };
                cfg.branch(&entry, 0, &bb_left).unwrap();
                cfg.branch(&entry, 1, &bb_right).unwrap();
                cfg.branch(&bb_left, 0, &bb_bottom).unwrap();
                cfg.branch(&bb_right, 0, &bb_bottom).unwrap();
                cfg.branch(&bb_bottom, 0, &entry).unwrap();
                cfg.branch(&bb_bottom, 1, &exit).unwrap();
                cfg.finish_sub_container().unwrap().outputs_arr()
            };
            outer.finish_hugr_with_outputs([unit, out]).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        remove_nonlocal_edges(&mut hugr).unwrap();
        println!("{}", hugr.mermaid_string());
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }
}

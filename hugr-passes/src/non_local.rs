//! This module provides functions for finding non-local edges
//! in a Hugr and converting them to local edges.
use delegate::delegate;
use std::{
    collections::{BTreeMap, HashMap},
    iter, mem,
};

use hugr_core::{HugrView, IncomingPort, core::HugrNode};
use itertools::{Either, Itertools as _};

use hugr_core::{
    Direction, Wire,
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    hugr::{HugrError, hugrmut::HugrMut},
    ops::{DataflowOpTrait as _, OpType, Tag, TailLoop},
    types::{EdgeKind, Type, TypeRow},
};

use crate::ComposablePass;

/// [ComposablePass] that converts all non-local edges in a Hugr
/// into local ones, by inserting extra inputs to container nodes
/// and extra outports to Input nodes.
struct LocalizeEdges;

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

// Analysis: determining all extra ports that must be added =============================
#[derive(Debug, Clone)]
// Map from (parent of target node) to source Wire to Type.
// `BB` is any container, not necessarily a Basic Block or in a CFG
struct BBNeedsSourcesMap<N: HugrNode>(BTreeMap<N, BTreeMap<Wire<N>, Type>>);

impl<N: HugrNode> Default for BBNeedsSourcesMap<N> {
    fn default() -> Self {
        Self(BTreeMap::default())
    }
}

impl<N: HugrNode> BBNeedsSourcesMap<N> {
    fn insert(&mut self, node: N, source: Wire<N>, ty: Type) -> bool {
        self.0.entry(node).or_default().insert(source, ty).is_none()
    }

    fn get(&self, node: N) -> impl Iterator<Item = (&Wire<N>, &Type)> + '_ {
        match self.0.get(&node) {
            Some(x) => Either::Left(x.iter()),
            None => Either::Right(iter::empty()),
        }
    }

    delegate! {
        to self.0 {
            fn keys(&self) -> impl Iterator<Item=&N>;
        }
    }
}

#[derive(Debug, Clone)]
struct BBNeedsSourcesMapBuilder<H: HugrView> {
    hugr: H,
    needs_sources: BBNeedsSourcesMap<H::Node>,
}

impl<H: HugrView> BBNeedsSourcesMapBuilder<H> {
    fn new(hugr: H) -> Self {
        Self {
            hugr,
            needs_sources: Default::default(),
        }
    }

    fn insert(&mut self, mut parent: H::Node, source: Wire<H::Node>, ty: Type) {
        let source_parent = self.hugr.get_parent(source.node()).unwrap();
        while source_parent != parent {
            if !self.needs_sources.insert(parent, source, ty.clone()) {
                break;
            }
            if self.hugr.get_optype(parent).is_conditional() {
                // One of these we must have just done on the previous iteration
                for case in self.hugr.children(parent) {
                    // Full recursion unnecessary as we've just added parent:
                    self.needs_sources.insert(case, source, ty.clone());
                }
            }
            // this will panic if source_parent is not an ancestor of target
            let parent_parent = self.hugr.get_parent(parent).unwrap();
            if self.hugr.get_optype(parent).is_dataflow_block() {
                assert!(self.hugr.get_optype(parent_parent).is_cfg());
                for pred in self.hugr.input_neighbours(parent).collect::<Vec<_>>() {
                    self.insert(pred, source, ty.clone());
                }
                if Some(parent) != self.hugr.children(parent_parent).next() {
                    // Recursive calls on predecessors will have traced back to entry block
                    // (or source_parent itself if a dominating Basic Block)
                    break;
                }
                // We've just added to entry node - so must add to CFG as well
            }
            parent = parent_parent;
        }
    }

    fn finish(self) -> BBNeedsSourcesMap<H::Node> {
        self.needs_sources
    }
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

// Transformation: adding extra ports, and wiring them up ===============================
impl<N: HugrNode> BBNeedsSourcesMap<N> {
    fn thread_node(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        locals: &HashMap<Wire<N>, Wire<N>>,
    ) {
        if self.get(node).next().is_none() {
            // No edges incoming into this subtree, but there could still be nonlocal edges internal to it
            for ch in hugr.children(node).collect::<Vec<_>>() {
                self.thread_node(hugr, ch, &HashMap::new())
            }
            return;
        }

        let sources: Vec<(Wire<N>, Type)> = self.get(node).map(|(w, t)| (*w, t.clone())).collect();
        let src_wires: Vec<Wire<N>> = sources.iter().map(|(w, _)| *w).collect();

        // `match` must deal with everything inside the node, and update the signature (per OpType)
        let start_new_port_index = match hugr.optype_mut(node) {
            OpType::DFG(dfg) => {
                let ins = dfg.signature.input.to_mut();
                let start_new_port_index = ins.len();
                ins.extend(just_types(&sources));

                self.thread_dataflow_parent(hugr, node, start_new_port_index, sources);
                start_new_port_index
            }
            OpType::Conditional(cond) => {
                let start_new_port_index = cond.signature().input.len();
                cond.other_inputs.to_mut().extend(just_types(&sources));

                self.thread_conditional(hugr, node, sources);
                start_new_port_index
            }
            OpType::TailLoop(tail_op) => {
                vec_prepend(tail_op.just_inputs.to_mut(), just_types(&sources));
                self.thread_tailloop(hugr, node, sources);
                0
            }
            OpType::CFG(cfg) => {
                vec_prepend(cfg.signature.input.to_mut(), just_types(&sources));
                assert_eq!(
                    self.get(node).collect::<Vec<_>>(),
                    self.get(hugr.children(node).next().unwrap())
                        .collect::<Vec<_>>()
                ); // Entry node
                for bb in hugr.children(node).collect::<Vec<_>>() {
                    if hugr.get_optype(bb).is_dataflow_block() {
                        self.thread_bb(hugr, bb);
                    }
                }
                0
            }
            _ => panic!(
                "All containers handled except Module/FuncDefn or root Case/DFB, which should not have incoming nonlocal edges"
            ),
        };

        let new_dfg_ports = hugr.insert_ports(
            node,
            Direction::Incoming,
            start_new_port_index,
            src_wires.len(),
        );
        let local_srcs = src_wires.into_iter().map(|w| *locals.get(&w).unwrap_or(&w));
        for (w, tgt_port) in local_srcs.zip_eq(new_dfg_ports) {
            assert_eq!(hugr.get_parent(w.node()), hugr.get_parent(node));
            hugr.connect(w.node(), w.source(), node, tgt_port)
        }
    }

    fn thread_dataflow_parent(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        start_new_port_index: usize,
        srcs: Vec<(Wire<N>, Type)>,
    ) -> HashMap<Wire<N>, Wire<N>> {
        let nlocals = if srcs.is_empty() {
            HashMap::new()
        } else {
            let (srcs, tys): (Vec<_>, Vec<Type>) = srcs.into_iter().unzip();
            let [inp, _] = hugr.get_io(node).unwrap();
            let OpType::Input(in_op) = hugr.optype_mut(inp) else {
                panic!("Expected Input node")
            };
            vec_insert(in_op.types.to_mut(), tys, start_new_port_index);
            let new_outports =
                hugr.insert_ports(inp, Direction::Outgoing, start_new_port_index, srcs.len());

            srcs.into_iter()
                .zip_eq(new_outports)
                .map(|(w, p)| (w, Wire::new(inp, p)))
                .collect()
        };
        for ch in hugr.children(node).collect::<Vec<_>>() {
            for (inp, _) in hugr.in_value_types(ch).collect::<Vec<_>>() {
                if let Some((src_n, src_p)) = hugr.single_linked_output(ch, inp) {
                    if hugr.get_parent(src_n) != Some(node) {
                        hugr.disconnect(ch, inp);
                        let new_p = nlocals.get(&Wire::new(src_n, src_p)).unwrap();
                        hugr.connect(new_p.node(), new_p.source(), ch, inp);
                    }
                }
            }
            self.thread_node(hugr, ch, &nlocals);
        }
        nlocals
    }

    fn thread_conditional(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        srcs: Vec<(Wire<N>, Type)>,
    ) {
        for case in hugr.children(node).collect::<Vec<_>>() {
            let OpType::Case(case_op) = hugr.optype_mut(case) else {
                continue;
            };
            let ins = case_op.signature.input.to_mut();
            let start_case_port_index = ins.len();
            ins.extend(just_types(&srcs));
            self.thread_dataflow_parent(hugr, case, start_case_port_index, srcs.clone());
        }
    }

    fn thread_tailloop(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        srcs: Vec<(Wire<N>, Type)>,
    ) {
        let [_, o] = hugr.get_io(node).unwrap();
        let new_sum_row_prefixes = {
            let mut v = vec![vec![]; 2];
            v[TailLoop::CONTINUE_TAG] = srcs.clone();
            v
        };
        add_control_prefixes(hugr, o, new_sum_row_prefixes);
        self.thread_dataflow_parent(hugr, node, 0, srcs);
    }

    fn thread_bb(&self, hugr: &mut impl HugrMut<Node = N>, node: N) {
        let OpType::DataflowBlock(this_dfb) = hugr.optype_mut(node) else {
            panic!("Expected dataflow block")
        };
        let my_inputs: Vec<_> = self.get(node).map(|(w, t)| (*w, t.clone())).collect();
        vec_prepend(this_dfb.inputs.to_mut(), just_types(&my_inputs));
        let locals = self.thread_dataflow_parent(hugr, node, 0, my_inputs);
        let variant_source_prefixes: Vec<Vec<(Wire<N>, Type)>> = hugr
            .output_neighbours(node)
            .map(|succ| {
                // The wires required for each successor block, should be available in the predecessor
                self.get(succ)
                    .map(|(w, ty)| {
                        (
                            if hugr.get_parent(w.node()) == Some(node) {
                                *w
                            } else {
                                *locals.get(w).unwrap()
                            },
                            ty.clone(),
                        )
                    })
                    .collect()
            })
            .collect();
        let OpType::DataflowBlock(this_dfb) = hugr.optype_mut(node) else {
            panic!("It worked earlier!")
        };
        for (source_prefix, sum_row) in variant_source_prefixes
            .iter()
            .zip_eq(this_dfb.sum_rows.iter_mut())
        {
            vec_prepend(sum_row.to_mut(), just_types(source_prefix));
        }
        let [_, output_node] = hugr.get_io(node).unwrap();
        add_control_prefixes(hugr, output_node, variant_source_prefixes);
    }
}

fn just_types<'a, X: 'a>(v: impl IntoIterator<Item = &'a (X, Type)>) -> impl Iterator<Item = Type> {
    v.into_iter().map(|(_, t)| t.clone())
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

#[derive(derive_more::Error, derive_more::Display, derive_more::From, Debug, PartialEq)]
#[non_exhaustive]
pub enum LocalizeEdgesError {
    HugrError(#[from] HugrError),
}

#[derive(Debug, Clone)]
struct WorkItem<N: HugrNode> {
    source: Wire<N>,
    target: (N, IncomingPort),
    ty: Type,
}

/// `variant_source_prefixes` are extra wires/types to prepend onto each variant
/// (must have one element per variant of control Sum)
fn add_control_prefixes<H: HugrMut>(
    hugr: &mut H,
    output_node: H::Node,
    variant_source_prefixes: Vec<Vec<(Wire<H::Node>, Type)>>,
) {
    debug_assert!(hugr.get_optype(output_node).is_output()); // Just to fail fast
    let parent = hugr.get_parent(output_node).unwrap();
    let mut needed_sources = BTreeMap::new();
    let (cond, new_control_type) = {
        let Some(EdgeKind::Value(control_type)) = hugr
            .get_optype(output_node)
            .port_kind(IncomingPort::from(0))
        else {
            panic!("impossible")
        };
        let Some(sum_type) = control_type.as_sum() else {
            panic!("impossible")
        };

        let mut type_for_source = |source: &(Wire<H::Node>, Type)| {
            let (w, t) = source;
            let replaced = needed_sources.insert(*w, (*w, t.clone()));
            debug_assert!(!replaced.is_some_and(|x| x != (*w, t.clone())));
            t.clone()
        };
        let old_sum_rows: Vec<TypeRow> = sum_type
            .variants()
            .map(|x| x.clone().try_into().unwrap())
            .collect_vec();
        let new_sum_rows: Vec<TypeRow> =
            itertools::zip_eq(variant_source_prefixes.iter(), old_sum_rows.iter())
                .map(|(new_sources, old_tys)| {
                    new_sources
                        .iter()
                        .map(&mut type_for_source)
                        .chain(old_tys.iter().cloned())
                        .collect_vec()
                        .into()
                })
                .collect_vec();

        let new_control_type = Type::new_sum(new_sum_rows.clone());
        let mut cond = ConditionalBuilder::new(
            old_sum_rows.clone(),
            just_types(needed_sources.values()).collect_vec(),
            new_control_type.clone(),
        )
        .unwrap();
        for (i, new_sources) in variant_source_prefixes.into_iter().enumerate() {
            let mut case = cond.case_builder(i).unwrap();
            let case_inputs = case.input_wires().collect_vec();
            let mut args = new_sources
                .into_iter()
                .map(|(s, _ty)| {
                    case_inputs[old_sum_rows[i].len()
                        + needed_sources
                            .iter()
                            .find_position(|(w, _)| **w == s)
                            .unwrap()
                            .0]
                })
                .collect_vec();
            args.extend(&case_inputs[..old_sum_rows[i].len()]);
            let case_outputs = case
                .add_dataflow_op(Tag::new(i, new_sum_rows.clone()), args)
                .unwrap()
                .outputs();
            case.finish_with_outputs(case_outputs).unwrap();
        }
        (cond.finish_hugr().unwrap(), new_control_type)
    };
    let cond_node = hugr.insert_hugr(parent, cond).inserted_entrypoint;
    let (old_output_source_node, old_output_source_port) =
        hugr.single_linked_output(output_node, 0).unwrap();
    debug_assert_eq!(hugr.get_parent(old_output_source_node).unwrap(), parent);
    hugr.connect(old_output_source_node, old_output_source_port, cond_node, 0);
    for (i, &(w, _)) in needed_sources.values().enumerate() {
        hugr.connect(w.node(), w.source(), cond_node, i + 1);
    }
    hugr.disconnect(output_node, IncomingPort::from(0));
    hugr.connect(cond_node, 0, output_node, 0);
    let OpType::Output(output) = hugr.optype_mut(output_node) else {
        panic!("impossible")
    };
    output.types.to_mut()[0] = new_control_type;
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

    bb_needs_sources_map.thread_node(hugr, hugr.entrypoint(), &HashMap::new());

    Ok(())
}

fn vec_prepend<T>(v: &mut Vec<T>, ts: impl IntoIterator<Item = T>) {
    vec_insert(v, ts, 0)
}

fn vec_insert<T>(v: &mut Vec<T>, ts: impl IntoIterator<Item = T>, index: usize) {
    let mut old_v_iter = mem::take(v).into_iter();
    v.extend(old_v_iter.by_ref().take(index).chain(ts));
    v.extend(old_v_iter);
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
    fn vec_insert0() {
        let mut v = vec![5, 7, 9];
        vec_insert(&mut v, [1, 2], 0);
        assert_eq!(v, [1, 2, 5, 7, 9]);
    }

    #[test]
    fn vec_insert1() {
        let mut v = vec![5, 7, 9];
        vec_insert(&mut v, [1, 2], 1);
        assert_eq!(v, [5, 1, 2, 7, 9]);
    }

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

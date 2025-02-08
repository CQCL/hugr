//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
use delegate::delegate;
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    iter, mem,
};

//TODO Add `remove_nonlocal_edges` and `add_nonlocal_edges` functions
use itertools::{Either, Itertools as _};

use hugr_core::{
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, HierarchyView},
        HugrError,
    },
    ops::{DataflowOpTrait as _, OpType, Tag, TailLoop},
    types::{EdgeKind, Type, TypeRow},
    HugrView, IncomingPort, Node, PortIndex, Wire,
};

use crate::validation::{ValidatePassError, ValidationLevel};

/// TODO docs
#[derive(Debug, Clone, Default)]
pub struct UnNonLocalPass {
    validation: ValidationLevel,
}

impl UnNonLocalPass {
    /// Sets the validation level used before and after the pass is run.
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Run the Monomorphization pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), NonLocalEdgesError> {
        let root = hugr.root();
        remove_nonlocal_edges(hugr, root)?;
        Ok(())
    }

    /// Run the pass using specified configuration.
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), NonLocalEdgesError> {
        self.validation
            .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
    }
}

/// Returns an iterator over all non local edges in a Hugr.
///
/// All `(node, in_port)` pairs are returned where `in_port` is a value port
/// connected to a node with a parent other than the parent of `node`.
pub fn nonlocal_edges(hugr: &impl HugrView) -> impl Iterator<Item = (Node, IncomingPort)> + '_ {
    hugr.nodes().flat_map(move |node| {
        hugr.in_value_types(node).filter_map(move |(in_p, _)| {
            let parent = hugr.get_parent(node);
            hugr.linked_outputs(node, in_p)
                .any(|(neighbour_node, _)| parent != hugr.get_parent(neighbour_node))
                .then_some((node, in_p))
        })
    })
}

#[derive(derive_more::Error, derive_more::From, derive_more::Display, Debug, PartialEq)]
#[non_exhaustive]
pub enum NonLocalEdgesError {
    #[display("Found {} nonlocal edges", _0.len())]
    #[error(ignore)]
    Edges(Vec<(Node, IncomingPort)>),
    #[from]
    ValidationError(ValidatePassError),
    #[from]
    HugrError(HugrError),
}

/// Verifies that there are no non local value edges in the Hugr.
pub fn ensure_no_nonlocal_edges(hugr: &impl HugrView) -> Result<(), NonLocalEdgesError> {
    let non_local_edges: Vec<_> = nonlocal_edges(hugr).collect_vec();
    if non_local_edges.is_empty() {
        Ok(())
    } else {
        Err(NonLocalEdgesError::Edges(non_local_edges))?
    }
}

#[derive(Debug, Clone)]
struct WorkItem {
    source: Wire,
    target: (Node, IncomingPort),
    ty: Type,
}

#[derive(Clone, Default, Debug)]
struct ParentSourceMap(HashMap<Node, HashMap<Wire, Wire>>);

impl ParentSourceMap {
    // fn contains_parent(&self, parent: Node) -> bool {
    //     self.0.contains_key(&parent)
    // }

    fn insert_sources_in_parent(
        &mut self,
        parent: Node,
        sources: impl IntoIterator<Item = (Wire, Wire)>,
    ) {
        debug_assert!(!self.0.contains_key(&parent));
        self.0.entry(parent).or_default().extend(sources);
    }

    fn get_source_in_parent(&self, parent: Node, source: Wire) -> Option<Wire> {
        self.0.get(&parent).and_then(|m| m.get(&source).cloned())
    }

    fn thread_dataflow_parent(
        &mut self,
        hugr: &mut impl HugrMut,
        parent: Node,
        start_port_index: usize,
        sources: impl IntoIterator<Item = (Wire, Type)>,
    ) -> impl Iterator<Item = Wire> {
        let [input_n, _] = hugr.get_io(parent).unwrap();
        let OpType::Input(mut input) = hugr.get_optype(input_n).clone() else {
            panic!("impossible")
        };
        let mut input_wires = vec![];
        self.0
            .entry(parent)
            .or_default()
            .extend(sources.into_iter().enumerate().map(|(i, (source, ty))| {
                input.types.to_mut().insert(start_port_index + i, ty);
                let input_wire = Wire::new(
                    input_n,
                    hugr.insert_outgoing_port(input_n, start_port_index + i),
                );
                input_wires.push(input_wire);
                (source, input_wire)
            }));
        hugr.replace_op(input_n, input).unwrap();
        input_wires.into_iter()
    }
}

#[derive(Clone, Debug)]
struct ThreadState<'a> {
    parent_source_map: ParentSourceMap,
    needs: &'a BBNeedsSourcesMap,
    worklist: Vec<WorkItem>,
}

impl<'a> ThreadState<'a> {
    delegate! {
        to self.parent_source_map {
            // fn contains_parent(&self, parent: Node) -> bool;
            fn get_source_in_parent(&self, parent: Node, source: Wire) -> Option<Wire>;
            fn insert_sources_in_parent(&mut self, parent: Node, sources: impl IntoIterator<Item = (Wire, Wire)>);
            fn thread_dataflow_parent(
                &mut self,
                hugr: &mut impl HugrMut,
                parent: Node,
                start_port_index: usize,
                sources: impl IntoIterator<Item=(Wire, Type)>,
            ) -> impl Iterator<Item=Wire>;
        }
    }

    fn new(bbnsm: &'a BBNeedsSourcesMap) -> Self {
        Self {
            parent_source_map: Default::default(),
            needs: bbnsm,
            worklist: vec![],
        }
    }

    fn do_dataflow_block(
        &mut self,
        hugr: &mut impl HugrMut,
        node: Node,
        sources: Vec<(Wire, Type)>,
    ) {
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        let new_sum_row_prefixes = {
            let mut dfb = hugr.get_optype(node).as_dataflow_block().unwrap().clone();
            let mut nsrp = vec![vec![]; dfb.sum_rows.len()];
            dfb.inputs.to_mut().extend(types.clone());
            for (this_p, succ_n) in hugr.node_outputs(node).filter_map(|out_p| {
                let (succ_n, _) = hugr.single_linked_input(node, out_p).unwrap();
                if hugr.get_optype(succ_n).is_exit_block() {
                    None
                } else {
                    Some((out_p.index(), succ_n))
                }
            }) {
                let succ_needs = &self.needs[&succ_n];
                let new_tys = succ_needs
                    .iter()
                    .map(|(&w, ty)| {
                        (
                            sources.iter().find_position(|(x, _)| x == &w).unwrap().0,
                            ty.clone(),
                        )
                    })
                    .collect_vec();
                nsrp[this_p] = new_tys.clone();
                let tys = dfb.sum_rows[this_p].to_mut();
                let old_tys = mem::replace(tys, new_tys.into_iter().map(|x| x.1).collect_vec());
                tys.extend(old_tys);
            }
            hugr.replace_op(node, dfb).unwrap();
            nsrp
        };

        let input_wires = self
            .thread_dataflow_parent(hugr, node, 0, sources.clone())
            .collect_vec();

        let [_, o] = hugr.get_io(node).unwrap();
        let (cond, new_control_type) = {
            let Some(EdgeKind::Value(control_type)) =
                hugr.get_optype(o).port_kind(IncomingPort::from(0))
            else {
                panic!("impossible")
            };
            let Some(sum_type) = control_type.as_sum_type() else {
                panic!("impossible")
            };

            let old_sum_rows: Vec<TypeRow> = sum_type
                .iter_variants()
                .map(|x| x.clone().try_into().unwrap())
                .collect_vec();
            let new_sum_rows: Vec<TypeRow> =
                itertools::zip_eq(new_sum_row_prefixes.clone(), old_sum_rows.iter())
                    .map(|(new, old)| {
                        new.into_iter()
                            .map(|x| x.1)
                            .chain(old.iter().cloned())
                            .collect_vec()
                            .into()
                    })
                    .collect_vec();

            let new_control_type = Type::new_sum(new_sum_rows.clone());
            let mut cond = ConditionalBuilder::new(
                old_sum_rows.clone(),
                types.clone(),
                new_control_type.clone(),
            )
            .unwrap();
            for (i, row) in new_sum_row_prefixes.iter().enumerate() {
                let mut case = cond.case_builder(i).unwrap();
                let case_inputs = case.input_wires().collect_vec();
                let mut args = vec![];
                for (source_i, _) in row {
                    args.push(case_inputs[old_sum_rows[i].len() + source_i]);
                }

                args.extend(&case_inputs[..old_sum_rows[i].len()]);

                let case_outputs = case
                    .add_dataflow_op(Tag::new(i, new_sum_rows.clone()), args)
                    .unwrap()
                    .outputs();
                case.finish_with_outputs(case_outputs).unwrap();
            }
            (cond.finish_hugr().unwrap(), new_control_type)
        };
        let cond_node = hugr.insert_hugr(node, cond).new_root;
        let (n, p) = hugr.single_linked_output(o, 0).unwrap();
        hugr.connect(n, p, cond_node, 0);
        for (i, w) in input_wires.into_iter().enumerate() {
            hugr.connect(w.node(), w.source(), cond_node, i + 1);
        }
        hugr.disconnect(o, IncomingPort::from(0));
        hugr.connect(cond_node, 0, o, 0);
        let mut output = hugr.get_optype(o).as_output().unwrap().clone();
        output.types.to_mut()[0] = new_control_type;
        hugr.replace_op(o, output).unwrap();
    }

    fn do_cfg(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        {
            let mut cfg = hugr.get_optype(node).as_cfg().unwrap().clone();
            let inputs = cfg.signature.input.to_mut();
            let old_inputs = mem::replace(inputs, types);
            inputs.extend(old_inputs);
            hugr.replace_op(node, cfg).unwrap();
        }
        let new_cond_ports = (0..sources.len())
            .map(|i| hugr.insert_incoming_port(node, i))
            .collect_vec();
        self.insert_sources_in_parent(node, iter::empty());
        self.worklist
            .extend(mk_workitems(node, sources, new_cond_ports))
    }

    fn do_dfg(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut dfg = hugr.get_optype(node).as_dfg().unwrap().clone();
        let start_new_port_index = dfg.signature.input().len();
        let new_dfg_ports = (0..sources.len())
            .map(|i| hugr.insert_incoming_port(node, start_new_port_index + i))
            .collect_vec();
        dfg.signature
            .input
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, dfg).unwrap();
        let _ =
            self.thread_dataflow_parent(hugr, node, start_new_port_index, sources.iter().cloned());
        self.worklist
            .extend(mk_workitems(node, sources, new_dfg_ports));
    }

    fn do_conditional(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut cond = hugr.get_optype(node).as_conditional().unwrap().clone();
        let start_new_port_index = cond.signature().input().len();
        cond.other_inputs
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, cond).unwrap();
        let new_cond_ports = (0..sources.len())
            .map(|i| hugr.insert_incoming_port(node, start_new_port_index + i))
            .collect_vec();
        self.insert_sources_in_parent(node, iter::empty());
        self.worklist
            .extend(mk_workitems(node, sources, new_cond_ports))
    }

    fn do_case(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut case = hugr.get_optype(node).as_case().unwrap().clone();
        let start_case_port_index = case.signature.input().len();
        case.signature
            .input
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, case).unwrap();
        let _ = self.thread_dataflow_parent(hugr, node, start_case_port_index, sources);
    }

    fn do_tailloop(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut tailloop = hugr.get_optype(node).as_tail_loop().unwrap().clone();
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        let start_port_index = tailloop.just_inputs.len();
        {
            tailloop.just_inputs.to_mut().extend(types.clone());
            hugr.replace_op(node, tailloop).unwrap();
        }
        let tailloop_ports = (0..sources.len())
            .map(|i| hugr.insert_incoming_port(node, start_port_index + i))
            .collect_vec();

        let input_wires = self
            .thread_dataflow_parent(hugr, node, start_port_index, sources.clone())
            .collect_vec();

        let [_, o] = hugr.get_io(node).unwrap();
        let (cond, new_control_type) = {
            let Some(EdgeKind::Value(control_type)) =
                hugr.get_optype(o).port_kind(IncomingPort::from(0))
            else {
                panic!("impossible")
            };
            let Some(sum_type) = control_type.as_sum_type() else {
                panic!("impossible")
            };

            let old_sum_rows: Vec<TypeRow> = sum_type
                .iter_variants()
                .map(|x| x.clone().try_into().unwrap())
                .collect_vec();
            let new_sum_rows = {
                let mut v = old_sum_rows.clone();
                v[TailLoop::CONTINUE_TAG]
                    .to_mut()
                    .extend(types.iter().cloned());
                v
            };

            let new_control_type = Type::new_sum(new_sum_rows.clone());
            let mut cond =
                ConditionalBuilder::new(old_sum_rows, types.clone(), new_control_type.clone())
                    .unwrap();
            for i in 0..2 {
                let mut case = cond.case_builder(i).unwrap();
                let inputs = {
                    let all_inputs = case.input_wires();
                    if i == TailLoop::CONTINUE_TAG {
                        Either::Left(all_inputs)
                    } else {
                        Either::Right(all_inputs.into_iter().dropping_back(types.len()))
                    }
                };

                let case_outputs = case
                    .add_dataflow_op(Tag::new(i, new_sum_rows.clone()), inputs)
                    .unwrap()
                    .outputs();
                case.finish_with_outputs(case_outputs).unwrap();
            }
            (cond.finish_hugr().unwrap(), new_control_type)
        };
        let cond_node = hugr.insert_hugr(node, cond).new_root;
        let (n, p) = hugr.single_linked_output(o, 0).unwrap();
        hugr.connect(n, p, cond_node, 0);
        for (i, w) in input_wires.into_iter().enumerate() {
            hugr.connect(w.node(), w.source(), cond_node, i + 1);
        }
        hugr.disconnect(o, IncomingPort::from(0));
        hugr.connect(cond_node, 0, o, 0);
        let mut output = hugr.get_optype(o).as_output().unwrap().clone();
        output.types.to_mut()[0] = new_control_type;
        hugr.replace_op(o, output).unwrap();
        self.worklist
            .extend(mk_workitems(node, sources, tailloop_ports))
    }

    fn finish(self, _hugr: &mut impl HugrMut) -> (Vec<WorkItem>, ParentSourceMap) {
        (self.worklist, self.parent_source_map)
    }
}

fn thread_sources(
    hugr: &mut impl HugrMut,
    bb_needs_sources_map: &BBNeedsSourcesMap,
) -> (Vec<WorkItem>, ParentSourceMap) {
    let mut state = ThreadState::new(bb_needs_sources_map);
    for (&bb, sources) in bb_needs_sources_map {
        let sources = sources
            .iter()
            .map(|(&w, ty)| (w, ty.clone()))
            .collect_vec();
        match hugr.get_optype(bb).clone() {
            OpType::DFG(_) => state.do_dfg(hugr, bb, sources),
            OpType::Conditional(_) => state.do_conditional(hugr, bb, sources),
            OpType::Case(_) => state.do_case(hugr, bb, sources),
            OpType::TailLoop(_) => state.do_tailloop(hugr, bb, sources),
            OpType::DataflowBlock(_) => state.do_dataflow_block(hugr, bb, sources),
            OpType::CFG(_) => state.do_cfg(hugr, bb, sources),
            _ => panic!("impossible"),
        }
    }

    state.finish(hugr)
}

fn mk_workitems(
    node: Node,
    sources: impl IntoIterator<Item = (Wire, Type)>,
    ports: impl IntoIterator<Item = IncomingPort>,
) -> impl Iterator<Item = WorkItem> {
    itertools::izip!(sources, ports).map(move |((source, ty), p)| WorkItem {
        source,
        target: (node, p),
        ty,
    })
}

type BBNeedsSourcesMap = HashMap<Node, BTreeMap<Wire, Type>>;

#[derive(Debug, Default, Clone)]
struct BBNeedsSourcesMapBuilder(BBNeedsSourcesMap);

impl BBNeedsSourcesMapBuilder {
    fn insert(&mut self, bb: Node, source: Wire, ty: Type) {
        self.0.entry(bb).or_default().insert(source, ty);
    }

    fn extend_parent_needs_for(&mut self, ref hugr: impl HugrView, child: Node) -> bool {
        let parent = hugr.get_parent(child).unwrap();
        let parent_needs = self
            .0
            .get(&child)
            .into_iter()
            .flat_map(move |m| {
                m.iter()
                    .filter(move |(w, _)| hugr.get_parent(w.node()).unwrap() != parent)
                    .map(|(&w, ty)| (w, ty.clone()))
            })
            .collect_vec();
        let any = !parent_needs.is_empty();
        if any {
            self.0.entry(parent).or_default().extend(parent_needs);
        }
        any
    }

    fn finish(mut self, hugr: impl HugrView) -> BBNeedsSourcesMap {
        {
            let conds = self
                .0
                .keys()
                .copied()
                .filter(|&n| hugr.get_optype(n).is_conditional())
                .collect_vec();
            for cond in conds {
                let cases = hugr
                    .children(cond)
                    .filter(|&child| hugr.get_optype(child).is_case())
                    .collect_vec();
                let all_needed: BTreeMap<_, _> = cases
                    .iter()
                    .flat_map(|&case| {
                        let case_needed = self.0.get(&case);
                        case_needed
                            .into_iter()
                            .flat_map(|m| m.iter().map(|(&w, ty)| (w, ty.clone())))
                    })
                    .collect();
                for case in cases {
                    let _ = self.0.insert(case, all_needed.clone());
                }
            }
        }
        {
            let cfgs = self
                .0
                .keys()
                .copied()
                .filter(|&n| hugr.get_optype(n).is_cfg() && self.0.contains_key(&n))
                .collect_vec();
            for cfg in cfgs {
                let dfbs = hugr
                    .children(cfg)
                    .filter(|&child| hugr.get_optype(child).is_dataflow_block())
                    .collect_vec();

                // let mut dfb_needs_map: HashMap<_, _> = dfbs
                //     .iter()
                //     .map(|&n| (n, self.0.get(&n).cloned().unwrap_or_default()))
                //     .collect();
                loop {
                    let mut any_change = false;
                    for &dfb in dfbs.iter() {
                        for succ_n in hugr.output_neighbours(dfb) {
                            for (w, ty) in self.0.get(&succ_n).cloned().unwrap_or_default() {
                                any_change |=
                                    self.0.entry(dfb).or_default().insert(w, ty).is_none();
                            }
                        }
                    }
                    if !any_change {
                        break;
                    }
                }
            }
        }

        self.0
    }
}

pub fn remove_nonlocal_edges(
    hugr: &mut impl HugrMut,
    root: Node,
) -> Result<(), NonLocalEdgesError> {
    let nonlocal_edges_map: HashMap<Node, WorkItem> =
        nonlocal_edges(&DescendantsGraph::<Node>::try_new(hugr, root)?)
            .map(|target @ (node, inport)| {
                let source = {
                    let (n, p) = hugr.single_linked_output(node, inport).unwrap();
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
                (node, WorkItem { source, target, ty })
            })
            .collect();

    if nonlocal_edges_map.is_empty() {
        return Ok(());
    }

    let bb_needs_sources_map = {
        let nonlocal_sorted = {
            let mut v = iter::successors(Some(vec![root]), |nodes| {
                let children = nodes.iter().flat_map(|&n| hugr.children(n)).collect_vec();
                (!children.is_empty()).then_some(children)
            })
            .flatten()
            .filter_map(|n| nonlocal_edges_map.get(&n))
            .collect_vec();
            v.reverse();
            v
        };
        let mut parent_set = HashSet::<Node>::new();
        // earlier items are deeper in the heirarchy
        let mut parent_worklist = VecDeque::<Node>::new();
        let mut add_parent = |p, wl: &mut VecDeque<_>| {
            if parent_set.insert(p) {
                wl.push_back(p);
            }
        };
        let mut bnsm = BBNeedsSourcesMapBuilder::default();
        for workitem in nonlocal_sorted {
            let parent = hugr.get_parent(workitem.target.0).unwrap();
            debug_assert!(hugr.get_parent(parent).is_some());
            bnsm.insert(parent, workitem.source, workitem.ty.clone());
            add_parent(parent, &mut parent_worklist);
        }

        while let Some(bb_node) = parent_worklist.pop_front() {
            let Some(parent) = hugr.get_parent(bb_node) else {
                continue;
            };
            if bnsm.extend_parent_needs_for(&hugr, bb_node) {
                add_parent(parent, &mut parent_worklist);
            }
        }
        bnsm.finish(&hugr)
    };

    #[cfg(debug_assertions)]
    {
        for (&n, wi) in nonlocal_edges_map.iter() {
            let mut m = n;
            loop {
                let parent = hugr.get_parent(m).unwrap();
                if hugr.get_parent(wi.source.node()).unwrap() == parent {
                    break;
                }
                assert!(bb_needs_sources_map[&parent].contains_key(&wi.source));
                m = parent;
            }
        }

        for &bb in bb_needs_sources_map.keys() {
            assert!(hugr.get_parent(bb).is_some());
        }
    }

    let (parent_source_map, worklist) = {
        let mut worklist = nonlocal_edges_map.into_values().collect_vec();
        let (wl, psm) = thread_sources(hugr, &bb_needs_sources_map);
        worklist.extend(wl);
        (psm, worklist)
    };

    for wi in worklist {
        let parent = hugr.get_parent(wi.target.0).unwrap();
        let source = if hugr.get_parent(wi.source.node()).unwrap() == parent {
            wi.source
        } else {
            parent_source_map
                .get_source_in_parent(parent, wi.source)
                .unwrap()
        };
        debug_assert_eq!(hugr.get_parent(source.node()), hugr.get_parent(wi.target.0));
        hugr.disconnect(wi.target.0, wi.target.1);
        hugr.connect(source.node(), source.source(), wi.target.0, wi.target.1);
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer},
        extension::prelude::{bool_t, Noop},
        ops::{handle::NodeHandle, Tag, TailLoop, Value},
        type_row,
        types::Signature,
    };

    use super::*;

    #[test]
    fn ensures_no_nonlocal_edges() {
        let hugr = {
            let mut builder =
                DFGBuilder::new(Signature::new_endo(bool_t()).with_prelude()).unwrap();
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
            let mut builder =
                DFGBuilder::new(Signature::new_endo(bool_t()).with_prelude()).unwrap();
            let [in_w] = builder.input_wires_arr();
            let ([out_w], edge) = {
                let mut dfg_builder = builder
                    .dfg_builder(Signature::new(type_row![], bool_t()).with_prelude(), [])
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
            NonLocalEdgesError::Edges(vec![edge])
        );
    }

    #[test]
    fn unnonlocal_dfg() {
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
            let [w0] = outer.input_wires_arr();
            let [w1] = {
                let inner = outer
                    .dfg_builder(Signature::new(type_row![], bool_t()), [])
                    .unwrap();
                inner.finish_with_outputs([w0]).unwrap().outputs_arr()
            };
            outer.finish_hugr_with_outputs([w1]).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        let root = hugr.root();
        remove_nonlocal_edges(&mut hugr, root).unwrap();
        hugr.validate().unwrap();
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn unnonlocal_tailloop() {
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
        let root = hugr.root();
        remove_nonlocal_edges(&mut hugr, root).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn unnonlocal_conditional() {
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
        let root = hugr.root();
        remove_nonlocal_edges(&mut hugr, root).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }

    #[test]
    fn unnonlocal_cfg() {
        let (t1, t2, t3) = (Type::UNIT, bool_t(), Type::new_unit_sum(3));
        // let out_variants = vec![t1.clone().into(), t2.clone().into()];
        let out_type = t1.clone();
        let mut hugr = {
            let mut outer = DFGBuilder::new(Signature::new(
                vec![t1.clone(), t2.clone(), t3.clone()],
                out_type.clone(),
            ))
            .unwrap();
            let [s1, s2, s3] = outer.input_wires_arr();
            let [out] = {
                let mut cfg = outer.cfg_builder([], out_type.into()).unwrap();

                let entry = {
                    let mut entry = cfg.entry_builder([type_row![]], type_row![]).unwrap();
                    let w = entry.add_load_value(Value::unit());
                    entry.finish_with_outputs(w, []).unwrap()
                };
                let exit = cfg.exit_block();

                let bb1 = {
                    let mut entry = cfg
                        .block_builder(type_row![], [type_row![]], t1.clone().into())
                        .unwrap();
                    let w = entry.add_load_value(Value::unit());
                    entry.finish_with_outputs(w, [s1]).unwrap()
                };
                cfg.branch(&entry, 0, &bb1).unwrap();
                cfg.branch(&bb1, 0, &exit).unwrap();
                cfg.finish_sub_container().unwrap().outputs_arr()
            };
            outer.finish_hugr_with_outputs([out]).unwrap()
        };
        assert!(ensure_no_nonlocal_edges(&hugr).is_err());
        let root = hugr.root();
        remove_nonlocal_edges(&mut hugr, root).unwrap();
        println!("{}", hugr.mermaid_string());
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }
}

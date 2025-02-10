//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
use delegate::delegate;
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    iter, mem,
};

//TODO Add `remove_nonlocal_edges` and `add_nonlocal_edges` functions
use itertools::Itertools as _;

use hugr_core::{
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, HierarchyView},
        HugrError,
    },
    ops::{DataflowOpTrait as _, OpType, Tag, TailLoop},
    types::{EdgeKind, Type, TypeRow},
    Direction, HugrView, IncomingPort, Node, PortIndex, Wire,
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
struct ParentSourceMap(HashMap<Node, HashMap<Wire, (Wire, Type)>>);

impl ParentSourceMap {
    fn insert_sources_in_parent(
        &mut self,
        parent: Node,
        sources: impl IntoIterator<Item = (Wire, Wire, Type)>,
    ) {
        debug_assert!(!self.0.contains_key(&parent));
        self.0
            .entry(parent)
            .or_default()
            .extend(sources.into_iter().map(|(s, p, t)| (s, (p, t))));
    }

    fn get_source_in_parent(
        &self,
        parent: Node,
        source: Wire,
        ref hugr: impl HugrView,
    ) -> (Wire, Type) {
        let r @ (w, _) = self
            .0
            .get(&parent)
            .and_then(|m| m.get(&source).cloned())
            .unwrap();
        debug_assert_eq!(hugr.get_parent(w.node()).unwrap(), parent);
        r
    }

    fn thread_dataflow_parent(
        &mut self,
        hugr: &mut impl HugrMut,
        parent: Node,
        start_port_index: usize,
        sources: impl IntoIterator<Item = (Wire, Type)>,
    ) {
        let (source_wires, source_types): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
        let input_wires = {
            let [input_n, _] = hugr.get_io(parent).unwrap();
            let Some(mut input) = hugr.get_optype(input_n).as_input().cloned() else {
                panic!("impossible")
            };
            vec_insert(input.types.to_mut(), source_types.clone(), start_port_index);
            hugr.replace_op(input_n, input).unwrap();
            hugr.insert_ports(
                input_n,
                Direction::Outgoing,
                start_port_index,
                source_wires.len(),
            )
            .map(move |new_port| Wire::new(input_n, new_port))
            .collect_vec()
        };
        self.insert_sources_in_parent(
            parent,
            itertools::izip!(source_wires, input_wires, source_types),
        );
    }
}

#[derive(Clone, Debug)]
struct ControlWorkItem {
    output_node: Node,
    variant_source_prefixes: Vec<Vec<Wire>>,
}

impl ControlWorkItem {
    fn go(self, hugr: &mut impl HugrMut, psm: &ParentSourceMap) {
        let parent = hugr.get_parent(self.output_node).unwrap();
        let Some(mut output) = hugr.get_optype(self.output_node).as_output().cloned() else {
            panic!("impossible")
        };
        let mut needed_sources = BTreeMap::new();
        let (cond, new_control_type) = {
            let Some(EdgeKind::Value(control_type)) = hugr
                .get_optype(self.output_node)
                .port_kind(IncomingPort::from(0))
            else {
                panic!("impossible")
            };
            let Some(sum_type) = control_type.as_sum() else {
                panic!("impossible")
            };

            let mut type_for_source = |source: &Wire| {
                let (w, t) = psm.get_source_in_parent(parent, *source, &hugr);
                let replaced = needed_sources.insert(*source, (w, t.clone()));
                debug_assert!(!replaced.is_some_and(|x| x != (w, t.clone())));
                t
            };
            let old_sum_rows: Vec<TypeRow> = sum_type
                .variants()
                .map(|x| x.clone().try_into().unwrap())
                .collect_vec();
            let new_sum_rows: Vec<TypeRow> =
                itertools::zip_eq(self.variant_source_prefixes.iter(), old_sum_rows.iter())
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
                needed_sources
                    .values()
                    .map(|(_, t)| t.clone())
                    .collect_vec(),
                new_control_type.clone(),
            )
            .unwrap();
            for (i, new_sources) in self.variant_source_prefixes.into_iter().enumerate() {
                let mut case = cond.case_builder(i).unwrap();
                let case_inputs = case.input_wires().collect_vec();
                let mut args = new_sources
                    .into_iter()
                    .map(|s| {
                        case_inputs[old_sum_rows[i].len()
                            + needed_sources
                                .iter()
                                .find_position(|(&w, _)| w == s)
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
        let cond_node = hugr.insert_hugr(parent, cond).new_root;
        let (old_output_source_node, old_output_source_port) =
            hugr.single_linked_output(self.output_node, 0).unwrap();
        debug_assert_eq!(hugr.get_parent(old_output_source_node).unwrap(), parent);
        hugr.connect(old_output_source_node, old_output_source_port, cond_node, 0);
        for (i, &(w, _)) in needed_sources.values().enumerate() {
            hugr.connect(w.node(), w.source(), cond_node, i + 1);
        }
        hugr.disconnect(self.output_node, IncomingPort::from(0));
        hugr.connect(cond_node, 0, self.output_node, 0);
        output.types.to_mut()[0] = new_control_type;
        hugr.replace_op(self.output_node, output).unwrap();
    }
}

#[derive(Clone, Debug)]
struct ThreadState<'a> {
    parent_source_map: ParentSourceMap,
    needs: &'a BBNeedsSourcesMap,
    worklist: Vec<WorkItem>,
    control_worklist: Vec<ControlWorkItem>,
}

impl<'a> ThreadState<'a> {
    delegate! {
        to self.parent_source_map {
            fn thread_dataflow_parent(
                &mut self,
                hugr: &mut impl HugrMut,
                parent: Node,
                start_port_index: usize,
                sources: impl IntoIterator<Item=(Wire, Type)>,
            );
        }
    }

    fn new(bbnsm: &'a BBNeedsSourcesMap) -> Self {
        Self {
            parent_source_map: Default::default(),
            needs: bbnsm,
            worklist: vec![],
            control_worklist: vec![],
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
            let mut this_dfb = hugr.get_optype(node).as_dataflow_block().unwrap().clone();
            let mut nsrp = vec![vec![]; this_dfb.sum_rows.len()];
            vec_prepend(this_dfb.inputs.to_mut(), types.clone());

            for (this_p, succ_n) in hugr.node_outputs(node).filter_map(|out_p| {
                let (succ_n, _) = hugr.single_linked_input(node, out_p).unwrap();
                hugr.get_optype(succ_n)
                    .is_dataflow_block()
                    .then_some((out_p.index(), succ_n))
            }) {
                let succ_needs = &self.needs[&succ_n];
                let succ_needs_source_indices = succ_needs
                    .iter()
                    .map(|(&w, _)| sources.iter().find_position(|(x, _)| x == &w).unwrap().0)
                    .collect_vec();
                let succ_needs_tys = succ_needs_source_indices
                    .iter()
                    .copied()
                    .map(|x| sources[x].1.clone())
                    .collect_vec();
                vec_prepend(this_dfb.sum_rows[this_p].to_mut(), succ_needs_tys);
                nsrp[this_p] = succ_needs_source_indices;
            }
            hugr.replace_op(node, this_dfb).unwrap();
            nsrp
        };

        self.thread_dataflow_parent(hugr, node, 0, sources.clone());

        let [_, o] = hugr.get_io(node).unwrap();
        self.control_worklist.push(ControlWorkItem {
            output_node: o,
            variant_source_prefixes: new_sum_row_prefixes
                .into_iter()
                .map(|v| v.into_iter().map(|i| sources[i].0).collect_vec())
                .collect_vec(),
        });
    }

    fn do_cfg(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        {
            let mut cfg = hugr.get_optype(node).as_cfg().unwrap().clone();
            vec_insert(cfg.signature.input.to_mut(), types, 0);
            hugr.replace_op(node, cfg).unwrap();
        }
        let new_cond_ports = hugr
            .insert_ports(node, Direction::Incoming, 0, sources.len())
            .map_into();
        self.worklist
            .extend(mk_workitems(node, sources, new_cond_ports))
    }

    fn do_dfg(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut dfg = hugr.get_optype(node).as_dfg().unwrap().clone();
        let start_new_port_index = dfg.signature.input().len();
        let new_dfg_ports = hugr
            .insert_ports(
                node,
                Direction::Incoming,
                start_new_port_index,
                sources.len(),
            )
            .map_into();
        dfg.signature
            .input
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, dfg).unwrap();
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
        let new_cond_ports = hugr
            .insert_ports(
                node,
                Direction::Incoming,
                start_new_port_index,
                sources.len(),
            )
            .map_into();
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
        self.thread_dataflow_parent(hugr, node, start_case_port_index, sources);
    }

    fn do_tailloop(&mut self, hugr: &mut impl HugrMut, node: Node, sources: Vec<(Wire, Type)>) {
        let mut tailloop = hugr.get_optype(node).as_tail_loop().unwrap().clone();
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        {
            vec_prepend(tailloop.just_inputs.to_mut(), types.clone());
            hugr.replace_op(node, tailloop).unwrap();
        }
        let tailloop_ports = hugr
            .insert_ports(node, Direction::Incoming, 0, sources.len())
            .map_into();

        self.thread_dataflow_parent(hugr, node, 0, sources.clone());

        let [_, o] = hugr.get_io(node).unwrap();
        let new_sum_row_prefixes = {
            let mut v = vec![vec![]; 2];
            v[TailLoop::CONTINUE_TAG].extend(sources.iter().map(|x| x.0));
            v
        };
        self.control_worklist.push(ControlWorkItem {
            output_node: o,
            variant_source_prefixes: new_sum_row_prefixes,
        });
        self.worklist
            .extend(mk_workitems(node, sources, tailloop_ports))
    }

    fn finish(
        self,
        _hugr: &mut impl HugrMut,
    ) -> (Vec<WorkItem>, ParentSourceMap, Vec<ControlWorkItem>) {
        (self.worklist, self.parent_source_map, self.control_worklist)
    }
}

fn thread_sources(
    hugr: &mut impl HugrMut,
    bb_needs_sources_map: &BBNeedsSourcesMap,
) -> (Vec<WorkItem>, ParentSourceMap, Vec<ControlWorkItem>) {
    let mut state = ThreadState::new(bb_needs_sources_map);
    for (&bb, sources) in bb_needs_sources_map {
        let sources = sources.iter().map(|(&w, ty)| (w, ty.clone())).collect_vec();
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

    let (parent_source_map, worklist, control_worklist) = {
        let mut worklist = nonlocal_edges_map.into_values().collect_vec();
        let (wl, psm, control_worklist) = thread_sources(hugr, &bb_needs_sources_map);
        worklist.extend(wl);
        (psm, worklist, control_worklist)
    };

    for wi in worklist {
        let parent = hugr.get_parent(wi.target.0).unwrap();
        let source = if hugr.get_parent(wi.source.node()).unwrap() == parent {
            wi.source
        } else {
            parent_source_map
                .get_source_in_parent(parent, wi.source, &hugr)
                .0
        };
        debug_assert_eq!(hugr.get_parent(source.node()), hugr.get_parent(wi.target.0));
        hugr.disconnect(wi.target.0, wi.target.1);
        hugr.connect(source.node(), source.source(), wi.target.0, wi.target.1);
    }

    for cwi in control_worklist {
        cwi.go(hugr, &parent_source_map)
    }

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
        extension::prelude::{bool_t, either_type, Noop},
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
                    .dfg_builder(Signature::new_endo(bool_t()), [w0])
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
        let root = hugr.root();
        remove_nonlocal_edges(&mut hugr, root).unwrap();
        println!("{}", hugr.mermaid_string());
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }
}

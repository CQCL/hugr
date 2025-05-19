//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
use delegate::delegate;
use std::{
    collections::{BTreeMap, HashMap},
    iter, mem,
};

use hugr_core::{HugrView, IncomingPort, core::HugrNode};
use itertools::{Either, Itertools as _};

use hugr_core::{
    Direction, PortIndex, Wire,
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    hugr::{HugrError, hugrmut::HugrMut},
    ops::{DataflowOpTrait as _, OpType, Tag, TailLoop},
    types::{EdgeKind, Type, TypeRow},
};

// use crate::validation::{ValidatePassError, ValidationLevel};

/// TODO docs
// #[derive(Debug, Clone, Default)]
// pub struct UnNonLocalPass {
//     validation: ValidationLevel,
// }

// impl UnNonLocalPass {
//     /// Sets the validation level used before and after the pass is run.
//     pub fn validation_level(mut self, level: ValidationLevel) -> Self {
//         self.validation = level;
//         self
//     }

//     /// Run the Monomorphization pass.
//     fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), NonLocalEdgesError> {
//         let root = hugr.root();
//         remove_nonlocal_edges(hugr, root)?;
//         Ok(())
//     }

//     /// Run the pass using specified configuration.
//     pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), NonLocalEdgesError> {
//         self.validation
//             .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
//     }
// }

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
// `BB` is any (dataflow?) container, not necessarily a Basic Block or in a CFG
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
        loop {
            if source_parent == parent {
                break;
            }
            if !self.needs_sources.insert(parent, source, ty.clone()) {
                break;
            }
            let Some(parent_of_parent) = self.hugr.get_parent(parent) else {
                break;
            };
            parent = parent_of_parent
        }
    }

    fn finish(mut self) -> BBNeedsSourcesMap<H::Node> {
        {
            // Conditionals. Any `Case` needing an input, means the parent Conditional needs it too.
            let conds = self
                .needs_sources
                .keys()
                .copied()
                .filter(|&n| self.hugr.get_optype(n).is_conditional())
                .collect_vec();
            for n in conds {
                let n_needs = self
                    .needs_sources
                    .get(n)
                    .map(|(&w, ty)| (w, ty.clone()))
                    .collect_vec();
                for case in self
                    .hugr
                    .children(n)
                    .filter(|&child| self.hugr.get_optype(child).is_case())
                {
                    for (w, ty) in n_needs.iter() {
                        self.needs_sources.insert(case, *w, ty.clone());
                    }
                }
            }
        }
        {
            let cfgs = self
                .needs_sources
                .keys()
                .copied()
                .filter(|&n| self.hugr.get_optype(n).is_cfg())
                .collect_vec();
            for n in cfgs {
                let dfbs = self
                    .hugr
                    .children(n)
                    .filter(|&child| self.hugr.get_optype(child).is_dataflow_block())
                    .collect_vec();
                loop {
                    let mut any_change = false;
                    for &dfb in dfbs.iter() {
                        for succ_n in self.hugr.output_neighbours(dfb) {
                            for (w, ty) in self
                                .needs_sources
                                .get(succ_n)
                                .map(|(w, ty)| (*w, ty.clone()))
                                .collect_vec()
                            {
                                // Do we need something like:
                                //  if w.node() == dfb: continue
                                any_change |= self.needs_sources.insert(dfb, w, ty.clone());
                            }
                        }
                    }
                    if !any_change {
                        break;
                    }
                }
            }
        }
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

#[derive(derive_more::Error, derive_more::From, derive_more::Display, Debug, PartialEq)]
#[non_exhaustive]
pub enum NonLocalEdgesError<N> {
    #[display("Found {} nonlocal edges", _0.len())]
    #[error(ignore)]
    Edges(Vec<(N, IncomingPort)>),
    #[from]
    HugrError(HugrError),
}

/// Verifies that there are no non local value edges in the Hugr.
pub fn ensure_no_nonlocal_edges<H: HugrView>(hugr: &H) -> Result<(), NonLocalEdgesError<H::Node>> {
    let non_local_edges: Vec<_> = nonlocal_edges(hugr).collect_vec();
    if non_local_edges.is_empty() {
        Ok(())
    } else {
        Err(NonLocalEdgesError::Edges(non_local_edges))?
    }
}

#[derive(Debug, Clone)]
struct WorkItem<N: HugrNode> {
    source: Wire<N>,
    target: (N, IncomingPort),
    ty: Type,
}

impl<N: HugrNode> WorkItem<N> {
    pub fn go(self, hugr: &mut impl HugrMut<Node = N>, parent_source_map: &ParentSourceMap<N>) {
        let parent = hugr.get_parent(self.target.0).unwrap();
        let source = if hugr.get_parent(self.source.node()).unwrap() == parent {
            self.source
        } else {
            parent_source_map
                .get_source_in_parent(parent, self.source, &hugr)
                .0
        };
        debug_assert_eq!(
            hugr.get_parent(source.node()),
            hugr.get_parent(self.target.0)
        );
        hugr.disconnect(self.target.0, self.target.1);
        hugr.connect(source.node(), source.source(), self.target.0, self.target.1);
    }
}

#[derive(Clone, Debug)]
struct ParentSourceMap<N: HugrNode>(BTreeMap<N, BTreeMap<Wire<N>, (Wire<N>, Type)>>);

impl<N: HugrNode> Default for ParentSourceMap<N> {
    fn default() -> Self {
        Self(BTreeMap::default())
    }
}

impl<N: HugrNode> ParentSourceMap<N> {
    fn insert_sources_in_parent(
        &mut self,
        parent: N,
        sources: impl IntoIterator<Item = (Wire<N>, Wire<N>, Type)>,
    ) {
        debug_assert!(!self.0.contains_key(&parent));
        self.0
            .entry(parent)
            .or_default()
            .extend(sources.into_iter().map(|(s, p, t)| (s, (p, t))));
    }

    fn get_source_in_parent(
        &self,
        parent: N,
        source: Wire<N>,
        hugr: impl HugrView<Node = N>,
    ) -> (Wire<N>, Type) {
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
        hugr: &mut impl HugrMut<Node = N>,
        parent: N,
        start_port_index: usize,
        sources: impl IntoIterator<Item = (Wire<N>, Type)>,
    ) {
        let (source_wires, source_types): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
        let input_wires = {
            let [input_n, _] = hugr.get_io(parent).unwrap();
            let Some(mut input) = hugr.get_optype(input_n).as_input().cloned() else {
                panic!("impossible")
            };
            vec_insert(input.types.to_mut(), source_types.clone(), start_port_index);
            hugr.replace_op(input_n, input);
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
struct ControlWorkItem<N: HugrNode> {
    output_node: N, // Output node of CFG / TailLoop
    variant_source_prefixes: Vec<Vec<Wire<N>>>, // prefixes to each element of Sum type
}

impl<N: HugrNode> ControlWorkItem<N> {
    fn go(self, hugr: &mut impl HugrMut<Node = N>, psm: &ParentSourceMap<N>) {
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

            let mut type_for_source = |source: &Wire<N>| {
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
            hugr.single_linked_output(self.output_node, 0).unwrap();
        debug_assert_eq!(hugr.get_parent(old_output_source_node).unwrap(), parent);
        hugr.connect(old_output_source_node, old_output_source_port, cond_node, 0);
        for (i, &(w, _)) in needed_sources.values().enumerate() {
            hugr.connect(w.node(), w.source(), cond_node, i + 1);
        }
        hugr.disconnect(self.output_node, IncomingPort::from(0));
        hugr.connect(cond_node, 0, self.output_node, 0);
        output.types.to_mut()[0] = new_control_type;
        hugr.replace_op(self.output_node, output);
    }
}

#[derive(Clone, Debug)]
struct ThreadState<'a, N: HugrNode> {
    parent_source_map: ParentSourceMap<N>,
    needs: &'a BBNeedsSourcesMap<N>,
    worklist: Vec<WorkItem<N>>,
    control_worklist: Vec<ControlWorkItem<N>>,
}

impl<'a, N: HugrNode> ThreadState<'a, N> {
    delegate! {
        to self.parent_source_map {
            fn thread_dataflow_parent(
                &mut self,
                hugr: &mut impl HugrMut<Node=N>,
                parent: N,
                start_port_index: usize,
                sources: impl IntoIterator<Item=(Wire<N>, Type)>,
            );
        }
    }

    fn new(bbnsm: &'a BBNeedsSourcesMap<N>) -> Self {
        Self {
            parent_source_map: ParentSourceMap::default(),
            needs: bbnsm,
            worklist: vec![],
            control_worklist: vec![],
        }
    }

    fn do_dataflow_block(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
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
                let succ_needs_source_indices = self
                    .needs
                    .get(succ_n)
                    .map(|(w, _)| sources.iter().find_position(|(x, _)| x == w).unwrap().0)
                    .collect_vec();
                let succ_needs_tys = succ_needs_source_indices
                    .iter()
                    .copied()
                    .map(|x| sources[x].1.clone())
                    .collect_vec();
                vec_prepend(this_dfb.sum_rows[this_p].to_mut(), succ_needs_tys);
                nsrp[this_p] = succ_needs_source_indices;
            }
            hugr.replace_op(node, this_dfb);
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

    fn do_cfg(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
    ) {
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        {
            let mut cfg = hugr.get_optype(node).as_cfg().unwrap().clone();
            vec_insert(cfg.signature.input.to_mut(), types, 0);
            hugr.replace_op(node, cfg);
        }
        let new_cond_ports = hugr
            .insert_ports(node, Direction::Incoming, 0, sources.len())
            .map_into();
        self.worklist
            .extend(mk_workitems(node, sources, new_cond_ports))
    }

    fn do_dfg(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
    ) {
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
        hugr.replace_op(node, dfg);
        self.thread_dataflow_parent(hugr, node, start_new_port_index, sources.iter().cloned());
        self.worklist
            .extend(mk_workitems(node, sources, new_dfg_ports));
    }

    fn do_conditional(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
    ) {
        let mut cond = hugr.get_optype(node).as_conditional().unwrap().clone();
        let start_new_port_index = cond.signature().input().len();
        cond.other_inputs
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, cond);
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

    fn do_case(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
    ) {
        let mut case = hugr.get_optype(node).as_case().unwrap().clone();
        let start_case_port_index = case.signature.input().len();
        case.signature
            .input
            .to_mut()
            .extend(sources.iter().map(|x| x.1.clone()));
        hugr.replace_op(node, case);
        self.thread_dataflow_parent(hugr, node, start_case_port_index, sources);
    }

    fn do_tailloop(
        &mut self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        sources: Vec<(Wire<N>, Type)>,
    ) {
        let mut tailloop = hugr.get_optype(node).as_tail_loop().unwrap().clone();
        let types = sources.iter().map(|x| x.1.clone()).collect_vec();
        {
            vec_prepend(tailloop.just_inputs.to_mut(), types.clone());
            hugr.replace_op(node, tailloop);
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
        _hugr: &mut impl HugrMut<Node = N>,
    ) -> (
        Vec<WorkItem<N>>,
        ParentSourceMap<N>,
        Vec<ControlWorkItem<N>>,
    ) {
        (self.worklist, self.parent_source_map, self.control_worklist)
    }
}

fn thread_sources<N: HugrNode>(
    hugr: &mut impl HugrMut<Node = N>,
    bb_needs_sources_map: &BBNeedsSourcesMap<N>,
) -> (
    Vec<WorkItem<N>>,
    ParentSourceMap<N>,
    Vec<ControlWorkItem<N>>,
) {
    let mut state = ThreadState::new(bb_needs_sources_map);
    for (&bb, sources) in bb_needs_sources_map.0.iter() {
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

fn mk_workitems<N: HugrNode>(
    node: N,
    sources: impl IntoIterator<Item = (Wire<N>, Type)>,
    ports: impl IntoIterator<Item = IncomingPort>,
) -> impl Iterator<Item = WorkItem<N>> {
    itertools::izip!(sources, ports).map(move |((source, ty), p)| WorkItem {
        source,
        target: (node, p),
        ty,
    })
}

pub fn remove_nonlocal_edges<H: HugrMut>(hugr: &mut H) -> Result<(), NonLocalEdgesError<H::Node>> {
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

    // Here we mutate the HUGR; adding ports to parent nodes and their Input nodes.
    // The result is:
    //  * parent_source_map: A map from parent and source to the wire that should substitute for that source in that parent.
    //  * worklist: a list of workitems. Each should be fulfilled by connecting the source, substituted through parent_source_map, to the target.
    //  * control_worklist: A list of control ports (i.e. 0th output port of DataflowBlock or TailLoop) that must be rewired.
    let (parent_source_map, worklist, control_worklist) = {
        let mut worklist = nonlocal_edges_map.into_values().collect_vec();
        let (wl, psm, control_worklist) = thread_sources(hugr, &bb_needs_sources_map);
        worklist.extend(wl);
        (psm, worklist, control_worklist)
    };

    for wi in worklist {
        wi.go(hugr, &parent_source_map)
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
        extension::prelude::{Noop, bool_t, either_type},
        ops::{Tag, TailLoop, Value, handle::NodeHandle},
        type_row,
        types::Signature,
    };

    use super::*;

    #[test]
    fn vec_insert0() {
        let mut v = vec![5,7,9];
        vec_insert(&mut v, [1,2], 0);
        assert_eq!(v, [1,2,5,7,9]);
    }

    #[test]
    fn vec_insert1() {
        let mut v = vec![5,7,9];
        vec_insert(&mut v, [1,2], 1);
        assert_eq!(v, [5,1,2,7,9]);
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
        remove_nonlocal_edges(&mut hugr).unwrap();
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
        remove_nonlocal_edges(&mut hugr).unwrap();
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
        remove_nonlocal_edges(&mut hugr).unwrap();
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
        remove_nonlocal_edges(&mut hugr).unwrap();
        println!("{}", hugr.mermaid_string());
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }
}

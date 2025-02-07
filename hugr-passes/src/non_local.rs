//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    iter,
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
    HugrView, IncomingPort, Node, Wire,
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

fn thread_dataflow_parent(
    hugr: &mut impl HugrMut,
    parent: Node,
    start_port_index: usize,
    types: Vec<Type>,
) -> impl Iterator<Item = Wire> {
    let [input_n, _] = hugr.get_io(parent).unwrap();
    let OpType::Input(mut input) = hugr.get_optype(input_n).clone() else {
        panic!("impossible")
    };
    let mut r = vec![];
    for (i, ty) in types.into_iter().enumerate() {
        input
            .types
            .to_mut()
            .insert(start_port_index + i, ty.clone());
        r.push(Wire::new(
            input_n,
            hugr.insert_outgoing_port(input_n, start_port_index + i),
        ));
    }
    hugr.replace_op(input_n, input).unwrap();
    r.into_iter()
}

fn do_tailloop(
    parent_source_map: &mut ParentSourceMap,
    hugr: &mut impl HugrMut,
    node: Node,
    sources: impl IntoIterator<Item = (Wire, Type)>,
) -> impl Iterator<Item = WorkItem> {
    let (sources, types): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
    let mut tailloop = hugr.get_optype(node).as_tail_loop().unwrap().clone();
    let start_port_index = tailloop.just_inputs.len();
    {
        tailloop.just_inputs.to_mut().extend(types.iter().cloned());
        hugr.replace_op(node, tailloop).unwrap();
    }
    let tailloop_ports = (0..sources.len())
        .map(|i| hugr.insert_incoming_port(node, start_port_index + i))
        .collect_vec();

    let input_wires =
        thread_dataflow_parent(hugr, node, start_port_index, types.clone()).collect_vec();
    parent_source_map.insert(
        node,
        iter::zip(sources.iter().copied(), input_wires.iter().copied()),
    );

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
            ConditionalBuilder::new(old_sum_rows, types.clone(), new_control_type.clone()).unwrap();
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
    mk_workitems(node, sources, tailloop_ports, types)
}

#[derive(Clone, Default, Debug)]
struct ParentSourceMap(HashMap<Node, HashMap<Wire, Wire>>);

impl ParentSourceMap {
    fn contains_parent(&self, parent: Node) -> bool {
        self.0.contains_key(&parent)
    }

    fn insert(&mut self, parent: Node, sources: impl IntoIterator<Item = (Wire, Wire)>) {
        debug_assert!(!self.0.contains_key(&parent));
        self.0.entry(parent).or_default().extend(sources);
    }

    fn get(&self, parent: Node, source: Wire) -> Option<Wire> {
        self.0.get(&parent).and_then(|m| m.get(&source).cloned())
    }
}

fn mk_workitems(
    node: Node,
    sources: impl IntoIterator<Item = Wire>,
    ports: impl IntoIterator<Item = IncomingPort>,
    types: impl IntoIterator<Item = Type>,
) -> impl Iterator<Item = WorkItem> {
    itertools::izip!(sources, ports, types).map(move |(source, p, ty)| WorkItem {
        source,
        target: (node, p),
        ty,
    })
}

fn thread_sources(
    parent_source_map: &mut ParentSourceMap,
    hugr: &mut impl HugrMut,
    bb: Node,
    sources: impl IntoIterator<Item = (Wire, Type)>,
) -> Vec<WorkItem> {
    let (source_wires, types): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
    match hugr.get_optype(bb).clone() {
        OpType::DFG(mut dfg) => {
            debug_assert!(!parent_source_map.contains_parent(bb));
            let start_new_port_index = dfg.signature.input().len();
            let new_dfg_ports = (0..source_wires.len())
                .map(|i| hugr.insert_incoming_port(bb, start_new_port_index + i))
                .collect_vec();
            dfg.signature.input.to_mut().extend(types.clone());
            hugr.replace_op(bb, dfg).unwrap();
            for (source, &target) in iter::zip(source_wires.iter(), new_dfg_ports.iter()) {
                hugr.connect(source.node(), source.source(), bb, target);
            }
            parent_source_map.insert(
                bb,
                iter::zip(
                    source_wires.iter().copied(),
                    thread_dataflow_parent(hugr, bb, start_new_port_index, types.clone()),
                ),
            );
            mk_workitems(bb, source_wires, new_dfg_ports, types).collect_vec()
        }
        OpType::Conditional(mut cond) => {
            debug_assert!(!parent_source_map.contains_parent(bb));
            let start_new_port_index = cond.signature().input().len();
            cond.other_inputs.to_mut().extend(types.clone());
            hugr.replace_op(bb, cond).unwrap();
            let new_cond_ports = (0..source_wires.len())
                .map(|i| hugr.insert_incoming_port(bb, start_new_port_index + i))
                .collect_vec();
            parent_source_map.insert(bb, iter::empty());
            mk_workitems(bb, source_wires, new_cond_ports, types).collect_vec()
        }
        OpType::Case(mut case) => {
            debug_assert!(!parent_source_map.contains_parent(bb));
            let start_case_port_index = case.signature.input().len();
            case.signature.input.to_mut().extend(types.clone());
            hugr.replace_op(bb, case).unwrap();
            parent_source_map.insert(
                bb,
                iter::zip(
                    source_wires.iter().copied(),
                    thread_dataflow_parent(hugr, bb, start_case_port_index, types),
                ),
            );
            vec![]
        }
        OpType::TailLoop(_) => {
            do_tailloop(parent_source_map, hugr, bb, iter::zip(source_wires, types)).collect_vec()
        }
        _ => panic!("impossible"),
    }
    //             _ => panic!("impossible"),
    //         };
    //         non_local_edges.push(workitem);
    //         wire
    //     };
    //     hugr.disconnect(target.0, target.1);
    //     hugr.connect(
    //         local_source.node(),
    //         local_source.source(),
    //         target.0,
    //         target.1,
    //     );
    // }
}

#[derive(Debug, Default, Clone)]
struct BBNeedsSourcesMapBuilder(HashMap<Node, BTreeMap<Wire, Type>>);

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
                m.iter().filter(move |(w, _)| hugr.get_parent(w.node()).unwrap() != parent)
                    .map(|(&w, ty)| (w, ty.clone()))
            })
            .collect_vec();
        let any = !parent_needs.is_empty();
        if any {
            self.0.entry(parent).or_default().extend(parent_needs);
        }
        any
    }

    fn finish(mut self, hugr: impl HugrView) -> HashMap<Node, BTreeMap<Wire, Type>> {
        let conds = self
            .0
            .keys()
            .copied()
            .filter(|&n| hugr.get_optype(n).is_conditional())
            .collect_vec();
        for cond in conds {
            if hugr.get_optype(cond).is_conditional() {
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
                let children = nodes
                    .iter()
                    .flat_map(|&n| hugr.children(n))
                    .collect_vec();
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

    let mut worklist = nonlocal_edges_map.into_values().collect_vec();
    let mut parent_source_map = ParentSourceMap::default();

    for (bb, needs_sources) in bb_needs_sources_map {
        worklist.extend(thread_sources(
            &mut parent_source_map,
            hugr,
            bb,
            needs_sources,
        ));
    }

    let parent_source_map = parent_source_map;

    while let Some(wi) = worklist.pop() {
        let parent = hugr.get_parent(wi.target.0).unwrap();
        let source = if hugr.get_parent(wi.source.node()).unwrap() == parent {
            wi.source
        } else {
            parent_source_map.get(parent, wi.source).unwrap()
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
    fn remove_nonlocal_edges_dfg() {
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
    fn remove_nonlocal_edges_tailloop() {
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
    fn remove_nonlocal_edges_cond() {
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
}

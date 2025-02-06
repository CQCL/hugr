//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
use ascent::hashbrown::HashMap;
//
//TODO Add `remove_nonlocal_edges` and `add_nonlocal_edges` functions
use itertools::{Either, Itertools as _};

use hugr_core::{
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    hugr::hugrmut::HugrMut,
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

impl UnNonLocalPass  {
    /// Sets the validation level used before and after the pass is run.
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Run the Monomorphization pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), NonLocalEdgesError> {
        remove_nonlocal_edges(hugr)?;
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
    port_index: usize,
    ty: Type,
) -> Wire {
    let [i, _] = hugr.get_io(parent).unwrap();
    let OpType::Input(mut input) = hugr.get_optype(i).clone() else {
        panic!("impossible")
    };
    input.types.to_mut().insert(port_index, ty);
    hugr.replace_op(i, input).unwrap();
    let input_port = hugr.insert_outgoing_port(i, port_index);
    Wire::new(i, input_port)
}

fn do_tailloop(hugr: &mut impl HugrMut, node: Node, source: Wire, ty: Type) -> (WorkItem, Wire) {
    let mut tailloop = hugr.get_optype(node).as_tail_loop().unwrap().clone();
    let new_port_index = tailloop.just_inputs.len();
    tailloop.just_inputs.to_mut().push(ty.clone());
    hugr.replace_op(node, tailloop).unwrap();
    let tailloop_port = hugr.insert_incoming_port(node, new_port_index);
    hugr.connect(source.node(), source.source(), node, tailloop_port);
    let workitem = WorkItem {
        source,
        target: (node, tailloop_port),
        ty: ty.clone(),
    };

    let input_wire = thread_dataflow_parent(hugr, node, tailloop_port.index(), ty.clone());

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
            v[TailLoop::CONTINUE_TAG].to_mut().push(ty.clone());
            v
        };

        let new_control_type = Type::new_sum(new_sum_rows.clone());
        let mut cond =
            ConditionalBuilder::new(old_sum_rows, ty.clone(), new_control_type.clone()).unwrap();
        for i in 0..2 {
            let mut case = cond.case_builder(i).unwrap();
            let inputs = {
                let all_inputs = case.input_wires();
                if i == TailLoop::CONTINUE_TAG {
                    Either::Left(all_inputs)
                } else {
                    Either::Right(all_inputs.into_iter().dropping_back(1))
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
    hugr.connect(input_wire.node(), input_wire.source(), cond_node, 1);
    hugr.disconnect(o, IncomingPort::from(0));
    hugr.connect(cond_node, 0, o, 0);
    let mut output = hugr.get_optype(o).as_output().unwrap().clone();
    output.types.to_mut()[0] = new_control_type;
    hugr.replace_op(o, output).unwrap();
    (workitem, input_wire)
}

pub fn remove_nonlocal_edges(hugr: &mut impl HugrMut) -> Result<(), NonLocalEdgesError> {
    let mut non_local_edges = nonlocal_edges(hugr)
        .map(|target @ (node, inport)| {
            let source = {
                let (n, p) = hugr.single_linked_output(node, inport).unwrap();
                Wire::new(n, p)
            };
            debug_assert!(
                hugr.get_parent(source.node()).unwrap() != hugr.get_parent(node).unwrap()
            );
            let Some(EdgeKind::Value(ty)) = hugr
                .get_optype(hugr.get_parent(source.node()).unwrap())
                .port_kind(source.source())
            else {
                panic!("impossible")
            };
            WorkItem { source, target, ty }
        })
        .collect_vec();

    if non_local_edges.is_empty() {
        return Ok(());
    }

    let mut parent_source_map = HashMap::new();

    while let Some(WorkItem { source, target, ty }) = non_local_edges.pop() {
        dbg!(&source, target, &ty);
        let parent = hugr.get_parent(target.0).unwrap();
        let local_source = if hugr.get_parent(source.node()).unwrap() == parent {
            &source
        } else {
            parent_source_map
                .entry((parent, source))
                .or_insert_with(|| {
                    let (workitem, wire) = match hugr.get_optype(parent).clone() {
                        OpType::DFG(mut dfg) => {
                            let new_port_index = dfg.signature.input.len();
                            dbg!(&dfg, new_port_index);
                            dfg.signature.input.to_mut().push(ty.clone());
                            hugr.replace_op(parent, dfg).unwrap();
                            let dfg_port = hugr.insert_incoming_port(parent, new_port_index);
                            hugr.connect(source.node(), source.source(), parent, dfg_port);
                            (
                                WorkItem {
                                    source,
                                    target: (parent, dfg_port),
                                    ty: ty.clone(),
                                },
                                thread_dataflow_parent(hugr, parent, dfg_port.index(), ty),
                            )
                        }
                        OpType::DataflowBlock(dataflow_block) => todo!(),
                        OpType::TailLoop(_) => do_tailloop(hugr, parent, source, ty),
                        OpType::Case(case) => todo!(),
                        _ => panic!("impossible"),
                    };
                    non_local_edges.push(workitem);
                    wire
                })
        };
        hugr.disconnect(target.0, target.1);
        hugr.connect(
            local_source.node(),
            local_source.source(),
            target.0,
            target.1,
        );
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer},
        extension::prelude::{bool_t, Noop},
        ops::{handle::NodeHandle, Tag, TailLoop},
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
        remove_nonlocal_edges(&mut hugr).unwrap();
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
        remove_nonlocal_edges(&mut hugr).unwrap();
        hugr.validate().unwrap_or_else(|e| panic!("{e}"));
        assert!(ensure_no_nonlocal_edges(&hugr).is_ok());
    }
}

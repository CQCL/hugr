//! This module provides functions for inspecting and modifying the nature of
//! non local edges in a Hugr.
//
//TODO Add `remove_nonlocal_edges` and `add_nonlocal_edges` functions
use itertools::Itertools as _;
use thiserror::Error;

use hugr_core::{HugrView, IncomingPort, Node};

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

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NonLocalEdgesError {
    #[error("Found {} nonlocal edges", .0.len())]
    Edges(Vec<(Node, IncomingPort)>),
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

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer},
        extension::{
            prelude::{bool_t, Noop},
            EMPTY_REG,
        },
        ops::handle::NodeHandle,
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
            builder
                .finish_hugr_with_outputs([out_w], &EMPTY_REG)
                .unwrap()
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
            (
                builder
                    .finish_hugr_with_outputs([out_w], &EMPTY_REG)
                    .unwrap(),
                edge,
            )
        };
        assert_eq!(
            ensure_no_nonlocal_edges(&hugr).unwrap_err(),
            NonLocalEdgesError::Edges(vec![edge])
        );
    }
}

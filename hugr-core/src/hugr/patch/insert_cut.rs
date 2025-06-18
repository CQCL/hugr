//! Patch for inserting a sub-HUGR as a "cut" across existing edges.

use std::collections::HashMap;
use std::iter;

use crate::core::HugrNode;
use crate::hugr::patch::inline_dfg::InlineDFG;
use crate::hugr::{HugrMut, Node};
use crate::ops::{DataflowOpTrait, OpTag, OpTrait, OpType};

use crate::{Hugr, HugrView, IncomingPort};

use super::inline_dfg::InlineDFGError;
use super::{Patch, PatchHugrMut, PatchVerification};

use itertools::Itertools;
use thiserror::Error;

/// Implementation of the `InsertCut` operation.
///
/// The `InsertCut` operation allows inserting a HUGR sub-graph as a "cut" between existing nodes in a dataflow graph.
/// It effectively intercepts connections between nodes by inserting the new Hugr in between them.
///
/// This patch operation works by:
/// 1. Inserting a new HUGR as a child of the specified parent
/// 2. Redirecting existing connections through the newly inserted HUGR.
pub struct InsertCut<N = Node> {
    /// The parent node to insert the new HUGR under.
    pub parent: N,
    /// The targets of the existing edges.
    pub targets: Vec<(N, IncomingPort)>,
    /// The HUGR to insert, must have  DFG root.
    pub insertion: Hugr,
}

impl<N> InsertCut<N> {
    /// Create a new [`InsertCut`] specification.
    pub fn new(parent: N, targets: Vec<(N, IncomingPort)>, insertion: Hugr) -> Self {
        Self {
            parent,
            targets,
            insertion,
        }
    }
}
/// Error from an [`InsertCut`] operation.
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum InsertCutError<N = Node> {
    /// Invalid parent node.
    #[error("Parent node is invalid.")]
    InvalidParentNode,
    /// Invalid node.
    #[error("HUGR graph does not contain node: {0}.")]
    InvalidNode(N),

    /// Replacement HUGR not a DFG.
    #[error("Parent node is not a DFG, found root optype: {0}.")]
    ReplaceNotDfg(OpType),

    /// Inline error.
    #[error("Inlining inserting DFG failed: {0}.")]
    InlineFailed(#[from] InlineDFGError),

    /// Port connection error.
    #[error("Incoming port has {0} connections, expected exactly 1.")]
    InvalidIncomingPort(usize),

    /// Target number mismatch.
    #[error("Target number mismatch, expected {0}, found {1}.")]
    TargetNumberMismatch(usize, usize),

    /// Input/Output mismatch.
    #[error("Replacement DFG must have the same number of inputs and outputs.")]
    InputOutputMismatch,
}

impl<N: HugrNode> PatchVerification for InsertCut<N> {
    type Error = InsertCutError<N>;
    type Node = N;

    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), Self::Error> {
        let insert_root = self.insertion.entrypoint_optype();
        let Some(dfg) = insert_root.as_dfg() else {
            return Err(InsertCutError::ReplaceNotDfg(insert_root.clone()));
        };

        let sig = dfg.signature();
        if sig.input().len() != sig.output().len() {
            return Err(InsertCutError::InputOutputMismatch);
        }
        if sig.input().len() != self.targets.len() {
            return Err(InsertCutError::TargetNumberMismatch(
                sig.input().len(),
                self.targets.len(),
            ));
        }
        if !h.contains_node(self.parent) {
            return Err(InsertCutError::InvalidNode(self.parent));
        }
        let parent_op = h.get_optype(self.parent);
        if !OpTag::DataflowParent.is_superset(parent_op.tag()) {
            return Err(InsertCutError::InvalidParentNode);
        }

        // Verify that each target node exists and each target port is valid
        for (node, port) in &self.targets {
            if !h.contains_node(*node) {
                return Err(InsertCutError::InvalidNode(*node));
            }

            let n_links = h.linked_outputs(*node, *port).count();
            if n_links != 1 {
                return Err(InsertCutError::InvalidIncomingPort(n_links));
            }
        }
        Ok(())
    }

    #[inline]
    fn invalidated_nodes(
        &self,
        _: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        iter::once(self.parent)
            .chain(self.targets.iter().map(|(n, _)| *n))
            .unique()
    }
}
impl PatchHugrMut for InsertCut<Node> {
    type Outcome = HashMap<Node, Node>;
    const UNCHANGED_ON_FAILURE: bool = false;

    fn apply_hugr_mut(
        self,
        h: &mut impl HugrMut<Node = Node>,
    ) -> Result<Self::Outcome, InsertCutError> {
        let insert_res = h.insert_hugr(self.parent, self.insertion);
        let inserted_entrypoint = insert_res.inserted_entrypoint;
        for (i, (target, port)) in self.targets.into_iter().enumerate() {
            let (src_n, src_p) = h
                .single_linked_output(target, port)
                .expect("Incoming value edge has single connection.");
            h.disconnect(target, port);
            h.connect(src_n, src_p, inserted_entrypoint, i);
            h.connect(inserted_entrypoint, i, target, port);
        }
        let inline = InlineDFG(inserted_entrypoint.into());

        inline.apply(h)?;
        Ok(insert_res.node_map)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::{Noop, bool_t, qb_t},
        types::Signature,
    };

    #[rstest]
    fn test_insert_cut() {
        let dfg_b = DFGBuilder::new(Signature::new_endo(vec![bool_t(), qb_t()])).unwrap();
        let inputs = dfg_b.input().outputs();
        let mut h = dfg_b.finish_hugr_with_outputs(inputs).unwrap();
        let [i, o] = h.get_io(h.entrypoint()).unwrap();

        let mut dfg_b = DFGBuilder::new(Signature::new_endo(vec![bool_t(), qb_t()])).unwrap();
        let [b, q] = dfg_b.input().outputs_arr();
        let noop1 = dfg_b.add_dataflow_op(Noop::new(bool_t()), [b]).unwrap();
        let noop2 = dfg_b.add_dataflow_op(Noop::new(qb_t()), [q]).unwrap();

        let replacement = dfg_b
            .finish_hugr_with_outputs([noop1.out_wire(0), noop2.out_wire(0)])
            .unwrap();

        let targets: Vec<_> = h.all_linked_inputs(i).collect();
        let inserter = InsertCut::new(h.entrypoint(), targets, replacement);
        assert_eq!(
            inserter.invalidated_nodes(&h).collect::<Vec<Node>>(),
            vec![h.entrypoint(), o]
        );

        inserter.verify(&h).unwrap();

        assert_eq!(h.entry_descendants().count(), 3);
        inserter.apply_hugr_mut(&mut h).unwrap();

        h.validate().unwrap();
        assert_eq!(h.entry_descendants().count(), 5);
    }
}

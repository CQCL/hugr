//! Implementation of the `InsertIdentity` operation.

use std::iter;

use crate::extension::prelude::Noop;
use crate::hugr::{HugrMut, Node};
use crate::ops::{OpTag, OpTrait};

use crate::types::EdgeKind;
use crate::{HugrView, IncomingPort};

use super::{PatchHugrMut, PatchVerification};

use thiserror::Error;

/// Specification of a identity-insertion operation.
#[derive(Debug, Clone)]
pub struct IdentityInsertion<N = Node> {
    /// The node following the identity to be inserted.
    pub post_node: N,
    /// The port following the identity to be inserted.
    pub post_port: IncomingPort,
}

impl<N> IdentityInsertion<N> {
    /// Create a new [`IdentityInsertion`] specification.
    pub fn new(post_node: N, post_port: IncomingPort) -> Self {
        Self {
            post_node,
            post_port,
        }
    }
}

/// Error from an [`IdentityInsertion`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdentityInsertionError {
    /// Invalid parent node.
    #[error("Parent node is invalid.")]
    InvalidParentNode,
    /// Invalid node.
    #[error("Node is invalid.")]
    InvalidNode(),
    /// Invalid port kind.
    #[error("post_port has invalid kind {}. Must be Value.", _0.as_ref().map_or("None".to_string(), ToString::to_string))]
    InvalidPortKind(Option<EdgeKind>),
}

impl<N: Copy> PatchVerification for IdentityInsertion<N> {
    type Error = IdentityInsertionError;
    type Node = N;

    fn verify(&self, _h: &impl HugrView) -> Result<(), IdentityInsertionError> {
        /*
        Assumptions:
        1. Value kind inputs can only have one connection.
        2. Node exists.
        Conditions:
        1. post_port is Value kind.
        2. post_port is connected to a sibling of post_node.
        3. post_port is input.
         */

        unimplemented!()
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = N> {
        iter::once(self.post_node)
    }
}

impl PatchHugrMut for IdentityInsertion {
    /// The inserted node.
    type Outcome = Node;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(self, h: &mut impl HugrMut) -> Result<Self::Outcome, IdentityInsertionError> {
        let kind = h.get_optype(self.post_node).port_kind(self.post_port);
        let Some(EdgeKind::Value(ty)) = kind else {
            return Err(IdentityInsertionError::InvalidPortKind(kind));
        };

        let (pre_node, pre_port) = h
            .single_linked_output(self.post_node, self.post_port)
            .expect("Value kind input can only have one connection.");

        h.disconnect(self.post_node, self.post_port);
        let parent = h
            .get_parent(self.post_node)
            .ok_or(IdentityInsertionError::InvalidParentNode)?;
        if !OpTag::DataflowParent.is_superset(h.get_optype(parent).tag()) {
            return Err(IdentityInsertionError::InvalidParentNode);
        }
        let new_node = h.add_node_with_parent(parent, Noop(ty));
        h.connect(pre_node, pre_port, new_node, 0);

        h.connect(new_node, 0, self.post_node, self.post_port);
        Ok(new_node)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::super::simple_replace::test::dfg_hugr;
    use super::*;
    use crate::{extension::prelude::qb_t, Hugr};

    #[rstest]
    fn correct_insertion(dfg_hugr: Hugr) {
        let mut h = dfg_hugr;

        assert_eq!(h.node_count(), 6);

        let final_node = h
            .input_neighbours(h.get_io(h.root()).unwrap()[1])
            .next()
            .unwrap();

        let final_node_port = h.node_inputs(final_node).next().unwrap();

        let rw = IdentityInsertion::new(final_node, final_node_port);

        let noop_node = h.apply_rewrite(rw).unwrap();

        assert_eq!(h.node_count(), 7);

        let noop: Noop = h.get_optype(noop_node).cast().unwrap();

        assert_eq!(noop, Noop(qb_t()));

        h.validate().unwrap();
    }
}

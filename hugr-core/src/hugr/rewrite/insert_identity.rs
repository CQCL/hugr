//! Implementation of the `InsertIdentity` operation.

use std::iter;

use crate::extension::prelude::leaf::Noop;
use crate::hugr::{HugrMut, Node};
use crate::ops::{OpTag, OpTrait};

use crate::types::EdgeKind;
use crate::{HugrView, IncomingPort};

use super::Rewrite;

use thiserror::Error;

/// Specification of a identity-insertion operation.
#[derive(Debug, Clone)]
pub struct IdentityInsertion {
    /// The node following the identity to be inserted.
    pub post_node: Node,
    /// The port following the identity to be inserted.
    pub post_port: IncomingPort,
}

impl IdentityInsertion {
    /// Create a new [`IdentityInsertion`] specification.
    pub fn new(post_node: Node, post_port: IncomingPort) -> Self {
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
    #[error("post_port has invalid kind {0:?}. Must be Value.")]
    InvalidPortKind(Option<EdgeKind>),
}

impl Rewrite for IdentityInsertion {
    type Error = IdentityInsertionError;
    /// The inserted node.
    type ApplyResult = Node;
    const UNCHANGED_ON_FAILURE: bool = true;
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
    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, IdentityInsertionError> {
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
        let new_node = h.add_node_with_parent(parent, Noop { ty });
        h.connect(pre_node, pre_port, new_node, 0);

        h.connect(new_node, 0, self.post_node, self.post_port);
        Ok(new_node)
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        iter::once(self.post_node)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::super::simple_replace::test::dfg_hugr;
    use super::*;
    use crate::{
        extension::{prelude::QB_T, PRELUDE_REGISTRY},
        Hugr,
    };

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

        assert_eq!(noop, Noop { ty: QB_T });

        h.update_validate(&PRELUDE_REGISTRY).unwrap();
    }
}

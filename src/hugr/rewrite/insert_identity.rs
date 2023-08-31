//! Implementation of the `InsertIdentity` operation.

use crate::hugr::{HugrMut, Node};
use crate::ops::LeafOp;
use crate::types::EdgeKind;
use crate::{Hugr, HugrView, Port};

use super::Rewrite;

use itertools::Itertools;
use thiserror::Error;

/// Specification of a identity-insertion operation.
#[derive(Debug, Clone)]
pub struct IdentityInsertion {
    /// The node following the identity to be inserted.
    pub post_node: Node,
    /// The port following the identity to be inserted.
    pub post_port: Port,
}

impl IdentityInsertion {
    /// Create a new [`IdentityInsertion`] specification.
    pub fn new(post_node: Node, post_port: Port) -> Self {
        Self {
            post_node,
            post_port,
        }
    }
}

/// Error from an [`IdentityInsertion`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum IdentityInsertionError {
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
    fn verify(&self, _h: &Hugr) -> Result<(), IdentityInsertionError> {
        /*
        Assumptions:
        1. Value kind inputs can only have one connection.
        2. Node exists.
        Conditions:
        1. post_port is Value kind.
        2. post_port is connected to a sibling of post_node.
         */

        unimplemented!()
    }
    fn apply(self, h: &mut Hugr) -> Result<Self::ApplyResult, IdentityInsertionError> {
        let (pre_node, pre_port) = h
            .linked_ports(self.post_node, self.post_port)
            .exactly_one()
            .expect("Value kind input can only have one connection.");

        let kind = h.get_optype(self.post_node).port_kind(self.post_port);
        let Some(EdgeKind::Value(ty)) = kind else {
            return Err(IdentityInsertionError::InvalidPortKind(kind));
        };

        h.disconnect(self.post_node, self.post_port).unwrap();
        let new_node = h.add_op(LeafOp::Noop { ty });
        h.connect(pre_node, pre_port.index(), new_node, 0)
            .expect("Should only fail if ports don't exist.");

        // TODO Check type, insert Noop...
        h.connect(new_node, 0, self.post_node, self.post_port.index())
            .expect("Should only fail if ports don't exist.");
        Ok(new_node)
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use crate::Hugr;

    #[fixture]
    fn sample() -> Hugr {
        todo!()
    }
    #[rstest]
    fn correct_insertion() {}
}

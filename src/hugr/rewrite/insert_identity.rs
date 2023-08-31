//! Implementation of the `InsertIdentity` operation.

use crate::extension::prelude::QB_T;
use crate::hugr::{HugrMut, Node};
use crate::ops::LeafOp;
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
    /// Invalid port.
    #[error("post_port is invalid.")]
    InvalidPort(),
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
        Conditions:
        1. post_port is Value kind
        2. post_port is connected to a sibling of post_node
         */

        unimplemented!()
    }
    fn apply(self, h: &mut Hugr) -> Result<Self::ApplyResult, IdentityInsertionError> {
        let (pre_node, pre_port) = h
            .linked_ports(self.post_node, self.post_port)
            .exactly_one()
            .expect("Value kind input can only have one connection.");
        h.disconnect(self.post_node, self.post_port).unwrap();
        let new_node = h.add_op(LeafOp::Noop { ty: QB_T });
        // TODO Check type, insert Noop...
        h.connect(
            pre_node,
            pre_port.index(),
            self.post_node,
            self.post_port.index(),
        )
        .unwrap();
        Ok(new_node)
    }
}

//! Implementation of the `InsertIdentity` operation.

use crate::hugr::{HugrMut, Node};
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
    const UNCHANGED_ON_FAILURE: bool = true;
    fn verify(&self, _h: &Hugr) -> Result<(), IdentityInsertionError> {
        unimplemented!()
    }
    fn apply(self, h: &mut Hugr) -> Result<(), IdentityInsertionError> {
        let (pre_node, pre_port) = h
            .linked_ports(self.post_node, self.post_port)
            .exactly_one()
            .unwrap();
        h.disconnect(self.post_node, self.post_port).unwrap();
        // TODO Check type, insert Noop...
        h.connect(
            pre_node,
            pre_port.index(),
            self.post_node,
            self.post_port.index(),
        )
        .unwrap();
        Ok(())
    }
}

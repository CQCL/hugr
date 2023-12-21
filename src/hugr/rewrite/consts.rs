//! Rewrite operations involving Const and LoadConst operations

use std::iter;

use crate::{
    hugr::{HugrError, HugrMut},
    HugrView, Node,
};
use itertools::Itertools;
use thiserror::Error;

use super::Rewrite;

/// Remove a [`crate::ops::LoadConstant`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConstIgnore(pub Node);

/// Error from an [`RemoveConstIgnore`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RemoveConstIgnoreError {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not LoadConst).")]
    InvalidNode(Node),
    /// Node in use.
    #[error("Node: {0:?} has non-zero outgoing connections.")]
    ValueUsed(Node),
    /// Not connected to a Const.
    #[error("Node: {0:?} is not connected to a Const node.")]
    NoConst(Node),
    /// Removal error
    #[error("Removing node caused error: {0:?}.")]
    RemoveFail(#[from] HugrError),
}

impl Rewrite for RemoveConstIgnore {
    type Error = RemoveConstIgnoreError;

    // The Const node the LoadConstant was connected to.
    type ApplyResult = Node;

    type InvalidationSet<'a> = iter::Once<Node>;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_load_constant()) {
            return Err(RemoveConstIgnoreError::InvalidNode(node));
        }

        if h.out_value_types(node)
            .next()
            .is_some_and(|(p, _)| h.linked_inputs(node, p).next().is_some())
        {
            return Err(RemoveConstIgnoreError::ValueUsed(node));
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let source = h
            .input_neighbours(node)
            .exactly_one()
            .map_err(|_| RemoveConstIgnoreError::NoConst(node))?;
        h.remove_node(node)?;

        Ok(source)
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        iter::once(self.0)
    }
}

/// Remove a [`crate::ops::Const`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConst(pub Node);

/// Error from an [`RemoveConst`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RemoveConstError {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not Const).")]
    InvalidNode(Node),
    /// Node in use.
    #[error("Node: {0:?} has non-zero outgoing connections.")]
    ValueUsed(Node),
    /// Removal error
    #[error("Removing node caused error: {0:?}.")]
    RemoveFail(#[from] HugrError),
}

impl Rewrite for RemoveConst {
    type Error = RemoveConstError;

    // The parent of the Const node.
    type ApplyResult = Node;

    type InvalidationSet<'a> = iter::Once<Node>;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_const()) {
            return Err(RemoveConstError::InvalidNode(node));
        }

        if h.output_neighbours(node).next().is_some() {
            return Err(RemoveConstError::ValueUsed(node));
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let source = h
            .get_parent(node)
            .expect("Const node without a parent shouldn't happen.");
        h.remove_node(node)?;

        Ok(source)
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        iter::once(self.0)
    }
}

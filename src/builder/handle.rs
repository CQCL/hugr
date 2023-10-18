//! Handles to nodes in HUGR used during the building phase.
//!
use crate::{
    hugr::OutgoingPort,
    ops::{
        handle::{BasicBlockID, CaseID, DfgID, FuncID, NodeHandle, TailLoopID},
        OpTag,
    },
};
use crate::{Node, Wire};

use itertools::Itertools;
use std::iter::FusedIterator;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Handle to a dataflow node which has a known number of value outputs
pub struct BuildHandle<T> {
    node_handle: T,
    num_value_outputs: usize,
}

impl<T: From<Node>> From<(Node, usize)> for BuildHandle<T> {
    fn from((node, num_value_outputs): (Node, usize)) -> Self {
        Self {
            node_handle: node.into(),
            num_value_outputs,
        }
    }
}

impl<T: NodeHandle> NodeHandle for BuildHandle<T> {
    const TAG: OpTag = T::TAG;

    fn node(&self) -> Node {
        self.node_handle.node()
    }
}

impl<T: NodeHandle> BuildHandle<T> {
    #[inline]
    /// Number of Value kind outputs from this node.
    fn num_value_outputs(&self) -> usize {
        self.num_value_outputs
    }

    #[inline]
    /// Return iterator over Value outputs.
    pub fn outputs(&self) -> Outputs {
        Outputs {
            node: self.node(),
            range: (0..self.num_value_outputs()),
        }
    }

    /// Attempt to cast outputs in to array of Wires.
    pub fn outputs_arr<const N: usize>(&self) -> [Wire; N] {
        self.outputs()
            .collect_vec()
            .try_into()
            .expect(&format!("Incorrect number of wires: {}", N)[..])
    }

    #[inline]
    /// Retrieve a [`Wire`] corresponding to the given offset.
    /// Does not check whether such a wire is valid for this node.
    pub fn out_wire(&self, offset: usize) -> Wire {
        Wire::new(self.node(), OutgoingPort::from(offset))
    }

    #[inline]
    /// Underlying node handle
    pub fn handle(&self) -> &T {
        &self.node_handle
    }
}

impl From<BuildHandle<DfgID>> for BuildHandle<FuncID<true>> {
    #[inline]
    fn from(value: BuildHandle<DfgID>) -> Self {
        Self {
            node_handle: value.node().into(),
            num_value_outputs: value.num_value_outputs,
        }
    }
}

impl From<BuildHandle<DfgID>> for BasicBlockID {
    #[inline]
    fn from(value: BuildHandle<DfgID>) -> Self {
        value.node().into()
    }
}

impl From<BuildHandle<DfgID>> for BuildHandle<CaseID> {
    #[inline]
    fn from(value: BuildHandle<DfgID>) -> Self {
        Self {
            node_handle: value.node().into(),
            num_value_outputs: value.num_value_outputs,
        }
    }
}

impl From<BuildHandle<DfgID>> for BuildHandle<TailLoopID> {
    #[inline]
    fn from(value: BuildHandle<DfgID>) -> Self {
        Self {
            node_handle: value.node().into(),
            num_value_outputs: value.num_value_outputs,
        }
    }
}

#[derive(Debug, Clone)]
/// Iterator over output wires of a [`BuildHandle`].
pub struct Outputs {
    node: Node,
    range: std::ops::Range<usize>,
}

impl Iterator for Outputs {
    type Item = Wire;

    fn next(&mut self) -> Option<Self::Item> {
        self.range
            .next()
            .map(|offset| Wire::new(self.node, OutgoingPort::from(offset)))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(|offset| Wire::new(self.node, offset))
    }

    #[inline]
    fn count(self) -> usize {
        self.range.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl ExactSizeIterator for Outputs {
    #[inline]
    fn len(&self) -> usize {
        self.range.len()
    }
}

impl DoubleEndedIterator for Outputs {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range
            .next_back()
            .map(|offset| Wire::new(self.node, offset))
    }
}

impl FusedIterator for Outputs {}

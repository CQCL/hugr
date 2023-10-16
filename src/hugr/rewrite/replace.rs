//! Implementation of the `Replace` operation.

use std::collections::hash_map::Values;
use std::collections::hash_set::Iter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::{Chain, Copied, Take};

use itertools::Itertools;
use thiserror::Error;

use crate::ops::OpTrait;
use crate::types::Type;
use crate::{Hugr, HugrView, Node};

use super::Rewrite;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewEdgeSpec {
    src: Node,
    tgt: Node,
    kind: NewEdgeKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NewEdgeKind {
    Order,
    Value {
        ty: Type,
        src_pos: usize,
        tgt_pos: usize,
    },
    Static {
        ty: Type,
        src_pos: usize,
        tgt_pos: usize,
    },
    ControlFlow {
        src_pos: usize,
    },
}

/// Specification of a `Replace` operation
#[derive(Debug, Clone)] // PartialEq? Eq probably doesn't make sense because of ordering.
pub struct Replacement {
    /// The nodes to remove from the existing Hugr (known as Gamma).
    /// These must all have a common parent (i.e. be siblings).  Called "S" in the spec.
    removal: HashSet<Node>,
    /// A hugr whose root is the same type as the parent of all the [nodes]. "G" in the spec.
    replacement: Hugr,
    /// Map from container nodes in [replacement] that have no children, to container nodes
    /// that are descended from [nodes]. The keys are the new parents for the children of the
    /// values, i.e. said children are transferred to new locations rather than removed from the graph.
    /// Note no value may be ancestor/descendant of another. "R" is the set of descendants of [nodes]
    /// that are not descendants of values here.
    transfers: HashMap<Node, Node>,
    /// Edges from nodes in the existing Hugr that are not removed ([NewEdgeSpec::src] in Gamma\R)
    /// to inserted nodes ([NewEdgeSpec::tgt] in [replacement]). `$\mu_\inp$` in the spec.
    in_edges: Vec<NewEdgeSpec>,
    /// Edges from inserted nodes ([NewEdgeSpec::src] in [replacement]) to existing nodes not removed
    /// ([NewEdgeSpec::tgt] in Gamma \ R). `$\mu_\out$` in the spec.
    out_edges: Vec<NewEdgeSpec>,
    /// Edges to add between existing nodes (both [NewEdgeSpec::src] and [NewEdgeSpec::tgt] in Gamma \ R).
    /// For example, in cases where the source had an edge to a removed node, and the target had an
    /// edge from a removed node, this would allow source to be directly connected to target.
    new_edges: Vec<NewEdgeSpec>,
}

impl Rewrite for Replacement {
    type Error = ReplaceError;

    type ApplyResult = ();

    type InvalidationSet<'a> = //<Vec<Node> as IntoIterator>::IntoIter
        Chain<
            Chain<
                Copied<Iter<'a, Node>>,
                Copied<Values<'a, Node, Node>>>,
            Copied<Take<Iter<'a, Node>>>>
    where
        Self: 'a;

    const UNCHANGED_ON_FAILURE: bool = false;

    fn verify(&self, h: &impl crate::HugrView) -> Result<(), Self::Error> {
        let parent = match self
            .removal
            .iter()
            .map(|n| h.get_parent(*n))
            .unique()
            .exactly_one()
        {
            Ok(Some(p)) => Ok(p),
            Ok(None) => Err(ReplaceError(
                "Cannot replace the top node of the Hugr".to_string(),
            )), // maybe we SHOULD be able to??
            Err(e) => Err(ReplaceError(format!(
                "Nodes to remove do not share a parent: {}",
                e
            ))),
        }?;
        // Should we require exactly equality?
        let expected_tag = h.get_optype(parent).tag();
        let found_tag = self.replacement.root_type().tag();
        if expected_tag != found_tag {
            return Err(ReplaceError(format!(
                "Expected replacement to have root node w/tag {} but found {}",
                expected_tag, found_tag
            )));
        };
        let mut transferred: HashSet<Node> = self.transfers.values().copied().collect();
        if transferred.len() != self.transfers.values().len() {
            return Err(ReplaceError(
                "Repeated nodes in RHS of transfer map".to_string(),
            ));
        }
        let mut removed = HashSet::new();
        let mut queue = VecDeque::from_iter(self.removal.iter().copied());
        while let Some(n) = queue.pop_front() {
            let new = removed.insert(n);
            debug_assert!(new); // Fails only if h's hierarchy has merges (is not a tree)
            if !transferred.remove(&n) {
                h.children(n).for_each(|ch| queue.push_back(ch))
            }
        }
        if !transferred.is_empty() {
            return Err(ReplaceError(
                "Some transferred nodes were not to be removed".to_string(),
            ));
        }
        Ok(())
    }

    fn apply(self, _h: &mut impl crate::hugr::HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        todo!()
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        self.removal
            .iter()
            .copied()
            .chain(self.transfers.values().copied())
            .chain(self.removal.iter().take(1).copied())
    }
}

/// Error in a [`Replacement`]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct ReplaceError(String);

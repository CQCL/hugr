//! Implementation of the `Replace` operation.

use std::collections::hash_map::Values;
use std::collections::hash_set::Iter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::{Chain, Copied, Take};

use itertools::Itertools;
use thiserror::Error;

use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::HugrMut;
use crate::ops::{OpTag, OpTrait};
use crate::types::EdgeKind;
use crate::{Direction, Hugr, HugrView, Node, Port};

use super::Rewrite;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewEdgeSpec {
    pub src: Node,
    pub tgt: Node,
    pub kind: NewEdgeKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NewEdgeKind {
    Order,
    Value { src_pos: usize, tgt_pos: usize },
    Static { src_pos: usize, tgt_pos: usize },
    ControlFlow { src_pos: usize },
}

/// Specification of a `Replace` operation
#[derive(Debug, Clone)] // PartialEq? Eq probably doesn't make sense because of ordering.
pub struct Replacement {
    /// The nodes to remove from the existing Hugr (known as Gamma).
    /// These must all have a common parent (i.e. be siblings).  Called "S" in the spec.
    /// Must be non-empty - otherwise there is no parent under which to place [replacement],
    /// and no possible [in_edges], [out_edges] or [transfers].
    pub removal: HashSet<Node>,
    /// A hugr whose root is the same type as the parent of all the [nodes]. "G" in the spec.
    pub replacement: Hugr,
    /// Map from container nodes in [replacement] that have no children, to container nodes
    /// that are descended from [nodes]. The keys are the new parents for the children of the
    /// values, i.e. said children are transferred to new locations rather than removed from the graph.
    /// Note no value may be ancestor/descendant of another. "R" is the set of descendants of [nodes]
    /// that are not descendants of values here.
    pub transfers: HashMap<Node, Node>,
    /// Edges from nodes in the existing Hugr that are not removed ([NewEdgeSpec::src] in Gamma\R)
    /// to inserted nodes ([NewEdgeSpec::tgt] in [replacement]). `$\mu_\inp$` in the spec.
    pub mu_inp: Vec<NewEdgeSpec>,
    /// Edges from inserted nodes ([NewEdgeSpec::src] in [replacement]) to existing nodes not removed
    /// ([NewEdgeSpec::tgt] in Gamma \ R). `$\mu_\out$` in the spec.
    pub mu_out: Vec<NewEdgeSpec>,
    /// Edges to add between existing nodes (both [NewEdgeSpec::src] and [NewEdgeSpec::tgt] in Gamma \ R).
    /// For example, in cases where the source had an edge to a removed node, and the target had an
    /// edge from a removed node, this would allow source to be directly connected to target.
    pub mu_new: Vec<NewEdgeSpec>,
}

impl Replacement {
    fn check_parent(&self, h: &impl HugrView) -> Result<Node, ReplaceError> {
        let parent = self
            .removal
            .iter()
            .map(|n| h.get_parent(*n))
            .unique()
            .exactly_one()
            .map_err(|ex_one| ReplaceError::MultipleParents(ex_one.flatten().collect()))?
            .ok_or(ReplaceError::CantReplaceRoot)?; // If no parent

        // Check replacement parent is of same tag. Should we require exactly OpType equality?
        let expected = h.get_optype(parent).tag();
        let actual = self.replacement.root_type().tag();
        if expected != actual {
            return Err(ReplaceError::WrongRootNodeTag { expected, actual });
        };
        Ok(parent)
    }
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
        let parent = self.check_parent(h)?;
        let mut transferred: HashSet<Node> = self.transfers.values().copied().collect();
        if transferred.len() != self.transfers.values().len() {
            return Err(ReplaceError::ConflictingTransfers(
                self.transfers
                    .values()
                    .filter(|v| !transferred.remove(v))
                    .copied()
                    .collect(),
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
            return Err(ReplaceError::TransfersNotSeparateDescendants(
                transferred.into_iter().collect(),
            ));
        }
        // Edge sources...
        for e in self.mu_inp.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.src) || removed.contains(&e.src) {
                return Err(ReplaceError::BadEdgeSpec(
                    "Edge source",
                    WhichHugr::Retained,
                    e.src,
                ));
            }
        }
        self.mu_out.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.src).map_err(|_| {
                ReplaceError::BadEdgeSpec("Out-edge source", WhichHugr::Replacement, e.src)
            })
        })?;
        // Edge targets...
        self.mu_inp.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.tgt).map_err(|_| {
                ReplaceError::BadEdgeSpec("In-edge target", WhichHugr::Replacement, e.tgt)
            })
        })?;
        for e in self.mu_out.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.tgt) || removed.contains(&e.tgt) {
                return Err(ReplaceError::BadEdgeSpec(
                    "Edge target",
                    WhichHugr::Retained,
                    e.tgt,
                ));
            }
            if let NewEdgeKind::Static { tgt_pos, .. } | NewEdgeKind::Value { tgt_pos, .. } = e.kind
            {
                fn descends(h: &impl HugrView, ancestor: Node, mut descendant: Node) -> bool {
                    while descendant != ancestor {
                        let Some(p) = h.get_parent(descendant) else {return false};
                        descendant = p;
                    }
                    true
                }
                let found_incoming = h
                    .linked_ports(e.tgt, Port::new(Direction::Incoming, tgt_pos))
                    .exactly_one()
                    .is_ok_and(|(src_n, _)| {
                        // The descendant check is to allow the case where the old edge is nonlocal
                        // from a part of the Hugr being moved (which may require changing source,
                        // depending on where the transplanted portion ends up). While this subsumes
                        // the first "removed.contains" check, we'll keep that as a common-case fast-path.
                        removed.contains(&src_n) || descends(h, parent, src_n)
                    });
                if !found_incoming {
                    return Err(ReplaceError::NoRemovedEdge(e.clone()));
                };
            }
        }
        // TODO check ports and/or node types appropriate for kind of edge added
        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<(), Self::Error> {
        let parent = self.check_parent(h)?;
        // 1. Add all the new nodes. Note this includes replacement.root(), which we don't want.
        // TODO what would an error here mean? e.g. malformed self.replacement??
        let InsertionResult { new_root, node_map } =
            h.insert_hugr(parent, self.replacement).unwrap();

        // 2. Add new edges from existing to copied nodes according to mu_in
        let translate_idx = |n| *node_map.get(&n).unwrap();
        let id = |n| n;
        transfer_edges(h, self.mu_inp.iter(), id, translate_idx, false)?;

        // 3. Add new edges from copied to existing nodes according to mu_out,
        // replacing existing value/static edges incoming to targets
        transfer_edges(h, self.mu_out.iter(), translate_idx, id, true)?;

        //4. Add new edges between existing nodes according to mu_new,
        // replacing existing value/static edges incoming to targets
        transfer_edges(h, self.mu_new.iter(), id, id, true)?;

        // 5. Put newly-added copies into correct places in hierarchy
        // (these will be correct places after removing nodes)
        let mut remove = self.removal.iter();
        for new_node in h.children(new_root).collect::<Vec<Node>>().into_iter() {
            if let Some(to_remove) = remove.next() {
                h.move_before_sibling(new_node, *to_remove).unwrap();
            } else {
                h.set_parent(new_node, parent).unwrap();
            }
        }
        debug_assert!(h.children(new_root).next().is_none());
        h.remove_node(new_root).unwrap();

        // 6. Transfer to keys of `transfers` children of the corresponding values.
        fn first_child(h: &impl HugrView, parent: Node) -> Option<crate::Node> {
            h.children(parent).next()
        }
        for (new_parent, old_parent) in self.transfers.iter() {
            debug_assert!(h.children(*old_parent).next().is_some());
            while let Some(ch) = first_child(h, *old_parent) {
                h.set_parent(ch, *new_parent).unwrap();
            }
        }

        // 7. Remove remaining nodes
        let mut to_remove = VecDeque::from_iter(self.removal);
        while let Some(n) = to_remove.pop_front() {
            // Nodes may have no children if they were moved in step 6
            to_remove.extend(h.children(n));
            h.remove_node(n).unwrap();
        }
        Ok(())
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        self.removal
            .iter()
            .copied()
            .chain(self.transfers.values().copied())
            .chain(self.removal.iter().take(1).copied())
    }
}

fn transfer_edges<'a>(
    h: &mut impl HugrMut,
    edges: impl Iterator<Item = &'a NewEdgeSpec>,
    trans_src: impl Fn(Node) -> Node,
    trans_tgt: impl Fn(Node) -> Node,
    remove_existing: bool,
) -> Result<(), ReplaceError> {
    for e in edges {
        let src = trans_src(e.src);
        let tgt = trans_tgt(e.tgt);
        match e.kind {
            NewEdgeKind::Order => {
                if h.get_optype(src).other_output() != Some(EdgeKind::StateOrder) {
                    return Err(ReplaceError::BadEdgeKind(e.clone()));
                }
                h.add_other_edge(src, tgt).unwrap();
            }
            NewEdgeKind::Value { src_pos, tgt_pos } | NewEdgeKind::Static { src_pos, tgt_pos } => {
                if remove_existing {
                    h.disconnect(tgt, Port::new(Direction::Incoming, tgt_pos))
                        .unwrap();
                }
                h.connect(src, src_pos, tgt, tgt_pos).unwrap();
            }
            NewEdgeKind::ControlFlow { src_pos } => h.connect(src, src_pos, tgt, 0).unwrap(),
        }
    }
    Ok(())
}

/// Error in a [`Replacement`]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReplaceError {
    /// The node(s) to replace had no parent i.e. were root(s).
    // (Perhaps if there is only one node to replace we should be able to?)
    #[error("Cannot replace the root node of the Hugr")]
    CantReplaceRoot,
    /// The nodes to replace did not have a unique common parent
    #[error("Removed nodes had different parents {0:?}")]
    MultipleParents(Vec<Node>),
    /// Replacement root node had different tag from parent of removed nodes
    #[error("Expected replacement root with tag {expected} but found {actual}")]
    #[allow(missing_docs)]
    WrongRootNodeTag { expected: OpTag, actual: OpTag },
    /// Values in transfer map were not unique - contains the repeated elements
    #[error("Nodes cannot be transferred to multiple locations: {0:?}")]
    ConflictingTransfers(Vec<Node>),
    /// Some values in the transfer map were either descendants of other values,
    /// or not descendants of the removed nodes
    #[error("Nodes not free to be moved into new locations: {0:?}")]
    TransfersNotSeparateDescendants(Vec<Node>),
    /// A node at one end of a [NewEdgeSpec] was not found
    #[error("{0} not in {1}: {2:?}")]
    BadEdgeSpec(&'static str, WhichHugr, Node),
    /// The target of the edge was found, but there was no existing edge to replace
    #[error("Target of edge {0:?} did not have a corresponding incoming edge being removed")]
    NoRemovedEdge(NewEdgeSpec),
    /// The {NewEdgeKind} was not applicable for the source/target node(s)
    #[error("The edge kind was not applicable to the node(s)")]
    BadEdgeKind(NewEdgeSpec),
}

/// A Hugr or portion thereof that is part of the [Replacement]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhichHugr {
    /// The newly-inserted nodes, i.e. the [Replacement::replacement]
    Replacement,
    /// Nodes in the existing Hugr that are not [Replacement::removal]
    Retained,
}

impl std::fmt::Display for WhichHugr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Replacement => "replacement Hugr",
            Self::Retained => "retained portion of Hugr",
        })
    }
}

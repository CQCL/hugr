//! Implementation of the `Replace` operation.

use std::collections::hash_map::Values;
use std::collections::hash_set::Iter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::iter::{Chain, Copied, Take};

use itertools::Itertools;
use thiserror::Error;

use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::{HugrError, HugrMut};
use crate::ops::OpTrait;
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
            Ok(None) => Err(ReplaceError::Msg(
                "Cannot replace the top node of the Hugr".to_string(),
            )), // maybe we SHOULD be able to??
            Err(e) => Err(ReplaceError::Msg(format!(
                "Nodes to remove do not share a parent: {}",
                e
            ))),
        }?;
        // Should we require exactly equality?
        let expected_tag = h.get_optype(parent).tag();
        let found_tag = self.replacement.root_type().tag();
        if expected_tag != found_tag {
            return Err(ReplaceError::Msg(format!(
                "Expected replacement to have root node w/tag {} but found {}",
                expected_tag, found_tag
            )));
        };
        let mut transferred: HashSet<Node> = self.transfers.values().copied().collect();
        if transferred.len() != self.transfers.values().len() {
            return Err(ReplaceError::Msg(
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
            // This also occurs if some RHS was reachable from another
            return Err(ReplaceError::Msg(
                "Some transferred nodes were not to be removed".to_string(),
            ));
        }
        // Edge sources...
        for e in self.mu_inp.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.src) || removed.contains(&e.src) {
                return Err(ReplaceError::Msg(format!(
                    "Edge source not in retained nodes: {:?}",
                    e.src
                )));
            }
        }
        self.mu_out.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.src).map_err(|_| {
                ReplaceError::Msg(format!(
                    "Out-edge source not in replacement Hugr: {:?}",
                    e.src
                ))
            })
        })?;
        // Edge targets...
        self.mu_inp.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.tgt).map_err(|_| {
                ReplaceError::Msg(format!(
                    "In-edge target not in replacement Hugr: {:?}",
                    e.tgt
                ))
            })
        })?;
        for e in self.mu_out.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.tgt) || removed.contains(&e.tgt) {
                return Err(ReplaceError::Msg(format!(
                    "Edge target not in retained nodes: {:?}",
                    e.tgt
                )));
            }
            fn strict_desc(h: &impl HugrView, ancestor: Node, mut descendant: Node) -> bool {
                while let Some(p) = h.get_parent(descendant) {
                    if ancestor == p {
                        return true;
                    };
                    descendant = p;
                }
                return false;
            }
            match e.kind {
                NewEdgeKind::Static { tgt_pos, .. } | NewEdgeKind::Value { tgt_pos, .. } => match h
                    .linked_ports(e.tgt, Port::new(Direction::Incoming, tgt_pos))
                    .exactly_one()
                {
                    // The descendant check is to allow the case where the old edge is nonlocal
                    // from a part of the Hugr being moved (which may require changing source,
                    // depending on where the transplanted portion ends up). While this subsumes
                    // the first "removed.contains" check, we'll keep that as a common-case fast-path.
                    Ok((src_n, _)) if removed.contains(&src_n) || strict_desc(h, parent, src_n) => {
                        ()
                    }
                    _ => {
                        return Err(ReplaceError::Msg(format!(
                            "Target of Edge {:?} did not have incoming edge being removed",
                            e
                        )))
                    }
                },
                _ => (),
            };
        }
        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<(), Self::Error> {
        let parent = h.get_parent(*self.removal.iter().next().unwrap()).unwrap();
        // 1. Add all the new nodes. Note this includes replacement.root(), which we don't want
        let InsertionResult { new_root, node_map } = h.insert_hugr(parent, self.replacement)?;

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
                    return Err(ReplaceError::Msg(
                        "Can't insert Order edge except from DFG node".to_string(),
                    ));
                }
                if h.get_parent(tgt) != h.get_parent(src) {
                    return Err(ReplaceError::Msg(
                        "Order edge target not at top level".to_string(),
                    ));
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
    #[error("{0}")]
    Msg(String),
    #[error(transparent)]
    Hugr(#[from] HugrError),
}

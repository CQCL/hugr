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

/// Specifies how to create a new edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewEdgeSpec {
    /// The source of the new edge. For [Replacement::mu_inp] and [Replacement::mu_new], this is in the
    /// existing Hugr; for edges in [Replacement::mu_out] this is in the [Replacement::replacement]
    pub src: Node,
    /// The target of the new edge. For [Replacement::mu_inp], this is in the [Replacement::replacement];
    /// for edges in [Replacement::mu_out] and [Replacement::mu_new], this is in the existing Hugr.
    pub tgt: Node,
    /// The kind of edge to create, and any port specifiers required
    pub kind: NewEdgeKind,
}

/// Describes an edge that should be created between two nodes already given
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NewEdgeKind {
    /// An [EdgeKind::StateOrder] edge (between DFG nodes only)
    Order,
    /// An [EdgeKind::Value] edge (between DFG nodes only)
    Value {
        /// The source port
        src_pos: usize,
        /// The target port
        tgt_pos: usize,
    },
    /// An [EdgeKind::Static] edge
    Static {
        /// The source port
        src_pos: usize,
        /// The target port
        tgt_pos: usize,
    },
    /// A [EdgeKind::ControlFlow] edge (between CFG nodes only)
    ControlFlow {
        /// Identifies a control-flow output (successor) of the source node.
        src_pos: usize,
    },
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

impl NewEdgeSpec {
    fn check_src(&self, h: &impl HugrView) -> Result<(), ReplaceError> {
        let optype = h.get_optype(self.src);
        let ok = match self.kind {
            NewEdgeKind::Order => optype.other_output() == Some(EdgeKind::StateOrder),
            NewEdgeKind::Value { src_pos, .. } => matches!(
                optype.port_kind(Port::new_outgoing(src_pos)),
                Some(EdgeKind::Value(_))
            ),
            NewEdgeKind::Static { src_pos, .. } => matches!(
                optype.port_kind(Port::new_outgoing(src_pos)),
                Some(EdgeKind::Static(_))
            ),
            NewEdgeKind::ControlFlow { src_pos } => matches!(
                optype.port_kind(Port::new_outgoing(src_pos)),
                Some(EdgeKind::ControlFlow)
            ),
        };
        ok.then_some(())
            .ok_or(ReplaceError::BadEdgeKind(Direction::Outgoing, self.clone()))
    }
    fn check_tgt(&self, h: &impl HugrView) -> Result<(), ReplaceError> {
        let optype = h.get_optype(self.tgt);
        let ok = match self.kind {
            NewEdgeKind::Order => optype.other_input() == Some(EdgeKind::StateOrder),
            NewEdgeKind::Value { tgt_pos, .. } => matches!(
                optype.port_kind(Port::new_incoming(tgt_pos)),
                Some(EdgeKind::Value(_))
            ),
            NewEdgeKind::Static { tgt_pos, .. } => matches!(
                optype.port_kind(Port::new_incoming(tgt_pos)),
                Some(EdgeKind::Static(_))
            ),
            NewEdgeKind::ControlFlow { .. } => matches!(
                optype.port_kind(Port::new_incoming(0)),
                Some(EdgeKind::ControlFlow)
            ),
        };
        ok.then_some(())
            .ok_or(ReplaceError::BadEdgeKind(Direction::Incoming, self.clone()))
    }

    fn check_existing_edge(
        &self,
        h: &impl HugrView,
        src_ok: impl Fn(Node) -> bool,
    ) -> Result<(), ReplaceError> {
        if let NewEdgeKind::Static { tgt_pos, .. } | NewEdgeKind::Value { tgt_pos, .. } = self.kind
        {
            let found_incoming = h
                .linked_ports(self.tgt, Port::new_incoming(tgt_pos))
                .exactly_one()
                .is_ok_and(|(src_n, _)| src_ok(src_n));
            if !found_incoming {
                return Err(ReplaceError::NoRemovedEdge(self.clone()));
            };
        };
        Ok(())
    }
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

    fn get_removed_nodes(&self, h: &impl HugrView) -> Result<HashSet<Node>, ReplaceError> {
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
        Ok(removed)
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
        let removed = self.get_removed_nodes(h)?;
        // Edge sources...
        for e in self.mu_inp.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.src) || removed.contains(&e.src) {
                return Err(ReplaceError::BadEdgeSpec(
                    Direction::Outgoing,
                    WhichHugr::Retained,
                    e.clone(),
                ));
            }
            e.check_src(h)?;
        }
        self.mu_out.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.src).map_err(|_| {
                ReplaceError::BadEdgeSpec(Direction::Outgoing, WhichHugr::Replacement, e.clone())
            })?;
            e.check_src(&self.replacement)
        })?;
        // Edge targets...
        self.mu_inp.iter().try_for_each(|e| {
            self.replacement.valid_non_root(e.tgt).map_err(|_| {
                ReplaceError::BadEdgeSpec(Direction::Incoming, WhichHugr::Replacement, e.clone())
            })?;
            e.check_tgt(&self.replacement)
        })?;
        for e in self.mu_out.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.tgt) || removed.contains(&e.tgt) {
                return Err(ReplaceError::BadEdgeSpec(
                    Direction::Incoming,
                    WhichHugr::Retained,
                    e.clone(),
                ));
            }
            e.check_tgt(h)?;
            // The descendant check is to allow the case where the old edge is nonlocal
            // from a part of the Hugr being moved (which may require changing source,
            // depending on where the transplanted portion ends up). While this subsumes
            // the first "removed.contains" check, we'll keep that as a common-case fast-path.
            e.check_existing_edge(h, |n| removed.contains(&n) || descends(h, parent, n))?;
        }
        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<(), Self::Error> {
        let parent = self.check_parent(h)?;
        // Calculate removed nodes here. (Does not include transfers, so enumerates only
        // nodes we are going to remove, individually, anyway; so no *asymptotic* speed penalty)
        let to_remove = self.get_removed_nodes(h)?;

        // 1. Add all the new nodes. Note this includes replacement.root(), which we don't want.
        // TODO what would an error here mean? e.g. malformed self.replacement??
        let InsertionResult { new_root, node_map } =
            h.insert_hugr(parent, self.replacement).unwrap();

        // 2. Add new edges from existing to copied nodes according to mu_in
        let translate_idx = |n| node_map.get(&n).copied().ok_or(WhichHugr::Replacement);
        let kept = |n| {
            let keep = !to_remove.contains(&n);
            keep.then_some(n).ok_or(WhichHugr::Retained)
        };
        transfer_edges(h, self.mu_inp.iter(), kept, translate_idx, None)?;

        // 3. Add new edges from copied to existing nodes according to mu_out,
        // replacing existing value/static edges incoming to targets
        transfer_edges(h, self.mu_out.iter(), translate_idx, kept, Some(parent))?;

        // 4. Add new edges between existing nodes according to mu_new,
        // replacing existing value/static edges incoming to targets.
        transfer_edges(h, self.mu_new.iter(), kept, kept, Some(parent))?;

        // 5. Put newly-added copies into correct places in hierarchy
        // (these will be correct places after removing nodes)
        let mut remove_top_sibs = self.removal.iter();
        for new_node in h.children(new_root).collect::<Vec<Node>>().into_iter() {
            if let Some(top_sib) = remove_top_sibs.next() {
                h.move_before_sibling(new_node, *top_sib).unwrap();
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
        to_remove
            .into_iter()
            .for_each(|n| h.remove_node(n).unwrap());
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

fn descends(h: &impl HugrView, ancestor: Node, mut descendant: Node) -> bool {
    while descendant != ancestor {
        let Some(p) = h.get_parent(descendant) else {return false};
        descendant = p;
    }
    true
}

fn transfer_edges<'a>(
    h: &mut impl HugrMut,
    edges: impl Iterator<Item = &'a NewEdgeSpec>,
    trans_src: impl Fn(Node) -> Result<Node, WhichHugr>,
    trans_tgt: impl Fn(Node) -> Result<Node, WhichHugr>,
    existing_src_ancestor: Option<Node>,
) -> Result<(), ReplaceError> {
    for oe in edges {
        let e = NewEdgeSpec {
            // Translation can only fail for Nodes that are supposed to be in the replacement
            src: trans_src(oe.src)
                .map_err(|h| ReplaceError::BadEdgeSpec(Direction::Outgoing, h, oe.clone()))?,
            tgt: trans_tgt(oe.tgt)
                .map_err(|h| ReplaceError::BadEdgeSpec(Direction::Incoming, h, oe.clone()))?,
            ..oe.clone()
        };
        h.valid_node(e.src).map_err(|_| {
            ReplaceError::BadEdgeSpec(Direction::Outgoing, WhichHugr::Retained, oe.clone())
        })?;
        h.valid_node(e.tgt).map_err(|_| {
            ReplaceError::BadEdgeSpec(Direction::Incoming, WhichHugr::Retained, oe.clone())
        })?;
        e.check_src(h)?;
        e.check_tgt(h)?;
        match e.kind {
            NewEdgeKind::Order => {
                h.add_other_edge(e.src, e.tgt).unwrap();
            }
            NewEdgeKind::Value { src_pos, tgt_pos } | NewEdgeKind::Static { src_pos, tgt_pos } => {
                if let Some(anc) = existing_src_ancestor {
                    e.check_existing_edge(h, |n| descends(h, anc, n))?;
                    h.disconnect(e.tgt, Port::new_incoming(tgt_pos)).unwrap();
                }
                h.connect(e.src, src_pos, e.tgt, tgt_pos).unwrap();
            }
            NewEdgeKind::ControlFlow { src_pos } => h.connect(e.src, src_pos, e.tgt, 0).unwrap(),
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
    #[error("{0:?} end of edge {2:?} not found in {1}")]
    BadEdgeSpec(Direction, WhichHugr, NewEdgeSpec),
    /// The target of the edge was found, but there was no existing edge to replace
    #[error("Target of edge {0:?} did not have a corresponding incoming edge being removed")]
    NoRemovedEdge(NewEdgeSpec),
    /// The [NewEdgeKind] was not applicable for the source/target node(s)
    #[error("The edge kind was not applicable to the {0:?} node: {1:?}")]
    BadEdgeKind(Direction, NewEdgeSpec),
}

/// A Hugr or portion thereof that is part of the [Replacement]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhichHugr {
    /// The newly-inserted nodes, i.e. the [Replacement::replacement]
    Replacement,
    /// Nodes in the existing Hugr that are not [Replacement::removal]
    /// (or are on the RHS of an entry in [Replacement::transfers])
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

#[cfg(test)]
mod test {
    use std::collections::{HashMap, HashSet};

    use crate::extension::prelude::USIZE_T;
    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE};
    use crate::hugr::hugrmut::sealed::HugrMutInternals;
    use crate::hugr::{HugrError, HugrMut, NodeType};
    use crate::ops::handle::NodeHandle;
    use crate::ops::{self, BasicBlock, LeafOp, OpTrait, OpType, DFG};
    use crate::std_extensions::collections;
    use crate::types::{FunctionType, Type, TypeArg, TypeRow};
    use crate::{type_row, Hugr, HugrView, Node};

    use super::{NewEdgeKind, NewEdgeSpec, Replacement};

    #[test]
    fn cfg() -> Result<(), Box<dyn std::error::Error>> {
        let reg: ExtensionRegistry = [PRELUDE.to_owned(), collections::EXTENSION.to_owned()].into();
        let listy = Type::new_extension(
            collections::EXTENSION
                .get_type(collections::LIST_TYPENAME.as_str())
                .unwrap()
                .instantiate_concrete([TypeArg::Type { ty: USIZE_T }])
                .unwrap(),
        );
        let pop: LeafOp = collections::EXTENSION
            .instantiate_extension_op("pop", [TypeArg::Type { ty: USIZE_T }], &reg)
            .unwrap()
            .into();
        let push: LeafOp = collections::EXTENSION
            .instantiate_extension_op("push", [TypeArg::Type { ty: USIZE_T }], &reg)
            .unwrap()
            .into();
        let just_list = TypeRow::from(vec![listy.clone()]);
        let exset = ExtensionSet::singleton(&collections::EXTENSION_NAME);
        let intermed = TypeRow::from(vec![listy.clone(), USIZE_T]);

        let mut h = Hugr::new(NodeType::open_extensions(ops::CFG {
            signature: FunctionType::new_linear(just_list.clone()).with_extension_delta(&exset),
        }));
        let pred_const = h.add_op_with_parent(h.root(), ops::Const::simple_unary_predicate())?;

        let entry = single_node_block(&mut h, pop, pred_const)?;
        let bb2 = single_node_block(&mut h, push, pred_const)?;

        let exit = h.add_node_with_parent(
            h.root(),
            NodeType::open_extensions(BasicBlock::Exit {
                cfg_outputs: just_list.clone(),
            }),
        )?;
        h.move_before_sibling(entry, pred_const)?;
        h.move_before_sibling(exit, pred_const)?;

        h.connect(entry, 0, bb2, 0)?;
        h.connect(bb2, 0, exit, 0)?;

        h.update_validate(&reg)?;
        // Replacement: one BB with two DFGs inside.
        // Use Hugr rather than Builder because DFGs must be empty (not even Input/Output).
        let mut replacement = Hugr::new(NodeType::open_extensions(BasicBlock::DFB {
            inputs: vec![listy.clone()].into(),
            predicate_variants: vec![type_row![]],
            other_outputs: vec![listy.clone()].into(),
            extension_delta: ExtensionSet::singleton(&collections::EXTENSION_NAME),
        }));
        let rroot = replacement.root();
        let inp = replacement.add_op_with_parent(
            rroot,
            ops::Input {
                types: vec![listy.clone()].into(),
            },
        )?;
        let df1 = replacement.add_op_with_parent(
            rroot,
            DFG {
                signature: FunctionType::new(vec![listy.clone()], intermed.clone()),
            },
        )?;
        replacement.connect(inp, 0, df1, 0)?;

        let df2 = replacement.add_op_with_parent(
            rroot,
            DFG {
                signature: FunctionType::new(intermed, vec![listy.clone()]),
            },
        )?;
        [0, 1]
            .iter()
            .try_for_each(|p| replacement.connect(df1, *p, df2, *p))?;

        let ex = replacement.add_op_with_parent(
            rroot,
            ops::Output {
                types: vec![listy.clone()].into(),
            },
        )?;
        replacement.connect(df2, 0, ex, 0)?;

        h.apply_rewrite(Replacement {
            removal: HashSet::from([entry.node(), bb2.node()]),
            replacement,
            transfers: HashMap::from([(df1.node(), entry.node()), (df2.node(), bb2.node())]),
            mu_inp: vec![],
            mu_out: vec![NewEdgeSpec {
                src: rroot,
                tgt: exit.node(),
                kind: NewEdgeKind::ControlFlow { src_pos: 0 },
            }],
            mu_new: vec![],
        })?;
        h.validate(&reg)?;
        Ok(())
    }

    fn single_node_block(
        hugr: &mut Hugr,
        op: impl Into<OpType>,
        pred_const: Node,
    ) -> Result<Node, HugrError> {
        let op: OpType = op.into();
        let op_sig = op.signature();

        let bb = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(BasicBlock::DFB {
                inputs: op_sig.input.clone(),
                other_outputs: op_sig.output.clone(),
                predicate_variants: vec![type_row![]],
                extension_delta: op_sig.extension_reqs.clone(),
            }),
        )?;
        let input = hugr.add_node_with_parent(
            bb,
            NodeType::open_extensions(ops::Input {
                types: op_sig.input.clone(),
            }),
        )?;
        let output = hugr.add_node_with_parent(
            bb,
            NodeType::open_extensions(ops::Output {
                types: op_sig.output.clone(),
            }),
        )?;
        let op = hugr.add_node_with_parent(bb, NodeType::open_extensions(op))?;

        for (p, _) in op_sig.input().iter().enumerate() {
            hugr.connect(input, p, op, p)?;
        }
        let pred = hugr.add_node_with_parent(
            bb,
            NodeType::open_extensions(ops::LoadConstant {
                datatype: Type::new_simple_predicate(1),
            }),
        )?;

        hugr.connect(pred_const, 0, pred, 0)?;
        hugr.connect(pred, 0, output, 0)?;
        for (p, _) in op_sig.output().iter().enumerate() {
            hugr.connect(op, p, output, p + 1)?;
        }
        Ok(bb)
    }
}

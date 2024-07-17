//! Implementation of the `Replace` operation.

use std::collections::{HashMap, HashSet, VecDeque};

use itertools::Itertools;
use thiserror::Error;

use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::HugrMut;
use crate::ops::{OpTag, OpTrait};
use crate::types::EdgeKind;
use crate::{Direction, Hugr, HugrView, IncomingPort, Node, OutgoingPort};

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
        src_pos: OutgoingPort,
        /// The target port
        tgt_pos: IncomingPort,
    },
    /// An [EdgeKind::Const] or [EdgeKind::Function] edge
    Static {
        /// The source port
        src_pos: OutgoingPort,
        /// The target port
        tgt_pos: IncomingPort,
    },
    /// A [EdgeKind::ControlFlow] edge (between CFG nodes only)
    ControlFlow {
        /// Identifies a control-flow output (successor) of the source node.
        src_pos: OutgoingPort,
    },
}

/// Specification of a `Replace` operation
#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    /// The nodes to remove from the existing Hugr (known as Gamma).
    /// These must all have a common parent (i.e. be siblings).  Called "S" in the spec.
    /// Must be non-empty - otherwise there is no parent under which to place [Self::replacement],
    /// and there would be no possible [Self::mu_inp], [Self::mu_out] or [Self::adoptions].
    pub removal: Vec<Node>,
    /// A hugr (not necessarily valid, as it may be missing edges and/or nodes), whose root
    /// is the same type as the root of [Self::replacement].  "G" in the spec.
    pub replacement: Hugr,
    /// Describes how parts of the Hugr that would otherwise be removed should instead be preserved but
    /// with new parents amongst the newly-inserted nodes.  This is a Map from container nodes in
    /// [Self::replacement] that have no children, to container nodes that are descended from [Self::removal].
    /// The keys are the new parents for the children of the values.  Note no value may be ancestor or
    /// descendant of another.  This is "B" in the spec; "R" is the set of descendants of [Self::removal]
    ///  that are not descendants of values here.
    pub adoptions: HashMap<Node, Node>,
    /// Edges from nodes in the existing Hugr that are not removed ([NewEdgeSpec::src] in Gamma\R)
    /// to inserted nodes ([NewEdgeSpec::tgt] in [Self::replacement]).
    pub mu_inp: Vec<NewEdgeSpec>,
    /// Edges from inserted nodes ([NewEdgeSpec::src] in [Self::replacement]) to existing nodes not removed
    /// ([NewEdgeSpec::tgt] in Gamma \ R).
    pub mu_out: Vec<NewEdgeSpec>,
    /// Edges to add between existing nodes (both [NewEdgeSpec::src] and [NewEdgeSpec::tgt] in Gamma \ R).
    /// For example, in cases where the source had an edge to a removed node, and the target had an
    /// edge from a removed node, this would allow source to be directly connected to target.
    pub mu_new: Vec<NewEdgeSpec>,
}

impl NewEdgeSpec {
    fn check_src(&self, h: &impl HugrView, err_spec: &NewEdgeSpec) -> Result<(), ReplaceError> {
        let optype = h.get_optype(self.src);
        let ok = match self.kind {
            NewEdgeKind::Order => optype.other_output() == Some(EdgeKind::StateOrder),
            NewEdgeKind::Value { src_pos, .. } => {
                matches!(optype.port_kind(src_pos), Some(EdgeKind::Value(_)))
            }
            NewEdgeKind::Static { src_pos, .. } => optype
                .port_kind(src_pos)
                .as_ref()
                .is_some_and(EdgeKind::is_static),
            NewEdgeKind::ControlFlow { src_pos } => {
                matches!(optype.port_kind(src_pos), Some(EdgeKind::ControlFlow))
            }
        };
        ok.then_some(())
            .ok_or_else(|| ReplaceError::BadEdgeKind(Direction::Outgoing, err_spec.clone()))
    }
    fn check_tgt(&self, h: &impl HugrView, err_spec: &NewEdgeSpec) -> Result<(), ReplaceError> {
        let optype = h.get_optype(self.tgt);
        let ok = match self.kind {
            NewEdgeKind::Order => optype.other_input() == Some(EdgeKind::StateOrder),
            NewEdgeKind::Value { tgt_pos, .. } => {
                matches!(optype.port_kind(tgt_pos), Some(EdgeKind::Value(_)))
            }
            NewEdgeKind::Static { tgt_pos, .. } => optype
                .port_kind(tgt_pos)
                .as_ref()
                .is_some_and(EdgeKind::is_static),
            NewEdgeKind::ControlFlow { .. } => matches!(
                optype.port_kind(IncomingPort::from(0)),
                Some(EdgeKind::ControlFlow)
            ),
        };
        ok.then_some(())
            .ok_or_else(|| ReplaceError::BadEdgeKind(Direction::Incoming, err_spec.clone()))
    }

    fn check_existing_edge(
        &self,
        h: &impl HugrView,
        legal_src_ancestors: &HashSet<Node>,
        err_edge: impl Fn() -> NewEdgeSpec,
    ) -> Result<(), ReplaceError> {
        if let NewEdgeKind::Static { tgt_pos, .. } | NewEdgeKind::Value { tgt_pos, .. } = self.kind
        {
            let descends_from_legal = |mut descendant: Node| -> bool {
                while !legal_src_ancestors.contains(&descendant) {
                    let Some(p) = h.get_parent(descendant) else {
                        return false;
                    };
                    descendant = p;
                }
                true
            };
            let found_incoming = h
                .single_linked_output(self.tgt, tgt_pos)
                .is_some_and(|(src_n, _)| descends_from_legal(src_n));
            if !found_incoming {
                return Err(ReplaceError::NoRemovedEdge(err_edge()));
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

        // Check replacement parent is of same tag. Note we do not require exact equality
        // of OpType/Signature, e.g. to ease changing of Input/Output node signatures too.
        let removed = h.get_optype(parent).tag();
        let replacement = self.replacement.root_type().tag();
        if removed != replacement {
            return Err(ReplaceError::WrongRootNodeTag {
                removed,
                replacement,
            });
        };
        Ok(parent)
    }

    fn get_removed_nodes(&self, h: &impl HugrView) -> Result<HashSet<Node>, ReplaceError> {
        // Check the keys of the transfer map too, the values we'll use imminently
        self.adoptions.keys().try_for_each(|&n| {
            (self.replacement.contains_node(n)
                && self.replacement.get_optype(n).is_container()
                && self.replacement.children(n).next().is_none())
            .then_some(())
            .ok_or(ReplaceError::InvalidAdoptingParent(n))
        })?;
        let mut transferred: HashSet<Node> = self.adoptions.values().copied().collect();
        if transferred.len() != self.adoptions.values().len() {
            return Err(ReplaceError::AdopteesNotSeparateDescendants(
                self.adoptions
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
            return Err(ReplaceError::AdopteesNotSeparateDescendants(
                transferred.into_iter().collect(),
            ));
        }
        Ok(removed)
    }
}
impl Rewrite for Replacement {
    type Error = ReplaceError;

    /// Map from Node in replacement to corresponding Node in the result Hugr
    type ApplyResult = HashMap<Node, Node>;

    const UNCHANGED_ON_FAILURE: bool = false;

    fn verify(&self, h: &impl crate::HugrView) -> Result<(), Self::Error> {
        self.check_parent(h)?;
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
            e.check_src(h, e)?;
        }
        self.mu_out
            .iter()
            .try_for_each(|e| match self.replacement.valid_non_root(e.src) {
                true => e.check_src(&self.replacement, e),
                false => Err(ReplaceError::BadEdgeSpec(
                    Direction::Outgoing,
                    WhichHugr::Replacement,
                    e.clone(),
                )),
            })?;
        // Edge targets...
        self.mu_inp
            .iter()
            .try_for_each(|e| match self.replacement.valid_non_root(e.tgt) {
                true => e.check_tgt(&self.replacement, e),
                false => Err(ReplaceError::BadEdgeSpec(
                    Direction::Incoming,
                    WhichHugr::Replacement,
                    e.clone(),
                )),
            })?;
        for e in self.mu_out.iter().chain(self.mu_new.iter()) {
            if !h.contains_node(e.tgt) || removed.contains(&e.tgt) {
                return Err(ReplaceError::BadEdgeSpec(
                    Direction::Incoming,
                    WhichHugr::Retained,
                    e.clone(),
                ));
            }
            e.check_tgt(h, e)?;
            // The descendant check is to allow the case where the old edge is nonlocal
            // from a part of the Hugr being moved (which may require changing source,
            // depending on where the transplanted portion ends up). While this subsumes
            // the first "removed.contains" check, we'll keep that as a common-case fast-path.
            e.check_existing_edge(h, &removed, || e.clone())?;
        }
        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        let parent = self.check_parent(h)?;
        // Calculate removed nodes here. (Does not include transfers, so enumerates only
        // nodes we are going to remove, individually, anyway; so no *asymptotic* speed penalty)
        let to_remove = self.get_removed_nodes(h)?;

        // 1. Add all the new nodes. Note this includes replacement.root(), which we don't want.
        // TODO what would an error here mean? e.g. malformed self.replacement??
        let InsertionResult { new_root, node_map } = h.insert_hugr(parent, self.replacement);

        // 2. Add new edges from existing to copied nodes according to mu_in
        let translate_idx = |n| node_map.get(&n).copied().ok_or(WhichHugr::Replacement);
        let kept = |n| {
            let keep = !to_remove.contains(&n);
            keep.then_some(n).ok_or(WhichHugr::Retained)
        };
        transfer_edges(h, self.mu_inp.iter(), kept, translate_idx, None)?;

        // 3. Add new edges from copied to existing nodes according to mu_out,
        // replacing existing value/static edges incoming to targets
        transfer_edges(h, self.mu_out.iter(), translate_idx, kept, Some(&to_remove))?;

        // 4. Add new edges between existing nodes according to mu_new,
        // replacing existing value/static edges incoming to targets.
        transfer_edges(h, self.mu_new.iter(), kept, kept, Some(&to_remove))?;

        // 5. Put newly-added copies into correct places in hierarchy
        // (these will be correct places after removing nodes)
        let mut remove_top_sibs = self.removal.iter();
        for new_node in h.children(new_root).collect::<Vec<Node>>().into_iter() {
            if let Some(top_sib) = remove_top_sibs.next() {
                h.move_before_sibling(new_node, *top_sib);
            } else {
                h.set_parent(new_node, parent);
            }
        }
        debug_assert!(h.children(new_root).next().is_none());
        h.remove_node(new_root);

        // 6. Transfer to keys of `transfers` children of the corresponding values.
        for (new_parent, &old_parent) in self.adoptions.iter() {
            let new_parent = node_map.get(new_parent).unwrap();
            debug_assert!(h.children(old_parent).next().is_some());
            loop {
                let ch = match h.children(old_parent).next() {
                    None => break,
                    Some(c) => c,
                };
                h.set_parent(ch, *new_parent);
            }
        }

        // 7. Remove remaining nodes
        to_remove.into_iter().for_each(|n| h.remove_node(n));
        Ok(node_map)
    }

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        self.removal.iter().copied()
    }
}

fn transfer_edges<'a>(
    h: &mut impl HugrMut,
    edges: impl Iterator<Item = &'a NewEdgeSpec>,
    trans_src: impl Fn(Node) -> Result<Node, WhichHugr>,
    trans_tgt: impl Fn(Node) -> Result<Node, WhichHugr>,
    legal_src_ancestors: Option<&HashSet<Node>>,
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
        if !h.valid_node(e.src) {
            return Err(ReplaceError::BadEdgeSpec(
                Direction::Outgoing,
                WhichHugr::Retained,
                oe.clone(),
            ));
        }
        if !h.valid_node(e.tgt) {
            return Err(ReplaceError::BadEdgeSpec(
                Direction::Incoming,
                WhichHugr::Retained,
                oe.clone(),
            ));
        };
        e.check_src(h, oe)?;
        e.check_tgt(h, oe)?;
        match e.kind {
            NewEdgeKind::Order => {
                h.add_other_edge(e.src, e.tgt);
            }
            NewEdgeKind::Value { src_pos, tgt_pos } | NewEdgeKind::Static { src_pos, tgt_pos } => {
                if let Some(legal_src_ancestors) = legal_src_ancestors {
                    e.check_existing_edge(h, legal_src_ancestors, || oe.clone())?;
                    h.disconnect(e.tgt, tgt_pos);
                }
                h.connect(e.src, src_pos, e.tgt, tgt_pos);
            }
            NewEdgeKind::ControlFlow { src_pos } => h.connect(e.src, src_pos, e.tgt, 0),
        }
    }
    Ok(())
}

/// Error in a [`Replacement`]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ReplaceError {
    /// The node(s) to replace had no parent i.e. were root(s).
    // (Perhaps if there is only one node to replace we should be able to?)
    #[error("Cannot replace the root node of the Hugr")]
    CantReplaceRoot,
    /// The nodes to replace did not have a unique common parent
    #[error("Removed nodes had different parents {0:?}")]
    MultipleParents(Vec<Node>),
    /// Replacement root node had different tag from parent of removed nodes
    #[error("Expected replacement root with tag {removed} but found {replacement}")]
    WrongRootNodeTag {
        /// The tag of the parent of the removed nodes
        removed: OpTag,
        /// The tag of the root in the replacement Hugr
        replacement: OpTag,
    },
    /// Keys in [Replacement::adoptions] were not valid container nodes in [Replacement::replacement]
    #[error("Node {0:?} was not an empty container node in the replacement")]
    InvalidAdoptingParent(Node),
    /// Some values in [Replacement::adoptions] were either descendants of other values, or not
    /// descendants of the [Replacement::removal]. The nodes are indicated on a best-effort basis.
    #[error("Nodes not free to be moved into new locations: {0:?}")]
    AdopteesNotSeparateDescendants(Vec<Node>),
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
    /// (or are on the RHS of an entry in [Replacement::adoptions])
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
    use std::collections::HashMap;

    use cool_asserts::assert_matches;
    use itertools::Itertools;

    use crate::builder::{
        endo_sig, BuildError, CFGBuilder, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, SubContainer,
    };
    use crate::extension::prelude::{BOOL_T, USIZE_T};
    use crate::extension::{ExtensionId, ExtensionRegistry, PRELUDE, PRELUDE_REGISTRY};
    use crate::hugr::internal::HugrMutInternals;
    use crate::hugr::rewrite::replace::WhichHugr;
    use crate::hugr::{HugrMut, Rewrite};
    use crate::ops::custom::{CustomOp, OpaqueOp};
    use crate::ops::dataflow::DataflowOpTrait;
    use crate::ops::handle::{BasicBlockID, ConstID, NodeHandle};
    use crate::ops::{self, Case, DataflowBlock, OpTag, OpType, DFG};
    use crate::std_extensions::collections;
    use crate::types::{Signature, Type, TypeArg, TypeRow};
    use crate::utils::depth;
    use crate::{type_row, Direction, Hugr, HugrView, OutgoingPort};

    use super::{NewEdgeKind, NewEdgeSpec, ReplaceError, Replacement};

    #[test]
    fn cfg() -> Result<(), Box<dyn std::error::Error>> {
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), collections::EXTENSION.to_owned()])
                .unwrap();
        let listy = Type::new_extension(
            collections::EXTENSION
                .get_type(&collections::LIST_TYPENAME)
                .unwrap()
                .instantiate([TypeArg::Type { ty: USIZE_T }])
                .unwrap(),
        );
        let pop: CustomOp = collections::EXTENSION
            .instantiate_extension_op("pop", [TypeArg::Type { ty: USIZE_T }], &reg)
            .unwrap()
            .into();
        let push: CustomOp = collections::EXTENSION
            .instantiate_extension_op("push", [TypeArg::Type { ty: USIZE_T }], &reg)
            .unwrap()
            .into();
        let just_list = TypeRow::from(vec![listy.clone()]);
        let intermed = TypeRow::from(vec![listy.clone(), USIZE_T]);

        let mut cfg = CFGBuilder::new(endo_sig(just_list.clone()))?;

        let pred_const = cfg.add_constant(ops::Value::unary_unit_sum());

        let entry = single_node_block(&mut cfg, pop, &pred_const, true)?;
        let bb2 = single_node_block(&mut cfg, push, &pred_const, false)?;

        let exit = cfg.exit_block();
        cfg.branch(&entry, 0, &bb2)?;
        cfg.branch(&bb2, 0, &exit)?;

        let mut h = cfg.finish_hugr(&reg).unwrap();
        {
            let pop = find_node(&h, "pop");
            let push = find_node(&h, "push");
            assert_eq!(depth(&h, pop), 2); // BB, CFG
            assert_eq!(depth(&h, push), 2);

            let popp = h.get_parent(pop).unwrap();
            let pushp = h.get_parent(push).unwrap();
            assert_ne!(popp, pushp); // Two different BBs
            assert!(h.get_optype(popp).is_dataflow_block());
            assert!(h.get_optype(pushp).is_dataflow_block());

            assert_eq!(h.get_parent(popp).unwrap(), h.get_parent(pushp).unwrap());
        }

        // Replacement: one BB with two DFGs inside.
        // Use Hugr rather than Builder because DFGs must be empty (not even Input/Output).
        let mut replacement = Hugr::new(ops::CFG {
            signature: Signature::new_endo(just_list.clone()),
        });
        let r_bb = replacement.add_node_with_parent(
            replacement.root(),
            DataflowBlock {
                inputs: vec![listy.clone()].into(),
                sum_rows: vec![type_row![]],
                other_outputs: vec![listy.clone()].into(),
                extension_delta: collections::EXTENSION_NAME.into(),
            },
        );
        let r_df1 = replacement.add_node_with_parent(
            r_bb,
            DFG {
                signature: Signature::new(vec![listy.clone()], simple_unary_plus(intermed.clone()))
                    .with_extension_delta(collections::EXTENSION_NAME),
            },
        );
        let r_df2 = replacement.add_node_with_parent(
            r_bb,
            DFG {
                signature: Signature::new(intermed, simple_unary_plus(just_list.clone()))
                    .with_extension_delta(collections::EXTENSION_NAME),
            },
        );
        [0, 1]
            .iter()
            .for_each(|p| replacement.connect(r_df1, *p + 1, r_df2, *p));

        {
            let inp = replacement.add_node_before(
                r_df1,
                ops::Input {
                    types: just_list.clone(),
                },
            );
            let out = replacement.add_node_before(
                r_df1,
                ops::Output {
                    types: simple_unary_plus(just_list),
                },
            );
            replacement.connect(inp, 0, r_df1, 0);
            replacement.connect(r_df2, 0, out, 0);
            replacement.connect(r_df2, 1, out, 1);
        }

        h.apply_rewrite(Replacement {
            removal: vec![entry.node(), bb2.node()],
            replacement,
            adoptions: HashMap::from([(r_df1.node(), entry.node()), (r_df2.node(), bb2.node())]),
            mu_inp: vec![],
            mu_out: vec![NewEdgeSpec {
                src: r_bb,
                tgt: exit.node(),
                kind: NewEdgeKind::ControlFlow {
                    src_pos: OutgoingPort::from(0),
                },
            }],
            mu_new: vec![],
        })?;
        h.update_validate(&reg)?;
        {
            let pop = find_node(&h, "pop");
            let push = find_node(&h, "push");
            assert_eq!(depth(&h, pop), 3); // DFG, BB, CFG
            assert_eq!(depth(&h, push), 3);

            let popp = h.get_parent(pop).unwrap();
            let pushp = h.get_parent(push).unwrap();
            assert_ne!(popp, pushp); // Two different DFGs
            assert!(h.get_optype(popp).is_dfg());
            assert!(h.get_optype(pushp).is_dfg());

            let grandp = h.get_parent(popp).unwrap();
            assert_eq!(grandp, h.get_parent(pushp).unwrap());
            assert!(h.get_optype(grandp).is_dataflow_block());
        }

        Ok(())
    }

    fn find_node(h: &Hugr, s: &str) -> crate::Node {
        h.nodes()
            .filter(|n| format!("{:?}", h.get_optype(*n)).contains(s))
            .exactly_one()
            .ok()
            .unwrap()
    }

    fn single_node_block<T: AsRef<Hugr> + AsMut<Hugr>, O: DataflowOpTrait + Into<OpType>>(
        h: &mut CFGBuilder<T>,
        op: O,
        pred_const: &ConstID,
        entry: bool,
    ) -> Result<BasicBlockID, BuildError> {
        let op_sig = op.signature();
        let mut bb = if entry {
            assert_eq!(
                match h.hugr().get_optype(h.container_node()) {
                    OpType::CFG(c) => &c.signature.input,
                    _ => panic!(),
                },
                op_sig.input()
            );
            h.simple_entry_builder_exts(op_sig.output, 1, op_sig.extension_reqs.clone())?
        } else {
            h.simple_block_builder(op_sig, 1)?
        };
        let op: OpType = op.into();
        let op = bb.add_dataflow_op(op, bb.input_wires())?;
        let load_pred = bb.load_const(pred_const);
        bb.finish_with_outputs(load_pred, op.outputs())
    }

    fn simple_unary_plus(t: TypeRow) -> TypeRow {
        let mut v = t.into_owned();
        v.insert(0, Type::new_unit_sum(1));
        v.into()
    }

    #[test]
    fn test_invalid() -> Result<(), Box<dyn std::error::Error>> {
        let utou = Signature::new_endo(vec![USIZE_T]);
        let mk_op = |s| {
            CustomOp::new_opaque(OpaqueOp::new(
                ExtensionId::new("unknown_ext").unwrap(),
                s,
                String::new(),
                vec![],
                utou.clone(),
            ))
        };
        let mut h = DFGBuilder::new(Signature::new(
            type_row![USIZE_T, BOOL_T],
            type_row![USIZE_T],
        ))?;
        let [i, b] = h.input_wires_arr();
        let mut cond = h.conditional_builder(
            (vec![type_row![]; 2], b),
            [(USIZE_T, i)],
            type_row![USIZE_T],
        )?;
        let mut case1 = cond.case_builder(0)?;
        let foo = case1.add_dataflow_op(mk_op("foo"), case1.input_wires())?;
        let case1 = case1.finish_with_outputs(foo.outputs())?.node();
        let mut case2 = cond.case_builder(1)?;
        let bar = case2.add_dataflow_op(mk_op("bar"), case2.input_wires())?;
        let mut baz_dfg = case2.dfg_builder(utou.clone(), bar.outputs())?;
        let baz = baz_dfg.add_dataflow_op(mk_op("baz"), baz_dfg.input_wires())?;
        let baz_dfg = baz_dfg.finish_with_outputs(baz.outputs())?;
        let case2 = case2.finish_with_outputs(baz_dfg.outputs())?.node();
        let cond = cond.finish_sub_container()?;
        let h = h.finish_hugr_with_outputs(cond.outputs(), &PRELUDE_REGISTRY)?;

        let mut r_hugr = Hugr::new(h.get_optype(cond.node()).clone());
        let r1 = r_hugr.add_node_with_parent(
            r_hugr.root(),
            Case {
                signature: utou.clone(),
            },
        );
        let r2 = r_hugr.add_node_with_parent(
            r_hugr.root(),
            Case {
                signature: utou.clone(),
            },
        );
        let rep: Replacement = Replacement {
            removal: vec![case1, case2],
            replacement: r_hugr,
            adoptions: HashMap::from_iter([(r1, case1), (r2, baz_dfg.node())]),
            mu_inp: vec![],
            mu_out: vec![],
            mu_new: vec![],
        };
        assert_eq!(h.get_parent(baz.node()), Some(baz_dfg.node()));
        rep.verify(&h).unwrap();
        {
            let mut target = h.clone();
            let node_map = rep.clone().apply(&mut target)?;
            let new_case2 = *node_map.get(&r2).unwrap();
            assert_eq!(target.get_parent(baz.node()), Some(new_case2));
        }

        // Test some bad Replacements (using variations of the `replacement` Hugr).
        let check_same_errors = |r: Replacement| {
            let verify_res = r.verify(&h).unwrap_err();
            let apply_res = r.apply(&mut h.clone()).unwrap_err();
            assert_eq!(verify_res, apply_res);
            apply_res
        };
        // Root node type needs to be that of common parent of the removed nodes:
        let mut rep2 = rep.clone();
        rep2.replacement
            .replace_op(rep2.replacement.root(), h.root_type().clone())?;
        assert_eq!(
            check_same_errors(rep2),
            ReplaceError::WrongRootNodeTag {
                removed: OpTag::Conditional,
                replacement: OpTag::Dfg
            }
        );
        // Removed nodes...
        assert_eq!(
            check_same_errors(Replacement {
                removal: vec![h.root()],
                ..rep.clone()
            }),
            ReplaceError::CantReplaceRoot
        );
        assert_eq!(
            check_same_errors(Replacement {
                removal: vec![case1, baz_dfg.node()],
                ..rep.clone()
            }),
            ReplaceError::MultipleParents(vec![cond.node(), case2])
        );
        // Adoptions...
        assert_eq!(
            check_same_errors(Replacement {
                adoptions: HashMap::from([(r1, case1), (rep.replacement.root(), case2)]),
                ..rep.clone()
            }),
            ReplaceError::InvalidAdoptingParent(rep.replacement.root())
        );
        assert_eq!(
            check_same_errors(Replacement {
                adoptions: HashMap::from_iter([(r1, case1), (r2, case1)]),
                ..rep.clone()
            }),
            ReplaceError::AdopteesNotSeparateDescendants(vec![case1])
        );
        assert_eq!(
            check_same_errors(Replacement {
                adoptions: HashMap::from_iter([(r1, case2), (r2, baz_dfg.node())]),
                ..rep.clone()
            }),
            ReplaceError::AdopteesNotSeparateDescendants(vec![baz_dfg.node()])
        );
        // Edges....
        let edge_from_removed = NewEdgeSpec {
            src: case1,
            tgt: r2,
            kind: NewEdgeKind::Order,
        };
        assert_eq!(
            check_same_errors(Replacement {
                mu_inp: vec![edge_from_removed.clone()],
                ..rep.clone()
            }),
            ReplaceError::BadEdgeSpec(Direction::Outgoing, WhichHugr::Retained, edge_from_removed)
        );
        let bad_out_edge = NewEdgeSpec {
            src: h.nodes().max().unwrap(), // not valid in replacement
            tgt: cond.node(),
            kind: NewEdgeKind::Order,
        };
        assert_eq!(
            check_same_errors(Replacement {
                mu_out: vec![bad_out_edge.clone()],
                ..rep.clone()
            }),
            ReplaceError::BadEdgeSpec(Direction::Outgoing, WhichHugr::Replacement, bad_out_edge)
        );
        let bad_order_edge = NewEdgeSpec {
            src: cond.node(),
            tgt: h.get_io(h.root()).unwrap()[1],
            kind: NewEdgeKind::ControlFlow { src_pos: 0.into() },
        };
        assert_matches!(
            check_same_errors(Replacement {
                mu_new: vec![bad_order_edge.clone()],
                ..rep.clone()
            }),
            ReplaceError::BadEdgeKind(_, e) => assert_eq!(e, bad_order_edge)
        );
        let op = OutgoingPort::from(0);
        let (tgt, ip) = h.linked_inputs(cond.node(), op).next().unwrap();
        let new_out_edge = NewEdgeSpec {
            src: r1.node(),
            tgt,
            kind: NewEdgeKind::Value {
                src_pos: op,
                tgt_pos: ip,
            },
        };
        assert_eq!(
            check_same_errors(Replacement {
                mu_out: vec![new_out_edge.clone()],
                ..rep.clone()
            }),
            ReplaceError::BadEdgeKind(Direction::Outgoing, new_out_edge)
        );
        Ok(())
    }
}

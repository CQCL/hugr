//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use itertools::Either;

use crate::{
    Hugr, HugrView, Node,
    core::HugrNode,
    hugr::{HugrMut, hugrmut::InsertedForest, internal::HugrMutInternals},
};

/// Methods that merge Hugrs, adding static edges between old and inserted nodes.
///
/// This is done by module-children from the inserted (source) Hugr replacing, or being replaced by,
/// module-children already in the target Hugr; static edges from the replaced node,
/// are transferred to come from the replacing node, and the replaced node(/subtree) then deleted.
pub trait HugrLinking: HugrMut {
    /// Copy and link nodes from another Hugr into this one, with linking specified by Node.
    ///
    /// If `parent` is non-None, then `other`'s entrypoint-subtree is copied under it.
    /// `children` of the Module root of `other` may also be inserted with their
    /// subtrees or linked according to their [NodeLinkingDirective].
    ///
    /// # Errors
    ///
    /// * If `children` are not `children` of the root of `other`
    /// * If `parent` is Some, and `other.entrypoint()` is either
    ///   * among `children`, or
    ///   * descends from an element of `children` with [NodeLinkingDirective::Add]
    ///
    /// # Panics
    ///
    /// If `parent` is `Some` but not in the graph.
    #[allow(clippy::type_complexity)]
    fn insert_link_view_by_node<H: HugrView>(
        &mut self,
        parent: Option<Self::Node>,
        other: &H,
        children: NodeLinkingDirectives<H::Node, Self::Node>,
    ) -> Result<InsertedForest<H::Node, Self::Node>, NodeLinkingError<H::Node, Self::Node>> {
        let transfers = check_directives(other, parent, &children)?;
        let nodes =
            parent
                .iter()
                .flat_map(|_| other.entry_descendants())
                .chain(children.iter().flat_map(|(&ch, dirv)| match dirv {
                    NodeLinkingDirective::Add { .. } => Either::Left(other.descendants(ch)),
                    NodeLinkingDirective::UseExisting(_) => Either::Right(std::iter::once(ch)),
                }));
        let mut roots = HashMap::new();
        if let Some(parent) = parent {
            roots.insert(other.entrypoint(), parent);
        }
        for ch in children.keys() {
            roots.insert(*ch, self.module_root());
        }
        let mut inserted = self
            .insert_view_forest(other, nodes, roots)
            .expect("NodeLinkingDirectives were checked for disjointness");
        link_by_node(self, transfers, &mut inserted.node_map);
        Ok(inserted)
    }

    /// Insert and link another Hugr into this one, with linking specified by Node.
    ///
    /// If `parent` is non-None, then `other`'s entrypoint-subtree is placed under it.
    /// `children` of the Module root of `other` may also be inserted with their
    /// subtrees or linked according to their [NodeLinkingDirective].
    ///
    /// # Errors
    ///
    /// * If `children` are not `children` of the root of `other`
    /// * If `other`s entrypoint is among `children`, or descends from an element
    ///   of `children` with [NodeLinkingDirective::Add]
    ///
    /// # Panics
    ///
    /// If `parent` is not in this graph.
    fn insert_link_hugr_by_node(
        &mut self,
        parent: Option<Self::Node>,
        mut other: Hugr,
        children: NodeLinkingDirectives<Node, Self::Node>,
    ) -> Result<InsertedForest<Node, Self::Node>, NodeLinkingError<Node, Self::Node>> {
        let transfers = check_directives(&other, parent, &children)?;
        let mut roots = HashMap::new();
        if let Some(parent) = parent {
            roots.insert(other.entrypoint(), parent);
            other.set_parent(other.entrypoint(), other.module_root());
        };
        for (ch, dirv) in children.iter() {
            roots.insert(*ch, self.module_root());
            if matches!(dirv, NodeLinkingDirective::UseExisting(_)) {
                // We do not need to copy the children of ch
                while let Some(gch) = other.first_child(*ch) {
                    // No point in deleting subtree, we won't copy disconnected nodes
                    other.remove_node(gch);
                }
            }
        }
        let mut inserted = self
            .insert_forest(other, roots)
            .expect("NodeLinkingDirectives were checked for disjointness");
        link_by_node(self, transfers, &mut inserted.node_map);
        Ok(inserted)
    }
}

impl<T: HugrMut> HugrLinking for T {}

/// An error resulting from an [NodeLinkingDirective] passed to [HugrLinking::insert_link_hugr_by_node]
/// or [HugrLinking::insert_link_view_by_node].
///
/// `SN` is the type of nodes in the source (inserted) Hugr; `TN` similarly for the target Hugr.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeLinkingError<SN: Display = Node, TN: Display = Node> {
    /// Inserting the whole Hugr, yet also asked to insert some of its children
    /// (so the inserted Hugr's entrypoint was its module-root).
    #[error(
        "Cannot insert children (e.g. {_0}) when already inserting whole Hugr (entrypoint == module_root)"
    )]
    ChildOfEntrypoint(SN),
    /// A module-child requested contained (or was) the entrypoint
    #[error("Requested to insert module-child {_0} but this contains the entrypoint")]
    ChildContainsEntrypoint(SN),
    /// A module-child requested was not a child of the module root
    #[error("{_0} was not a child of the module root")]
    NotChildOfRoot(SN),
    /// A node in the target Hugr was in a [NodeLinkingDirective::Add::replace] for multiple
    /// inserted nodes (it is not clear to which we should transfer edges).
    #[error("Target node {_0} is to be replaced by two source nodes {_1} and {_2}")]
    NodeMultiplyReplaced(TN, SN, SN),
}

/// Directive for how to treat a particular module-child in the source Hugr.
/// (TN is a node in the target Hugr.)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeLinkingDirective<TN = Node> {
    /// Insert the module-child (with subtree if any) into the target Hugr.
    Add {
        // TODO If non-None, change the name of the inserted function
        //rename: Option<String>,
        /// Existing/old nodes in the target which will be removed (with their subtrees),
        /// and any static ([EdgeKind::Function]/[EdgeKind::Const]) edges from them changed
        /// to leave the newly-inserted node instead. (Typically, this `Vec` would contain
        /// at most one [FuncDefn], or perhaps-multiple, aliased, [FuncDecl]s.)
        ///
        /// [FuncDecl]: crate::ops::FuncDecl
        /// [FuncDefn]: crate::ops::FuncDefn
        /// [EdgeKind::Const]: crate::types::EdgeKind::Const
        /// [EdgeKind::Function]: crate::types::EdgeKind::Function
        replace: Vec<TN>,
    },
    /// Do not insert the node/subtree from the source, but for any static edge from it
    /// to an inserted node, instead add an edge from the specified node already existing
    /// in the target Hugr. (Static edges are [EdgeKind::Function] and [EdgeKind::Const].)
    ///
    /// [EdgeKind::Const]: crate::types::EdgeKind::Const
    /// [EdgeKind::Function]: crate::types::EdgeKind::Function
    UseExisting(TN),
}

impl<TN> NodeLinkingDirective<TN> {
    /// Just add the node (and any subtree) into the target.
    /// (Could lead to an invalid Hugr if the target Hugr
    /// already has another with the same name and both are [Public])
    ///
    /// [Public]: crate::Visibility::Public
    pub const fn add() -> Self {
        Self::Add { replace: vec![] }
    }

    /// The new node should replace the specified node(s) already existing
    /// in the target.
    ///
    /// (Could lead to an invalid Hugr if they have different signatures,
    /// or if the target already has another function with the same name and both are public.)
    pub fn replace(nodes: impl IntoIterator<Item = TN>) -> Self {
        Self::Add {
            replace: nodes.into_iter().collect(),
        }
    }
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr.
///
/// For use with [HugrLinking::insert_link_hugr_by_node] and [HugrLinking::insert_link_view_by_node].
pub type NodeLinkingDirectives<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

/// Invariant: no SourceNode can be in both maps (by type of [NodeLinkingDirective])
/// TargetNodes can be (in RHS of multiple directives)
struct Transfers<SourceNode, TargetNode> {
    use_existing: HashMap<SourceNode, TargetNode>,
    replace: HashMap<TargetNode, SourceNode>,
}

fn check_directives<SRC: HugrView, TN: HugrNode>(
    other: &SRC,
    parent: Option<TN>,
    children: &HashMap<SRC::Node, NodeLinkingDirective<TN>>,
) -> Result<Transfers<SRC::Node, TN>, NodeLinkingError<SRC::Node, TN>> {
    if parent.is_some() {
        if other.entrypoint() == other.module_root() {
            if let Some(c) = children.keys().next() {
                return Err(NodeLinkingError::ChildOfEntrypoint(*c));
            }
        } else {
            let mut n = other.entrypoint();
            if children.contains_key(&n) {
                // If parent == hugr.module_root() and the directive is to Add, we could
                // allow that - it amounts to two instructions to do the same thing.
                // (If the directive is to UseExisting, then we'd have nothing to add
                //  beneath parent! And if parent != hugr.module_root(), then not only
                //  would we have to double-copy the entrypoint-subtree, but also
                //  (unless n is a Const!) we would be creating an illegal Hugr.)
                return Err(NodeLinkingError::ChildContainsEntrypoint(n));
            }
            while let Some(p) = other.get_parent(n) {
                if matches!(children.get(&p), Some(NodeLinkingDirective::Add { .. })) {
                    return Err(NodeLinkingError::ChildContainsEntrypoint(p));
                }
                n = p
            }
        }
    }
    let mut trns = Transfers {
        replace: HashMap::default(),
        use_existing: HashMap::default(),
    };
    for (&sn, dirv) in children {
        if other.get_parent(sn) != Some(other.module_root()) {
            return Err(NodeLinkingError::NotChildOfRoot(sn));
        }
        match dirv {
            NodeLinkingDirective::Add { replace } => {
                for &r in replace {
                    if let Some(old_sn) = trns.replace.insert(r, sn) {
                        return Err(NodeLinkingError::NodeMultiplyReplaced(r, old_sn, sn));
                    }
                }
            }
            NodeLinkingDirective::UseExisting(tn) => {
                trns.use_existing.insert(sn, *tn);
            }
        }
    }
    Ok(trns)
}

fn link_by_node<SN: HugrNode, TGT: HugrLinking + ?Sized>(
    hugr: &mut TGT,
    transfers: Transfers<SN, TGT::Node>,
    node_map: &mut HashMap<SN, TGT::Node>,
) {
    // Resolve `use_existing` first in case the existing node is also replaced by
    // a new node (which we know will not be in RHS of any entry in `replace`).
    for (sn, tn) in transfers.use_existing {
        let copy = node_map.remove(&sn).unwrap();
        // Because of `UseExisting` we avoided adding `sn`s descendants
        debug_assert_eq!(hugr.children(copy).next(), None);
        replace_static_src(hugr, copy, tn);
    }
    for (tn, sn) in transfers.replace {
        let new_node = *node_map.get(&sn).unwrap();
        replace_static_src(hugr, tn, new_node);
    }
}

fn replace_static_src<H: HugrMut + ?Sized>(hugr: &mut H, old_src: H::Node, new_src: H::Node) {
    let targets = hugr.all_linked_inputs(old_src).collect::<Vec<_>>();
    for (target, inport) in targets {
        let (src_node, outport) = hugr.single_linked_output(target, inport).unwrap();
        debug_assert_eq!(src_node, old_src);
        hugr.disconnect(target, inport);
        hugr.connect(new_src, outport, target, inport);
    }
    hugr.remove_subtree(old_src);
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use cool_asserts::assert_matches;
    use itertools::Itertools;

    use super::{HugrLinking, NodeLinkingDirective, NodeLinkingError};
    use crate::builder::test::{dfg_calling_defn_decl, simple_dfg_hugr};
    use crate::hugr::hugrmut::test::check_calls_defn_decl;
    use crate::ops::{FuncDecl, OpTag, OpTrait, handle::NodeHandle};
    use crate::{HugrView, hugr::HugrMut, types::Signature};

    #[test]
    fn test_insert_link_nodes_add() {
        // Default (non-linking) methods...just for comparison
        let (insert, _, _) = dfg_calling_defn_decl();

        let mut h = simple_dfg_hugr();
        h.insert_from_view(h.entrypoint(), &insert);
        check_calls_defn_decl(&h, false, false);

        let mut h = simple_dfg_hugr();
        h.insert_hugr(h.entrypoint(), insert);
        check_calls_defn_decl(&h, false, false);

        // Specify which decls to transfer. No real "linking" here though.
        for (call1, call2) in [(false, false), (false, true), (true, false), (true, true)] {
            let (insert, defn, decl) = dfg_calling_defn_decl();
            let mod_children = HashMap::from_iter(
                call1
                    .then_some((defn.node(), NodeLinkingDirective::add()))
                    .into_iter()
                    .chain(call2.then_some((decl.node(), NodeLinkingDirective::add()))),
            );

            let mut h = simple_dfg_hugr();
            h.insert_link_view_by_node(Some(h.entrypoint()), &insert, mod_children.clone())
                .unwrap();
            check_calls_defn_decl(&h, call1, call2);

            let mut h = simple_dfg_hugr();
            h.insert_link_hugr_by_node(Some(h.entrypoint()), insert, mod_children)
                .unwrap();
            check_calls_defn_decl(&h, call1, call2);
        }
    }

    #[test]
    fn insert_link_nodes_replace() {
        let (mut host, defn, decl) = dfg_calling_defn_decl();
        assert_eq!(
            host.children(host.module_root())
                .map(|n| host.get_optype(n).tag())
                .collect_vec(),
            vec![OpTag::FuncDefn, OpTag::FuncDefn, OpTag::Function]
        );
        let insert = simple_dfg_hugr();
        let dirvs = HashMap::from([(
            insert
                .children(insert.module_root())
                .exactly_one()
                .ok()
                .unwrap(),
            NodeLinkingDirective::Add {
                replace: vec![defn.node(), decl.node()],
            },
        )]);
        host.insert_link_hugr_by_node(None, insert, dirvs).unwrap();
        host.validate().unwrap();
        assert_eq!(
            host.children(host.module_root())
                .map(|n| host.get_optype(n).tag())
                .collect_vec(),
            vec![OpTag::FuncDefn; 2]
        );
    }

    #[test]
    fn insert_link_nodes_use_existing() {
        let (insert, defn, decl) = dfg_calling_defn_decl();
        let mut chmap =
            HashMap::from([defn.node(), decl.node()].map(|n| (n, NodeLinkingDirective::add())));
        let (h, node_map) = {
            let mut h = simple_dfg_hugr();
            let res = h
                .insert_link_view_by_node(Some(h.entrypoint()), &insert, chmap.clone())
                .unwrap();
            (h, res.node_map)
        };
        h.validate().unwrap();
        let num_nodes = h.num_nodes();
        let num_ep_nodes = h.descendants(node_map[&insert.entrypoint()]).count();
        let [inserted_defn, inserted_decl] = [defn.node(), decl.node()].map(|n| node_map[&n]);

        // No reason we can't add the decl again, or replace the defn with the decl,
        // but here we'll limit to the "interesting" (likely) cases
        for decl_replacement in [inserted_defn, inserted_decl] {
            let decl_mode = NodeLinkingDirective::UseExisting(decl_replacement);
            chmap.insert(decl.node(), decl_mode);
            for defn_mode in [
                NodeLinkingDirective::add(),
                NodeLinkingDirective::UseExisting(inserted_defn),
            ] {
                chmap.insert(defn.node(), defn_mode.clone());
                let mut h = h.clone();
                h.insert_link_hugr_by_node(Some(h.entrypoint()), insert.clone(), chmap.clone())
                    .unwrap();
                h.validate().unwrap();
                if defn_mode != NodeLinkingDirective::add() {
                    assert_eq!(h.num_nodes(), num_nodes + num_ep_nodes);
                }
                assert_eq!(
                    h.children(h.module_root()).count(),
                    3 + (defn_mode == NodeLinkingDirective::add()) as usize
                );
                let expected_defn_uses = 1
                    + (defn_mode == NodeLinkingDirective::UseExisting(inserted_defn)) as usize
                    + (decl_replacement == inserted_defn) as usize;
                assert_eq!(
                    h.static_targets(inserted_defn).unwrap().count(),
                    expected_defn_uses
                );
                assert_eq!(
                    h.static_targets(inserted_decl).unwrap().count(),
                    1 + (decl_replacement == inserted_decl) as usize
                );
            }
        }
    }

    #[test]
    fn bad_insert_link_nodes() {
        let backup = simple_dfg_hugr();
        let mut h = backup.clone();

        let (insert, defn, decl) = dfg_calling_defn_decl();
        let (defn, decl) = (defn.node(), decl.node());

        let epp = insert.get_parent(insert.entrypoint()).unwrap();
        let r = h.insert_link_view_by_node(
            Some(h.entrypoint()),
            &insert,
            HashMap::from([(epp, NodeLinkingDirective::add())]),
        );
        assert_eq!(
            r.err().unwrap(),
            NodeLinkingError::ChildContainsEntrypoint(epp)
        );
        assert_eq!(h, backup);

        let [inp, _] = insert.get_io(defn).unwrap();
        let r = h.insert_link_view_by_node(
            Some(h.entrypoint()),
            &insert,
            HashMap::from([(inp, NodeLinkingDirective::add())]),
        );
        assert_eq!(r.err().unwrap(), NodeLinkingError::NotChildOfRoot(inp));
        assert_eq!(h, backup);

        let mut insert = insert;
        insert.set_entrypoint(defn);
        let r = h.insert_link_view_by_node(
            Some(h.module_root()),
            &insert,
            HashMap::from([(
                defn,
                NodeLinkingDirective::UseExisting(h.get_parent(h.entrypoint()).unwrap()),
            )]),
        );
        assert_eq!(
            r.err().unwrap(),
            NodeLinkingError::ChildContainsEntrypoint(defn)
        );
        assert_eq!(h, backup);

        insert.set_entrypoint(insert.module_root());
        let r = h.insert_link_hugr_by_node(
            Some(h.module_root()),
            insert,
            HashMap::from([(decl, NodeLinkingDirective::add())]),
        );
        assert_eq!(r.err().unwrap(), NodeLinkingError::ChildOfEntrypoint(decl));
        assert_eq!(h, backup);

        let (insert, defn, decl) = dfg_calling_defn_decl();
        let sig = insert
            .get_optype(defn.node())
            .as_func_defn()
            .unwrap()
            .signature()
            .clone();
        let tmp = h.add_node_with_parent(h.module_root(), FuncDecl::new("replaced", sig));
        let r = h.insert_link_hugr_by_node(
            Some(h.entrypoint()),
            insert,
            HashMap::from([
                (decl.node(), NodeLinkingDirective::replace([tmp])),
                (defn.node(), NodeLinkingDirective::replace([tmp])),
            ]),
        );
        assert_matches!(
            r.err().unwrap(),
            NodeLinkingError::NodeMultiplyReplaced(tn, sn1, sn2) => {
                assert_eq!(tmp, tn);
                assert_eq!([sn1,sn2].into_iter().sorted().collect_vec(), [defn.node(), decl.node()]);
        });
    }

    #[test]
    fn test_replace_used() {
        let mut h = simple_dfg_hugr();
        let temp = h.add_node_with_parent(
            h.module_root(),
            FuncDecl::new("temp", Signature::new_endo(vec![])),
        );

        let (insert, defn, decl) = dfg_calling_defn_decl();
        let node_map = h
            .insert_link_hugr_by_node(
                Some(h.entrypoint()),
                insert,
                HashMap::from([
                    (defn.node(), NodeLinkingDirective::replace([temp])),
                    (decl.node(), NodeLinkingDirective::UseExisting(temp)),
                ]),
            )
            .unwrap()
            .node_map;
        let defn = node_map[&defn.node()];
        assert_eq!(node_map.get(&decl.node()), None);
        assert!(!h.contains_node(temp));

        assert!(
            h.children(h.module_root())
                .all(|n| h.get_optype(n).is_func_defn())
        );
        for call in h.nodes().filter(|n| h.get_optype(*n).is_call()) {
            assert_eq!(h.static_source(call), Some(defn));
        }
    }
}

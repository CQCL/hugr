//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use itertools::Either;

use crate::{
    Hugr, HugrView, Node,
    hugr::{HugrMut, hugrmut::insert_hugr_internal},
};

pub trait LinkHugr: HugrMut {
    /// Copy nodes from another Hugr into this one, with linking directives specified by Node.
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
    fn insert_from_view_link_nodes<H: HugrView>(
        &mut self,
        parent: Option<Self::Node>,
        other: &H,
        children: NodeLinkingDirectives<H::Node, Self::Node>,
    ) -> Result<HashMap<H::Node, Self::Node>, NodeLinkingError<H::Node>> {
        todo!()
    }

    /// Insert another Hugr into this one, with linking directives specified by Node.
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
    fn insert_hugr_link_nodes(
        &mut self,
        parent: Option<Self::Node>,
        other: Hugr,
        children: NodeLinkingDirectives<Node, Self::Node>,
    ) -> Result<HashMap<Node, Self::Node>, NodeLinkingError<Node>> {
        todo!()
    }
}

impl<T: HugrMut> LinkHugr for T {}

/// An error resulting from an [NodeLinkingDirective] passed to [insert_hugr_link_nodes]
/// or [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeLinkingError<N: Display> {
    /// Inserting the whole Hugr, yet also asked to insert some of its children
    /// (so the inserted Hugr's entrypoint was its module-root).
    #[error(
        "Cannot insert children (e.g. {_0}) when already inserting whole Hugr (entrypoint == module_root)"
    )]
    ChildOfEntrypoint(N),
    /// A module-child requested contained (or was) the entrypoint
    #[error("Requested to insert module-child {_0} but this contains the entrypoint")]
    ChildContainsEntrypoint(N),
    /// A module-child requested was not a child of the module root
    #[error("{_0} was not a child of the module root")]
    NotChildOfRoot(N),
}

/// Directive for how to treat a particular FuncDefn/FuncDecl in the source Hugr.
/// (TN is a node in the target Hugr.)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum NodeLinkingDirective<TN = Node> {
    /// Insert the FuncDecl, or the FuncDefn and its subtree, into the target Hugr.
    Add {
        // TODO If non-None, change the name of the inserted function
        //rename: Option<String>,
        /// Existing/old nodes in the target which will be removed (with their subtrees),
        /// and any ([EdgeKind::Function]) edges from them changed to leave the newly-inserted
        /// node instead. (Typically, this `Vec` would contain at most one `FuncDefn`,
        /// or perhaps-multiple, aliased, `FuncDecl`s.)
        ///
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
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr (beneath the module root). For use with [insert_hugr_link_nodes] and
/// [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
pub type NodeLinkingDirectives<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

fn insert_link_by_node<H: HugrView>(
    hugr: &mut Hugr,
    parent: Option<Node>,
    other: &H,
    children: HashMap<H::Node, NodeLinkingDirective>,
) -> Result<HashMap<H::Node, Node>, NodeLinkingError<H::Node>> {
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
    for &c in children.keys() {
        if other.get_parent(c) != Some(other.module_root()) {
            return Err(NodeLinkingError::NotChildOfRoot(c));
        }
    }
    // In fact we'll copy all `children`, but only including subtrees
    // for children that should be `Add`ed. This ensures we copy
    // edges from any of those children to any other copied nodes.
    let nodes = children
        .iter()
        .flat_map(|(&ch, m)| match m {
            NodeLinkingDirective::Add { .. } => Either::Left(other.descendants(ch)),
            NodeLinkingDirective::UseExisting(_) => Either::Right(std::iter::once(ch)),
        })
        .chain(parent.iter().flat_map(|_| other.entry_descendants()));
    let hugr_root = hugr.module_root();
    let mut node_map = insert_hugr_internal(hugr, &other, nodes, |&n| {
        if n == other.entrypoint() {
            parent // If parent is None, quite possible this case will not be used
        } else {
            children.contains_key(&n).then_some(hugr_root)
        }
    });
    // Now enact any `Add`s with replaces, and `UseExisting`s, removing the copied children
    for (ch, m) in children {
        match m {
            NodeLinkingDirective::UseExisting(replace_with) => {
                let copy = node_map.remove(&ch).unwrap();
                // Because of `UseExisting` we avoided adding `ch`s descendants above
                debug_assert_eq!(hugr.children(copy).next(), None);
                replace_static_src(hugr, copy, replace_with);
            }
            NodeLinkingDirective::Add { replace } => {
                let new_node = *node_map.get(&ch).unwrap();
                for replace in replace {
                    replace_static_src(hugr, replace, new_node);
                }
            }
        }
    }
    Ok(node_map)
}

fn replace_static_src(hugr: &mut Hugr, old_src: Node, new_src: Node) {
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

    use itertools::Itertools;

    use super::{LinkHugr, NodeLinkingDirective, NodeLinkingError};
    use crate::builder::test::{dfg_calling_defn_decl, simple_dfg_hugr};
    use crate::hugr::{HugrMut, ValidationError};
    use crate::ops::{OpTag, OpTrait, handle::NodeHandle};
    use crate::{Hugr, HugrView};

    #[test]
    fn test_insert_link_nodes_add() {
        let mut h = simple_dfg_hugr();
        let (insert, _, _) = dfg_calling_defn_decl();

        // Defaults
        h.insert_from_view(h.entrypoint(), &insert);
        check_insertion(h, false, false);

        let mut h = simple_dfg_hugr();
        h.insert_hugr(h.entrypoint(), insert);
        check_insertion(h, true, true);

        // Specify which decls to transfer
        for (call1, call2) in [(false, false), (false, true), (true, false), (true, true)] {
            let (insert, defn, decl) = dfg_calling_defn_decl();
            let mod_children = HashMap::from_iter(
                call1
                    .then_some((defn.node(), NodeLinkingDirective::add()))
                    .into_iter()
                    .chain(call2.then_some((decl.node(), NodeLinkingDirective::add()))),
            );

            let mut h = simple_dfg_hugr();
            h.insert_from_view_link_nodes(Some(h.entrypoint()), &insert, mod_children.clone())
                .unwrap();
            check_insertion(h, call1, call2);

            let mut h = simple_dfg_hugr();
            h.insert_hugr_link_nodes(Some(h.entrypoint()), insert, mod_children)
                .unwrap();
            check_insertion(h, call1, call2);
        }
    }

    fn check_insertion(h: Hugr, call1_ok: bool, call2_ok: bool) {
        if call1_ok && call2_ok {
            h.validate().unwrap();
        } else {
            assert!(matches!(
                h.validate(),
                Err(ValidationError::UnconnectedPort { .. })
            ));
        }
        assert_eq!(
            h.children(h.module_root()).count(),
            1 + (call1_ok as usize) + (call2_ok as usize)
        );
        let [call1, call2] = h
            .nodes()
            .filter(|n| h.get_optype(*n).is_call())
            .collect_array()
            .unwrap();

        let tgt1 = h.nodes().find(|n| {
            h.get_optype(*n)
                .as_func_defn()
                .is_some_and(|fd| fd.func_name() == "helper_id")
        });
        assert_eq!(tgt1.is_some(), call1_ok);
        assert_eq!(h.static_source(call1), tgt1);

        let tgt2 = h.nodes().find(|n| {
            h.get_optype(*n)
                .as_func_decl()
                .is_some_and(|fd| fd.func_name() == "helper2")
        });
        assert_eq!(tgt2.is_some(), call2_ok);
        assert_eq!(h.static_source(call2), tgt2);
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
        let pol = HashMap::from([(
            insert
                .children(insert.module_root())
                .exactly_one()
                .ok()
                .unwrap(),
            NodeLinkingDirective::Add {
                replace: vec![defn.node(), decl.node()],
            },
        )]);
        host.insert_hugr_link_nodes(None, insert, pol).unwrap();
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
                .insert_from_view_link_nodes(Some(h.entrypoint()), &insert, chmap.clone())
                .unwrap();
            (h, res)
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
                h.insert_hugr_link_nodes(Some(h.entrypoint()), insert.clone(), chmap.clone())
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
        let r = h.insert_from_view_link_nodes(
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
        let r = h.insert_from_view_link_nodes(
            Some(h.entrypoint()),
            &insert,
            HashMap::from([(inp, NodeLinkingDirective::add())]),
        );
        assert_eq!(r.err().unwrap(), NodeLinkingError::NotChildOfRoot(inp));
        assert_eq!(h, backup);

        let mut insert = insert;
        insert.set_entrypoint(defn);
        let r = h.insert_from_view_link_nodes(
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
        let r = h.insert_hugr_link_nodes(
            Some(h.module_root()),
            insert,
            HashMap::from([(decl, NodeLinkingDirective::add())]),
        );
        assert_eq!(r.err().unwrap(), NodeLinkingError::ChildOfEntrypoint(decl));
    }
}

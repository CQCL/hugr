//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use itertools::{Either, Itertools};

use crate::{
    Hugr, HugrView, Node, Visibility,
    core::HugrNode,
    hugr::{HugrMut, hugrmut::InsertedForest, internal::HugrMutInternals},
    ops::OpType,
    types::PolyFuncType,
};

/// Methods that merge Hugrs, adding static edges between old and inserted nodes.
///
/// This is done by module-children from the inserted (source) Hugr replacing, or being replaced by,
/// module-children already in the target Hugr; static edges from the replaced node,
/// are transferred to come from the replacing node, and the replaced node(/subtree) then deleted.
pub trait HugrLinking: HugrMut {
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
    fn add_view_link_nodes<H: HugrView>(
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
    fn add_hugr_link_nodes(
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

    /// Copy module-children from another Hugr into this one according to a [NameLinkingPolicy].
    ///
    /// All [Visibility::Public] module-children are copied, or linked, according to the
    /// specified policy; private children will also be copied, at least including all those
    /// used by the copied public children.
    // Yes at present we copy all private children, i.e. a safe over-approximation!
    ///
    /// # Errors
    ///
    /// * If [NameLinkingPolicy::error_on_conflicting_sig] is true and there are public
    ///   functions with the same name but different signatures
    ///
    /// * If [MultipleImplHandling::ErrorDontInsert] is used
    ///   and both `self` and `other` have public [FuncDefn]s with the same name and signature
    ///
    /// [Visibility::Public]: crate::Visibility::Public
    /// [MultipleImplHandling::ErrorDontInsert]: crate::hugr::linking::MultipleImplHandling::ErrorDontInsert
    #[allow(clippy::type_complexity)]
    fn link_hugr(
        &mut self,
        other: Hugr,
        policy: NameLinkingPolicy,
    ) -> Result<InsertedForest<Node, Self::Node>, NameLinkingError<Node, Self::Node>> {
        let per_node = policy.to_node_linking(self, &other)?;
        Ok(self
            .add_hugr_link_nodes(None, other, per_node)
            .expect("NodeLinkingPolicy was constructed to avoid any error"))
    }
}

impl<T: HugrMut> HugrLinking for T {}

/// An error resulting from an [NodeLinkingDirective] passed to [HugrLinking::add_hugr_link_nodes]
/// or [HugrLinking::add_view_link_nodes].
///
/// `SN` is the type of nodes in the source (inserted) Hugr; `TN` similarly for the target Hugr.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeLinkingError<SN: Display, TN: Display> {
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

/// Describes ways to link a "Source" Hugr being inserted into a target Hugr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameLinkingPolicy {
    // TODO: consider pub-funcs-to-add? (With others, optionally filtered by callgraph, made private)
    // copy_private_funcs: bool, // TODO: allow filtering private funcs to only those reachable in callgraph
    /// How to handle cases where the same (public) name is present in both
    /// inserted and target Hugr but with different signatures.
    sig_conflict: SignatureConflictHandling,
    /// How to handle cases where both target and inserted Hugr have a FuncDefn
    /// with the same name and signature.
    // TODO consider Set of names where to prefer new? Or optional map from name?
    multi_impls: MultipleImplHandling,
    // TODO Renames to apply to public functions in the inserted Hugr. These take effect
    // before [error_on_conflicting_sig] or [take_existing_and_new_impls].
    // rename_map: HashMap<String, String>
}

/// What to do when both target and inserted Hugr have a
/// [Visibility::Public] function with the same name but different signatures.
// ALAN Note: we *could* combine with MultipleImplHandling; the UseExisting/UseNew variants
// would lead to invalid edges (between ports of different EdgeKind).

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[non_exhaustive] // could consider e.g. disconnections
pub enum SignatureConflictHandling {
    /// Do not link the Hugrs together; raise a [NameLinkingError::SignatureConflict] instead.
    ErrorDontInsert,
    /// Add the new function alongside the existing one in the target Hugr,
    /// preserving (separately) uses of both. (The Hugr will be invalid because
    /// of duplicate names.)
    UseBoth,
}

/// What to do when both target and inserted Hugr
/// have a [Visibility::Public] FuncDefn with the same name and signature.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[non_exhaustive] // could consider e.g. disconnections
pub enum MultipleImplHandling {
    /// Do not link the Hugrs together; raise a [NameLinkingError::MultipleImpls] instead
    ErrorDontInsert,
    /// Keep the implementation already in the target Hugr. (Edges in the source
    /// Hugr will be redirected to use the function from the target.)
    UseExisting,
    /// Keep the implementation in the source Hugr. (Edges in the target Hugr
    /// will be redirected to use the function from the source; the previously-existing
    /// function in the target Hugr will be removed.)
    UseNew,
    /// Add the new function alongside the existing one in the target Hugr,
    /// preserving (separately) uses of both. (The Hugr will be invalid because
    /// of duplicate names.)
    UseBoth,
}

impl NameLinkingPolicy {
    pub fn err_on_conflict(multi_impls: MultipleImplHandling) -> Self {
        Self {
            multi_impls,
            sig_conflict: SignatureConflictHandling::ErrorDontInsert,
        }
    }

    pub fn keep_both_invalid() -> Self {
        Self {
            multi_impls: MultipleImplHandling::UseBoth,
            sig_conflict: SignatureConflictHandling::UseBoth,
        }
    }

    pub fn on_signature_conflict(&mut self, s: SignatureConflictHandling) {
        self.sig_conflict = s;
    }

    pub fn on_multiple_impls(&mut self, mih: MultipleImplHandling) {
        self.multi_impls = mih;
    }
}

impl Default for NameLinkingPolicy {
    fn default() -> Self {
        Self::err_on_conflict(MultipleImplHandling::ErrorDontInsert)
    }
}

/// An error in using names to determine how to link functions in source and target Hugrs.
/// (SN = Source Node, TN = Target Node)
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum NameLinkingError<SN: Display, TN: Display + std::fmt::Debug> {
    /// Both source and target contained a [FuncDefn] (public and with same name
    /// and signature).
    ///
    /// [FuncDefn]: crate::ops::FuncDefn
    #[error("Source ({_1}) and target ({_2}) both contained FuncDefn with same public name {_0}")]
    MultipleImpls(String, SN, TN),
    /// Source and target containing public functions with conflicting signatures
    // TODO ALAN Should we indicate which were decls or defns? via an extra enum?
    #[error(
        "Conflicting signatures for name {name} - Source ({src_node}) has {src_sig}, Target ({tgt_node}) has ({tgt_sig})"
    )]
    #[allow(missing_docs)]
    SignatureConflict {
        name: String,
        src_node: SN,
        src_sig: Box<PolyFuncType>,
        tgt_node: TN,
        tgt_sig: Box<PolyFuncType>,
    },
    /// A [Visibility::Public] function in the source, whose body is being added
    /// to the target, contained the entrypoint (which needs to be added
    /// in a different place).
    #[error("The entrypoint is contained within function {_0} which will be added as {_1:?}")]
    AddFunctionContainingEntrypoint(SN, NodeLinkingDirective<TN>),
}

impl NameLinkingPolicy {
    /// Builds an explicit map of [NodeLinkingDirective]s that implements this policy for a given
    /// source (inserted) and target (inserted-into) Hugr.
    /// The map should be such that no [NodeLinkingError] will occur.
    #[allow(clippy::type_complexity)]
    pub fn to_node_linking<T: HugrView + ?Sized, S: HugrView + ?Sized>(
        &self,
        target: &T,
        source: &S,
    ) -> Result<NodeLinkingDirectives<S::Node, T::Node>, NameLinkingError<S::Node, T::Node>> {
        let existing = gather_existing(target);
        let mut res = NodeLinkingDirectives::new();

        let NameLinkingPolicy {
            sig_conflict,
            multi_impls,
        } = self;
        for n in source.children(source.module_root()) {
            if let Some((name, is_defn, vis, sig)) = link_sig(source, n) {
                let dirv = if let Some((ex_ns, ex_sig)) =
                    vis.is_public().then(|| existing.get(name)).flatten()
                {
                    match *sig_conflict {
                        _ if sig == *ex_sig => directive(name, n, is_defn, ex_ns, multi_impls)?,
                        SignatureConflictHandling::ErrorDontInsert => {
                            return Err(NameLinkingError::SignatureConflict {
                                name: name.clone(),
                                src_node: n,
                                src_sig: Box::new(sig.clone()),
                                tgt_node: *ex_ns.as_ref().left_or_else(|(n, _)| n),
                                tgt_sig: Box::new((*ex_sig).clone()),
                            });
                        }
                        SignatureConflictHandling::UseBoth => NodeLinkingDirective::add(),
                    }
                } else {
                    NodeLinkingDirective::add()
                };
                res.insert(n, dirv);
            }
        }

        Ok(res)
    }
}

fn directive<SN: Display, TN: HugrNode>(
    name: &str,
    new_n: SN,
    new_defn: bool,
    ex_ns: &Either<TN, (TN, Vec<TN>)>,
    multi_impls: &MultipleImplHandling,
) -> Result<NodeLinkingDirective<TN>, NameLinkingError<SN, TN>> {
    Ok(match (new_defn, ex_ns) {
        (false, Either::Right(_)) => NodeLinkingDirective::add(), // another alias
        (false, Either::Left(defn)) => NodeLinkingDirective::UseExisting(*defn), // resolve decl
        (true, Either::Right((decl, decls))) => {
            NodeLinkingDirective::replace(std::iter::once(decl).chain(decls).cloned())
        }
        (true, &Either::Left(defn)) => match multi_impls {
            MultipleImplHandling::UseExisting => NodeLinkingDirective::UseExisting(defn),
            MultipleImplHandling::UseNew => NodeLinkingDirective::replace([defn]),
            MultipleImplHandling::ErrorDontInsert => {
                return Err(NameLinkingError::MultipleImpls(
                    name.to_owned(),
                    new_n,
                    defn,
                ));
            }
            MultipleImplHandling::UseBoth => NodeLinkingDirective::add(),
        },
    })
}

type PubFuncs<'a, N> = (Either<N, (N, Vec<N>)>, &'a PolyFuncType);

fn link_sig<H: HugrView + ?Sized>(
    h: &H,
    n: H::Node,
) -> Option<(&String, bool, &Visibility, &PolyFuncType)> {
    match h.get_optype(n) {
        OpType::FuncDecl(fd) => Some((fd.func_name(), false, fd.visibility(), fd.signature())),
        OpType::FuncDefn(fd) => Some((fd.func_name(), true, fd.visibility(), fd.signature())),
        _ => None,
    }
}

fn gather_existing<H: HugrView + ?Sized>(h: &H) -> HashMap<&String, PubFuncs<H::Node>> {
    let left_if = |b| if b { Either::Left } else { Either::Right };
    h.children(h.module_root())
        .filter_map(|n| {
            link_sig(h, n).and_then(|(fname, is_defn, vis, sig)| {
                vis.is_public()
                    .then_some((fname, (left_if(is_defn)(n), sig)))
            })
        })
        .into_grouping_map()
        .aggregate(|acc: Option<PubFuncs<H::Node>>, name, (new, sig2)| {
            let Some((mut acc, sig1)) = acc else {
                return Some((new.map_right(|n| (n, vec![])), sig2));
            };
            assert_eq!(sig1, sig2, "Invalid Hugr: different signatures for {name}");
            let (Either::Right((_, decls)), Either::Right(ndecl)) = (&mut acc, &new) else {
                let err = match acc.is_left() && new.is_left() {
                    true => "Multiple FuncDefns",
                    false => "FuncDefn and FuncDecl(s)",
                };
                panic!("Invalid Hugr: {err} for {name}");
            };
            decls.push(*ndecl);
            Some((acc, sig2))
        })
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr.
///
/// For use with [HugrLinking::add_hugr_link_nodes] and [HugrLinking::add_view_link_nodes].
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
    use rstest::rstest;

    use super::{HugrLinking, NodeLinkingDirective, NodeLinkingError};
    use crate::builder::test::{dfg_calling_defn_decl, simple_dfg_hugr};
    use crate::builder::{
        Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::{ConstUsize, usize_t};
    use crate::hugr::hugrmut::test::check_calls_defn_decl;
    use crate::hugr::linking::{
        MultipleImplHandling, NameLinkingError, NameLinkingPolicy, SignatureConflictHandling,
    };
    use crate::hugr::{ValidationError, hugrmut::HugrMut};
    use crate::ops::{FuncDecl, OpTag, OpTrait, OpType, handle::NodeHandle};
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use crate::{Hugr, HugrView, Visibility, types::Signature};

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
            h.add_view_link_nodes(Some(h.entrypoint()), &insert, mod_children.clone())
                .unwrap();
            check_calls_defn_decl(&h, call1, call2);

            let mut h = simple_dfg_hugr();
            h.add_hugr_link_nodes(Some(h.entrypoint()), insert, mod_children)
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
        host.add_hugr_link_nodes(None, insert, dirvs).unwrap();
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
                .add_view_link_nodes(Some(h.entrypoint()), &insert, chmap.clone())
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
                h.add_hugr_link_nodes(Some(h.entrypoint()), insert.clone(), chmap.clone())
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
        let r = h.add_view_link_nodes(
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
        let r = h.add_view_link_nodes(
            Some(h.entrypoint()),
            &insert,
            HashMap::from([(inp, NodeLinkingDirective::add())]),
        );
        assert_eq!(r.err().unwrap(), NodeLinkingError::NotChildOfRoot(inp));
        assert_eq!(h, backup);

        let mut insert = insert;
        insert.set_entrypoint(defn);
        let r = h.add_view_link_nodes(
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
        let r = h.add_hugr_link_nodes(
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
        let r = h.add_hugr_link_nodes(
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
            .add_hugr_link_nodes(
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

    fn list_decls_defns<H: HugrView>(h: &H) -> (HashMap<H::Node, &str>, HashMap<H::Node, &str>) {
        let mut decls = HashMap::new();
        let mut defns = HashMap::new();
        for n in h.children(h.module_root()) {
            match h.get_optype(n) {
                OpType::FuncDecl(fd) => decls.insert(n, fd.func_name().as_str()),
                OpType::FuncDefn(fd) => defns.insert(n, fd.func_name().as_str()),
                _ => None,
            };
        }
        (decls, defns)
    }

    fn call_targets<H: HugrView>(h: &H) -> HashMap<H::Node, H::Node> {
        h.nodes()
            .filter(|n| h.get_optype(*n).is_call())
            .map(|n| {
                let tgt = h.static_source(n).expect(format!("For node {n}").as_str());
                (n, tgt)
            })
            .collect()
    }

    #[test]
    fn combines_decls_defn() {
        let i64_t = || INT_TYPES[6].to_owned();
        let foo_sig = Signature::new_endo(i64_t());
        let bar_sig = Signature::new(vec![i64_t(); 2], i64_t());
        let orig_target = {
            let mut fb =
                FunctionBuilder::new_vis("foo", foo_sig.clone(), Visibility::Public).unwrap();
            let mut mb = fb.module_root_builder();
            let bar1 = mb.declare("bar", bar_sig.clone().into()).unwrap();
            let bar2 = mb.declare("bar", bar_sig.clone().into()).unwrap(); // alias
            let [i] = fb.input_wires_arr();
            let [c] = fb.call(&bar1, &[], [i, i]).unwrap().outputs_arr();
            let r = fb.call(&bar2, &[], [i, c]).unwrap();
            let h = fb.finish_hugr_with_outputs(r.outputs()).unwrap();
            assert_eq!(
                list_decls_defns(&h),
                (
                    HashMap::from([(bar1.node(), "bar"), (bar2.node(), "bar")]),
                    HashMap::from([(h.entrypoint(), "foo")])
                )
            );
            h
        };

        let inserted = {
            let mut main_b = FunctionBuilder::new("main", Signature::new(vec![], i64_t())).unwrap();
            let mut mb = main_b.module_root_builder();
            let foo1 = mb.declare("foo", foo_sig.clone().into()).unwrap();
            let foo2 = mb.declare("foo", foo_sig.clone().into()).unwrap();
            let mut bar = mb
                .define_function_vis("bar", bar_sig.clone(), Visibility::Public)
                .unwrap();
            let res = bar
                .add_dataflow_op(IntOpDef::iadd.with_log_width(6), bar.input_wires())
                .unwrap();
            let bar = bar.finish_with_outputs(res.outputs()).unwrap();
            let i = main_b.add_load_value(ConstInt::new_u(6, 257).unwrap());
            let c = main_b.call(&foo1, &[], [i]).unwrap();
            let r = main_b.call(&foo2, &[], c.outputs()).unwrap();
            let h = main_b.finish_hugr_with_outputs(r.outputs()).unwrap();
            assert_eq!(
                list_decls_defns(&h),
                (
                    HashMap::from([(foo1.node(), "foo"), (foo2.node(), "foo")]),
                    HashMap::from([(h.entrypoint(), "main"), (bar.node(), "bar")])
                )
            );
            h
        };

        // Linking by name...neither of the looped-over params should make any difference:
        for sig_conflict in [
            SignatureConflictHandling::ErrorDontInsert,
            SignatureConflictHandling::UseBoth,
        ] {
            for multi_impls in [
                MultipleImplHandling::ErrorDontInsert,
                MultipleImplHandling::UseNew,
                MultipleImplHandling::UseExisting,
                MultipleImplHandling::UseBoth,
            ] {
                let pol = NameLinkingPolicy {
                    sig_conflict,
                    multi_impls,
                };
                let mut target = orig_target.clone();

                target.link_hugr(inserted.clone(), pol).unwrap();
                target.validate().unwrap();
                let (decls, defns) = list_decls_defns(&target);
                assert_eq!(decls, HashMap::new());
                assert_eq!(
                    defns.values().copied().sorted().collect_vec(),
                    ["bar", "foo", "main"]
                );
                let call_tgts = call_targets(&target);
                for (defn, name) in defns {
                    if name != "main" {
                        // Defns now have two calls each (was one to each alias)
                        assert_eq!(call_tgts.values().filter(|tgt| **tgt == defn).count(), 2);
                    }
                }
            }
        }
    }

    #[rstest]
    fn sig_conflict(
        #[values(false, true)] host_defn: bool,
        #[values(false, true)] inserted_defn: bool,
    ) {
        let mk_def_or_decl = |n, sig: Signature, defn| {
            let mut mb = ModuleBuilder::new();
            let node = if defn {
                let fb = mb.define_function_vis(n, sig, Visibility::Public).unwrap();
                let ins = fb.input_wires();
                fb.finish_with_outputs(ins).unwrap().node()
            } else {
                mb.declare(n, sig.into()).unwrap().node()
            };
            (mb.finish_hugr().unwrap(), node)
        };

        let old_sig = Signature::new_endo(usize_t());
        let (orig_host, orig_fn) = mk_def_or_decl("foo", old_sig.clone(), host_defn);
        let new_sig = Signature::new_endo(INT_TYPES[3].clone());
        let (inserted, inserted_fn) = mk_def_or_decl("foo", new_sig.clone(), inserted_defn);

        let mut pol = NameLinkingPolicy::err_on_conflict(MultipleImplHandling::ErrorDontInsert);
        let mut host = orig_host.clone();
        let res = host.link_hugr(inserted.clone(), pol.clone());
        assert_eq!(host, orig_host); // Did nothing
        assert_eq!(
            res.err(),
            Some(NameLinkingError::SignatureConflict {
                name: "foo".to_string(),
                src_node: inserted_fn,
                src_sig: Box::new(new_sig.into()),
                tgt_node: orig_fn,
                tgt_sig: Box::new(old_sig.into())
            })
        );

        pol.on_signature_conflict(SignatureConflictHandling::UseBoth);
        let node_map = host.link_hugr(inserted, pol).unwrap().node_map;
        assert_eq!(
            host.validate(),
            Err(ValidationError::DuplicateExport {
                link_name: "foo".to_string(),
                children: [orig_fn, node_map[&inserted_fn]]
            })
        );
    }

    #[rstest]
    #[case(MultipleImplHandling::UseNew, vec![11])]
    #[case(MultipleImplHandling::UseExisting, vec![5])]
    #[case(MultipleImplHandling::UseBoth, vec![5, 11])]
    #[case(MultipleImplHandling::ErrorDontInsert, vec![])]
    fn impl_conflict(#[case] multi_impls: MultipleImplHandling, #[case] expected: Vec<u64>) {
        fn build_hugr(cst: u64) -> Hugr {
            let mut mb = ModuleBuilder::new();
            let mut fb = mb
                .define_function_vis("foo", Signature::new(vec![], usize_t()), Visibility::Public)
                .unwrap();
            let c = fb.add_load_value(ConstUsize::new(cst));
            fb.finish_with_outputs([c]).unwrap();
            mb.finish_hugr().unwrap()
        }
        let backup = build_hugr(5);
        let mut host = backup.clone();
        let inserted = build_hugr(11);

        let mut pol = NameLinkingPolicy::keep_both_invalid();
        pol.on_multiple_impls(multi_impls);
        let res = host.link_hugr(inserted, pol);
        if multi_impls == MultipleImplHandling::ErrorDontInsert {
            assert!(matches!(res, Err(NameLinkingError::MultipleImpls(n, _, _)) if n == "foo"));
            assert_eq!(host, backup);
            return;
        }
        res.unwrap();
        let res = host.validate();
        if multi_impls == MultipleImplHandling::UseBoth {
            assert!(
                matches!(res, Err(ValidationError::DuplicateExport { link_name, .. }) if link_name == "foo")
            );
        } else {
            res.unwrap();
        }
        let func_consts = host
            .children(host.module_root())
            .filter(|n| host.get_optype(*n).is_func_defn())
            .map(|n| {
                host.children(n)
                    .filter_map(|ch| host.get_optype(ch).as_const())
                    .exactly_one()
                    .ok()
                    .unwrap()
                    .get_custom_value::<ConstUsize>()
                    .unwrap()
                    .value()
            })
            .collect_vec();
        assert_eq!(func_consts, expected);
    }
}

//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use itertools::{Either, Itertools};

use crate::{HugrView, Node, Visibility, core::HugrNode, ops::OpType, types::PolyFuncType};

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
        "Cannot insert children (e.g. {_0}) when already inserting whole Hugr (with module entrypoint)"
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
    /// Do not insert the node/subtree from the source, but for any inserted node
    /// with an ([EdgeKind::Function]) edge from it, change that edge to come from
    /// the specified existing node instead.
    ///
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

    /// The new node should replace the specified node (already existing)
    /// in the target. (Could lead to an invalid Hugr if they have
    /// different signatures.)
    pub fn replace(nodes: impl IntoIterator<Item = TN>) -> Self {
        Self::Add {
            replace: nodes.into_iter().collect(),
        }
    }
}

/// Describes ways to link a "Source" Hugr being inserted into a target Hugr.
pub enum NameLinkingPolicy {
    /// Do not use linking - just insert all functions from source Hugr into target.
    /// (This can lead to an invalid Hugr if there are name/signature conflicts on public functions).
    AddAll,
    /// Do not use linking - break any edges from functions in the source Hugr to the insert part.
    AddNone,
    /// Identify public functions in source and target Hugr by name.
    /// Multiple FuncDecls, and FuncDecl+FuncDefn pairs, with the same name and signature
    /// will be combined, taking the FuncDefn from either Hugr.
    LinkByName {
        /// If true, all private functions from the source hugr are inserted into the target.
        /// (Since these are private, name conflicts do not make the Hugr invalid.)
        /// If false, instead edges from said private functions to any inserted parts
        /// of the source Hugr will be broken, making the target Hugr invalid.
        copy_private_funcs: bool,
        /// How to handle cases where the same (public) name is present in both
        /// inserted and target Hugr but with different signatures.
        /// `true` means an error is raised and nothing is added to the target Hugr.
        /// `false` means the new function will be added alongside the existing one
        ///   - this will give an invalid Hugr (duplicate names).
        // NOTE there are other possible handling schemes, both where we don't insert the new function, both leading to an invalid Hugr:
        //   * don't insert but break edges --> Unconnected ports (or, replace and break existing edges)
        //   * use (or replace) the existing function --> incompatible ports
        // but given you'll need to patch the Hugr up afterwards, you can get there just
        // by setting this to `false` (and maybe removing one FuncDefn), or via explicit node linking.
        error_on_conflicting_sig: bool,
        /// How to handle cases where both target and inserted Hugr have a FuncDefn
        /// with the same name and signature.
        multi_impls: MultipleImplHandling,
        // TODO Renames to apply to public functions in the inserted Hugr. These take effect
        // before [error_on_conflicting_sig] or [take_existing_and_new_impls].
        // rename_map: HashMap<String, String>
    },
}

/// What to do when [NameLinkingPolicy::LinkByName] finds both target and inserted Hugr
/// have a [Visibility::Public] FuncDefn with the same name and signature.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum MultipleImplHandling {
    /// Do not perform insertion; raise an error instead
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
    Signatures {
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
    /// The default policy used by [HugrMut::insert_hugr].
    /// * All private functions are copied
    /// * If public functions have conflicting signatures, we keep both
    /// * If both existing and inserted Hugrs have same-signature public FuncDefns,
    ///   the newly inserted one replaces the original
    ///
    /// [HugrMut::insert_hugr]: crate::hugr::HugrMut::insert_hugr
    pub fn default_for_hugr() -> Self {
        Self::LinkByName {
            copy_private_funcs: true,
            error_on_conflicting_sig: false,
            multi_impls: MultipleImplHandling::UseNew,
        }
    }

    /// The default policy used by [HugrMut::insert_from_view].
    /// * Private functions are not copied, i.e. edges from them into the inserted portion, are disconnected in the target
    /// * If public functions have conflicting signatures, we keep both
    /// * If both existing and inserted Hugrs have same-signature public FuncDefns,
    ///   the original is used in place of the new.
    ///
    /// [HugrMut::insert_from_view]: crate::hugr::HugrMut::insert_from_view
    pub fn default_for_view() -> Self {
        Self::LinkByName {
            copy_private_funcs: false,
            error_on_conflicting_sig: false,
            multi_impls: MultipleImplHandling::UseExisting,
        }
    }

    /// Builds an explicit map of [NodeLinkingDirective]s that implements this policy for a given
    /// source (inserted) and target (inserted-into) Hugr.
    /// The map should be such that no [NodeLinkingError] will occur.
    #[allow(clippy::type_complexity)]
    pub fn to_node_linking<T: HugrView + ?Sized, S: HugrView + ?Sized>(
        &self,
        target: &T,
        source: &S,
    ) -> Result<NodeLinkingPolicy<S::Node, T::Node>, NameLinkingError<S::Node, T::Node>> {
        let existing = gather_existing(target);
        let mut res = NodeLinkingPolicy::new();

        for n in source.children(source.module_root()) {
            if let Some((name, is_defn, vis, sig)) = link_sig(source, n) {
                let mut dirv = NodeLinkingDirective::add();
                match self {
                    NameLinkingPolicy::AddAll => (),
                    NameLinkingPolicy::AddNone => continue,
                    NameLinkingPolicy::LinkByName {
                        copy_private_funcs,
                        error_on_conflicting_sig,
                        multi_impls,
                    } => {
                        if !vis.is_public() {
                            if !copy_private_funcs {
                                continue;
                            };
                        } else if let Some((ex_ns, ex_sig)) = existing.get(name) {
                            if sig == *ex_sig {
                                dirv = directive(name, n, is_defn, ex_ns, multi_impls)?
                            } else if *error_on_conflicting_sig {
                                return Err(NameLinkingError::Signatures {
                                    name: name.clone(),
                                    src_node: n,
                                    src_sig: Box::new(sig.clone()),
                                    tgt_node: *ex_ns.as_ref().left_or_else(|(n, _)| n),
                                    tgt_sig: Box::new((*ex_sig).clone()),
                                });
                            }
                        };
                    }
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
/// target Hugr (beneath the module root). For use with [insert_hugr_link_nodes] and
/// [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
pub type NodeLinkingPolicy<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

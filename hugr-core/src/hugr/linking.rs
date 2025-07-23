//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use crate::{HugrView, Node, Visibility, ops::OpType, types::PolyFuncType};

/// An error resulting from an [NodeLinkingDirective] passed to [insert_hugr_link_nodes]
/// or [insert_from_view_link_nodes].
///
/// [NodeLinkingDirective]: crate::hugr::hugrmut::NodeLinkingDirective
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
        /// If non-None, the specified node+subtree in the target Hugr will be removed,
        /// with any ([EdgeKind::Function]) edges from it changed to come from the newly-inserted node instead.
        replace: Option<TN>,
    },
    /// Do not insert the node/subtree from the source, but for any inserted node
    /// with an ([EdgeKind::Function]) edge from it, change that edge to come from
    /// the specified existing node instead.
    UseExisting(TN),
}

impl<TN> NodeLinkingDirective<TN> {
    /// Just add the node (and any subtree) into the target.
    /// (Could lead to an invalid Hugr if the target Hugr
    /// already has another with the same name and both are [Public])
    ///
    /// [Public]: crate::Visibility::Public
    pub const fn add() -> Self {
        Self::Add { replace: None }
    }

    /// The new node should replace the specified node (already existing)
    /// in the target. (Could lead to an invalid Hugr if they have
    /// different signatures.)
    pub const fn replace(node: TN) -> Self {
        Self::Add {
            replace: Some(node),
        }
    }
}

/// Describes ways to link a "Source" Hugr being inserted into a target Hugr.
/// Note SN = Source Node, TN=Target Node
pub enum NameLinkingPolicy {
    /// Do not use linking - just insert all functions from source Hugr into target.
    /// (This can lead to an invalid Hugr if there are name/signature conflicts on public functions).
    AddAll,
    /// Do not use linking - break any edges from functions in the source Hugr to the insert part.
    AddNone,
    /// Identify public functions in source and target Hugr by name.
    /// Multiple FuncDecls, and FuncDecl+FuncDefn pairs, with the same name and signature
    /// will be combined, taking the FuncDefn from either Hugr. This is the default.
    LinkByName {
        /// If true, all private functions from the source hugr are inserted into the target.
        /// (Since these are private, name conflicts do not make the Hugr invalid.)
        /// If false, instead edges from said private functions to any inserted parts
        /// of the source Hugr will be broken, making the target Hugr invalid.
        /// Defaults to `true`.
        copy_private_funcs: bool,
        /// How to handle cases where the same (public) name is present in both
        /// inserted and target Hugr but with different signatures.
        /// `true` means an error is raised and nothing is added to the target Hugr.
        /// `false` means the new function will be added alongside the existing one
        ///   - this will give an invalid Hugr (duplicate names). This is the default.
        // NOTE there are other possible handling schemes, both where we don't insert the new function, both leading to an invalid Hugr:
        //   * don't insert but break edges --> Unconnected ports (or, replace and break existing edges)
        //   * use (or replace) the existing function --> incompatible ports
        // but given you'll need to patch the Hugr up afterwards, you can get there just
        // by setting this to `false` (and maybe removing one FuncDefn), or via [NameLinkingPolicy::Explicit].
        error_on_conflicting_sig: bool,
        /// How to handle cases where both target and inserted Hugr have a FuncDefn
        /// with the same name and signature.
        multi_impls: MultipleImplHandling,
        // TODO Renames to apply to public functions in the inserted Hugr. These take effect
        // before [error_on_conflicting_sig] or [take_existing_and_new_impls].
        // rename_map: HashMap<String, String>
    },
}

impl Default for NameLinkingPolicy {
    fn default() -> Self {
        Self::LinkByName {
            copy_private_funcs: true,
            error_on_conflicting_sig: false,
            multi_impls: MultipleImplHandling::default(),
        }
    }
}

/// What to do when [NameLinkingPolicy::LinkByName] finds both target and inserted Hugr
/// have a [Visibility::Public] FuncDefn with the same name and signature.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub enum MultipleImplHandling {
    /// Do not perform insertion; raise an error instead
    ErrorDontInsert,
    /// Keep the implementation already in the target Hugr. (Edges in the source
    /// Hugr will be redirected to use the function from the target.)
    /// This is the default.
    #[default]
    UseExisting,
    /// Keep the implementation in the source Hugr. (Edges in the target Hugr
    /// will be redirected to use the funtion from the source; the previously-existing
    /// function in the target Hugr will be removed.)
    UseNew,
    /// Add the new function alongside the existing one in the target Hugr,
    /// preserving (separately) uses of both. (The Hugr will be invalid because
    /// of duplicate names.)
    UseBoth,
}

/// A conflict between [Visibility::Public] functions in source and target Hugrs.
/// (SN = Source Node, TN = Target Node)
// ALAN not quite right, "containing entrypoint" is not such an error
pub enum ConflictError<SN, TN> {
    /// Both source and target contained a [FuncDefn] (public and with same name
    /// and signature).
    MultipleImpls(SN, TN),
    /// Source and target containing public functions with conflicting signatures
    // should we indicate which were decls or defns? via an extra enum?
    Signatures(SN, Box<PolyFuncType>, TN, Box<PolyFuncType>),
    /// A [Visibility::Public] function in the source, whose body is being added
    /// to the target, contained the entrypoint (which needs to be added
    /// in a different place).
    AddFunctionContainingEntrypoint(SN, NodeLinkingDirective<TN>),
}

impl NameLinkingPolicy {
    /// Builds an explicit map of [NodeLinkingDirectives] that implements this policy for a given
    /// source (inserted) and target (inserted-into) Hugr.
    /// The map should be such that no [NodeLinkingError] will occur.
    #[allow(clippy::type_complexity)]
    pub fn to_node_linking<T: HugrView, S: HugrView>(
        &self,
        target: &T,
        source: &S,
    ) -> Result<NodeLinkingPolicy<S::Node, T::Node>, ConflictError<S::Node, T::Node>> {
        // Get some easy cases out of the way first
        let (copy_private, err_conf_sig, multi_impls) = match self {
            Self::AddAll => {
                return Ok(source
                    .children(source.module_root())
                    .map(|n| (n, NodeLinkingDirective::add()))
                    .collect());
            }
            Self::AddNone => return Ok(NodeLinkingPolicy::new()),
            Self::LinkByName {
                copy_private_funcs,
                error_on_conflicting_sig,
                multi_impls,
            } => (*copy_private_funcs, *error_on_conflicting_sig, *multi_impls),
        };

        let existing = target
            .children(target.module_root())
            .filter_map(|n| {
                link_sig(target, n).and_then(|(fname, is_defn, vis, sig)| {
                    vis.is_public().then_some((fname, (n, is_defn, sig)))
                })
            })
            .collect::<HashMap<_, _>>();
        let mut res = NodeLinkingPolicy::new();

        for n in source.children(source.module_root()) {
            let Some((name, is_defn, vis, sig)) = link_sig(source, n) else {
                continue;
            };
            let mut dirv = NodeLinkingDirective::add();
            if !vis.is_public() {
                if !copy_private {
                    continue;
                }
            } else if let Some(&(ex_n, ex_is_defn, ex_sig)) = existing.get(name) {
                if sig != ex_sig {
                    if err_conf_sig {
                        return Err(ConflictError::Signatures(
                            n,
                            Box::new(sig.clone()),
                            ex_n,
                            Box::new(ex_sig.clone()),
                        ));
                    }
                } else {
                    dirv = match (is_defn, ex_is_defn, multi_impls) {
                        (false, _, _) | (_, true, MultipleImplHandling::UseExisting) => {
                            NodeLinkingDirective::UseExisting(ex_n)
                        }
                        (_, false, _) | (_, _, MultipleImplHandling::UseNew) => {
                            NodeLinkingDirective::replace(ex_n)
                        }
                        (_, _, MultipleImplHandling::ErrorDontInsert) => {
                            return Err(ConflictError::MultipleImpls(n, ex_n));
                        }
                        (_, _, MultipleImplHandling::UseBoth) => NodeLinkingDirective::add(),
                    }
                }
            };
            res.insert(n, dirv);
        }
        let mut n = source.entrypoint();
        while let Some(p) = source.get_parent(n) {
            if p == source.module_root() {
                if let Some(dirv) = res.get(&n) {
                    // Would need to add entrypoint-subtree in two places.
                    // TODO allow if entrypoint *is* module child and inserting under
                    // target module-root ??
                    return Err(ConflictError::AddFunctionContainingEntrypoint(
                        n,
                        dirv.clone(),
                    ));
                }
            }
            n = p;
        }
        Ok(res)
    }
}

fn link_sig<H: HugrView>(h: &H, n: H::Node) -> Option<(&String, bool, &Visibility, &PolyFuncType)> {
    match h.get_optype(n) {
        OpType::FuncDecl(fd) => Some((fd.func_name(), false, fd.visibility(), fd.signature())),
        OpType::FuncDefn(fd) => Some((fd.func_name(), true, fd.visibility(), fd.signature())),
        _ => None,
    }
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr (beneath the module root). For use with [insert_hugr_link_nodes] and
/// [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
pub type NodeLinkingPolicy<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

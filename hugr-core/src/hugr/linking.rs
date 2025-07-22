//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use crate::Node;

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
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr (beneath the module root). For use with [insert_hugr_link_nodes] and
/// [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
pub type NodeLinkingPolicy<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

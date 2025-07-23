//! Directives and errors relating to linking Hugrs.

use std::fmt::Display;

/// An error resulting from an [InsertDefnMode] passed to [insert_hugr_link_nodes]
/// or [insert_from_view_link_nodes].
///
/// [InsertDefnMode]: crate::hugr::hugrmut::InsertDefnMode
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

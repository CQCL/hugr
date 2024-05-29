//! Validation routines for instantiations of a extension ops and types in a
//! Hugr.

use thiserror::Error;

use super::ExtensionSet;
use crate::{Node, Port};

/// Errors that can occur while validating a Hugr.
#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum ExtensionError {
    /// Missing lift node
    #[error("Extensions at target node {to:?} ({to_extensions}) exceed those at source {from:?} ({from_extensions})")]
    TgtExceedsSrcExtensions {
        from: Node,
        from_extensions: ExtensionSet,
        to: Node,
        to_extensions: ExtensionSet,
    },
    /// A version of the above which includes port info
    #[error("Extensions at target node {to:?} ({to_offset:?}) ({to_extensions}) exceed those at source {from:?} ({from_offset:?}) ({from_extensions})")]
    TgtExceedsSrcExtensionsAtPort {
        from: Node,
        from_offset: Port,
        from_extensions: ExtensionSet,
        to: Node,
        to_offset: Port,
        to_extensions: ExtensionSet,
    },
    /// Too many extension requirements coming from src
    #[error("Extensions at source node {from:?} ({from_extensions}) exceed those at target {to:?} ({to_extensions})")]
    SrcExceedsTgtExtensions {
        from: Node,
        from_extensions: ExtensionSet,
        to: Node,
        to_extensions: ExtensionSet,
    },
    /// A version of the above which includes port info
    #[error("Extensions at source node {from:?} ({from_offset:?}) ({from_extensions}) exceed those at target {to:?} ({to_offset:?}) ({to_extensions})")]
    SrcExceedsTgtExtensionsAtPort {
        from: Node,
        from_offset: Port,
        from_extensions: ExtensionSet,
        to: Node,
        to_offset: Port,
        to_extensions: ExtensionSet,
    },
    #[error("Missing input extensions for node {0:?}")]
    MissingInputExtensions(Node),
    #[error("Extensions of I/O node ({child:?}) {child_extensions:?} don't match those expected by parent node ({parent:?}): {parent_extensions:?}")]
    ParentIOExtensionMismatch {
        parent: Node,
        parent_extensions: ExtensionSet,
        child: Node,
        child_extensions: ExtensionSet,
    },
}

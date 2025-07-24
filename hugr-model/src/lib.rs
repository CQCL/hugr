//! The data model of the HUGR intermediate representation.
//! This crate defines data structures that capture the structure of a HUGR graph and
//! all its associated information in a form that can be stored on disk. The data structures
//! are not designed for efficient traversal or modification, but for simplicity and serialization.
mod capnp;

pub mod v0;

// This is required here since the generated code assumes it's in the package root.
use capnp::hugr_v0_capnp;
use derive_more::derive::Display;

/// A version number.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[display("{major}.{minor}")]
pub struct Version {
    /// The major part of the version.
    pub major: u32,
    /// The minor part of the version.
    pub minor: u32,
}

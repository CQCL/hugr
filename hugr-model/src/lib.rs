//! The data model of the HUGR intermediate representation.
//! This crate defines data structures that capture the structure of a HUGR graph and
//! all its associated information in a form that can be stored on disk. The data structures
//! are not designed for efficient traversal or modification, but for simplicity and serialization.
mod capnp;

pub mod v0;

// This is required here since the generated code assumes it's in the package root.
use capnp::hugr_v0_capnp;

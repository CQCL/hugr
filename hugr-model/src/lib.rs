//! The data model of the HUGR intermediate representation.
//! This crate defines data structures that capture the structure of a HUGR graph and
//! all its associated information in a form that can be stored on disk. The data structures
//! are not designed for efficient traversal or modification, but for simplicity and serialization.
pub mod serialization;
pub mod v0;

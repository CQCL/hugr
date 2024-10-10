//! The data model of the HUGR intermediate representation.
//! This crate defines data structures that capture the structure of a HUGR graph and
//! all its associated information in a form that can be stored on disk. The data structures
//! are not designed for efficient traversal or modification, but for simplicity and serialization.
pub mod v0;

pub(crate) mod hugr_v0_capnp {
    include!(concat!(env!("OUT_DIR"), "/hugr_v0_capnp.rs"));
}

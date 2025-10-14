//! The data model of the HUGR intermediate representation.
//!
//! This crate defines data structures that capture the structure of a HUGR graph and
//! all its associated information in a form that can be stored on disk. The data structures
//! are not designed for efficient traversal or modification, but for simplicity and serialization.
//!
//! This crate supports version `
#![doc = include_str!("../FORMAT_VERSION")]
//! ` of the HUGR model format.
mod capnp;

pub mod v0;

use std::sync::LazyLock;

// This is required here since the generated code assumes it's in the package root.
use capnp::hugr_v0_capnp;

/// The current version of the HUGR model format.
pub static CURRENT_VERSION: LazyLock<semver::Version> = LazyLock::new(|| {
    // We allow non-zero patch versions, but ignore them for compatibility checks.
    let v = semver::Version::parse(include_str!("../FORMAT_VERSION").trim())
        .expect("`FORMAT_VERSION` in `hugr-model` contains version that fails to parse");
    assert!(
        v.pre.is_empty(),
        "`FORMAT_VERSION` in `hugr-model` should not have a pre-release version"
    );
    v
});

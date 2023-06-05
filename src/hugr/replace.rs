//! Rewrite operations on the HUGR.

#[allow(clippy::module_inception)] // TODO: Rename?
pub mod rewrite;

pub use rewrite::{OpenHugr, Rewrite, RewriteError};

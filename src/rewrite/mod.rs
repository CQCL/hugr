//! Pattern matching and rewrite operations on the HUGR.

pub mod pattern;
#[allow(clippy::module_inception)] // TODO: Rename?
pub mod rewrite;

pub use rewrite::{OpenHugr, Rewrite, RewriteError};

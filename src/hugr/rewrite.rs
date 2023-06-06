//! Replace operations on the HUGR.

#[allow(clippy::module_inception)] // TODO: Rename?
pub mod replace;
use crate::Hugr;
pub use replace::{OpenHugr, Replace, ReplaceError};

/// An operation that can be applied to mutate a Hugr
pub trait Rewrite<E> {
    /// Mutate the specified Hugr, or fail with an error.
    /// Implementations are strongly encouraged not to mutate the Hugr
    /// if they return an Err.
    fn apply(self, h: &mut Hugr) -> Result<(), E>;
}

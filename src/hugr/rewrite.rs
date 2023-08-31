//! Rewrite operations on the HUGR - replacement, outlining, etc.

pub mod insert_identity;
pub mod outline_cfg;
pub mod simple_replace;
use std::mem;

use crate::Hugr;
pub use simple_replace::{SimpleReplacement, SimpleReplacementError};

/// An operation that can be applied to mutate a Hugr
pub trait Rewrite {
    /// The type of Error with which this Rewrite may fail
    type Error: std::error::Error;
    /// The type returned on successful application of the rewrite.
    type ApplyResult;
    /// If `true`, [self.apply]'s of this rewrite guarantee that they do not mutate the Hugr when they return an Err.
    /// If `false`, there is no guarantee; the Hugr should be assumed invalid when Err is returned.
    const UNCHANGED_ON_FAILURE: bool;

    /// Checks whether the rewrite would succeed on the specified Hugr.
    /// If this call succeeds, [self.apply] should also succeed on the same `h`
    /// If this calls fails, [self.apply] would fail with the same error.
    fn verify(&self, h: &Hugr) -> Result<(), Self::Error>;

    /// Mutate the specified Hugr, or fail with an error.
    /// Returns [`Self::ApplyResult`] if successful.
    /// If [self.unchanged_on_failure] is true, then `h` must be unchanged if Err is returned.
    /// See also [self.verify]
    /// # Panics
    /// May panic if-and-only-if `h` would have failed [Hugr::validate]; that is,
    /// implementations may begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply(self, h: &mut Hugr) -> Result<Self::ApplyResult, Self::Error>;
}

/// Wraps any rewrite into a transaction (i.e. that has no effect upon failure)
pub struct Transactional<R> {
    underlying: R,
}

// Note we might like to constrain R to Rewrite<unchanged_on_failure=false> but this
// is not yet supported, https://github.com/rust-lang/rust/issues/92827
impl<R: Rewrite> Rewrite for Transactional<R> {
    type Error = R::Error;
    type ApplyResult = R::ApplyResult;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &Hugr) -> Result<(), Self::Error> {
        self.underlying.verify(h)
    }

    fn apply(self, h: &mut Hugr) -> Result<Self::ApplyResult, Self::Error> {
        if R::UNCHANGED_ON_FAILURE {
            return self.underlying.apply(h);
        }
        let backup = h.clone();
        let r = self.underlying.apply(h);
        if r.is_err() {
            // drop the old h, it was undefined
            let _ = mem::replace(h, backup);
        }
        r
    }
}

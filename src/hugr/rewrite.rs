//! Replace operations on the HUGR.

#[allow(clippy::module_inception)] // TODO: Rename?
pub mod replace;
use std::mem;

use crate::Hugr;
pub use replace::{OpenHugr, Replace, ReplaceError};

/// An operation that can be applied to mutate a Hugr
pub trait Rewrite<E> {
    /// Check that this operation can be applied to the specified Hugr in a way
    /// that will not destroy it.  See contract for [self.apply].
    fn may_fail_destructively(&self, _h: &Hugr) -> bool {
        true // The least guarantee possible.
    }

    /// Mutate the specified Hugr, or fail with an error.
    /// [self.may_fail_destructively] may be used before calling in order to rule out destructive failure.
    /// If [self.may_fail_destructively] returns true, or `h` has since been mutated,
    /// or `h` does not [Hugr::validate], then no guarantees are given here.
    /// However if [self.may_fail_destructively] returns false, `h` has not been mutated since then,
    /// and `h` [Hugr::validate]s, then this method must *either* succeed, *or* fail leaving `h` unchanged.
    fn apply(self, h: &mut Hugr) -> Result<(), E>;
}

/// Wraps any rewrite into a transaction (i.e. that has no effect upon failure)
pub struct Transactional<R> {
    underlying: R,
}

impl<E, R: Rewrite<E>> Rewrite<E> for Transactional<R> {
    fn may_fail_destructively(&self, _h: &Hugr) -> bool {
        false
    }

    fn apply(self, h: &mut Hugr) -> Result<(), E> {
        // note that if underlying.may_fail_destructively(h) is false, we don't need a backup...
        let backup = h.clone();
        let r = self.underlying.apply(h);
        if let Err(_) = r {
            // drop the old h, it was undefined
            let _ = mem::replace(h, backup);
        }
        r
    }
}

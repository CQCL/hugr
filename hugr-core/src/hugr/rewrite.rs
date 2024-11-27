//! Rewrite operations on the HUGR - replacement, outlining, etc.

pub mod consts;
pub mod inline_dfg;
pub mod insert_identity;
pub mod outline_cfg;
pub mod replace;
pub mod simple_replace;

use crate::{Hugr, HugrView, Node};
pub use simple_replace::{SimpleReplacement, SimpleReplacementError};

use super::HugrMut;

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
    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error>;

    /// Mutate the specified Hugr, or fail with an error.
    /// Returns [`Self::ApplyResult`] if successful.
    /// If [self.unchanged_on_failure] is true, then `h` must be unchanged if Err is returned.
    /// See also [self.verify]
    /// # Panics
    /// May panic if-and-only-if `h` would have failed [Hugr::validate]; that is,
    /// implementations may begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error>;

    /// Returns a set of nodes referenced by the rewrite. Modifying any of these
    /// nodes will invalidate it.
    ///
    /// Two `impl Rewrite`s can be composed if their invalidation sets are
    /// disjoint.
    fn invalidation_set(&self) -> impl Iterator<Item = Node>;
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

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        self.underlying.verify(h)
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        if R::UNCHANGED_ON_FAILURE {
            return self.underlying.apply(h);
        }
        // Try to backup just the contents of this HugrMut.
        let mut backup = Hugr::new(h.root_type().clone());
        backup.insert_from_view(backup.root(), h);
        let r = self.underlying.apply(h);
        if r.is_err() {
            // Try to restore backup.
            h.replace_op(h.root(), backup.root_type().clone())
                .expect("The root replacement should always match the old root type");
            while let Some(child) = h.first_child(h.root()) {
                h.remove_node(child);
            }
            h.insert_from_view(h.root(), &backup);
        }
        r
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        self.underlying.invalidation_set()
    }
}

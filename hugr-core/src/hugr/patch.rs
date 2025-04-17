//! Rewrite operations on the HUGR - replacement, outlining, etc.

pub mod consts;
pub mod inline_call;
pub mod inline_dfg;
pub mod insert_identity;
pub mod outline_cfg;
mod port_types;
pub mod replace;
pub mod simple_replace;

use crate::{Hugr, HugrView, Node};
pub use port_types::{BoundaryPort, HostPort, ReplacementPort};
pub use simple_replace::{SimpleReplacement, SimpleReplacementError};

use super::HugrMut;

/// Verify that a patch application would succeed.
pub trait VerifyPatch {
    /// The type of Error with which this Rewrite may fail
    type Error: std::error::Error;

    /// The node type of the HugrView that this patch applies to.
    type Node;

    /// Checks whether the rewrite would succeed on the specified Hugr.
    /// If this call succeeds, [self.apply] should also succeed on the same `h`
    /// If this calls fails, [self.apply] would fail with the same error.
    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error>;

    /// Returns a set of nodes referenced by the rewrite. Modifying any of these
    /// nodes will invalidate it.
    ///
    /// Two `impl Rewrite`s can be composed if their invalidation sets are
    /// disjoint.
    fn invalidation_set(&self) -> impl Iterator<Item = Self::Node>;
}

/// A patch that can be applied to a Hugr of type `H`.
///
/// ### When to use
///
/// Use this trait whenever possible in bounds for the most generality. Note
/// that this will require specifying which types `H` the patch applies to.
///
/// ### When to implement
///
/// For `H: HugrMut`, prefer implementing [`ApplyPatchHugrMut`] instead. This
/// will automatically implement this trait.
pub trait ApplyPatch<H: HugrView>: VerifyPatch<Node = H::Node> {
    /// The type returned on successful application of the rewrite.
    type Outcome;

    /// If `true`, [self.apply]'s of this rewrite guarantee that they do not mutate the Hugr when they return an Err.
    /// If `false`, there is no guarantee; the Hugr should be assumed invalid when Err is returned.
    const UNCHANGED_ON_FAILURE: bool;

    /// Mutate the specified Hugr, or fail with an error.
    ///
    /// Returns [`Self::Outcome`] if successful.
    /// If [self.unchanged_on_failure] is true, then `h` must be unchanged if Err is returned.
    /// See also [self.verify]
    /// # Panics
    /// May panic if-and-only-if `h` would have failed [Hugr::validate]; that is,
    /// implementations may begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply(self, h: &mut H) -> Result<Self::Outcome, Self::Error>;
}

/// A patch that can be applied to any HugrMut.
///
/// This trait is a generalisation of [`ApplyPatch`] in that it guarantees that
/// the patch can be applied to any type implementing [`HugrMut`].
///
/// ### When to use
///
/// Prefer using the more general [`ApplyPatch`] trait in bounds where the
/// type `H` is known. Resort to this trait if patches must be applicable to
/// any [`HugrMut`] instance.
///
/// ### When to implement
///
/// Always implement this trait when possible, to define how a patch is applied
/// to any type implementing [`HugrMut`]. A blanket implementation ensures that
/// any type implementing this trait also implements [`ApplyPatch`].
pub trait ApplyPatchHugrMut: VerifyPatch<Node = Node> {
    /// The type returned on successful application of the rewrite.
    type Outcome;

    /// If `true`, [self.apply]'s of this rewrite guarantee that they do not mutate the Hugr when they return an Err.
    /// If `false`, there is no guarantee; the Hugr should be assumed invalid when Err is returned.
    const UNCHANGED_ON_FAILURE: bool;

    /// Mutate the specified Hugr, or fail with an error.
    ///
    /// Returns [`Self::Outcome`] if successful.
    /// If [self.unchanged_on_failure] is true, then `h` must be unchanged if Err is returned.
    /// See also [self.verify]
    /// # Panics
    /// May panic if-and-only-if `h` would have failed [Hugr::validate]; that is,
    /// implementations may begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply_hugr_mut(self, h: &mut impl HugrMut) -> Result<Self::Outcome, Self::Error>;
}

impl<H: HugrMut, R: ApplyPatchHugrMut> ApplyPatch<H> for R {
    type Outcome = R::Outcome;
    const UNCHANGED_ON_FAILURE: bool = R::UNCHANGED_ON_FAILURE;

    fn apply(self, h: &mut H) -> Result<Self::Outcome, Self::Error> {
        self.apply_hugr_mut(h)
    }
}

/// Wraps any rewrite into a transaction (i.e. that has no effect upon failure)
pub struct Transactional<R> {
    underlying: R,
}

impl<R: VerifyPatch> VerifyPatch for Transactional<R> {
    type Error = R::Error;
    type Node = R::Node;

    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error> {
        self.underlying.verify(h)
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = Self::Node> {
        self.underlying.invalidation_set()
    }
}

// Note we might like to constrain R to Rewrite<unchanged_on_failure=false> but this
// is not yet supported, https://github.com/rust-lang/rust/issues/92827
impl<R: ApplyPatchHugrMut> ApplyPatchHugrMut for Transactional<R> {
    type Outcome = R::Outcome;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(
        self,
        h: &mut impl HugrMut<Node = Self::Node>,
    ) -> Result<Self::Outcome, Self::Error> {
        if R::UNCHANGED_ON_FAILURE {
            return self.underlying.apply_hugr_mut(h);
        }
        // Try to backup just the contents of this HugrMut.
        let mut backup = Hugr::new(h.root_type().clone());
        backup.insert_from_view(backup.root(), h);
        let r = self.underlying.apply_hugr_mut(h);
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
}

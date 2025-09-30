//! Rewrite operations on the HUGR - replacement, outlining, etc.

pub mod consts;
pub mod inline_call;
pub mod inline_dfg;
pub mod insert_cut;
pub mod insert_identity;
pub mod outline_cfg;
pub mod peel_loop;
mod port_types;
pub mod replace;
pub mod simple_replace;

use crate::HugrView;
use crate::core::HugrNode;
use crate::hugr::views::NodesIter;
use itertools::Itertools;
pub use port_types::{BoundaryPort, HostPort, ReplacementPort};
pub use simple_replace::{SimpleReplacement, SimpleReplacementError};

use super::HugrMut;
use super::views::ExtractionResult;

/// Verify that a patch application would succeed.
// TODO: This trait should be parametrised on `H: NodesIter` to match the
// generality of [`Patch`], see https://github.com/CQCL/hugr/issues/2546.
pub trait PatchVerification {
    /// The type of Error with which this Rewrite may fail
    type Error: std::error::Error;

    /// The node type of the `HugrView` that this patch applies to.
    type Node: HugrNode;

    /// Checks whether the rewrite would succeed on the specified Hugr.
    /// If this call succeeds, [`Patch::apply`] should also succeed on the same
    /// `h` If this calls fails, [`Patch::apply`] would fail with the same
    /// error.
    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error>;

    /// The nodes invalidated by the rewrite. Deprecated: implement
    /// [Self::invalidated_nodes] instead. The default returns the empty
    /// iterator; this should be fine as there are no external calls.
    #[deprecated(note = "Use/implement invalidated_nodes instead", since = "0.20.2")]
    fn invalidation_set(&self) -> impl Iterator<Item = Self::Node> {
        std::iter::empty()
    }

    /// Returns the nodes removed or altered by the rewrite. Modifying any of these
    /// nodes will invalidate the rewrite.
    ///
    /// Two `impl Rewrite`s can be composed if their `invalidated_nodes` are
    /// disjoint.
    fn invalidated_nodes(
        &self,
        h: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        let _ = h;
        #[expect(deprecated)]
        self.invalidation_set()
    }
}

/// A patch that can be applied to a mutable Hugr of type `H`.
///
/// ### When to use
///
/// Use this trait whenever possible in bounds for the most generality. Note
/// that this will require specifying which type `H` the patch applies to.
///
/// ### When to implement
///
/// For patches that work on any `H: HugrMut`, prefer implementing
/// [`PatchHugrMut`] instead. This will automatically implement this trait.
pub trait Patch<H: NodesIter>: PatchVerification<Node = H::Node> {
    /// The type returned on successful application of the rewrite.
    type Outcome;

    /// If `true`, [`Patch::apply`]'s of this rewrite guarantee that they do not
    /// mutate the Hugr when they return an Err. If `false`, there is no
    /// guarantee; the Hugr should be assumed invalid when Err is returned.
    const UNCHANGED_ON_FAILURE: bool;

    /// Mutate the specified Hugr, or fail with an error.
    ///
    /// Returns [`Self::Outcome`] if successful. If
    /// [`Patch::UNCHANGED_ON_FAILURE`] is true, then `h` must be unchanged if Err
    /// is returned. See also [`PatchVerification::verify`]
    ///
    /// # Panics
    ///
    /// May panic if-and-only-if `h` would have failed
    /// [`Hugr::validate`][crate::Hugr::validate]; that is, implementations may
    /// begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply(self, h: &mut H) -> Result<Self::Outcome, Self::Error>;
}

/// A patch that can be applied to any [`HugrMut`].
///
/// This trait is a generalisation of [`Patch`] in that it guarantees that
/// the patch can be applied to any type implementing [`HugrMut`].
///
/// ### When to use
///
/// Prefer using the more general [`Patch`] trait in bounds where the
/// type `H` is known. Resort to this trait if patches must be applicable to
/// any [`HugrMut`] instance.
///
/// ### When to implement
///
/// Always implement this trait when possible, to define how a patch is applied
/// to any type implementing [`HugrMut`]. A blanket implementation ensures that
/// any type implementing this trait also implements [`Patch`].
pub trait PatchHugrMut: PatchVerification {
    /// The type returned on successful application of the rewrite.
    type Outcome;

    /// If `true`, [self.apply]'s of this rewrite guarantee that they do not
    /// mutate the Hugr when they return an Err. If `false`, there is no
    /// guarantee; the Hugr should be assumed invalid when Err is returned.
    const UNCHANGED_ON_FAILURE: bool;

    /// Mutate the specified Hugr, or fail with an error.
    ///
    /// Returns [`Self::Outcome`] if successful. If [`self.unchanged_on_failure`]
    /// is true, then `h` must be unchanged if Err is returned. See also
    /// [self.verify]
    ///
    /// # Panics
    ///
    /// May panic if-and-only-if `h` would have failed
    /// [`Hugr::validate`][crate::Hugr::validate]; that is, implementations may
    /// begin with `assert!(h.validate())`, with `debug_assert!(h.validate())`
    /// being preferred.
    fn apply_hugr_mut(
        self,
        h: &mut impl HugrMut<Node = Self::Node>,
    ) -> Result<Self::Outcome, Self::Error>;
}

impl<R: PatchHugrMut, H: HugrMut<Node = R::Node>> Patch<H> for R {
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

impl<R: PatchVerification> PatchVerification for Transactional<R> {
    type Error = R::Error;
    type Node = R::Node;

    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error> {
        self.underlying.verify(h)
    }

    #[inline]
    fn invalidated_nodes(
        &self,
        h: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        self.underlying.invalidated_nodes(h)
    }
}

// Note we might like to constrain R to Rewrite<unchanged_on_failure=false> but
// this is not yet supported, https://github.com/rust-lang/rust/issues/92827
impl<R: PatchHugrMut> PatchHugrMut for Transactional<R> {
    type Outcome = R::Outcome;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(
        self,
        h: &mut impl HugrMut<Node = Self::Node>,
    ) -> Result<Self::Outcome, Self::Error> {
        if R::UNCHANGED_ON_FAILURE {
            return self.underlying.apply_hugr_mut(h);
        }
        // Backup the whole Hugr.
        // Temporarily move the entrypoint so every node gets copied.
        //
        // TODO: This requires a full graph copy on each application.
        // Ideally we'd be able to just restore modified nodes, perhaps using a `HugrMut` wrapper
        // that keeps track of them.
        let (backup, backup_map) = h.extract_hugr(h.module_root());
        let backup_root = backup_map.extracted_node(h.module_root());
        let backup_entrypoint = backup_map.extracted_node(h.entrypoint());

        let r = self.underlying.apply_hugr_mut(h);
        if r.is_err() {
            // Restore the backup.
            let h_root = h.module_root();
            for f in h.children(h_root).collect_vec() {
                h.remove_subtree(f);
            }
            let insert_map = h.insert_hugr(h_root, backup).node_map;
            let inserted_root = insert_map[&backup_root];
            let inserted_entrypoint = insert_map[&backup_entrypoint];
            h.remove_node(h_root);
            h.set_module_root(inserted_root);
            h.set_entrypoint(inserted_entrypoint);
        }
        r
    }
}

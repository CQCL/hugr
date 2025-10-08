#![doc(hidden)] // TODO: remove when stable

//! Persistent data structure for HUGR mutations.
//!
//! This crate provides a persistent data structure [`PersistentHugr`] that
//! implements [`HugrView`](hugr_core::HugrView). A persistent data structure:
//!  - is cheap to clone,
//!  - shares as much of the underlying data with other versions of the data
//!    structure as possible, even as it gets mutated (unlike, say the Rust
//!    `Cow` type),
//!  - in our case, multiple versions of the same data structure can even be
//!    merged together (with a notion of "compatibility" between versions).
//!
//! Persistent data structures are sometimes called immutable data structures,
//! as mutations can always be implemented as cheap clone + cheap mutation.
//! Mutations to the data are stored persistently as a set of [`Commit`]s along
//! with the dependencies between the commits.
//!
//! As a result of persistency, the entire mutation history of a HUGR can be
//! traversed and references to previous versions of the data remain valid even
//! as the HUGR graph is "mutated" by applying patches: the patches are in
//! effect added to the history as new commits.
//!
//! Multiple [`PersistentHugr`] may use commits from a common
//! [`CommitStateSpace`]: [`PersistentHugr`]s derived from mutations of previous
//! [`PersistentHugr`]s will always share a common state space. A [`Walker`]
//! can be used on a state space to explore all possible [`PersistentHugr`]s
//! that can be created from commits in the space.
//!
//! ## Overlapping commits
//!
//! In general, [`CommitStateSpace`] may contain overlapping commits. Such
//! mutations are mutually exclusive as they modify the same nodes. It is
//! therefore not possible to apply all commits in a [`CommitStateSpace`]
//! simultaneously. A [`PersistentHugr`] on the other hand always contains a
//! a subset of the commits of a [`CommitStateSpace`], with the guarantee
//! that they are all non-overlapping, compatible commits. By applying all
//! commits in a [`PersistentHugr`], we can materialize a
//! [`Hugr`](hugr_core::Hugr). Traversing the materialized HUGR is equivalent to
//! using the [`HugrView`](hugr_core::HugrView) implementation of the
//! corresponding [`PersistentHugr`].
//!
//! ## Summary of data types
//!
//! - [`Commit`] A modification to a [`Hugr`](hugr_core::Hugr) (currently a
//!   [`SimpleReplacement`](hugr_core::SimpleReplacement)) that forms the atomic
//!   unit of change for a [`PersistentHugr`] (like a commit in git). This is a
//!   reference-counted value that is cheap to clone and will be freed when all
//!   [`PersistentHugr`] and [`CommitStateSpace`] that refer to it are dropped.
//! - [`PersistentHugr`] A data structure that implements
//!   [`HugrView`][hugr_core::HugrView] and can be used as a drop-in replacement
//!   for a [`Hugr`][hugr_core::Hugr] for read-only access and mutations through
//!   the [`PatchVerification`](hugr_core::hugr::patch::PatchVerification) and
//!   [`Patch`](hugr_core::hugr::Patch) traits. Mutations are stored as a
//!   history of commits. Unlike [`CommitStateSpace`], it maintains the
//!   invariant that all contained commits are compatible with eachother.
//! - [`CommitStateSpace`] A registry of all commits within the
//!   [`PersistentHugr`]s of the state space. Includes the base HUGR and any
//!   number of possibly incompatible (overlapping) commits. Unlike a
//!   [`PersistentHugr`], a state space can contain mutually exclusive commits.
//!
//! ## Usage
//!
//! A [`PersistentHugr`] can be created from a base HUGR using
//! [`PersistentHugr::with_base`]. Replacements can then be applied to it
//! using [`PersistentHugr::add_replacement`]. Alternatively, if you already
//! have a populated state space, use [`PersistentHugr::try_new`] to create a
//! new HUGR with those commits.
//!
//! To obtain a [`PersistentHugr`] from your state space, use
//! [`CommitStateSpace::try_create`]. A [`PersistentHugr`] can always be
//! materialized into a [`Hugr`][hugr_core::Hugr] type using
//! [`PersistentHugr::to_hugr`].

pub mod commit;
pub use commit::{Commit, InvalidCommit};

mod parents_view;

pub mod persistent_hugr;
pub use persistent_hugr::PersistentHugr;

pub mod state_space;
use state_space::CommitData;
pub use state_space::{CommitId, CommitStateSpace, PatchNode};

pub mod subgraph;
pub use subgraph::PinnedSubgraph;

mod trait_impls;

pub mod walker;
pub use walker::Walker;

mod wire;
pub use wire::PersistentWire;

/// A replacement operation that can be applied to a [`PersistentHugr`].
pub type PersistentReplacement = hugr_core::SimpleReplacement<PatchNode>;

pub mod serial {
    //! Serialized formats for commits, state spaces and persistent HUGRs.
    pub use super::persistent_hugr::serial::*;
    pub use super::state_space::serial::*;
}

#[cfg(test)]
mod tests;

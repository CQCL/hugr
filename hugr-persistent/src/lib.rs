#![doc(hidden)] // TODO: remove when stable

//! Persistent data structure for HUGR mutations.
//!
//! This crate provides a persistent data structure [`PersistentHugr`] that
//! implements [`HugrView`](hugr_core::HugrView); mutations to the data are
//! stored persistently as a set of [`Commit`]s along with the dependencies
//! between the commits.
//!
//! As a result of persistency, the entire mutation history of a HUGR can be
//! traversed and references to previous versions of the data remain valid even
//! as the HUGR graph is "mutated" by applying patches: the patches are in
//! effect added to the history as new commits.
//!
//! The data structure underlying [`PersistentHugr`], which stores the history
//! of all commits, is [`CommitStateSpace`]. Multiple [`PersistentHugr`] can be
//! stored within a single [`CommitStateSpace`], which allows for the efficient
//! exploration of the space of all possible graph rewrites.
//!
//! ## Overlapping commits
//!
//! In general, [`CommitStateSpace`] may contain overlapping commits. Such
//! mutations are mutually exclusive as they modify the same nodes. It is
//! therefore not possible to apply all commits in a [`CommitStateSpace`]
//! simultaneously. A [`PersistentHugr`] on the other hand always corresponds to
//! a subgraph of a [`CommitStateSpace`] that is guaranteed to contain only
//! non-overlapping, compatible commits. By applying all commits in a
//! [`PersistentHugr`], we can materialize a [`Hugr`](hugr_core::Hugr).
//! Traversing the materialized HUGR is equivalent to using the
//! [`HugrView`](hugr_core::HugrView) implementation of the corresponding
//! [`PersistentHugr`].
//!
//! ## Summary of data types
//!
//! - [`Commit`] A modification to a [`Hugr`](hugr_core::Hugr) (currently a
//!   [`SimpleReplacement`](hugr_core::SimpleReplacement)) that forms the atomic
//!   unit of change for a [`PersistentHugr`] (like a commit in git). This is a
//!   reference-counted value that is cheap to clone and will be freed when the
//!   last reference is dropped.
//! - [`PersistentHugr`] A data structure that implements
//!   [`HugrView`][hugr_core::HugrView] and can be used as a drop-in replacement
//!   for a [`Hugr`][hugr_core::Hugr] for read-only access and mutations through
//!   the [`PatchVerification`](hugr_core::hugr::patch::PatchVerification) and
//!   [`Patch`](hugr_core::hugr::Patch) traits. Mutations are stored as a
//!   history of commits. Unlike [`CommitStateSpace`], it maintains the
//!   invariant that all contained commits are compatible with eachother.
//! - [`CommitStateSpace`] Stores commits, recording the dependencies between
//!   them. Includes the base HUGR and any number of possibly incompatible
//!   (overlapping) commits. Unlike a [`PersistentHugr`], a state space can
//!   contain mutually exclusive commits.
//!
//! ## Usage
//!
//! A [`PersistentHugr`] can be created from a base HUGR using
//! [`PersistentHugr::with_base`]. Replacements can then be applied to it
//! using [`PersistentHugr::add_replacement`]. Alternatively, if you already
//! have a populated state space, use [`PersistentHugr::try_new`] to create a
//! new HUGR with those commits.
//!
//! Add a sequence of commits to a state space by merging a [`PersistentHugr`]
//! into it using [`CommitStateSpace::extend`] or directly using
//! [`CommitStateSpace::try_add_commit`].
//!
//! To obtain a [`PersistentHugr`] from your state space, use
//! [`CommitStateSpace::try_extract_hugr`]. A [`PersistentHugr`] can always be
//! materialized into a [`Hugr`][hugr_core::Hugr] type using
//! [`PersistentHugr::to_hugr`].

mod parents_view;
mod persistent_hugr;
mod resolver;
pub mod state_space;
pub mod subgraph;
mod trait_impls;
pub mod walker;
mod wire;

pub use persistent_hugr::{Commit, PersistentHugr};
pub use resolver::{PointerEqResolver, Resolver, SerdeHashResolver};
pub use state_space::{CommitId, CommitStateSpace, InvalidCommit, PatchNode};
pub use subgraph::PinnedSubgraph;
pub use walker::Walker;
pub use wire::PersistentWire;

/// A replacement operation that can be applied to a [`PersistentHugr`].
pub type PersistentReplacement = hugr_core::SimpleReplacement<PatchNode>;

use persistent_hugr::find_conflicting_node;
use state_space::CommitData;

pub mod serial {
    //! Serialized formats for commits, state spaces and persistent HUGRs.
    pub use super::persistent_hugr::serial::*;
    pub use super::state_space::serial::*;
}

#[cfg(test)]
mod tests;

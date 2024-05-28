//! Extensible, graph-based program representation with first-class support for linear types.
//!
//! This crate contains the core definitions for the HUGR representation.
//! See the [top-level crate documentation](https://docs.rs/hugr/latest/hugr/) for more information.

// Unstable check, may cause false positives.
// https://github.com/rust-lang/rust-clippy/issues/5112
#![warn(clippy::debug_assert_with_mut_call)]
// proptest-derive generates many of these warnings.
// https://github.com/rust-lang/rust/issues/120363
// https://github.com/proptest-rs/proptest/issues/447
#![cfg_attr(test, allow(non_local_definitions))]

pub mod builder;
pub mod core;
pub mod extension;
pub mod hugr;
pub mod macros;
pub mod ops;
pub mod std_extensions;
pub mod types;
pub mod utils;

pub use crate::core::{
    CircuitUnit, Direction, IncomingPort, Node, NodeIndex, OutgoingPort, Port, PortIndex, Wire,
};
pub use crate::extension::Extension;
pub use crate::hugr::{Hugr, HugrView, SimpleReplacement};

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(test)]
pub mod proptest;

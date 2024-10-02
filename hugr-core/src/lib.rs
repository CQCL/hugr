//! Extensible, graph-based program representation with first-class support for linear types.
//!
//! This crate contains the core definitions for the HUGR representation.
//! See the [top-level crate documentation](https://docs.rs/hugr/latest/hugr/) for more information.

// proptest-derive generates many of these warnings.
// https://github.com/rust-lang/rust/issues/120363
// https://github.com/proptest-rs/proptest/issues/447
#![cfg_attr(test, allow(non_local_definitions))]
pub mod builder;
pub mod core;
#[cfg(feature = "model")]
pub mod export;
pub mod extension;
pub mod hugr;
#[cfg(feature = "model")]
pub mod import;
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

#[cfg(test)]
pub mod proptest;

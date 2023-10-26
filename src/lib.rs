//! `hugr` is the Hierarchical Unified Graph Representation of quantum circuits
//! and operations in the Quantinuum ecosystem.
//!
//! # Features
//!
//! - `serde` enables serialization and deserialization of the components and
//!   structures.
//!

#![warn(missing_docs)]
// Unstable check, may cause false positives.
// https://github.com/rust-lang/rust-clippy/issues/5112
#![warn(clippy::debug_assert_with_mut_call)]

pub mod algorithm;
pub mod builder;
pub mod extension;
pub mod hugr;
pub mod macros;
pub mod ops;
pub mod std_extensions;
pub mod types;
mod utils;
pub mod values;

pub use crate::extension::Extension;
pub use crate::hugr::{Direction, Hugr, HugrView, Node, Port, SimpleReplacement, Wire};

pub mod walker;

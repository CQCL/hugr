#![warn(missing_docs)]

//! `hugr` is the Hierarchical Unified Graph Representation of quantum circuits
//! and operations in the Quantinuum ecosystem.
//!
//! # Features
//!
//! - `serde` enables serialization and deserialization of the components and
//!   structures.
//!

pub mod builder;
pub mod extensions;
pub mod hugr;
pub mod macros;
pub mod ops;
pub mod resource;
pub mod rewrite;
pub mod types;
mod utils;

pub use crate::hugr::{Direction, Hugr, Node, Port};
pub use crate::resource::Resource;
pub use crate::rewrite::{Rewrite, RewriteError};

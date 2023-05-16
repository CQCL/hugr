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

pub use crate::hugr::Hugr;
pub use portgraph::{NodeIndex, PortIndex};
pub use resource::Resource;
pub use rewrite::{Rewrite, RewriteError};

#[cfg(test)]
mod test {
    // Example test
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

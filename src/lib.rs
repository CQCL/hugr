//#![warn(missing_docs)] TODO: Re-enable this once the library is more mature
//#![feature(rustc_attrs)]

//! `hugr` is the Hierarchical Unified Graph Representation of quantum circuits
//! and operations in the Quantinuum ecosystem.
//!
//! # Features
//!
//! - `serde` enables serialization and deserialization of the components and
//!   structures.
//!

pub mod extensions;
pub mod hugr;
pub mod macros;
pub mod ops;
pub mod resource;
pub mod rewrite;
pub mod types;

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

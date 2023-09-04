#![warn(missing_docs)]

//! `hugr` is the Hierarchical Unified Graph Representation of quantum circuits
//! and operations in the Quantinuum ecosystem.
//!
//! # Features
//!
//! - `serde` enables serialization and deserialization of the components and
//!   structures.
//!

// Declares a 'const' member of a test module that's a new ExtensionId.
// field_name should be an UPPERCASE i.e. name of a const.
// This is here so that it is usable throughout the crate, but is not
// visible from outside (as it would be if we used macro_export)
// - as it won't *work* from outside because of the call to new_unchecked.
#[allow(unused_macros)] // Because not used *in this file*
macro_rules! test_const_ext_id {
    ($field_name:ident, $ext_name:expr) => {
        const $field_name: crate::extension::ExtensionId =
            crate::extension::ExtensionId::new_unchecked($ext_name);

        paste::paste! {
            #[test]
            fn [<check_ $field_name:lower _wellformed>]() {
                ExtensionId::new($ext_name).unwrap();
            }
        }
    };
}

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

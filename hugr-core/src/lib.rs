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
#[cfg(feature = "model_unstable")]
pub mod export;
pub mod extension;
pub mod hugr;
#[cfg(feature = "model_unstable")]
pub mod import;
pub mod macros;
pub mod ops;
pub mod package;
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

#[test]
#[cfg(feature = "model_unstable")]
#[should_panic] // BUG: see https://github.com/CQCL/hugr/issues/1876
fn bounds() {
    use crate::package::Package;
    use crate::std_extensions::STD_REG;
    use export::export_hugr;
    use import::import_hugr;

    let json = include_str!("../../hugr.json");
    let package: Package = serde_json::from_str(json.trim()).unwrap();

    // Extension registry combining standard extensions with those defined in the package.
    let mut exts = STD_REG.clone();
    exts.extend(package.extensions);

    // Export and import the first and only module in the package.
    assert_eq!(package.modules.len(), 1);
    let bump = bumpalo::Bump::new();
    let exported = export_hugr(&package.modules[0], &bump);
    import_hugr(&exported, &exts).unwrap();
}

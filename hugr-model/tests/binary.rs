#![allow(missing_docs)]

use bumpalo::Bump;
use hugr_model::v0 as model;
use pretty_assertions::assert_eq;

/// Reads a module from a string, serializes it to binary, and then deserializes it back to a module.
/// The original and deserialized modules are compared for equality.
pub fn binary_roundtrip(input: &str) {
    let bump = Bump::new();
    let parsed_module = model::text::parse(input, &bump).unwrap();
    let bytes = model::binary::write_to_vec(&parsed_module.module);
    let deserialized_module = model::binary::read_from_slice(&bytes, &bump).unwrap();
    assert_eq!(parsed_module.module, deserialized_module);
}

#[test]
pub fn test_add() {
    binary_roundtrip(include_str!("fixtures/model-add.edn"));
}

#[test]
pub fn test_alias() {
    binary_roundtrip(include_str!("fixtures/model-alias.edn"));
}

#[test]
pub fn test_call() {
    binary_roundtrip(include_str!("fixtures/model-call.edn"));
}

#[test]
pub fn test_cfg() {
    binary_roundtrip(include_str!("fixtures/model-cfg.edn"));
}

#[test]
pub fn test_cond() {
    binary_roundtrip(include_str!("fixtures/model-cond.edn"));
}

#[test]
pub fn test_loop() {
    binary_roundtrip(include_str!("fixtures/model-loop.edn"));
}

#[test]
pub fn test_params() {
    binary_roundtrip(include_str!("fixtures/model-params.edn"));
}

#[test]
pub fn test_decl_exts() {
    binary_roundtrip(include_str!("fixtures/model-decl-exts.edn"));
}

#[test]
pub fn test_constraints() {
    binary_roundtrip(include_str!("fixtures/model-constraints.edn"));
}

#[test]
pub fn test_lists() {
    binary_roundtrip(include_str!("fixtures/model-lists.edn"));
}

#[test]
pub fn test_const() {
    binary_roundtrip(include_str!("fixtures/model-const.edn"));
}

#[test]
pub fn test_literals() {
    binary_roundtrip(include_str!("fixtures/model-literals.edn"));
}

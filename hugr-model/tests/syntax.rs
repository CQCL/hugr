#![allow(missing_docs)]

use hugr_model::v0::syntax;
use std::str::FromStr as _;

fn roundtrip(source: &str) -> String {
    let module = syntax::Module::from_str(source).unwrap();
    format!("{}", module)
}

#[test]
pub fn test_add() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-add.edn")));
}

#[test]
pub fn test_alias() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-alias.edn")));
}

#[test]
pub fn test_call() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-call.edn")));
}

#[test]
pub fn test_cfg() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-cfg.edn")));
}

#[test]
pub fn test_cond() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-cond.edn")));
}

#[test]
pub fn test_loop() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-loop.edn")));
}

#[test]
pub fn test_params() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-params.edn")));
}

#[test]
pub fn test_decl_exts() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-decl-exts.edn")));
}

#[test]
pub fn test_constraints() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-constraints.edn")));
}

#[test]
pub fn test_lists() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-lists.edn")));
}

#[test]
pub fn test_const() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-const.edn")));
}

#[test]
pub fn test_literals() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-literals.edn")));
}

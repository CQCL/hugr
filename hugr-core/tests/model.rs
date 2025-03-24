#![allow(missing_docs)]

use std::str::FromStr;

use hugr::{package::Package, std_extensions::std_reg};
use hugr_model::v0 as model;

fn roundtrip(source: &str) -> String {
    let bump = model::bumpalo::Bump::new();
    let package_ast = model::ast::Package::from_str(source).unwrap();
    let package_table = package_ast.resolve(&bump).unwrap();
    let core = Package::from_model(&package_table, &std_reg()).unwrap();
    let exported_table = core.to_model(&bump);
    let exported_ast = exported_table.as_ast().unwrap();
    exported_ast.to_string()
}

#[test]
pub fn test_roundtrip_add() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-add.edn"
    )));
}

#[test]
pub fn test_roundtrip_call() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-call.edn"
    )));
}

#[test]
pub fn test_roundtrip_alias() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-alias.edn"
    )));
}

#[test]
pub fn test_roundtrip_cfg() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-cfg.edn"
    )));
}

#[test]
pub fn test_roundtrip_cond() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-cond.edn"
    )));
}

#[test]
pub fn test_roundtrip_loop() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-loop.edn"
    )));
}

#[test]
pub fn test_roundtrip_params() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-params.edn"
    )));
}

#[test]
pub fn test_roundtrip_constraints() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-constraints.edn"
    )));
}

#[test]
pub fn test_roundtrip_const() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-const.edn"
    )));
}

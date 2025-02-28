#![allow(missing_docs)]

use std::str::FromStr;

use hugr::std_extensions::std_reg;
use hugr_core::{export::export_hugr, import::import_hugr};
use hugr_model::v0::{self as model};

fn roundtrip(source: &str) -> String {
    let bump = model::bumpalo::Bump::new();
    let module_ast = model::ast::Module::from_str(source).unwrap();
    let module_table = module_ast.resolve(&bump).unwrap();
    let hugr = import_hugr(&module_table, &std_reg()).unwrap();
    let exported_table = export_hugr(&hugr, &bump);
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

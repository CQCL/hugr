#![allow(missing_docs)]

use std::str::FromStr;

use hugr::std_extensions::std_reg;
use hugr_core::{export::export_package, import::import_package};
use hugr_model::v0 as model;

fn roundtrip(source: &str) -> String {
    let bump = model::bumpalo::Bump::new();
    let package_ast = model::ast::Package::from_str(source).unwrap();
    let package_table = package_ast.resolve(&bump).unwrap();
    let core = import_package(&package_table, &std_reg()).unwrap();
    let exported_table = export_package(&core, &bump);
    let exported_ast = exported_table.as_ast().unwrap();
    exported_ast.to_string()
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_add() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-add.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_call() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-call.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_alias() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-alias.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_cfg() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-cfg.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_cond() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-cond.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_loop() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-loop.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_params() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-params.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_constraints() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-constraints.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_const() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-const.edn"
    )));
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_roundtrip_order() {
    insta::assert_snapshot!(roundtrip(include_str!(
        "../../hugr-model/tests/fixtures/model-order.edn"
    )));
}

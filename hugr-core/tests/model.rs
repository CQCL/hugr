use hugr::std_extensions::std_reg;
use hugr_core::{export::export_hugr, import::import_hugr};
use hugr_model::v0 as model;

fn roundtrip(source: &str) -> String {
    let bump = bumpalo::Bump::new();
    let parsed_model = model::text::parse(source, &bump).unwrap();
    let imported_hugr = import_hugr(&parsed_model.module, &std_reg()).unwrap();
    let exported_model = export_hugr(&imported_hugr, &bump);
    model::text::print_to_string(&exported_model, 80).unwrap()
}

#[test]
pub fn test_roundtrip_add() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-add.edn")));
}

#[test]
pub fn test_roundtrip_call() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-call.edn")));
}

#[test]
pub fn test_roundtrip_alias() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-alias.edn")));
}

#[test]
pub fn test_roundtrip_cfg() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-cfg.edn")));
}

#[test]
pub fn test_roundtrip_cond() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-cond.edn")));
}

#[test]
pub fn test_roundtrip_loop() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-loop.edn")));
}

#[test]
pub fn test_roundtrip_params() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-params.edn")));
}

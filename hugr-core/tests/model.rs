use hugr::std_extensions::std_reg;
use hugr_core::{export::export_hugr, import::import_hugr};
use hugr_model::v0 as model;

#[test]
pub fn test_import_export() {
    let bump = bumpalo::Bump::new();
    let parsed_module = model::text::parse(include_str!("fixtures/model-1.edn"), &bump).unwrap();
    let extensions = std_reg();
    let hugr = import_hugr(&parsed_module.module, &extensions).unwrap();
    let roundtrip = export_hugr(&hugr, &bump);
    panic!("{}:", model::text::print_to_string(&roundtrip, 80).unwrap());
}

#[cfg(feature = "model")]
use hugr::std_extensions::std_reg;
#[cfg(feature = "model")]
use hugr_core::{export::export_hugr, import::import_hugr};
#[cfg(feature = "model")]
use hugr_model::v0 as model;

#[cfg(feature = "model")]
#[test]
pub fn test_import_export() {
    let bump = bumpalo::Bump::new();
    let parsed_module = model::text::parse(include_str!("fixtures/model-1.edn"), &bump).unwrap();
    let extensions = std_reg();

    let hugr = import_hugr(&parsed_module.module, &extensions).unwrap();

    let roundtrip = export_hugr(&hugr, &bump);
    let roundtrip_str = model::text::print_to_string(&roundtrip, 80).unwrap();
    insta::assert_snapshot!(roundtrip_str);
}

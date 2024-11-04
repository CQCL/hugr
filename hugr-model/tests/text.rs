use hugr_model::v0 as model;

fn roundtrip(source: &str) -> String {
    let bump = bumpalo::Bump::new();
    let parsed_model = model::text::parse(source, &bump).unwrap();
    model::text::print_to_string(&parsed_model.module, 80).unwrap()
}

#[test]
pub fn test_declarative_extensions() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-decl-exts.edn")))
}

#[test]
pub fn test_literals() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-literals.edn")))
}

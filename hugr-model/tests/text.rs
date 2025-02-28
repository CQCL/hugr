#![allow(missing_docs)]

use hugr_model::v0::syntax;
use std::str::FromStr as _;

fn roundtrip(source: &str) -> String {
    let module = syntax::Module::from_str(source).unwrap();
    module.to_string()
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_declarative_extensions() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-decl-exts.edn")))
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
pub fn test_literals() {
    insta::assert_snapshot!(roundtrip(include_str!("fixtures/model-literals.edn")))
}

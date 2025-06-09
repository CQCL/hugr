#![no_main]

use hugr_model::v0 as model;
use libfuzzer_sys::fuzz_target;
use model::bumpalo::Bump;
use hugr_model::v0::ast::Package;

fuzz_target!(|package: Package| {
    let bump = Bump::new();
    let package = package.resolve(&bump).unwrap();
    let bytes = model::binary::write_to_vec(&package);
    let deserialized_package = model::binary::read_from_slice(&bytes, &bump).unwrap();
    assert_eq!(package, deserialized_package);
});

#![no_main]

use libfuzzer_sys::fuzz_target;
use hugr_model::v0 as model;
use std::str::FromStr;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _package_ast = model::ast::Package::from_str(&s);
    }
});

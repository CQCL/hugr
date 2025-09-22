//! The hugr-llvm build script.
//!
//! For debug builds, we compile the `test_panic_runtime` library to enable
//! LLVM execution tests involving panics.

fn main() {
    println!("cargo::rerun-if-changed=src/emit/test/panic_runtime.c");
    if std::env::var("PROFILE").unwrap() == "debug" {
        cc::Build::new()
            .file("src/emit/test/panic_runtime.c")
            .compile("test_panic_runtime");
    }
}

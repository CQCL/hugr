//! The hugr-llvm build script.
//!
//! For test builds, we compile the `test_panic_runtime` library to enable
//! LLVM execution tests involving panics.

fn main() {
    #[cfg(feature = "test-utils")]
    compile_panic_runtime();
}

#[cfg(feature = "test-utils")]
fn compile_panic_runtime() {
    println!("cargo::rerun-if-changed=src/emit/test/panic_runtime.c");
    cc::Build::new()
        .file("src/emit/test/panic_runtime.c")
        .compile("test_panic_runtime");
}

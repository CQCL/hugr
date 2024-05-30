//! Build script for the `hugr` crate.

fn main() {
    // We use a `ci_run` RUSTFLAG to indicate that we are running a CI check,
    // so we can reject debug code using some tools defined in `utils.rs`.
    println!("cargo:rustc-check-cfg=cfg(ci_run)");
}

//! Tests for the CLI extension writing.
//!
//! Miri is globally disabled for these tests because they mostly involve
//! calling the CLI binary, which Miri doesn't support.
#![cfg(all(test, not(miri)))]

use assert_cmd::Command;
use rstest::{fixture, rstest};

#[fixture]
fn cmd() -> Command {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("hugr");
    cmd.arg("gen-extensions");
    cmd
}

#[rstest]
fn test_extension_dump(mut cmd: Command) {
    let temp_dir = assert_fs::TempDir::new()
        .expect("temp dir creation failure.")
        .into_persistent_if(std::env::var_os("HUGR_CLI_TEST_PERSIST_FILES").is_some());
    cmd.arg("-o");
    cmd.arg(temp_dir.path());
    cmd.assert().success();

    let expected_paths = [
        "logic.json",
        "prelude.json",
        "ptr.json",
        "arithmetic/int/types.json",
        "arithmetic/float/types.json",
        "arithmetic/int.json",
        "arithmetic/float.json",
        "arithmetic/conversions.json",
        "collections/array.json",
        "collections/list.json",
    ];
    // check all paths exist
    for path in &expected_paths {
        let full_path = temp_dir.join(path);
        assert!(full_path.exists());
    }

    // temp dir deleted when dropped here
}

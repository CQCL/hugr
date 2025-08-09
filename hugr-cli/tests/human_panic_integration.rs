//! Integration test for `human-panic`.
//!
//! Builds the release binary with the `panic-test` feature, runs it with a
//! test-only panic trigger and an isolated temp dir, asserts the banner,
//! and verifies a TOML report is written. Temp dir is removed at the end.

//! Black-box integration test for `human-panic`.

use predicates::str::contains; // for cargo_bin()
use std::process::Command;
use tempfile::TempDir;

#[test]
fn human_panic_writes_report() {
    // Isolated temp dir for the crash report.
    let tmp = TempDir::new().expect("create tempdir");
    let tmp_path = tmp.path();

    // Run the release CLI binary from the workspace root.
    // No features needed: main() installs human-panic in release builds.
    let mut cmd = Command::new("cargo");
    cmd.args([
        "run",
        "--release",
        "-p",
        "hugr-cli",
        "--bin",
        "hugr",
        "--", // end cargo args; program args would follow
    ]);

    // Isolate temp location & trigger the test panic in the child process.
    if cfg!(windows) {
        cmd.env("TEMP", tmp_path).env("TMP", tmp_path);
    } else {
        cmd.env("TMPDIR", tmp_path);
    }
    cmd.env("PANIC_FOR_TESTS", "1");

    // Expect non-zero exit and the banner on stderr (release only).
    assert_cmd::Command::from_std(cmd)
        .assert()
        .failure()
        .stderr(contains("Well, this is embarrassing."));

    // Confirm a .toml report exists in our temp dir.
    let made_report = std::fs::read_dir(tmp_path)
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        });
    assert!(
        made_report,
        "expected a human-panic report in {:?}",
        tmp_path
    );

    // Explicit cleanup; surface any removal errors.
    tmp.close().expect("failed to remove temp dir");
}

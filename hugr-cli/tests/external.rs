//! Tests for external subcommand support in hugr-cli.
#![cfg(all(test, not(miri)))]

use predicates::str::contains;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

#[test]
fn test_missing_external_command() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("hugr");
    cmd.arg("idontexist");
    cmd.assert()
        .failure()
        .stderr(contains("no such subcommand"));
}

#[test]
#[cfg_attr(not(unix), ignore = "Dummy program supported on Unix-like systems")]
fn test_external_command_invocation() {
    // Create a dummy external command in a temp dir
    let tempdir = TempDir::new().unwrap();
    let bin_path = tempdir.path().join("hugr-dummy");
    fs::write(&bin_path, b"#!/bin/sh\necho dummy called: $@\nexit 42\n").unwrap();
    let mut perms = fs::metadata(&bin_path).unwrap().permissions();
    #[cfg(unix)]
    perms.set_mode(0o755);
    fs::set_permissions(&bin_path, perms).unwrap();

    // Prepend tempdir to PATH
    let orig_path = env::var("PATH").unwrap();
    let new_path = format!("{}:{}", tempdir.path().display(), orig_path);
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("hugr");
    cmd.env("PATH", new_path);
    cmd.arg("dummy");
    cmd.arg("foo");
    cmd.arg("bar");
    cmd.assert()
        .failure()
        .stdout(contains("dummy called: foo bar"))
        .code(42);
}

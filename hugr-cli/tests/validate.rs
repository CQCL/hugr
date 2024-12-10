//! Tests for the CLI
//!
//! Miri is globally disabled for these tests because they mostly involve
//! calling the CLI binary, which Miri doesn't support.
#![cfg(all(test, not(miri)))]

use std::sync::Arc;

use assert_cmd::Command;
use assert_fs::{fixture::FileWriteStr, NamedTempFile};
use hugr::builder::{DFGBuilder, DataflowSubContainer, ModuleBuilder};
use hugr::types::Type;
use hugr::{
    builder::{Container, Dataflow},
    extension::prelude::{bool_t, qb_t},
    std_extensions::arithmetic::float_types::float64_type,
    types::Signature,
    Hugr,
};
use hugr_cli::{validate::VALID_PRINT, Package};
use predicates::{prelude::*, str::contains};
use rstest::{fixture, rstest};

#[fixture]
fn cmd() -> Command {
    Command::cargo_bin("hugr").unwrap()
}

#[fixture]
fn val_cmd(mut cmd: Command) -> Command {
    cmd.arg("validate");
    cmd
}

// path to the fully serialized float extension
const FLOAT_EXT_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../specification/std_extensions/arithmetic/float/types.json"
);

/// A test package, containing a module-rooted HUGR.
#[fixture]
fn test_package(#[default(bool_t())] id_type: Type) -> Package {
    let mut module = ModuleBuilder::new();
    let df = module
        .define_function("test", Signature::new_endo(id_type))
        .unwrap();
    let [i] = df.input_wires_arr();
    df.finish_with_outputs([i]).unwrap();
    let hugr = module.hugr().clone(); // unvalidated

    let rdr = std::fs::File::open(FLOAT_EXT_FILE).unwrap();
    let float_ext: Arc<hugr::Extension> = serde_json::from_reader(rdr).unwrap();
    Package::new(vec![hugr], vec![float_ext]).unwrap()
}

/// A DFG-rooted HUGR.
#[fixture]
fn test_hugr(#[default(bool_t())] id_type: Type) -> Hugr {
    let mut df = DFGBuilder::new(Signature::new_endo(id_type)).unwrap();
    let [i] = df.input_wires_arr();
    df.set_outputs([i]).unwrap();
    df.hugr().clone() // unvalidated
}

#[fixture]
fn test_hugr_string(test_hugr: Hugr) -> String {
    serde_json::to_string(&test_hugr).unwrap()
}

#[fixture]
fn test_hugr_file(test_hugr_string: String) -> NamedTempFile {
    let file = assert_fs::NamedTempFile::new("sample.hugr").unwrap();
    file.write_str(&test_hugr_string).unwrap();
    file
}

#[rstest]
fn test_doesnt_exist(mut val_cmd: Command) {
    val_cmd.arg("foobar");
    val_cmd
        .assert()
        .failure()
        .stderr(contains("No such file or directory"));
}

#[rstest]
fn test_validate(test_hugr_file: NamedTempFile, mut val_cmd: Command) {
    val_cmd.arg(test_hugr_file.path());
    val_cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[rstest]
fn test_stdin(test_hugr_string: String, mut val_cmd: Command) {
    val_cmd.write_stdin(test_hugr_string);
    val_cmd.arg("-");

    val_cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[rstest]
fn test_stdin_silent(test_hugr_string: String, mut val_cmd: Command) {
    val_cmd.args(["-", "-q"]);
    val_cmd.write_stdin(test_hugr_string);

    val_cmd
        .assert()
        .success()
        .stderr(contains(VALID_PRINT).not());
}

#[rstest]
fn test_mermaid(test_hugr_file: NamedTempFile, mut cmd: Command) {
    const MERMAID: &str = "graph LR\n    subgraph 0 [\"(0) DFG\"]";
    cmd.arg("mermaid");
    cmd.arg(test_hugr_file.path());
    cmd.assert().success().stdout(contains(MERMAID));
}

#[fixture]
fn bad_hugr_string() -> String {
    let df = DFGBuilder::new(Signature::new_endo(vec![qb_t()])).unwrap();
    let bad_hugr = df.hugr().clone();

    serde_json::to_string(&bad_hugr).unwrap()
}

#[rstest]
fn test_mermaid_invalid(bad_hugr_string: String, mut cmd: Command) {
    cmd.arg("mermaid");
    cmd.arg("--validate");
    cmd.write_stdin(bad_hugr_string);
    cmd.assert().failure().stderr(contains("UnconnectedPort"));
}

#[rstest]
fn test_bad_hugr(bad_hugr_string: String, mut val_cmd: Command) {
    val_cmd.write_stdin(bad_hugr_string);
    val_cmd.arg("-");

    val_cmd
        .assert()
        .failure()
        .stderr(contains("Error validating HUGR").and(contains("unconnected port")));
}

#[rstest]
fn test_bad_json(mut val_cmd: Command) {
    val_cmd.write_stdin(r#"{"foo": "bar"}"#);
    val_cmd.arg("-");

    val_cmd
        .assert()
        .failure()
        .stderr(contains("Error parsing package"));
}

#[rstest]
fn test_bad_json_silent(mut val_cmd: Command) {
    val_cmd.write_stdin(r#"{"foo": "bar"}"#);
    val_cmd.args(["-", "-qqq"]);

    val_cmd
        .assert()
        .failure()
        .stderr(contains("Error parsing package").not());
}

#[rstest]
fn test_no_std(test_hugr_string: String, mut val_cmd: Command) {
    val_cmd.write_stdin(test_hugr_string);
    val_cmd.arg("-");
    val_cmd.arg("--no-std");
    // test hugr doesn't have any standard extensions, so this should succceed

    val_cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[fixture]
fn float_hugr_string(#[with(float64_type())] test_hugr: Hugr) -> String {
    serde_json::to_string(&test_hugr).unwrap()
}

#[rstest]
fn test_no_std_fail(float_hugr_string: String, mut val_cmd: Command) {
    val_cmd.write_stdin(float_hugr_string);
    val_cmd.arg("-");
    val_cmd.arg("--no-std");

    val_cmd
        .assert()
        .failure()
        .stderr(contains(" requires extension arithmetic.float.types"));
}

#[rstest]
fn test_float_extension(float_hugr_string: String, mut val_cmd: Command) {
    val_cmd.write_stdin(float_hugr_string);
    val_cmd.arg("-");
    val_cmd.arg("--no-std");
    val_cmd.arg("--extensions");
    val_cmd.arg(FLOAT_EXT_FILE);

    val_cmd.assert().success().stderr(contains(VALID_PRINT));
}
#[fixture]
fn package_string(#[with(float64_type())] test_package: Package) -> String {
    serde_json::to_string(&test_package).unwrap()
}

#[rstest]
fn test_package_validation(package_string: String, mut val_cmd: Command) {
    // package with float extension and hugr that uses floats can validate
    val_cmd.write_stdin(package_string);
    val_cmd.arg("-");
    val_cmd.arg("--no-std");

    val_cmd.assert().success().stderr(contains(VALID_PRINT));
}

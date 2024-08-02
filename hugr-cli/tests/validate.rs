//! Tests for the CLI
//!
//! Miri is globally disabled for these tests because they mostly involve
//! calling the CLI binary, which Miri doesn't support.
#![cfg(all(test, not(miri)))]

use assert_cmd::Command;
use assert_fs::{fixture::FileWriteStr, NamedTempFile};
use hugr_cli::validate::{Package, VALID_PRINT};
use hugr_core::builder::DFGBuilder;
use hugr_core::types::Type;
use hugr_core::{
    builder::{Container, Dataflow},
    extension::prelude::{BOOL_T, QB_T},
    std_extensions::arithmetic::float_types::FLOAT64_TYPE,
    type_row,
    types::Signature,
    Hugr,
};
use predicates::{prelude::*, str::contains};
use rstest::{fixture, rstest};

#[fixture]
fn cmd() -> Command {
    let mut cmd = Command::cargo_bin("hugr").unwrap();
    cmd.arg("validate");
    cmd
}

#[fixture]
fn test_hugr(#[default(BOOL_T)] id_type: Type) -> Hugr {
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
fn test_doesnt_exist(mut cmd: Command) {
    cmd.arg("foobar");
    cmd.assert()
        .failure()
        .stderr(contains("No such file or directory"));
}

#[rstest]
fn test_validate(test_hugr_file: NamedTempFile, mut cmd: Command) {
    cmd.arg(test_hugr_file.path());
    cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[rstest]
fn test_stdin(test_hugr_string: String, mut cmd: Command) {
    cmd.write_stdin(test_hugr_string);
    cmd.arg("-");

    cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[rstest]
fn test_stdin_silent(test_hugr_string: String, mut cmd: Command) {
    cmd.args(["-", "-q"]);
    cmd.write_stdin(test_hugr_string);

    cmd.assert().success().stderr(contains(VALID_PRINT).not());
}

#[rstest]
fn test_mermaid(test_hugr_file: NamedTempFile, mut cmd: Command) {
    const MERMAID: &str = "graph LR\n    subgraph 0 [\"(0) DFG\"]";
    cmd.arg(test_hugr_file.path());
    cmd.arg("--mermaid");
    cmd.arg("--no-validate");
    cmd.assert().success().stdout(contains(MERMAID));
}

#[rstest]
fn test_bad_hugr(mut cmd: Command) {
    let df = DFGBuilder::new(Signature::new_endo(type_row![QB_T])).unwrap();
    let bad_hugr = df.hugr().clone();

    let bad_hugr_string = serde_json::to_string(&bad_hugr).unwrap();
    cmd.write_stdin(bad_hugr_string);
    cmd.arg("-");

    cmd.assert()
        .failure()
        .stderr(contains("Error validating HUGR").and(contains("unconnected port")));
}

#[rstest]
fn test_bad_json(mut cmd: Command) {
    cmd.write_stdin(r#"{"foo": "bar"}"#);
    cmd.arg("-");

    cmd.assert()
        .failure()
        .stderr(contains("Error parsing input"));
}

#[rstest]
fn test_bad_json_silent(mut cmd: Command) {
    cmd.write_stdin(r#"{"foo": "bar"}"#);
    cmd.args(["-", "-qqq"]);

    cmd.assert()
        .failure()
        .stderr(contains("Error parsing input").not());
}

#[rstest]
fn test_no_std(test_hugr_string: String, mut cmd: Command) {
    cmd.write_stdin(test_hugr_string);
    cmd.arg("-");
    cmd.arg("--no-std");
    // test hugr doesn't have any standard extensions, so this should succceed

    cmd.assert().success().stderr(contains(VALID_PRINT));
}

#[fixture]
fn float_hugr_string(#[with(FLOAT64_TYPE)] test_hugr: Hugr) -> String {
    serde_json::to_string(&test_hugr).unwrap()
}

#[rstest]
fn test_no_std_fail(float_hugr_string: String, mut cmd: Command) {
    cmd.write_stdin(float_hugr_string);
    cmd.arg("-");
    cmd.arg("--no-std");

    cmd.assert()
        .failure()
        .stderr(contains(" Extension 'arithmetic.float.types' not found"));
}

// path to the fully serialized float extension
const FLOAT_EXT_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../specification/std_extensions/arithmetic/float/types.json"
);

#[rstest]
fn test_float_extension(float_hugr_string: String, mut cmd: Command) {
    cmd.write_stdin(float_hugr_string);
    cmd.arg("-");
    cmd.arg("--no-std");
    cmd.arg("--extensions");
    cmd.arg(FLOAT_EXT_FILE);

    cmd.assert().success().stderr(contains(VALID_PRINT));
}
#[fixture]
fn package_string(#[with(FLOAT64_TYPE)] test_hugr: Hugr) -> String {
    let rdr = std::fs::File::open(FLOAT_EXT_FILE).unwrap();
    let float_ext: hugr_core::Extension = serde_json::from_reader(rdr).unwrap();
    let package = Package::new(vec![test_hugr], vec![float_ext]);
    serde_json::to_string(&package).unwrap()
}

#[rstest]
fn test_package(package_string: String, mut cmd: Command) {
    // package with float extension and hugr that uses floats can validate
    cmd.write_stdin(package_string);
    cmd.arg("-");
    cmd.arg("--no-std");

    cmd.assert().success().stderr(contains(VALID_PRINT));
}

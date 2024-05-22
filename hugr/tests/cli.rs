#![cfg(feature = "cli")]

use assert_cmd::Command;
use assert_fs::{fixture::FileWriteStr, NamedTempFile};
use hugr::{
    builder::{Dataflow, DataflowHugr},
    extension::prelude::BOOL_T,
    type_row,
    types::FunctionType,
    Hugr,
};
use predicates::prelude::*;
use rstest::{fixture, rstest};

use hugr::cli::VALID_PRINT;
#[fixture]
fn cmd() -> Command {
    Command::cargo_bin(env!("CARGO_PKG_NAME")).unwrap()
}

#[fixture]
fn test_hugr() -> Hugr {
    use hugr::builder::DFGBuilder;

    let df = DFGBuilder::new(FunctionType::new_endo(type_row![BOOL_T])).unwrap();
    let [i] = df.input_wires_arr();
    df.finish_prelude_hugr_with_outputs([i]).unwrap()
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
fn test_doesnt_exist(mut cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    cmd.arg("foobar");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("No such file or directory"));

    Ok(())
}

#[rstest]
fn test_validate(
    test_hugr_file: NamedTempFile,
    mut cmd: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    cmd.arg(test_hugr_file.path());
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(VALID_PRINT));

    Ok(())
}

#[rstest]
fn test_stdin(
    test_hugr_string: String,
    mut cmd: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    cmd.write_stdin(test_hugr_string);
    cmd.arg("-");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains(VALID_PRINT));

    Ok(())
}

#[rstest]
fn test_mermaid(
    test_hugr_file: NamedTempFile,
    mut cmd: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    const MERMAID: &str = "graph LR\n    subgraph 0 [\"(0) DFG\"]";
    cmd.arg(test_hugr_file.path());
    cmd.arg("--mermaid");
    cmd.arg("--no-validate");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(MERMAID));

    Ok(())
}

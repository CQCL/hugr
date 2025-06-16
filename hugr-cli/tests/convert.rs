//! Tests for the convert subcommand
//!
//! Miri is globally disabled for these tests because they mostly involve
//! calling the CLI binary, which Miri doesn't support.
#![cfg(all(test, not(miri)))]

use assert_cmd::Command;
use assert_fs::{NamedTempFile, fixture::FileWriteStr};
use hugr::builder::{DataflowSubContainer, ModuleBuilder};
use hugr::envelope::{EnvelopeConfig, EnvelopeFormat, read_envelope};
use hugr::package::Package;
use hugr::types::Type;
use hugr::{
    builder::{Container, Dataflow},
    extension::ExtensionRegistry,
    extension::prelude::bool_t,
    types::Signature,
    HugrView
};
use predicates::str::contains;
use rstest::{fixture, rstest};
use std::io::BufReader;

#[fixture]
fn cmd() -> Command {
    Command::cargo_bin("hugr").unwrap()
}

#[fixture]
fn convert_cmd(mut cmd: Command) -> Command {
    cmd.arg("convert");
    cmd
}

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

    Package::new(vec![hugr])
}

#[fixture]
fn test_envelope_text(test_package: Package) -> (String, Package) {
    let config = EnvelopeConfig::text();
    (test_package.store_str(config).unwrap(), test_package)
}

#[fixture]
fn test_envelope_file(test_envelope_text: (String, Package)) -> NamedTempFile {
    let file = assert_fs::NamedTempFile::new("sample.hugr").unwrap();
    file.write_str(&test_envelope_text.0).unwrap();
    file
}

#[rstest]
fn test_convert_to_json(test_envelope_file: NamedTempFile, mut convert_cmd: Command) {
    // Create output file
    let output_file = assert_fs::NamedTempFile::new("output.hugr").unwrap();

    convert_cmd.args([
        test_envelope_file.path().to_str().unwrap(),
        "-o",
        output_file.path().to_str().unwrap(),
        "--format",
        "json",
    ]);

    convert_cmd.assert().success();

    // Verify the output exists and is a valid hugr envelope
    let output_content = std::fs::read(output_file.path()).expect("Failed to read output file");
    let reader = BufReader::new(output_content.as_slice());
    let registry = ExtensionRegistry::default();
    let (config, _) = read_envelope(reader, &registry).expect("Failed to read output envelope");

    // Verify the format is correct
    assert_eq!(config.format, EnvelopeFormat::PackageJson);
}

#[rstest]
fn test_convert_to_model(test_envelope_file: NamedTempFile, mut convert_cmd: Command) {
    // Create output file
    let output_file = assert_fs::NamedTempFile::new("output.hugr").unwrap();

    convert_cmd.args([
        test_envelope_file.path().to_str().unwrap(),
        "-o",
        output_file.path().to_str().unwrap(),
        "--format",
        "model",
    ]);

    convert_cmd.assert().success();

    // Verify the output exists and is a valid hugr envelope
    let output_content = std::fs::read(output_file.path()).expect("Failed to read output file");
    let reader = BufReader::new(output_content.as_slice());
    let registry = ExtensionRegistry::default();
    let (config, _) = read_envelope(reader, &registry).expect("Failed to read output envelope");

    // Verify the format is correct
    assert_eq!(config.format, EnvelopeFormat::Model);
}

#[rstest]
fn test_convert_invalid_format(test_envelope_file: NamedTempFile, mut convert_cmd: Command) {
    // Create output file
    let output_file = assert_fs::NamedTempFile::new("output.hugr").unwrap();

    convert_cmd.args([
        test_envelope_file.path().to_str().unwrap(),
        "-o",
        output_file.path().to_str().unwrap(),
        "--format",
        "invalid-format",
    ]);

    // This should fail with an error message about the invalid format
    convert_cmd
        .assert()
        .failure()
        .stderr(contains("Invalid format"));
}

#[rstest]
fn test_convert_with_compression(test_envelope_file: NamedTempFile, mut convert_cmd: Command) {
    // Create output file
    let output_file = assert_fs::NamedTempFile::new("output.hugr").unwrap();

    convert_cmd.args([
        test_envelope_file.path().to_str().unwrap(),
        "-o",
        output_file.path().to_str().unwrap(),
        "--compress",
        "--compression-level",
        "5",
    ]);

    convert_cmd.assert().success();
}

#[rstest]
fn test_convert_stdin_stdout(test_envelope_text: (String, Package), mut convert_cmd: Command) {
    // Use stdin/stdout
    convert_cmd.args(["-", "--format", "model-exts"]);
    convert_cmd.write_stdin(test_envelope_text.0);

    // Should succeed and produce output to stdout
    convert_cmd.assert().success();
}

#[rstest]
fn test_convert_model_text_format(test_envelope_file: NamedTempFile, mut convert_cmd: Command) {
    // Create output file
    let output_file = assert_fs::NamedTempFile::new("output.hugr").unwrap();

    convert_cmd.args([
        test_envelope_file.path().to_str().unwrap(),
        "-o",
        output_file.path().to_str().unwrap(),
        "--format",
        "model-text",
    ]);

    convert_cmd.assert().success();

    // Verify the output exists and is a valid hugr envelope
    let output_content = std::fs::read(output_file.path()).expect("Failed to read output file");
    let reader = BufReader::new(output_content.as_slice());
    let registry = ExtensionRegistry::default();
    let (config, _) = read_envelope(reader, &registry).expect("Failed to read output envelope");

    // Verify the format is correct
    assert_eq!(config.format, EnvelopeFormat::ModelText);
}

#[rstest]
fn test_format_roundtrip(test_package: Package) {
    // Test conversion between all formats in a roundtrip
    // Start with JSON format
    let config_json = EnvelopeConfig::new(EnvelopeFormat::PackageJson);
    let mut json_data = Vec::new();
    hugr::envelope::write_envelope(&mut json_data, &test_package, config_json).unwrap();

    // Convert to Model format
    let config_model = EnvelopeConfig::new(EnvelopeFormat::Model);
    let reader = BufReader::new(json_data.as_slice());
    let registry = ExtensionRegistry::default();
    let (_, package) = read_envelope(reader, &registry).unwrap();

    let mut model_data = Vec::new();
    hugr::envelope::write_envelope(&mut model_data, &package, config_model).unwrap();

    // Convert back to JSON
    let reader = BufReader::new(model_data.as_slice());
    let (_, package_back) = read_envelope(reader, &registry).unwrap();

    // Package should be the same after roundtrip conversion
    let [h1] = test_package.modules.as_slice() else {panic!()};
    let [h2] = package_back.modules.iter().collect::<Vec<_>>().try_into().unwrap();
    for n in h1.nodes() {
        let op1 = h1.get_optype(n);
        let op2 = h2.get_optype(n);
        if op1!=op2 { eprintln!("ALAN node {n} original {:?} roundtrip {:?}", op1, op2)}
    }
    assert_eq!(test_package, package_back);
}

#[rstest]
fn test_convert_text_flag(test_envelope_text: (String, Package), mut convert_cmd: Command) {
    convert_cmd.args(["-", "--text"]);
    convert_cmd.write_stdin(test_envelope_text.0);

    let output = convert_cmd.assert().success().get_output().to_owned();
    let stdout = output.stdout.clone();

    let reader = BufReader::new(stdout.as_slice());
    let registry = ExtensionRegistry::default();
    let (config, _) = read_envelope(reader, &registry).expect("Failed to read output envelope");

    // Verify it's a text-based format
    assert!(config.format.ascii_printable());
}

#[rstest]
fn test_convert_binary_flag(test_envelope_text: (String, Package), mut convert_cmd: Command) {
    convert_cmd.args(["-", "--binary"]);
    convert_cmd.write_stdin(test_envelope_text.0);

    let output = convert_cmd.assert().success().get_output().to_owned();
    let stdout = output.stdout.clone();

    let reader = BufReader::new(stdout.as_slice());
    let registry = ExtensionRegistry::default();
    let (config, _) = read_envelope(reader, &registry).expect("Failed to read output envelope");

    // Verify it's a binary format (not ASCII printable)
    assert!(!config.format.ascii_printable());
    assert!(config.zstd.is_some());
}

#[rstest]
fn test_format_conflicts(mut convert_cmd: Command) {
    // Test that --format and --text cannot be combined
    convert_cmd.args(["-", "--format", "json", "--text"]);

    // Should fail due to conflicting options
    convert_cmd
        .assert()
        .failure()
        .stderr(contains("cannot be used with"));

    // Test that --text and --binary cannot be combined
    let mut convert_cmd = Command::cargo_bin("hugr").unwrap();
    convert_cmd.arg("convert");
    convert_cmd.args(["-", "--text", "--binary"]);

    // Should fail due to conflicting options
    convert_cmd
        .assert()
        .failure()
        .stderr(contains("cannot be used with"));
}

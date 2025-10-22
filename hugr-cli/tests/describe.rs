//! Tests for describe subcommand of the CLI
//!
//! Miri is globally disabled for these tests because they mostly involve
//! calling the CLI binary, which Miri doesn't support.
#![cfg(all(test, not(miri)))]

use std::sync::Arc;

use assert_cmd::Command;
use assert_fs::NamedTempFile;
use assert_fs::assert::PathAssert;
use assert_fs::fixture::FileWriteBin;
use hugr::Extension;
use hugr::builder::ModuleBuilder;
use hugr::builder::{Dataflow, DataflowSubContainer, HugrBuilder};
use hugr::core::Visibility;
use hugr::envelope::{EnvelopeConfig, GENERATOR_KEY, USED_EXTENSIONS_KEY};
use hugr::extension::prelude::bool_t;
use hugr::extension::{ExtensionId, ExtensionRegistry, Version};
use hugr::hugr::HugrView;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::handle::NodeHandle;
use hugr::package::Package;
use hugr::types::Signature;
use predicates::{prelude::*, str::contains};
use rstest::{fixture, rstest};
use serde_json::Value;
use serde_json::json;
#[fixture]
fn cmd() -> Command {
    Command::cargo_bin("hugr").unwrap()
}

#[fixture]
fn describe_cmd(mut cmd: Command) -> Command {
    cmd.arg("describe");
    cmd
}

#[fixture]
fn empty_package() -> Package {
    Package::new(vec![])
}

#[fixture]
fn empty_package_file(empty_package: Package) -> NamedTempFile {
    let file = assert_fs::NamedTempFile::new("valid.hugr").unwrap();
    let mut buffer = Vec::new();
    empty_package
        .store(&mut buffer, EnvelopeConfig::default())
        .unwrap();
    file.write_binary(&buffer).unwrap();
    file
}

#[fixture]
fn package_with_exts() -> Vec<u8> {
    let test_ext = Extension::new_arc(
        ExtensionId::new_unchecked("resolved_ext"),
        Version::new(0, 1, 0),
        |ext, extension_ref| {
            ext.add_op(
                "Id".into(),
                "".into(),
                Signature::new_endo(bool_t()),
                extension_ref,
            )
            .unwrap();
        },
    );
    let mut module = ModuleBuilder::new();
    let mut df = module
        .define_function("entry_fn", Signature::new_endo(bool_t()))
        .unwrap();
    let [i] = df.input_wires_arr();
    let i = df
        .add_dataflow_op(test_ext.instantiate_extension_op("Id", []).unwrap(), [i])
        .unwrap()
        .out_wire(0);
    let f_n = df.finish_with_outputs([i]).unwrap().node();
    module
        .declare_vis(
            "public_fn",
            Signature::new_endo(bool_t()).into(),
            Visibility::Public,
        )
        .unwrap();
    let mut hugr = module.finish_hugr().unwrap();
    hugr.set_entrypoint(f_n);
    hugr.set_metadata(
        hugr.module_root(),
        USED_EXTENSIONS_KEY,
        json!([{ "name": "used_ext", "version": "1.0.0" }]),
    );
    hugr.set_metadata(
        hugr.module_root(),
        GENERATOR_KEY,
        json!({ "name": "my_generator", "version": "2.0.0" }),
    );
    let mut package = Package::new(vec![hugr]);
    let packed_ext = Extension::new(
        ExtensionId::new_unchecked("packed_ext"),
        Version::new(0, 1, 0),
    );
    package.extensions = ExtensionRegistry::new([Arc::new(packed_ext), test_ext]);
    let mut buffer = Vec::new();
    package
        .store(&mut buffer, EnvelopeConfig::default())
        .unwrap();
    buffer
}

#[fixture]
fn package_with_exts_file(package_with_exts: Vec<u8>) -> NamedTempFile {
    let file = assert_fs::NamedTempFile::new("valid_with_extensions.hugr").unwrap();

    file.write_binary(&package_with_exts).unwrap();
    file
}

#[rstest]
fn test_describe_basic(empty_package_file: NamedTempFile, mut describe_cmd: Command) {
    describe_cmd.arg(empty_package_file.path());
    describe_cmd
        .assert()
        .success()
        .stdout(contains("Package contains 0 module(s) and 0 extension(s)"));
}

#[rstest]
fn test_describe_json(package_with_exts_file: NamedTempFile, mut describe_cmd: Command) {
    describe_cmd.arg(package_with_exts_file.path());
    describe_cmd.arg("--json");
    describe_cmd.arg("--packaged-extensions");
    let output = describe_cmd.assert().success().get_output().stdout.clone();
    let json: Value = serde_json::from_slice(&output).unwrap();
    let expected_json = json!({
      "header": "EnvelopeHeader(PackageJson)",
      "modules": [
        {
          "entrypoint": {
            "node": 1,
            "optype": "FuncDefn(entry_fn: [Bool] -> [Bool])"
          },
          "generator": "my_generator-v2.0.0",
          "num_nodes": 6,
          "used_extensions_resolved": [
            {
              "name": "resolved_ext",
              "version": "0.1.0"
            }
          ]
        }
      ],
      "packaged_extensions": [
        {
          "name": "packed_ext",
          "version": "0.1.0"
        },
        {
          "name": "resolved_ext",
          "version": "0.1.0"
        }
      ]
    });
    assert_eq!(json, expected_json);
}

#[rstest]
fn test_describe_packaged_extensions(
    package_with_exts_file: NamedTempFile,
    mut describe_cmd: Command,
) {
    describe_cmd.args([
        package_with_exts_file.path(),
        std::path::Path::new("--packaged-extensions"),
    ]);
    describe_cmd
        .assert()
        .success()
        .stdout(contains("my_generator-v2.0.0"))
        .stdout(contains("Resolved extensions:"))
        .stdout(contains("Packaged extensions:"))
        .stdout(contains("used_ext").not())
        .stdout(contains("public_fn").not())
        .stdout(contains("resolved_ext"))
        .stdout(contains("Generator claimed extensions").not())
        .stdout(contains("packed_ext"));
}

#[rstest]
fn test_describe_output_redirection(empty_package_file: NamedTempFile, mut describe_cmd: Command) {
    let output_file = assert_fs::NamedTempFile::new("output.txt").unwrap();
    describe_cmd.args([
        empty_package_file.path(),
        std::path::Path::new("--output"),
        output_file.path(),
    ]);
    describe_cmd.assert().success();
    output_file.assert(contains("Package contains 0 module(s) and 0 extension(s)"));
}

#[rstest]
fn test_no_resolved_extensions(package_with_exts: Vec<u8>, mut describe_cmd: Command) {
    describe_cmd.write_stdin(package_with_exts);

    describe_cmd.arg("--no-resolved-extensions");
    describe_cmd
        .assert()
        .success()
        .stdout(contains("Packaged extensions:").not())
        .stdout(contains("Resolved extensions:").not());
}

#[rstest]
fn test_public_symbols(package_with_exts: Vec<u8>, mut describe_cmd: Command) {
    describe_cmd.write_stdin(package_with_exts);

    describe_cmd.arg("--public-symbols");
    describe_cmd
        .assert()
        .success()
        .stdout(contains("Public symbols:"))
        .stdout(contains("public_fn"));
}

#[rstest]
fn test_generator_claimed_extensions(package_with_exts: Vec<u8>, mut describe_cmd: Command) {
    describe_cmd.write_stdin(package_with_exts);

    describe_cmd.arg("--generator-claimed-extensions");
    describe_cmd
        .assert()
        .success()
        .stdout(contains("Generator claimed extensions"))
        .stdout(contains("used_ext"));
}

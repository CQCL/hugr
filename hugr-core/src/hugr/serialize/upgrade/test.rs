use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::BOOL_T,
    hugr::serialize::test::check_hugr_deserialize,
    std_extensions::logic::NaryLogic,
    type_row,
    types::Signature,
};
use lazy_static::lazy_static;
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
};

use crate::Hugr;

struct TestCase {
    name: String,
    hugr: Hugr,
}

lazy_static! {
    static ref TEST_CASE_DIR: PathBuf = {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(file!())
            .parent()
            .unwrap()
            .join("testcases")
    };
}

#[test]
fn test_case_dir_exists() {
    let test_case_dir: &Path = &TEST_CASE_DIR;
    assert!(
        test_case_dir.exists(),
        "Upgrade test case directory does not exist: {:?}",
        test_case_dir
    );
}

impl TestCase {
    fn new(name: impl Into<String>, hugr: Hugr) -> Self {
        Self {
            name: name.into(),
            hugr,
        }
    }

    fn run(&self) {
        let path = TEST_CASE_DIR.join(format!("{}.json", &self.name));
        if !path.exists() {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&path)
                .unwrap();
            serde_json::to_writer_pretty(f, &self.hugr).unwrap();
        }

        let val = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        // we do not expect `val` to satisfy any schemas, it is a non-latest
        // version.
        check_hugr_deserialize(&self.hugr, val, false);
    }
}

#[test]
fn empty_hugr() {
    TestCase::new("empty_hugr", Hugr::default()).run();
}

#[test]
#[cfg_attr(feature = "extension_inference", ignore = "Fails extension inference")]
fn hugr_with_named_op() {
    let mut builder =
        DFGBuilder::new(Signature::new(type_row![BOOL_T, BOOL_T], type_row![BOOL_T])).unwrap();
    let [a, b] = builder.input_wires_arr();
    let x = builder
        .add_dataflow_op(NaryLogic::And.with_n_inputs(2), [a, b])
        .unwrap();
    let h = builder
        .finish_prelude_hugr_with_outputs(x.outputs())
        .unwrap();
    TestCase::new("hugr_with_named_op", h).run();
}

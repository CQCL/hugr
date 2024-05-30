use std::{path::{Path, PathBuf}, thread::panicking};
use lazy_static::lazy_static;
use crate::hugr::serialize::test::check_hugr_deserialize;

use crate::Hugr;

struct TestCase {
    name: String,
    hugr: Hugr
}

lazy_static! {
    static ref TEST_CASE_DIR: PathBuf = {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(file!()).parent().unwrap().join("testcases")
    };
}

#[test]
fn test_case_dir_exists() {
    let test_case_dir: &Path = &TEST_CASE_DIR;
    assert!(test_case_dir.exists(), "Upgrade test case directory does not exist: {:?}", test_case_dir);
}


impl TestCase {
    fn new(name: impl Into<String>, hugr: Hugr) -> Self {
        Self {
            name: name.into(),
            hugr
        }
    }

    fn run(&self) {
        let path = TEST_CASE_DIR.join(format!("{}.json", &self.name));
        if !path.exists() {
            let f = std::fs::File::create_new(&path).unwrap();
            serde_json::to_writer_pretty(f, &self.hugr).unwrap();
        }

        let val = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        check_hugr_deserialize(&self.hugr, val, true);
    }
}

#[test]
fn empty_hugr() {
    TestCase::new("empty_hugr", Hugr::default()).run();
}

#![allow(missing_docs)]

use bumpalo::Bump;
use hugr::{package::Package, std_extensions::STD_REG, Hugr, HugrView};
use hugr_core::{export::export_hugr, import::import_hugr};
use hugr_model::v0 as model;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bump = Bump::new();
    let hugr = load_qv();
    let module = export_hugr(&hugr, &bump);
    std::fs::write(
        "qv_hugr.edn",
        model::text::print_to_string(&module, 120).unwrap(),
    )
    .unwrap();

    let binary = model::binary::write_to_vec(&module);
    println!("size in bytes: {}", binary.len());
    std::fs::write("qv_hugr.hugr", binary).unwrap();

    // println!("{:#?}", module);
    Ok(())
}

fn load_qv() -> Hugr {
    let qv = include_str!("../qv_hugr.json");
    let package: Package = serde_json::from_str(qv).unwrap();
    package.modules[0].clone()
}

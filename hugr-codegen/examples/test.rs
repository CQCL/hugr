#![allow(missing_docs)]
use bumpalo::Bump;
use hugr_codegen::rust::Generator;
use hugr_model::v0 as model;

pub fn main() {
    let mut gen = Generator::new();

    let files = [
        include_str!("../../hugr-model/extensions/core.edn"),
        include_str!("../../hugr-model/extensions/int.edn"),
        include_str!("../../hugr-model/extensions/array.edn"),
    ];

    for file in files {
        let bump = Bump::new();
        let module = model::text::parse(file, &bump).unwrap().module;
        gen.add_module(&module);
    }

    println!("{}", gen.as_str());
}

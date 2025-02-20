use bumpalo::Bump;
use hugr_model::v0 as model;

pub fn main() {
    let ext = include_str!("../../hugr-model/extensions/core.edn");
    let bump = Bump::new();
    let module = model::text::parse(ext, &bump).unwrap().module;
    println!("{}", hugr_codegen::rust::generate(&module));
}

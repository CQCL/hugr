//! compile the capnp schema
fn main() {
    ::capnpc::CompilerCommand::new()
        .file("src/serialization/hugr.capnp")
        .src_prefix("src/serialization")
        .default_parent_module(vec!["serialization".to_string()])
        .run()
        .expect("compiling schema");
}

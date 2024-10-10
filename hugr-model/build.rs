//! Build scripts for `hugr-model`.

/// Build the capnp schema files.
fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("capnp")
        .file("capnp/hugr-v0.capnp")
        .run()
        .expect("compiling schema");

    println!("cargo:rerun-if-changed=capnp/hugr-v0.capnp");
}

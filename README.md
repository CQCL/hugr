# hugr-llvm

A general, extensible, rust crate for lowering `HUGR`s into `LLVM`-ir. 

# Building

Requires `llvm`. At present, only `llvm-14` is supported, but this limitation can easily be lifted.

See the `llvm-sys` crate for details on how to use your preferred llvm installation.

A `devenv.sh` nix environment is provided, in which `cargo build && cargo test`
should work without any further configuration.

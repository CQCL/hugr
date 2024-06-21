# hugr-llvm

[![build_status][]](https://github.com/CQCL/hugr-llvm/actions)
[![codecov](https://codecov.io/github/CQCL/hugr-llvm/graph/badge.svg?token=TN3DSNHF43)](https://codecov.io/github/CQCL/hugr-llvm)
[![msrv][]](https://github.com/CQCL/hugr-llvm)


A general, extensible, rust crate for lowering `HUGR`s into `LLVM`-ir. 

# Building

Requires `llvm`. At present, only `llvm-14` is supported, but this limitation can easily be lifted.

See the `llvm-sys` crate for details on how to use your preferred llvm installation.

A `devenv.sh` nix environment is provided, in which `cargo build && cargo test`
should work without any further configuration.

  [build_status]: https://github.com/CQCL/hugr-llvm/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg

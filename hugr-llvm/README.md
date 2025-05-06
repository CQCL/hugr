# hugr-llvm

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![codecov](https://codecov.io/github/CQCL/hugr/graph/badge.svg?token=TN3DSNHF43)](https://codecov.io/github/CQCL/hugr)
[![msrv][]](https://github.com/CQCL/hugr/tree/main/hugr-llvm)

A general, extensible, rust crate for lowering `HUGR`s into `LLVM` IR. Built on [hugr][], [inkwell][], and [llvm][].

## Usage

You'll need to point your `Cargo.toml` to use a single LLVM version feature flag corresponding to your LLVM version, by calling

```bash
cargo add hugr-llvm --features llvm14-0
```

At present only `llvm14-0` is supported but we expect to introduce supported versions as required. Contributions are welcome.

See the [llvm-sys][] crate for details on how to use your preferred llvm installation.

## Recent Changes

See [CHANGELOG](CHANGELOG.md) for a list of changes. The minimum supported rust
version will only change on major releases.

## Developing hugr-llvm

See [DEVELOPMENT](../DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENCE](LICENCE) or <http://www.apache.org/licenses/LICENSE-2.0>).

  [build_status]: https://github.com/CQCL/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/crates/msrv/hugr-llvm
  [hugr]: https://lib.rs/crates/hugr
  [inkwell]: https://thedan64.github.io/inkwell/inkwell/index.html
  [llvm-sys]: https://crates.io/crates/llvm-sys
  [llvm]: https://llvm.org/

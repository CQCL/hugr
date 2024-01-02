quantinuum-hugr
===============

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![msrv][]](https://github.com/CQCL/hugr)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

The Hierarchical Unified Graph Representation (HUGR, pronounced _hugger_) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

It provides a high-fidelity representation of operations, that facilitates
compilation and encodes runnable programs.

The HUGR specification is [here](specification/hugr.md).

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
quantinuum-hugr = "0.1"
```

The library crate is called `hugr`.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [build_status]: https://github.com/CQCL/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
  [LICENSE]: LICENCE

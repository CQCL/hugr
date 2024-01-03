quantinuum-hugr
===============

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![crates][]](https://crates.io/crates/quantinuum-hugr)
[![msrv][]](https://github.com/CQCL/hugr)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

The Hierarchical Unified Graph Representation (HUGR, pronounced _hugger_) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

It provides a high-fidelity representation of operations, that facilitates
compilation and encodes runnable programs.

The HUGR specification is [here](specification/hugr.md).

## Usage

Add the dependency to your project:

```bash
cargo add quantinuum-hugr
```

The library crate is called `hugr`.

Please read the [API documentation here][].

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [API documentation here]: https://docs.rs/quantinuum-hugr/
  [build_status]: https://github.com/CQCL/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.70.0%2B-blue.svg
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
  [LICENSE]: LICENCE
  [CHANGELOG]: CHANGELOG.md

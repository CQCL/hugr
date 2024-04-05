hugr
===============

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![crates][]](https://crates.io/crates/hugr)
[![msrv][]](https://github.com/CQCL/hugr)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

The Hierarchical Unified Graph Representation (HUGR, pronounced _hugger_) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

It provides a high-fidelity representation of operations, that facilitates
compilation and encodes runnable programs.

The HUGR specification is [here](https://github.com/CQCL/hugr/blob/main/specification/hugr.md).

## Usage

Add the dependency to your project:

```bash
cargo add hugr
```

Please read the [API documentation here][].

## Experimental Features

- `extension_inference`:
  Experimental feature which allows automatic inference of extension usages and
  requirements in a HUGR and validation that extensions are correctly specified.
  Not enabled by default.

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](https://github.com/CQCL/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [API documentation here]: https://docs.rs/hugr/
  [build_status]: https://github.com/CQCL/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
  [crates]: https://img.shields.io/crates/v/hugr
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
  [LICENSE]: https://github.com/CQCL/hugr/blob/main/LICENCE
  [CHANGELOG]: https://github.com/CQCL/hugr/blob/main/CHANGELOG.md

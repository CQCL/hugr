![](/hugr/assets/hugr_logo.svg)

# hugr

[![build_status][]](https://github.com/quantinuum/hugr/actions)
[![crates][]](https://crates.io/crates/hugr)
[![msrv][]](https://github.com/quantinuum/hugr)
[![codecov][]](https://codecov.io/gh/quantinuum/hugr)

The Hierarchical Unified Graph Representation (HUGR, pronounced _hugger_) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

It provides a high-fidelity representation of operations, that facilitates
compilation and encodes runnable programs.

The HUGR specification is [here](https://github.com/quantinuum/hugr/blob/main/specification/hugr.md).

## Usage

Add the dependency to your project:

```bash
cargo add hugr
```

Please read the [API documentation here][].

## Experimental Features

- `declarative`:
  Experimental support for declaring extensions in YAML files, support is limited.

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](https://github.com/quantinuum/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment and on making releases.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [API documentation here]: https://docs.rs/hugr/
  [build_status]: https://github.com/quantinuum/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/crates/msrv/hugr
  [crates]: https://img.shields.io/crates/v/hugr
  [codecov]: https://img.shields.io/codecov/c/gh/quantinuum/hugr?logo=codecov
  [LICENSE]: https://github.com/quantinuum/hugr/blob/main/LICENCE
  [CHANGELOG]: https://github.com/quantinuum/hugr/blob/main/hugr/CHANGELOG.md

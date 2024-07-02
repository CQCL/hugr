![](/hugr/assets/hugr_logo.svg)

hugr-cli
===============

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![crates][]](https://crates.io/crates/hugr-cli)
[![msrv][]](https://github.com/CQCL/hugr)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

`hugr` CLI tool for common tasks on serialized HUGR (e.g. validation,
visualisation).

Refer to the [main HUGR crate](http://crates.io/crates/hugr) for more information.

## Usage

Install using `cargo`:

```bash
cargo install hugr-cli
```

This will install the `hugr` binary. Running `hugr --help` shows:

```
Validate a HUGR.

Usage: hugr [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -m, --mermaid      Visualise with mermaid.
  -n, --no-validate  Skip validation.
  -v, --verbose...   Increase logging verbosity
  -q, --quiet...     Decrease logging verbosity
  -h, --help         Print help
  -V, --version      Print version
```


To extend the CLI you can also add the project as a library dependency:

```bash
cargo add hugr-cli
```

Please read the [API documentation here][].

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](https://github.com/CQCL/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [API documentation here]: https://docs.rs/hugr-cli/
  [build_status]: https://github.com/CQCL/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
  [crates]: https://img.shields.io/crates/v/hugr-cli
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
  [LICENSE]: https://github.com/CQCL/hugr/blob/main/LICENCE
  [CHANGELOG]: https://github.com/CQCL/hugr/blob/main/hugr-cli/CHANGELOG.md

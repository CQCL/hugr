![](/hugr/assets/hugr_logo.svg)

# hugr-core

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![crates][]](https://crates.io/crates/hugr-core)
[![msrv][]](https://github.com/CQCL/hugr)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

Internal core definitions for the `hugr` package.
Refer to the [main crate](http://crates.io/crates/hugr) for more information.

Please read the [API documentation here][].

## Experimental Features

- `extension_inference`:
  Experimental feature which allows automatic inference of which extra extensions
  are required at runtime by a HUGR when validating it.
  Not enabled by default.
- `declarative`:
  Experimental support for declaring extensions in YAML files, support is limited.

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](https://github.com/CQCL/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

[API documentation here]: https://docs.rs/hugr-core/
[build_status]: https://github.com/CQCL/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
[msrv]: https://img.shields.io/badge/rust-1.75.0%2B-blue.svg
[crates]: https://img.shields.io/crates/v/hugr-core
[codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
[LICENSE]: https://github.com/CQCL/hugr/blob/main/LICENCE
[CHANGELOG]: https://github.com/CQCL/hugr/blob/main/hugr-core/CHANGELOG.md

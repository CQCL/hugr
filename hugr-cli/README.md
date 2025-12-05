![](/hugr/assets/hugr_logo.svg)

hugr-cli
===============

[![build_status][]](https://github.com/quantinuum/hugr/actions)
[![crates][]](https://crates.io/crates/hugr-cli)
[![msrv][]](https://github.com/quantinuum/hugr)
[![codecov][]](https://codecov.io/gh/quantinuum/hugr)

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
HUGR CLI tools.

Usage: hugr [OPTIONS] <COMMAND>

Commands:
  validate        Validate a HUGR package
  gen-extensions  Write standard extensions out in serialized form
  mermaid         Write HUGR as mermaid diagrams
  convert         Convert between different HUGR envelope formats
  help            Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose...  Increase logging verbosity
  -q, --quiet...    Decrease logging verbosity
  -h, --help        Print help
  -V, --version     Print version
```

Refer to the help for each subcommand for more information, e.g.

```
hugr validate --help
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

See [DEVELOPMENT.md](https://github.com/quantinuum/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [API documentation here]: https://docs.rs/hugr-cli/
  [build_status]: https://github.com/quantinuum/hugr/actions/workflows/ci-rs.yml/badge.svg?branch=main
  [msrv]: https://img.shields.io/crates/msrv/hugr-cli
  [crates]: https://img.shields.io/crates/v/hugr-cli
  [codecov]: https://img.shields.io/codecov/c/gh/quantinuum/hugr?logo=codecov
  [LICENSE]: https://github.com/quantinuum/hugr/blob/main/LICENCE
  [CHANGELOG]: https://github.com/quantinuum/hugr/blob/main/hugr-cli/CHANGELOG.md

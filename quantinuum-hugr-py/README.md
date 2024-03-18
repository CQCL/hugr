quantinuum-hugr
===============

[![build_status][]](https://github.com/CQCL/hugr/actions)
[![codecov][]](https://codecov.io/gh/CQCL/hugr)

The Hierarchical Unified Graph Representation (HUGR, pronounced _hugger_) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

This library provides a pure-python implementation of the HUGR data model, and
a low-level API for constructing HUGR objects.

This library is intended to be used as a dependency for other high-level tools.
See [`guppylang`][] and [`tket2`][] for examples of such tools.

The HUGR specification is [here](https://github.com/CQCL/hugr/blob/main/specification/hugr.md).

  [`guppylang`]: https://pypi.org/project/guppylang/
  [`tket2`]: https://github.com/CQCL/tket2


## Install

`quantinuum-hugr` can be installed via `pip`. Requires Python >= 3.10.

```sh
pip install quantinuum-hugr
```

## Usage

TODO

## Recent Changes

See [CHANGELOG][] for a list of changes. The minimum supported rust
version will only change on major releases.

## Development

See [DEVELOPMENT.md](https://github.com/CQCL/hugr/blob/main/DEVELOPMENT.md) for instructions on setting up the development environment.

## License

This project is licensed under Apache License, Version 2.0 ([LICENSE][] or http://www.apache.org/licenses/LICENSE-2.0).

  [build_status]: https://github.com/CQCL/hugr/workflows/Continuous%20integration/badge.svg?branch=main
  [codecov]: https://img.shields.io/codecov/c/gh/CQCL/hugr?logo=codecov
  [LICENSE]: https://github.com/CQCL/hugr/blob/main/LICENCE
  [CHANGELOG]: https://github.com/CQCL/hugr/blob/main/quantinuum-hugr-py/CHANGELOG.md

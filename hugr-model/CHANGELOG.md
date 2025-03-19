# Changelog

## [0.18.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.17.1...hugr-model-v0.18.0) - 2025-03-14

### Bug Fixes

- Hugr-model using undeclared derive_more features ([#1940](https://github.com/CQCL/hugr/pull/1940))

### New Features

- *(hugr-model)* [**breaking**] Add `read_from_reader` and `write_to_writer` for streaming reads and writes. ([#1871](https://github.com/CQCL/hugr/pull/1871))
- `hugr-model` AST ([#1953](https://github.com/CQCL/hugr/pull/1953))

### Refactor

- *(hugr-model)* Reexport `bumpalo` from `hugr-model` ([#1870](https://github.com/CQCL/hugr/pull/1870))

## [0.17.1](https://github.com/CQCL/hugr/compare/hugr-model-v0.17.0...hugr-model-v0.17.1) - 2025-02-05

### Bug Fixes

- determine correct bounds of custom types (#1888)

### New Features

- Special cased array, float and int constants in hugr-model export (#1857)
- Simplify hugr-model (#1893)
- Do not require `capnp` to be installed to compile `hugr-model` (#1907)

## [0.17.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.16.0...hugr-model-v0.17.0) - 2025-01-20

### Bug Fixes

- Three bugfixes in model import and export. (#1844)

### New Features

- Constant values in `hugr-model` (#1838)
- Bytes literal in hugr-model. (#1845)
- Improved representation for metadata in `hugr-model` (#1849)

## [0.16.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.15.0...hugr-model-v0.16.0) - 2024-12-18

### New Features

- Scoping rules and utilities for symbols, links and variables (#1754)

## [0.15.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.14.0...hugr-model-v0.15.0) - 2024-12-16

### Bug Fixes

- Ignare lint warnings in capnproto generated code (#1728)

### New Features

- Export/import of JSON metadata (#1622)
- Emulate `TypeBound`s on parameters via constraints. (#1624)
- Lists and extension sets with splicing (#1657)
- [**breaking**] Have `CustomType`s reference their `Extension` definition (#1723)

### Performance

- Faster singleton SiblingSubgraph construction (#1654)

## [0.14.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.13.2...hugr-model-v0.14.0) - 2024-11-06

### New Features

- Operation and constructor declarations in `hugr-model` ([#1605](https://github.com/CQCL/hugr/pull/1605))

## [0.13.2](https://github.com/CQCL/hugr/compare/hugr-model-v0.13.1...hugr-model-v0.13.2) - 2024-10-22

### New Features

- make errors more readable with Display impls ([#1597](https://github.com/CQCL/hugr/pull/1597))

## [0.13.1](https://github.com/CQCL/hugr/compare/hugr-model-v0.1.0...hugr-model-v0.13.1) - 2024-10-14

This release bumps the version to align with the other `hugr-*` crates.

### New Features

- Binary serialisation format for hugr-model based on capnproto. ([#1557](https://github.com/CQCL/hugr/pull/1557))

# Changelog

## [0.15.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.14.0...hugr-model-v0.15.0) - 2024-12-10

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

# Changelog


## [0.23.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.22.4...hugr-model-v0.23.0) - 2025-09-30

### Bug Fixes

- [**breaking**] Appease `cargo-audit` by replacing unmaintained dependencies ([#2572](https://github.com/CQCL/hugr/pull/2572))

### New Features

- Documentation and error hints ([#2523](https://github.com/CQCL/hugr/pull/2523))

## [0.22.4](https://github.com/CQCL/hugr/compare/hugr-model-v0.22.3...hugr-model-v0.22.4) - 2025-09-24

### New Features

- Documentation and error hints ([#2523](https://github.com/CQCL/hugr/pull/2523))

## [0.22.2](https://github.com/CQCL/hugr/compare/hugr-model-v0.22.1...hugr-model-v0.22.2) - 2025-08-06

### New Features

- Type of constants in `core` `Term`s. ([#2411](https://github.com/CQCL/hugr/pull/2411))

## [0.22.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.21.0...hugr-model-v0.22.0) - 2025-07-24

### New Features

- Names of private functions become `core.title` metadata. ([#2448](https://github.com/CQCL/hugr/pull/2448))
- include generator metatada in model import and cli validate errors ([#2452](https://github.com/CQCL/hugr/pull/2452))
- Version number in hugr binary format. ([#2468](https://github.com/CQCL/hugr/pull/2468))
- Use semver crate for -model version, and include in docs ([#2471](https://github.com/CQCL/hugr/pull/2471))
## [0.21.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.20.2...hugr-model-v0.21.0) - 2025-07-09

### Bug Fixes

- Model import should perform extension resolution ([#2326](https://github.com/CQCL/hugr/pull/2326))
- [**breaking**] Fixed bugs in model CFG handling and improved CFG signatures ([#2334](https://github.com/CQCL/hugr/pull/2334))
- [**breaking**] Fix panic in model resolver when variable is used outside of symbol. ([#2362](https://github.com/CQCL/hugr/pull/2362))
- Order hints on input and output nodes. ([#2422](https://github.com/CQCL/hugr/pull/2422))

### New Features

- [**breaking**] Added float and bytes literal to core and python bindings. ([#2289](https://github.com/CQCL/hugr/pull/2289))
- [**breaking**] Add Visibility to FuncDefn/FuncDecl. ([#2143](https://github.com/CQCL/hugr/pull/2143))
- [**breaking**] hugr-model use explicit Option<Visibility>, with ::Unspecified in capnp ([#2424](https://github.com/CQCL/hugr/pull/2424))

## [0.20.2](https://github.com/CQCL/hugr/compare/hugr-model-v0.20.1...hugr-model-v0.20.2) - 2025-06-25

### New Features

- better errors using metadata from generator ([#2368](https://github.com/CQCL/hugr/pull/2368))

## [0.20.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.19.0...hugr-model-v0.20.0) - 2025-05-14

### New Features

- [**breaking**] Mark all Error enums as non_exhaustive ([#2056](https://github.com/CQCL/hugr/pull/2056))
- [**breaking**] Bump MSRV to 1.85 ([#2136](https://github.com/CQCL/hugr/pull/2136))
- Export and import entrypoints via metadata in `hugr-model`. ([#2172](https://github.com/CQCL/hugr/pull/2172))
- Define text-model envelope formats ([#2188](https://github.com/CQCL/hugr/pull/2188))
- Import CFG regions without adding an entry block. ([#2200](https://github.com/CQCL/hugr/pull/2200))
- Symbol applications can leave out prefixes of wildcards. ([#2201](https://github.com/CQCL/hugr/pull/2201))

## [0.19.0](https://github.com/CQCL/hugr/compare/hugr-model-v0.18.1...hugr-model-v0.19.0) - 2025-05-07

### New Features

- Python bindings for `hugr-model`. ([#1959](https://github.com/CQCL/hugr/pull/1959))
- Remove extension sets from `hugr-model`. ([#2031](https://github.com/CQCL/hugr/pull/2031))
- Packages in `hugr-model` and envelope support. ([#2026](https://github.com/CQCL/hugr/pull/2026))
- Represent order edges in `hugr-model` as metadata. ([#2027](https://github.com/CQCL/hugr/pull/2027))

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

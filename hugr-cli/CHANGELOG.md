# Changelog


## [0.14.0](https://github.com/CQCL/hugr/compare/hugr-cli-v0.13.3...hugr-cli-v0.14.0) - 2024-11-19

### Performance

- Faster singleton SiblingSubgraph construction ([#1654](https://github.com/CQCL/hugr/pull/1654))

## [0.13.2](https://github.com/CQCL/hugr/compare/hugr-cli-v0.13.1...hugr-cli-v0.13.2) - 2024-10-22

### New Features

- Add `Package` definition on `hugr-core` ([#1587](https://github.com/CQCL/hugr/pull/1587))
- Ensure packages always have modules at the root ([#1589](https://github.com/CQCL/hugr/pull/1589))

## 0.13.1 (2024-10-14)

This release bumps the version to align with the other `hugr-*` crates.


## 0.6.0 (2024-09-04)

### Features

- [**breaking**] Allow registry specification in `run_dump` ([#1501](https://github.com/CQCL/hugr/pull/1501))
- [**breaking**] Add `Package::validate` and return `ExtensionRegistry` in helpers. ([#1507](https://github.com/CQCL/hugr/pull/1507))


## 0.5.0 (2024-08-30)

### Features

- [**breaking**] Add collections to serialized standard extensions ([#1452](https://github.com/CQCL/hugr/pull/1452))


## 0.4.0 (2024-08-12)

### Features

- Serialised standard extensions ([#1377](https://github.com/CQCL/hugr/pull/1377))
- Validate with extra extensions and packages ([#1389](https://github.com/CQCL/hugr/pull/1389))
- [**breaking**] Move mermaid to own sub-command ([#1390](https://github.com/CQCL/hugr/pull/1390))


## 0.3.0 (2024-07-26)

### Features

- [**breaking**] Created `validate` CLI subcommand. ([#1312](https://github.com/CQCL/hugr/pull/1312))


## 0.2.1 (2024-07-25)

- Updated `hugr` dependencies.


## 0.2.0 (2024-07-19)

### Refactor

- [**breaking**] Separate Signature from FuncValueType by parametrizing Type(/Row)/etc. ([#1138](https://github.com/CQCL/hugr/pull/1138))



## 0.1.3 (2024-07-10)

### Styling

- Change "serialise" etc to "serialize" etc. ([#1251](https://github.com/CQCL/hugr/pull/1251))



## 0.1.1 (2024-06-07)

### Features

- Reexport `clap::Parser` and `clap_verbosity_flag::Level` from hugr_cli ([#1146](https://github.com/CQCL/hugr/pull/1146))

### Refactor

- Move binary to hugr-cli ([#1134](https://github.com/CQCL/hugr/pull/1134))


## 0.1.0 (2024-05-29)

Initial release, ported from `hugr::cli` module.

### Bug Fixes

- Set initial version of hugr-core to 0.1.0 ([#1129](https://github.com/CQCL/hugr/pull/1129))

### Features

- [**breaking**] Move cli in to hugr-cli sub-crate ([#1107](https://github.com/CQCL/hugr/pull/1107))
- Add verbosity, return `Hugr` from `run`. ([#1116](https://github.com/CQCL/hugr/pull/1116))

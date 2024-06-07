# Changelog

## 0.2.0 (2024-06-07)

### Bug Fixes

- [**breaking**] Validate that control-flow outputs have exactly one successor ([#1144](https://github.com/CQCL/hugr/pull/1144))
- Do not require matching extension_reqs when creating a replacement ([#1177](https://github.com/CQCL/hugr/pull/1177))

### Features

- Add `ConstExternalSymbol` to prelude ([#1123](https://github.com/CQCL/hugr/pull/1123))
- `HugrView::extract_hugr` to extract regions into owned hugrs. ([#1173](https://github.com/CQCL/hugr/pull/1173))

### Testing

- Serialisation round trip testing for `OpDef` ([#999](https://github.com/CQCL/hugr/pull/999))


## 0.1.0 (2024-05-29)

### Bug Fixes

- Set initial version of hugr-core to 0.1.0 ([#1129](https://github.com/CQCL/hugr/pull/1129))

### Features

- [**breaking**] Move cli in to hugr-cli sub-crate ([#1107](https://github.com/CQCL/hugr/pull/1107))
- Make internals of int ops and the "int" CustomType more public. ([#1114](https://github.com/CQCL/hugr/pull/1114))
- Unseal and make public the traits `HugrInternals` and `HugrMutInternals` ([#1122](https://github.com/CQCL/hugr/pull/1122))

### Refactor

- Add a `hugr-core` crate ([#1108](https://github.com/CQCL/hugr/pull/1108))

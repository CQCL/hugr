# Changelog

## 0.6.0 (2024-07-19)

### Refactor

- [**breaking**] Separate Signature from FuncValueType by parametrizing Type(/Row)/etc. ([#1138](https://github.com/CQCL/hugr/pull/1138))


## 0.5.0 (2024-07-16)

### Bug Fixes

- [**breaking**] Ops require their own extension ([#1226](https://github.com/CQCL/hugr/pull/1226))
- [**breaking**] Force_order failing on Const nodes, add arg to rank. ([#1300](https://github.com/CQCL/hugr/pull/1300))

### Documentation

- Attempt to correct force_order docs ([#1299](https://github.com/CQCL/hugr/pull/1299))

### Refactor

- [**breaking**] Rename builder helpers: ft1->endo_ft, ft2->inout_ft ([#1297](https://github.com/CQCL/hugr/pull/1297))


## 0.4.0 (2024-07-10)

### Features

- Add `force_order` pass. ([#1285](https://github.com/CQCL/hugr/pull/1285))

### Refactor

- [**breaking**] Remove `Value::Tuple` ([#1255](https://github.com/CQCL/hugr/pull/1255))


## 0.3.0 (2024-06-28)

### Features

- [**breaking**] Validate Extensions using hierarchy, ignore input_extensions, RIP inference ([#1142](https://github.com/CQCL/hugr/pull/1142))
- Helper functions for requesting inference, use with builder in tests ([#1219](https://github.com/CQCL/hugr/pull/1219))


## 0.2.0 (2024-06-07)

### Features

- Add `ValidationLevel` tooling and apply to `constant_fold_pass` ([#1035](https://github.com/CQCL/hugr/pull/1035))


## 0.1.0 (2024-05-29)

Initial release, with functions ported from the `hugr::algorithms` module.

### Bug Fixes

- Set initial version of hugr-core to 0.1.0 ([#1129](https://github.com/CQCL/hugr/pull/1129))

### Features

- [**breaking**] Move passes from `algorithms` into a separate crate ([#1100](https://github.com/CQCL/hugr/pull/1100))
- [**breaking**] Move cli in to hugr-cli sub-crate ([#1107](https://github.com/CQCL/hugr/pull/1107))

### Refactor

- Add a `hugr-core` crate ([#1108](https://github.com/CQCL/hugr/pull/1108))

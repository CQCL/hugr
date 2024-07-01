# Changelog

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

# Changelog

## 0.8.1 (2024-09-04)

### Features

- Op replacement and lowering functions ([#1509](https://github.com/CQCL/hugr/pull/1509))


## 0.8.0 (2024-08-30)

### Features

- [**breaking**] Variadic logic ops now binary ([#1451](https://github.com/CQCL/hugr/pull/1451))
- [**breaking**] Int operations other than widen/narrow have only one width arg ([#1455](https://github.com/CQCL/hugr/pull/1455))
- [**breaking**] Move `Lift`, `MakeTuple`, `UnpackTuple` and `Lift` to prelude ([#1475](https://github.com/CQCL/hugr/pull/1475))
- [**breaking**] Add more list operations ([#1474](https://github.com/CQCL/hugr/pull/1474))
- [**breaking**] Move int conversions to `conversions` ext, add to/from usize ([#1490](https://github.com/CQCL/hugr/pull/1490))

### Refactor

- [**breaking**] Flatten `CustomOp` in to `OpType` ([#1429](https://github.com/CQCL/hugr/pull/1429))
- [**breaking**] Bring the collections ext in line with other extension defs ([#1469](https://github.com/CQCL/hugr/pull/1469))
- [**breaking**] Make Either::Right the "success" case ([#1489](https://github.com/CQCL/hugr/pull/1489))


## 0.7.0 (2024-08-12)

### Features

- [**breaking**] `Extension` requires a version ([#1367](https://github.com/CQCL/hugr/pull/1367))


## 0.6.2 (2024-07-26)

### Features

- Add `nonlocal_edges` and `ensure_no_nonlocal_edges` ([#1345](https://github.com/CQCL/hugr/pull/1345))


## 0.6.1 (2024-07-25)

- Updated `hugr` dependencies.


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

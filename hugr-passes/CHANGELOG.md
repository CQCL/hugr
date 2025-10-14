# Changelog


## [0.24.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.23.0...hugr-passes-v0.24.0) - 2025-10-13

### New Features

- Add handler for copying / discarding borrow arrays to default lineariser ([#2602](https://github.com/CQCL/hugr/pull/2602))

## [0.23.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.22.4...hugr-passes-v0.23.0) - 2025-09-30

### Bug Fixes

- DeadCodeElim keeps consumers of linear outputs ([#2560](https://github.com/CQCL/hugr/pull/2560))
- [**breaking**] Appease `cargo-audit` by replacing unmaintained dependencies ([#2572](https://github.com/CQCL/hugr/pull/2572))

### Miscellaneous Tasks

- [**breaking**] Cleanup deprecated definitions ([#2594](https://github.com/CQCL/hugr/pull/2594))

### New Features

- [**breaking**] DeadCodeElimPass reports error on non-existent entry_points ([#2566](https://github.com/CQCL/hugr/pull/2566))
- Normalize CFGs ([#2591](https://github.com/CQCL/hugr/pull/2591))

### Refactor

- [**breaking**] Replace lazy_static with std::sync::LazyLock ([#2567](https://github.com/CQCL/hugr/pull/2567))

## [0.22.4](https://github.com/CQCL/hugr/compare/hugr-passes-v0.22.3...hugr-passes-v0.22.4) - 2025-09-24

### Bug Fixes

- DeadCodeElim keeps consumers of linear outputs ([#2560](https://github.com/CQCL/hugr/pull/2560))

## [0.22.3](https://github.com/CQCL/hugr/compare/hugr-passes-v0.22.2...hugr-passes-v0.22.3) - 2025-09-11

### New Features

- SiblingSubgraph supports function calls ([#2528](https://github.com/CQCL/hugr/pull/2528))

## [0.22.1](https://github.com/CQCL/hugr/compare/hugr-passes-v0.22.0...hugr-passes-v0.22.1) - 2025-07-28

### New Features

- Include copy_discard_array in DelegatingLinearizer::default ([#2479](https://github.com/CQCL/hugr/pull/2479))
- Inline calls to functions not on cycles in the call graph ([#2450](https://github.com/CQCL/hugr/pull/2450))

## [0.22.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.21.0...hugr-passes-v0.22.0) - 2025-07-24

### New Features

- ReplaceTypes allows linearizing inside Op replacements ([#2435](https://github.com/CQCL/hugr/pull/2435))
- Add pass for DFG inlining ([#2460](https://github.com/CQCL/hugr/pull/2460))

## [0.21.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.20.2...hugr-passes-v0.21.0) - 2025-07-09

### Bug Fixes

- DeadFuncElimPass+CallGraph w/ non-module-child entrypoint ([#2390](https://github.com/CQCL/hugr/pull/2390))

### New Features

- [**breaking**] No nested FuncDefns (or AliasDefns) ([#2256](https://github.com/CQCL/hugr/pull/2256))
- [**breaking**] Split `TypeArg::Sequence` into tuples and lists. ([#2140](https://github.com/CQCL/hugr/pull/2140))
- [**breaking**] Merge `TypeParam` and `TypeArg` into one `Term` type in Rust ([#2309](https://github.com/CQCL/hugr/pull/2309))
- [**breaking**] Rename 'Any' type bound to 'Linear' ([#2421](https://github.com/CQCL/hugr/pull/2421))

### Refactor

- [**breaking**] Reduce error type sizes ([#2420](https://github.com/CQCL/hugr/pull/2420))

## [0.20.2](https://github.com/CQCL/hugr/compare/hugr-passes-v0.20.1...hugr-passes-v0.20.2) - 2025-06-25

### Bug Fixes

- update CallGraph and remove_dead_funcs for module-only FuncDefns ([#2336](https://github.com/CQCL/hugr/pull/2336))

## [0.20.1](https://github.com/CQCL/hugr/compare/hugr-passes-v0.20.0...hugr-passes-v0.20.1) - 2025-06-03

### Bug Fixes

- Dataflow analysis produces unsound results on Hugrs with entrypoint ([#2255](https://github.com/CQCL/hugr/pull/2255))

### New Features

- LocalizeEdges pass ([#2237](https://github.com/CQCL/hugr/pull/2237))

## [0.20.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.15.4...hugr-passes-v0.20.0) - 2025-05-14

### New Features

- [**breaking**] Mark all Error enums as non_exhaustive ([#2056](https://github.com/CQCL/hugr/pull/2056))
- [**breaking**] Handle CallIndirect in Dataflow Analysis ([#2059](https://github.com/CQCL/hugr/pull/2059))
- [**breaking**] ComposablePass trait allowing sequencing and validation ([#1895](https://github.com/CQCL/hugr/pull/1895))
- [**breaking**] ReplaceTypes: allow lowering ops into a Call to a function already in the Hugr ([#2094](https://github.com/CQCL/hugr/pull/2094))
- [**breaking**] Hugrmut on generic nodes ([#2111](https://github.com/CQCL/hugr/pull/2111))
- [**breaking**] Remove `RootTagged` from the hugr view trait hierarchy ([#2122](https://github.com/CQCL/hugr/pull/2122))
- [**breaking**] Split Rewrite trait into VerifyPatch and ApplyPatch ([#2070](https://github.com/CQCL/hugr/pull/2070))
- [**breaking**] Bump MSRV to 1.85 ([#2136](https://github.com/CQCL/hugr/pull/2136))
- [**breaking**] Cleanup core trait definitions ([#2126](https://github.com/CQCL/hugr/pull/2126))
- [**breaking**] Removed runtime extension sets. ([#2145](https://github.com/CQCL/hugr/pull/2145))
- [**breaking**] Improved array lowering ([#2109](https://github.com/CQCL/hugr/pull/2109))
- export mangle name function ([#2152](https://github.com/CQCL/hugr/pull/2152))
- [**breaking**] Make `NamedOp` private. Add `MakeExtensionOp::name` and `MakeOpDef::opdef_name` ([#2138](https://github.com/CQCL/hugr/pull/2138))
- [**breaking**] Add Hugr entrypoints ([#2147](https://github.com/CQCL/hugr/pull/2147))
- [**breaking**] Return a node mapping in HugrInternals::region_portgraph ([#2164](https://github.com/CQCL/hugr/pull/2164))
- [**breaking**] Validate any HugrView, make errors generic ([#2155](https://github.com/CQCL/hugr/pull/2155))
- [**breaking**] Explicit hugr type param to ComposablePass ([#2179](https://github.com/CQCL/hugr/pull/2179))
- [**breaking**] Hide FuncDefn/cl fields, add accessors and ::new(...) ([#2213](https://github.com/CQCL/hugr/pull/2213))

### Refactor

- [**breaking**] Removed global portgraph-related methods from `HugrInternals` ([#2180](https://github.com/CQCL/hugr/pull/2180))

## [0.15.4](https://github.com/CQCL/hugr/compare/hugr-passes-v0.15.3...hugr-passes-v0.15.4) - 2025-05-07

### New Features

- ReplaceTypes: handlers for array constants + linearization ([#2023](https://github.com/CQCL/hugr/pull/2023))

## [0.15.3](https://github.com/CQCL/hugr/compare/hugr-passes-v0.15.2...hugr-passes-v0.15.3) - 2025-04-02

### New Features

- ReplaceTypes pass allows replacing extension types and ops ([#1989](https://github.com/CQCL/hugr/pull/1989))
- MakeTuple->UnpackTuple elision pass ([#2012](https://github.com/CQCL/hugr/pull/2012))
- Extend LowerTypes pass to linearize by inserting copy/discard ([#2018](https://github.com/CQCL/hugr/pull/2018))

## [0.15.1](https://github.com/CQCL/hugr/compare/hugr-passes-v0.15.0...hugr-passes-v0.15.1) - 2025-03-21

### Bug Fixes

- correct `CallIndirect` tag from `FnCall` to `DataflowChild` ([#2006](https://github.com/CQCL/hugr/pull/2006))

## [0.15.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.14.4...hugr-passes-v0.15.0) - 2025-03-14

### New Features

- add separate DCE pass ([#1902](https://github.com/CQCL/hugr/pull/1902))
- [**breaking**] replace `Lift` with `Barrier` ([#1952](https://github.com/CQCL/hugr/pull/1952))
- [**breaking**] don't assume "main"-function in dataflow + constant folding ([#1896](https://github.com/CQCL/hugr/pull/1896))

## [0.14.3](https://github.com/CQCL/hugr/compare/hugr-passes-v0.14.2...hugr-passes-v0.14.3) - 2025-02-05

### Bug Fixes

- Export `RemoveDeadFuncsError` (#1883)
- const-folding Module keeps at least "main" (#1901)

### Documentation

- Fix deprecation warning messages (#1891)

## [0.14.2](https://github.com/CQCL/hugr/compare/hugr-passes-v0.14.1...hugr-passes-v0.14.2) - 2025-01-20

### New Features

- Add CallGraph struct, and dead-function-removal pass (#1796)

## [0.14.1](https://github.com/CQCL/hugr/compare/hugr-passes-v0.14.0...hugr-passes-v0.14.1) - 2024-12-18

### Bug Fixes

- Constant folding now tolerates root nodes without input/output nodes (#1799)

### New Features

- Cleanup `Display` of types and arguments (#1802)
- add MonomorphizePass and deprecate monomorphize (#1809)

## [0.14.0](https://github.com/CQCL/hugr/compare/hugr-passes-v0.13.3...hugr-passes-v0.14.0) - 2024-12-16

### ⚠ BREAKING CHANGES

- Updated to `hugr 0.14`, which includes breaking changes to the serialization format.

- `ConstantFoldPass` is no longer `UnwindSafe`, `RefUnwindSafe`, nor `Copy`.
- `fold_leaf_op` and `find_consts` have been removed.
- `ConstantFoldPass::new` function removed. Instead use `ConstantFoldPass::default`.
- Variant `ConstFoldError::SimpleReplacementError` was removed.

### Bug Fixes

- allow disconnected outputs in SiblingSubgraph::from_node (#1769)

### New Features

- [**breaking**] Replace GATs with `impl Iterator` returns (RPITIT) on `HugrView` (#1660)
- [**breaking**] Share `Extension`s under `Arc`s (#1647)
- [**breaking**] OpDefs and TypeDefs keep a reference to their extension (#1719)
- [**breaking**] Have `CustomType`s reference their `Extension` definition (#1723)
- [**breaking**] Resolve OpaqueOps and CustomType extensions  (#1735)
- Dataflow analysis framework (#1476)
- [**breaking**] `used_extensions` calls for both ops and signatures (#1739)
- [**breaking**] Hugrs now keep a `ExtensionRegistry` with their requirements (#1738)
- *(hugr-passes)* [**breaking**] Rewrite constant_fold_pass using dataflow framework (#1603)
- [**breaking**] Rename `collections` extension to `collections.list` (#1764)
- [**breaking**] Add `monomorphization` pass (#1733)
- [**breaking**] rename `extension_reqs` to `runtime_reqs` (#1776)
- [**breaking**] Don't require explicit extension registers for validation (#1784)
- [**breaking**] Remove ExtensionRegistry args in UnwrapBuilder and ListOp (#1785)

### Performance

- Faster singleton SiblingSubgraph construction (#1654)
- Return `Cow<Signature>` where possible (#1743)

## [0.13.2](https://github.com/CQCL/hugr/compare/hugr-passes-v0.13.1...hugr-passes-v0.13.2) - 2024-10-22

### New Features

- make errors more readable with Display impls ([#1597](https://github.com/CQCL/hugr/pull/1597))

## [0.13.1](https://github.com/CQCL/hugr/compare/hugr-passes-v0.8.2...hugr-passes-v0.13.1) - 2024-10-14

This release bumps the version to align with the other `hugr-*` crates.

### New Features

- return replaced ops from lowering ([#1568](https://github.com/CQCL/hugr/pull/1568))

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

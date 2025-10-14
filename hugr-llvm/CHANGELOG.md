# Changelog


## [0.24.0](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.23.0...hugr-llvm-v0.24.0) - 2025-10-13

### New Features

- LLVM lowering for borrow arrays using bitmasks ([#2574](https://github.com/CQCL/hugr/pull/2574))
- *(py, core, llvm)* add `is_borrowed` op for BorrowArray ([#2610](https://github.com/CQCL/hugr/pull/2610))

### Refactor

- [**breaking**] consistent inout order in borrow array ([#2621](https://github.com/CQCL/hugr/pull/2621))

## [0.23.0](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.22.4...hugr-llvm-v0.23.0) - 2025-09-30

### Miscellaneous Tasks

- [**breaking**] Cleanup deprecated definitions ([#2594](https://github.com/CQCL/hugr/pull/2594))

### Refactor

- [**breaking**] Replace lazy_static with std::sync::LazyLock ([#2567](https://github.com/CQCL/hugr/pull/2567))

### Testing

- Add framework for LLVM execution tests involving panics ([#2568](https://github.com/CQCL/hugr/pull/2568))

## [0.22.2](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.22.1...hugr-llvm-v0.22.2) - 2025-08-06

### Bug Fixes

- added public func getter for EmitFuncContext ([#2482](https://github.com/CQCL/hugr/pull/2482))
- *(hugr-llvm)* Set llvm function linkage based on Visibility hugr node field ([#2502](https://github.com/CQCL/hugr/pull/2502))
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.21.0](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.20.2...hugr-llvm-v0.21.0) - 2025-07-09

### New Features

- [**breaking**] No nested FuncDefns (or AliasDefns) ([#2256](https://github.com/CQCL/hugr/pull/2256))
- [**breaking**] Split `TypeArg::Sequence` into tuples and lists. ([#2140](https://github.com/CQCL/hugr/pull/2140))
- [**breaking**] More helpful error messages in model import ([#2272](https://github.com/CQCL/hugr/pull/2272))
- [**breaking**] Merge `TypeParam` and `TypeArg` into one `Term` type in Rust ([#2309](https://github.com/CQCL/hugr/pull/2309))
- Add `MakeError` op ([#2377](https://github.com/CQCL/hugr/pull/2377))

## [0.20.2](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.20.1...hugr-llvm-v0.20.2) - 2025-06-25

### New Features

- *(core, llvm)* add array unpack operations ([#2339](https://github.com/CQCL/hugr/pull/2339))

### Refactor

- *(llvm)* replace HashMap with BTreeMap ([#2313](https://github.com/CQCL/hugr/pull/2313))

## [0.20.1](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.20.0...hugr-llvm-v0.20.1) - 2025-06-03

### Bug Fixes

- Make SumType::Unit(N) equal to SumType::General([(); N]) ([#2250](https://github.com/CQCL/hugr/pull/2250))

### Testing

- Add exec tests for widen op ([#2043](https://github.com/CQCL/hugr/pull/2043))

## [0.20.0](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.15.4...hugr-llvm-v0.20.0) - 2025-05-14

### Bug Fixes

- Fix `inline_constant_functions` pass ([#2135](https://github.com/CQCL/hugr/pull/2135))

### New Features

- [**breaking**] Hugrmut on generic nodes ([#2111](https://github.com/CQCL/hugr/pull/2111))
- [**breaking**] Remove `RootTagged` from the hugr view trait hierarchy ([#2122](https://github.com/CQCL/hugr/pull/2122))
- [**breaking**] Bump MSRV to 1.85 ([#2136](https://github.com/CQCL/hugr/pull/2136))
- [**breaking**] Cleanup core trait definitions ([#2126](https://github.com/CQCL/hugr/pull/2126))
- [**breaking**] Removed runtime extension sets. ([#2145](https://github.com/CQCL/hugr/pull/2145))
- [**breaking**] Improved array lowering ([#2109](https://github.com/CQCL/hugr/pull/2109))
- [**breaking**] Make `NamedOp` private. Add `MakeExtensionOp::name` and `MakeOpDef::opdef_name` ([#2138](https://github.com/CQCL/hugr/pull/2138))
- Make `hugr_llvm::extension::collections::array::build_array_alloc` public ([#2165](https://github.com/CQCL/hugr/pull/2165))
- Add LLVM emission for prelude.noop ([#2160](https://github.com/CQCL/hugr/pull/2160))
- [**breaking**] Add Hugr entrypoints ([#2147](https://github.com/CQCL/hugr/pull/2147))
- [**breaking**] Return a node mapping in HugrInternals::region_portgraph ([#2164](https://github.com/CQCL/hugr/pull/2164))
- Restore old array lowering ([#2194](https://github.com/CQCL/hugr/pull/2194))
- [**breaking**] Hide FuncDefn/cl fields, add accessors and ::new(...) ([#2213](https://github.com/CQCL/hugr/pull/2213))

## [0.15.4](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.15.3...hugr-llvm-v0.15.4) - 2025-05-07

### New Features

- move `ArrayOpBuilder` to hugr-core ([#2115](https://github.com/CQCL/hugr/pull/2115))

## [0.15.3](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.15.2...hugr-llvm-v0.15.3) - 2025-04-02

### New Features

- *(hugr-llvm)* Add llvm codegen for `arithmetic.float.fpow` ([#2042](https://github.com/CQCL/hugr/pull/2042))
- *(hugr-llvm)* Emit divmod and mod operations ([#2025](https://github.com/CQCL/hugr/pull/2025))

## [0.15.1](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.15.0...hugr-llvm-v0.15.1) - 2025-03-21

### Bug Fixes

- Remove return from val_or_panic ([#1999](https://github.com/CQCL/hugr/pull/1999))

### New Features

- add exit operation to prelude ([#2008](https://github.com/CQCL/hugr/pull/2008))
- Add llvm codegen for collections.static_array ([#2003](https://github.com/CQCL/hugr/pull/2003))

## [0.15.0](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.14.4...hugr-llvm-v0.15.0) - 2025-03-14

### Bug Fixes

- Rename widen insta tests ([#1949](https://github.com/CQCL/hugr/pull/1949))

### New Features

- Emit `widen` ops from the int ops extension ([#1946](https://github.com/CQCL/hugr/pull/1946))
- [**breaking**] replace `Lift` with `Barrier` ([#1952](https://github.com/CQCL/hugr/pull/1952))
- *(hugr-llvm)* Emit narrow ops ([#1955](https://github.com/CQCL/hugr/pull/1955))
- Add float <--> int bytecasting ops to conversions extension ([#1956](https://github.com/CQCL/hugr/pull/1956))
- *(hugr-llvm)* Emit iu_to_s and is_to_u ([#1978](https://github.com/CQCL/hugr/pull/1978))

### Refactor

- [**breaking**] remove unused dependencies ([#1935](https://github.com/CQCL/hugr/pull/1935))

## [0.14.4](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.14.3...hugr-llvm-v0.14.4) - 2025-02-24

### New Features

- add xor to logic extension (#1911)
- *(hugr-llvm)* Add extension points to `PreludeCodegen` for customising string lowering (#1918)

## [0.14.2](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.14.1...hugr-llvm-v0.14.2) - 2025-01-20

### New Features

- *(hugr-llvm)* Emit more int ops (#1835)
- Constant values in `hugr-model` (#1838)
- *(hugr-llvm)* Emit ipow (#1839)

### Refactor

- *(hugr-llvm)* [**breaking**] Optimise the llvm types used to represent hugr sums. (#1855)

### Testing

- Fix failing inot test (#1841)

## [0.14.1](https://github.com/CQCL/hugr/compare/hugr-llvm-v0.14.0...hugr-llvm-v0.14.1) - 2024-12-18

### Bug Fixes

- Add LLVM lowering for `logic.Not` (#1812)

### New Features

- Lower LoadNat to LLVM (#1801)
- add ArrayValue to python, rust and lowering (#1773)
## [0.13.3](https://github.com/CQCL/hugr-llvm/compare/v0.6.0...v0.6.1) - 2024-11-25
No changes - version bump to catch up with other hugr crates in repository move.

## [0.6.1](https://github.com/CQCL/hugr-llvm/compare/v0.6.0...v0.6.1) - 2024-10-23

### Bug Fixes

- don't normalise half turns ([#137](https://github.com/CQCL/hugr-llvm/pull/137))

## [0.6.0](https://github.com/CQCL/hugr-llvm/compare/v0.5.1...v0.6.0) - 2024-10-21

### Bug Fixes

- Conversion operations having poison results  ([#131](https://github.com/CQCL/hugr-llvm/pull/131))

### New Features

- [**breaking**] Allow extension callbacks to have non-`'static` lifetimes ([#128](https://github.com/CQCL/hugr-llvm/pull/128))
- [**breaking**] Support `tket2.rotation.from_halfturns_unchecked` ([#133](https://github.com/CQCL/hugr-llvm/pull/133))

### Refactor

- [**breaking**] remove trait emit op ([#104](https://github.com/CQCL/hugr-llvm/pull/104))
- [**breaking**] rework extensions interface ([#119](https://github.com/CQCL/hugr-llvm/pull/119))
- [**breaking**] move packaged extensions from `crate::custom` to `crate::extension` ([#126](https://github.com/CQCL/hugr-llvm/pull/126))

## [0.5.1](https://github.com/CQCL/hugr-llvm/compare/v0.5.0...v0.5.1) - 2024-09-23

### New Features

- provide `inline_constant_functions` ([#108](https://github.com/CQCL/hugr-llvm/pull/108))

## [0.5.0](https://github.com/CQCL/hugr-llvm/compare/v0.4.0...v0.5.0) - 2024-09-16

### New Features

- Add emitters for int <-> float/usize conversions ([#94](https://github.com/CQCL/hugr-llvm/pull/94))
- [**breaking**] array ops ([#96](https://github.com/CQCL/hugr-llvm/pull/96))
- Add conversions itobool, ifrombool ([#101](https://github.com/CQCL/hugr-llvm/pull/101))
- Add `tket2` feature and lowerings for `tket2.rotation` extension ([#100](https://github.com/CQCL/hugr-llvm/pull/100))

### Testing

- Add execution test framework ([#97](https://github.com/CQCL/hugr-llvm/pull/97))

## [0.3.1](https://github.com/CQCL/hugr-llvm/compare/v0.3.0...v0.3.1) - 2024-08-28

### New Features
- Emit more int operations ([#87](https://github.com/CQCL/hugr-llvm/pull/87))

## [0.3.0](https://github.com/CQCL/hugr-llvm/compare/v0.2.1...v0.3.0) - 2024-08-27

### New Features
- [**breaking**] Lower string, print, and panic ([#78](https://github.com/CQCL/hugr-llvm/pull/78))
- Lower float operations ([#83](https://github.com/CQCL/hugr-llvm/pull/83))
- Lower logic extension ([#81](https://github.com/CQCL/hugr-llvm/pull/81))
- Lower arrays ([#82](https://github.com/CQCL/hugr-llvm/pull/82))

## [0.2.1](https://github.com/CQCL/hugr-llvm/compare/v0.2.0...v0.2.1) - 2024-08-19

### Documentation
- Remove fixed crate version in usage instructions ([#68](https://github.com/CQCL/hugr-llvm/pull/68))

### New Features
- Add lowering for LoadFunction ([#65](https://github.com/CQCL/hugr-llvm/pull/65))
- Emission for CallIndirect nodes ([#73](https://github.com/CQCL/hugr-llvm/pull/73))

## [0.2.0](https://github.com/CQCL/hugr-llvm/compare/v0.1.0...v0.2.0) - 2024-07-31

### New Features
- make EmitFuncContext::iw_context pub ([#55](https://github.com/CQCL/hugr-llvm/pull/55))

### Refactor
- use HugrFuncType/HugrType/HugrSumType ([#56](https://github.com/CQCL/hugr-llvm/pull/56))
- remove unneeded `HugrView` constraints  ([#59](https://github.com/CQCL/hugr-llvm/pull/59))
- [**breaking**] add `LLVMSumValue` ([#63](https://github.com/CQCL/hugr-llvm/pull/63))

### Testing
- Add test for LoadFunction Op ([#60](https://github.com/CQCL/hugr-llvm/pull/60))

## [0.1.0](https://github.com/CQCL/hugr-llvm/releases/tag/v0.1.0) - 2024-07-10

### Bug Fixes
- Syntax error
- sum type tag elision logic reversed
- [**breaking**] Allow Const and FuncDecl as children of Modules, Dataflow Parents, and CFG nodes ([#46](https://github.com/CQCL/hugr-llvm/pull/46))

### Documentation
- fix bad grammar ([#34](https://github.com/CQCL/hugr-llvm/pull/34))

### New Features
- Emission for Call nodes
- Support  values
- add `get_extern_func` ([#28](https://github.com/CQCL/hugr-llvm/pull/28))
- lower CFGs ([#26](https://github.com/CQCL/hugr-llvm/pull/26))
- Add initial codegen extension for `prelude` ([#29](https://github.com/CQCL/hugr-llvm/pull/29))
- [**breaking**] `Namer` optionally appends node index to mangled names. ([#32](https://github.com/CQCL/hugr-llvm/pull/32))
- Implement lowerings for ieq,ilt_s,sub in int codegen extension ([#33](https://github.com/CQCL/hugr-llvm/pull/33))
- Add initial `float` extension ([#31](https://github.com/CQCL/hugr-llvm/pull/31))
- Emit more int comparison operators ([#47](https://github.com/CQCL/hugr-llvm/pull/47))

### Refactor
- clean up fat.rs ([#38](https://github.com/CQCL/hugr-llvm/pull/38))

### Testing
- add a test for sum type tags
- Add integration tests lowering guppy programs ([#35](https://github.com/CQCL/hugr-llvm/pull/35))

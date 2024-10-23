# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

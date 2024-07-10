# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

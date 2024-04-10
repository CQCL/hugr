# Changelog

## Unreleased (xxxx-xx-xx)

This release includes multiple breaking changes to the API, including the rename of the crate from `quantinuum_hugr` to `hugr`.

### New Contributors

* @Cobord made their first contribution in [#889](https://github.com/CQCL/hugr/pull/889)
* @qartik made their first contribution in [#843](https://github.com/CQCL/hugr/pull/843)

### Bug Fixes

- [**breaking**] Polymorphic calls ([#901](https://github.com/CQCL/hugr/pull/901))

### Documentation

- HUGR spec copyediting ([#843](https://github.com/CQCL/hugr/pull/843))
- Add builder module docs + example ([#853](https://github.com/CQCL/hugr/pull/853))
- Add note on serialized hugr node order ([#849](https://github.com/CQCL/hugr/pull/849))

### Features

- Arbitrary types in the yaml extension definition ([#839](https://github.com/CQCL/hugr/pull/839))
- Make all core types serializable ([#850](https://github.com/CQCL/hugr/pull/850))
- Don't include empty type args in Display ([#857](https://github.com/CQCL/hugr/pull/857))
- Mermaid renderer for hugrs ([#852](https://github.com/CQCL/hugr/pull/852))
- Add impl TryFrom<&OpType> for ops::* ([#856](https://github.com/CQCL/hugr/pull/856))
- [**breaking**] Infallible `HugrMut` methods ([#869](https://github.com/CQCL/hugr/pull/869))
- Ancilla support in CircuitBuilder ([#867](https://github.com/CQCL/hugr/pull/867))
- `CircuitBuilder::append_with_output_arr` ([#871](https://github.com/CQCL/hugr/pull/871))
- [**breaking**] Make some `Container` methods infallible ([#872](https://github.com/CQCL/hugr/pull/872))
- [**breaking**] Cleaner error on wiring errors while building ([#873](https://github.com/CQCL/hugr/pull/873))
- [**breaking**] Change sums to be over TypeRows rather than Types ([#863](https://github.com/CQCL/hugr/pull/863))
- Make various data publicly accessible ([#875](https://github.com/CQCL/hugr/pull/875))
- [**breaking**] CustomConst is not restricted to being CustomType ([#878](https://github.com/CQCL/hugr/pull/878))
- [**breaking**] Return the type of FuncDecl in `HugrView::get_function_type` ([#880](https://github.com/CQCL/hugr/pull/880))
- [**breaking**] Merge `Value` into `Const` ([#881](https://github.com/CQCL/hugr/pull/881))
- Replace `Tuple` with unary sums ([#891](https://github.com/CQCL/hugr/pull/891))

### Performance

- Add some simple benchmarks ([#892](https://github.com/CQCL/hugr/pull/892))

### Refactor

- Add `From` conversion from ExtensionId to ExtensionSet ([#855](https://github.com/CQCL/hugr/pull/855))
- Remove clone in `ExtensionSet::union` ([#859](https://github.com/CQCL/hugr/pull/859))
- Extension Inference: make fewer things public, rm Meta::new ([#883](https://github.com/CQCL/hugr/pull/883))
- [**breaking**] Return impl trait in Rewrite trait ([#889](https://github.com/CQCL/hugr/pull/889))

## 0.2.0 (2024-02-20)

### Documentation

- Fix crates.io badge in README ([#809](https://github.com/CQCL/hugr/pull/809))
- Use absolute links in the README ([#811](https://github.com/CQCL/hugr/pull/811))
- Remove input->const order edges from spec diagrams ([#812](https://github.com/CQCL/hugr/pull/812))
- Remove incorrect indentation in spec ([#813](https://github.com/CQCL/hugr/pull/813))
- Tweaks to main example ([#825](https://github.com/CQCL/hugr/pull/825))
- Add example to `CFGBuilder` ([#826](https://github.com/CQCL/hugr/pull/826))

### Features

- Add InlineDFG rewrite ([#828](https://github.com/CQCL/hugr/pull/828))
- [**breaking**] Impls of CustomConst must be able to report their type ([#827](https://github.com/CQCL/hugr/pull/827))
- Minimal implementation for YAML extensions ([#833](https://github.com/CQCL/hugr/pull/833))

## 0.1.0 (2024-01-15)

This is the initial release of the Hierarchical Unified Graph Representation.
See the representation specification available at [hugr.md](https://github.com/CQCL/hugr/blob/main/specification/hugr.md).

This release includes an up-to-date implementation of the spec, including the core definitions (control flow, data flow and module structures) as well as the Prelude extension with support for basic classical operations and types.

HUGRs can be loaded and stored using the versioned serialization format, or they can be constructed programmatically using the builder utility.
The modules `hugr::hugr::view` and `hugr::hugr::rewrite` provide an API for querying and mutating the HUGR.
For more complex operations, some algorithms are provided in `hugr::algorithms`.

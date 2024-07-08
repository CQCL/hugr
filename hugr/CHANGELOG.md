# Changelog

## 0.6.1 (2024-07-08)

### Bug Fixes

- Bring back input_extensions serialized field in rust NodeSer ([#1275](https://github.com/CQCL/hugr/pull/1275))


## 0.6.0 (2024-06-28)

### Bug Fixes

- SimpleReplacement panic on multiports ([#1191](https://github.com/CQCL/hugr/pull/1191))
- Add some validation for const nodes ([#1222](https://github.com/CQCL/hugr/pull/1222))
- Cfg not validating entry/exit types ([#1229](https://github.com/CQCL/hugr/pull/1229))
- `extract_hugr` not removing root node ports ([#1239](https://github.com/CQCL/hugr/pull/1239))

### Documentation

- Fix documentation of `ValidationError::ConstTypeError` ([#1227](https://github.com/CQCL/hugr/pull/1227))

### Features

- CircuitBuilder::add_constant ([#1168](https://github.com/CQCL/hugr/pull/1168))
- [**breaking**] Make the rewrite errors more useful ([#1174](https://github.com/CQCL/hugr/pull/1174))
- [**breaking**] Validate Extensions using hierarchy, ignore input_extensions, RIP inference ([#1142](https://github.com/CQCL/hugr/pull/1142))
- [**breaking**] Infer extension deltas for Case, Cfg, Conditional, DataflowBlock, Dfg, TailLoop  ([#1195](https://github.com/CQCL/hugr/pull/1195))
- Helper functions for requesting inference, use with builder in tests ([#1219](https://github.com/CQCL/hugr/pull/1219))

### Refactor

- [**breaking**] FunctionBuilder takes impl Into<PolyFuncType> ([#1220](https://github.com/CQCL/hugr/pull/1220))
- [**breaking**] Remove NodeType and input_extensions ([#1183](https://github.com/CQCL/hugr/pull/1183))


## 0.5.1 (2024-06-07)

### Bug Fixes

- Validate that control-flow outputs have exactly one successor ([#1144](https://github.com/CQCL/hugr/pull/1144))
- Do not require matching extension_reqs when creating a replacement ([#1177](https://github.com/CQCL/hugr/pull/1177))

### Features

- Add `ConstExternalSymbol` to prelude ([#1123](https://github.com/CQCL/hugr/pull/1123))
- `HugrView::extract_hugr` to extract regions into owned hugrs. ([#1173](https://github.com/CQCL/hugr/pull/1173))

### Testing

- Serialisation round trip testing for `OpDef` ([#999](https://github.com/CQCL/hugr/pull/999))

### Refactor

- Move binary to hugr-cli ([#1134](https://github.com/CQCL/hugr/pull/1134))


## 0.5.0 (2024-05-29)

### Bug Fixes

- Missing re-exports in `hugr::hugr` ([#1127](https://github.com/CQCL/hugr/pull/1127))
- Set initial version of hugr-core to 0.1.0 ([#1129](https://github.com/CQCL/hugr/pull/1129))

### Features

- [**breaking**] Remove `PartialEq` impl for `ConstF64` ([#1079](https://github.com/CQCL/hugr/pull/1079))
- [**breaking**] Allow "Row Variables" declared as List<Type> ([#804](https://github.com/CQCL/hugr/pull/804))
- Hugr binary cli tool ([#1096](https://github.com/CQCL/hugr/pull/1096))
- [**breaking**] Move passes from `algorithms` into a separate crate ([#1100](https://github.com/CQCL/hugr/pull/1100))
- [**breaking**] Disallow nonlocal value edges into FuncDefn's ([#1061](https://github.com/CQCL/hugr/pull/1061))
- [**breaking**] Move cli in to hugr-cli sub-crate ([#1107](https://github.com/CQCL/hugr/pull/1107))
- Add verbosity, return `Hugr` from `run`. ([#1116](https://github.com/CQCL/hugr/pull/1116))
- Unseal and make public the traits `HugrInternals` and `HugrMutInternals` ([#1122](https://github.com/CQCL/hugr/pull/1122))

### Refactor

- [**breaking**] No Ports in TypeRow ([#1087](https://github.com/CQCL/hugr/pull/1087))
- Add a `hugr-core` crate ([#1108](https://github.com/CQCL/hugr/pull/1108))


## 0.4.0 (2024-05-20)

### Bug Fixes

- Disallow non-finite values for `ConstF64` ([#1075](https://github.com/CQCL/hugr/pull/1075))
- Serialization round-trips ([#948](https://github.com/CQCL/hugr/pull/948))
- [**breaking**] Combine `ConstIntU` and `ConstIntS` ([#974](https://github.com/CQCL/hugr/pull/974))
- Disable serialisation tests when miri is active ([#977](https://github.com/CQCL/hugr/pull/977))
- [**breaking**] Serialisation schema ([#968](https://github.com/CQCL/hugr/pull/968))
- Correct constant fold for `fne`. ([#995](https://github.com/CQCL/hugr/pull/995))
- [**breaking**] Serialisation fixes ([#997](https://github.com/CQCL/hugr/pull/997))
- [**breaking**] OpDef serialisation ([#1013](https://github.com/CQCL/hugr/pull/1013))
- NaryLogicOp constant folding ([#1026](https://github.com/CQCL/hugr/pull/1026))

### Features

- Add verification to constant folding ([#1030](https://github.com/CQCL/hugr/pull/1030))
- Add `Const::get_custom_value` ([#1037](https://github.com/CQCL/hugr/pull/1037))
- Add serialization schema for metadata ([#1038](https://github.com/CQCL/hugr/pull/1038))
- 'Replace' rewrite returns node map ([#929](https://github.com/CQCL/hugr/pull/929))
- `new` methods for leaf ops ([#940](https://github.com/CQCL/hugr/pull/940))
- Add `string` type and `print` function to `prelude` ([#942](https://github.com/CQCL/hugr/pull/942))
- `CustomOp::extension` utility function ([#951](https://github.com/CQCL/hugr/pull/951))
- [**breaking**] Add `non_exhaustive` to various enums ([#952](https://github.com/CQCL/hugr/pull/952))
- Encoder metadata in serialized hugr ([#955](https://github.com/CQCL/hugr/pull/955))
- [**breaking**] Bring back Value ([#967](https://github.com/CQCL/hugr/pull/967))
- Add LoadFunction node ([#947](https://github.com/CQCL/hugr/pull/947))
- Add From impls for TypeArg ([#1002](https://github.com/CQCL/hugr/pull/1002))
- Constant-folding of integer and logic operations ([#1009](https://github.com/CQCL/hugr/pull/1009))
- [**breaking**] Update serialisation schema, implement `CustomConst` serialisation ([#1005](https://github.com/CQCL/hugr/pull/1005))
- Merge basic blocks algorithm ([#956](https://github.com/CQCL/hugr/pull/956))
- [**breaking**] Allow panic operation to have any input and output wires ([#1024](https://github.com/CQCL/hugr/pull/1024))

### Refactor

- [**breaking**] Rename `crate::ops::constant::ExtensionValue` => `OpaqueValue` ([#1036](https://github.com/CQCL/hugr/pull/1036))
- Outline hugr::serialize::test ([#976](https://github.com/CQCL/hugr/pull/976))
- [**breaking**] Replace SmolStr identifiers with wrapper types. ([#959](https://github.com/CQCL/hugr/pull/959))
- Separate extension validation from the rest ([#1011](https://github.com/CQCL/hugr/pull/1011))
- Remove "trait TypeParametrised" ([#1019](https://github.com/CQCL/hugr/pull/1019))

### Testing

- Reorg OutlineCfg/nest_cfgs tests so hugr doesn't depend on algorithm ([#1007](https://github.com/CQCL/hugr/pull/1007))
- Ignore tests which depend on typetag when cfg(miri) ([#1051](https://github.com/CQCL/hugr/pull/1051))
- Really ignore tests which depend on typetag when cfg(miri) ([#1058](https://github.com/CQCL/hugr/pull/1058))
- Proptests for round trip serialisation of `Type`s and `Op`s. ([#981](https://github.com/CQCL/hugr/pull/981))
- Add a test of instantiating an extension set ([#939](https://github.com/CQCL/hugr/pull/939))
- Ignore serialisation tests when using miri ([#975](https://github.com/CQCL/hugr/pull/975))
- [**breaking**] Test roundtrip serialisation against strict + lax schema ([#982](https://github.com/CQCL/hugr/pull/982))
- Fix some bad assert_matches ([#1006](https://github.com/CQCL/hugr/pull/1006))
- Expand test of instantiating extension sets ([#1003](https://github.com/CQCL/hugr/pull/1003))
- Fix unconnected ports in extension test ([#1010](https://github.com/CQCL/hugr/pull/1010))


## 0.3.1 (2024-04-23)

### Features

- `new` methods for leaf ops ([#940](https://github.com/CQCL/hugr/pull/940))
- `CustomOp::extension` utility function ([#951](https://github.com/CQCL/hugr/pull/951))
- Encoder metadata in serialized hugr ([#955](https://github.com/CQCL/hugr/pull/955))

### Testing

- Add a test of instantiating an extension set ([#939](https://github.com/CQCL/hugr/pull/939))


## 0.3.0 (2024-04-15)

### Main changes

This release includes a long list of breaking changes to the API.

- The crate was renamed from `quantinuum_hugr` to `hugr`.
- The API has been simplified, flattening structures and reworking unnecessarily
  fallible operations where possible.
- Includes version `1` of the hugr serialization schema. Older pre-v1 serialized
  hugrs are no longer supported. Starting with `v1`, backward compatibility for
  loading older versions will be maintained.

### New Contributors

* @Cobord made their first contribution in [#889](https://github.com/CQCL/hugr/pull/889)
* @qartik made their first contribution in [#843](https://github.com/CQCL/hugr/pull/843)

### Bug Fixes

- [**breaking**] Polymorphic calls ([#901](https://github.com/CQCL/hugr/pull/901))

### Documentation

- HUGR spec copyediting ([#843](https://github.com/CQCL/hugr/pull/843))
- Add builder module docs + example ([#853](https://github.com/CQCL/hugr/pull/853))
- Add note on serialized hugr node order ([#849](https://github.com/CQCL/hugr/pull/849))
- Specify direct children in `HugrView::children` ([#921](https://github.com/CQCL/hugr/pull/921))
- Add logo svg to readme and spec ([#925](https://github.com/CQCL/hugr/pull/925))

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
- [**breaking**] No polymorphic closures ([#906](https://github.com/CQCL/hugr/pull/906))
- [**breaking**] Flatten `LeafOp` ([#922](https://github.com/CQCL/hugr/pull/922))

### Performance

- Add some simple benchmarks ([#892](https://github.com/CQCL/hugr/pull/892))

### Refactor

- Add `From` conversion from ExtensionId to ExtensionSet ([#855](https://github.com/CQCL/hugr/pull/855))
- Remove clone in `ExtensionSet::union` ([#859](https://github.com/CQCL/hugr/pull/859))
- Extension Inference: make fewer things public, rm Meta::new ([#883](https://github.com/CQCL/hugr/pull/883))
- [**breaking**] Return impl trait in Rewrite trait ([#889](https://github.com/CQCL/hugr/pull/889))
- Combine ExtensionSolutions (no separate closure) ([#884](https://github.com/CQCL/hugr/pull/884))
- [**breaking**] Merge `CustomOp` and `ExternalOp`. ([#923](https://github.com/CQCL/hugr/pull/923))

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

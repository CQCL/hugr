# Changelog

## 0.7.0 (2024-07-26)

### Bug Fixes

- [**breaking**] Bump serialisation version with no upgrade path ([#1352](https://github.com/CQCL/hugr/pull/1352))

### Features

- Serialization upgrade path ([#1327](https://github.com/CQCL/hugr/pull/1327))
- [**breaking**] Replace opaque type arguments with String ([#1328](https://github.com/CQCL/hugr/pull/1328))
- Add `impl Hash for Type` ([#1347](https://github.com/CQCL/hugr/pull/1347))
- `HasDef` and `HasConcrete` traits for def/concrete op design pattern ([#1336](https://github.com/CQCL/hugr/pull/1336))
- Add pointer standard extension ([#1337](https://github.com/CQCL/hugr/pull/1337))
- [**breaking**] Remove the `Eq` type bound. ([#1364](https://github.com/CQCL/hugr/pull/1364))

### Refactor

- [**breaking**] Use JSON rather than YAML in opaque fields. ([#1338](https://github.com/CQCL/hugr/pull/1338))
- [**breaking**] Declarative module behind optional feature flag ([#1341](https://github.com/CQCL/hugr/pull/1341))

### Testing

- Miri gate serialization upgrades ([#1349](https://github.com/CQCL/hugr/pull/1349))


## 0.6.1 (2024-07-25)

### Bug Fixes

- Dfg wrapper build handles incorrect output wire numbers ([#1332](https://github.com/CQCL/hugr/pull/1332))
- Sibling extension panics while computing signature with non-dataflow nodes ([#1350](https://github.com/CQCL/hugr/pull/1350))


## 0.6.0 (2024-07-19)

### Bug Fixes

- Add op's extension to signature check in `resolve_opaque_op` ([#1317](https://github.com/CQCL/hugr/pull/1317))
- Panic on `SimpleReplace` with multiports ([#1324](https://github.com/CQCL/hugr/pull/1324))

### Refactor

- [**breaking**] Separate Signature from FuncValueType by parametrizing Type(/Row)/etc. ([#1138](https://github.com/CQCL/hugr/pull/1138))

### Testing

- Verify order edges ([#1293](https://github.com/CQCL/hugr/pull/1293))
- Add failing test case for [#1315](https://github.com/CQCL/hugr/pull/1315) ([#1316](https://github.com/CQCL/hugr/pull/1316))


## 0.5.0 (2024-07-16)

### Bug Fixes

- NonConvex error on SiblingSubgraph::from_nodes with multiports ([#1295](https://github.com/CQCL/hugr/pull/1295))
- [**breaking**] Ops require their own extension ([#1226](https://github.com/CQCL/hugr/pull/1226))

### Features

- Make `DataflowOpTrait` public ([#1283](https://github.com/CQCL/hugr/pull/1283))
- Make op members consistently public ([#1274](https://github.com/CQCL/hugr/pull/1274))

### Refactor

- [**breaking**] Rename builder helpers: ft1->endo_ft, ft2->inout_ft ([#1297](https://github.com/CQCL/hugr/pull/1297))


## 0.4.0 (2024-07-10)

### Bug Fixes

- Bring back input_extensions serialized field in rust NodeSer ([#1275](https://github.com/CQCL/hugr/pull/1275))
- [**breaking**] `ops::Module` now empty struct rather than unit struct ([#1271](https://github.com/CQCL/hugr/pull/1271))

### Features

- [**breaking**] `MakeOpDef` has new `extension` method. ([#1266](https://github.com/CQCL/hugr/pull/1266))

### Refactor

- [**breaking**] Remove `Value::Tuple` ([#1255](https://github.com/CQCL/hugr/pull/1255))
- [**breaking**] Rename `HugrView` function type methods + simplify behaviour ([#1265](https://github.com/CQCL/hugr/pull/1265))

### Styling

- Change "serialise" etc to "serialize" etc. ([#1251](https://github.com/CQCL/hugr/pull/1251))

### Testing

- Add a test for [#1257](https://github.com/CQCL/hugr/pull/1257) ([#1260](https://github.com/CQCL/hugr/pull/1260))


## 0.3.1 (2024-07-08)

### Bug Fixes

- Bring back input_extensions serialized field in rust NodeSer ([#1275](https://github.com/CQCL/hugr/pull/1275))


## 0.3.0 (2024-06-28)

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

- [**breaking**] Remove NodeType and input_extensions ([#1183](https://github.com/CQCL/hugr/pull/1183))
- [**breaking**] FunctionBuilder takes impl Into<PolyFuncType> ([#1220](https://github.com/CQCL/hugr/pull/1220))


## 0.2.0 (2024-06-07)

### Bug Fixes

- [**breaking**] Validate that control-flow outputs have exactly one successor ([#1144](https://github.com/CQCL/hugr/pull/1144))
- Do not require matching extension_reqs when creating a replacement ([#1177](https://github.com/CQCL/hugr/pull/1177))

### Features

- Add `ConstExternalSymbol` to prelude ([#1123](https://github.com/CQCL/hugr/pull/1123))
- `HugrView::extract_hugr` to extract regions into owned hugrs. ([#1173](https://github.com/CQCL/hugr/pull/1173))

### Testing

- Serialization round trip testing for `OpDef` ([#999](https://github.com/CQCL/hugr/pull/999))


## 0.1.0 (2024-05-29)

### Bug Fixes

- Set initial version of hugr-core to 0.1.0 ([#1129](https://github.com/CQCL/hugr/pull/1129))

### Features

- [**breaking**] Move cli in to hugr-cli sub-crate ([#1107](https://github.com/CQCL/hugr/pull/1107))
- Make internals of int ops and the "int" CustomType more public. ([#1114](https://github.com/CQCL/hugr/pull/1114))
- Unseal and make public the traits `HugrInternals` and `HugrMutInternals` ([#1122](https://github.com/CQCL/hugr/pull/1122))

### Refactor

- Add a `hugr-core` crate ([#1108](https://github.com/CQCL/hugr/pull/1108))

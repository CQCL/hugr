# Changelog

## [0.14.2](https://github.com/CQCL/hugr/compare/hugr-v0.14.1...hugr-v0.14.2) - 2025-01-16

### Bug Fixes

- Three bugfixes in model import and export. (#1844)

### New Features

- Add CallGraph struct, and dead-function-removal pass (#1796)
- `Value::some`, `::none`, and `SumType::new_option` helpers (#1828)
- Constant values in `hugr-model` (#1838)
- *(hugr-llvm)* Emit ipow (#1839)
- Bytes literal in hugr-model. (#1845)

### Testing

- Add tests for constant value deserialization (#1822)

## [0.14.1](https://github.com/CQCL/hugr/compare/hugr-v0.14.0...hugr-v0.14.1) - 2024-12-18

### Bug Fixes

- Constant folding now tolerates root nodes without input/output nodes (#1799)
- `Call` ops not tracking their parameter extensions (#1805)

### New Features

- add MonomorphizePass and deprecate monomorphize (#1809)
- Lower LoadNat to LLVM (#1801)
- Cleanup `Display` of types and arguments (#1802)
- add ArrayValue to python, rust and lowering (#1773)
- Scoping rules and utilities for symbols, links and variables (#1754)

## [0.14.0](https://github.com/CQCL/hugr/compare/hugr-v0.13.3...hugr-v0.14.0) - 2024-12-16

This release includes a long list of breaking changes that simplify the API, specially around
extensions and extension registry management.

Extension definitions are now tracked by each operation and type inside the hugr, so there is no
need to pass around extension registries any more.

### ⚠ BREAKING CHANGES

#### Core

- The `LoadFunction::signature` field has been removed. Replaced with `DataflowOpTrait::signature()`.
- Types which implement `AsRef` - both library ones such as `Rc` and custom ones - no longer get a blanket impl of `HugrView`. Workaround by manually calling `as_ref()` and using the `&Hugr` yourself.
- Removed `Array` type variant from the serialization format.
- `Optrait` now requires `Sized` and `Clone` and is no longer object safe.
- `DataflowOptrait` now requires `Sized` and has an additional required method `substitute`.
- `::extension::resolve` operations now use `WeakExtensionRegistry`es.
- `ExtensionRegistry` and `Package` now wrap Extensions in `Arc`s.
- Renamed `OpDef::extension` and `TypeDef::extension` to `extension_id`. extension now returns weak references to the Extension defining them.
- `Extension::with_reqs` moved to `set_reqs`, which takes `&mut self` instead of `self`.
- `Extension::add_type` and `Extension::add_op` now take an extra parameter. See docs for example usage.
- `ExtensionRegistry::register_updated` and `register_updated_ref` are no longer fallible.
- Removed `CustomType::new_simple`. Custom types can no longer be const-constructed.
- Added `init_signature` and `extension_ref` methods to the `MakeOpDef` trait.
- Redefined the const types in the prelude to generator functions.
- Removed `resolve_opaque_op` and `resolve_extension_ops`. Use `Hugr::resolve_extension_defs` instead.
- Removed `ExtensionRegistry::try_new`. Use `new` instead, and call `ExtensionRegistry::validate` to validate.
- `ExtensionSet::insert` and `singleton` take extension ids by value instead of cloning internally.
- Removed `update_validate`. The hugr extensions should be resolved at load time, so we can use validate instead.
- The builder `finish_hugr` function family no longer takes a registry as parameter, and the `_prelude` variants have been removed.
- `extension_reqs` field in `FunctionType` and `Extension` renamed to `runtime_reqs`.
- Removed the extension registry argument from validate calls.
- Removed the extension registry argument from operation instantiation methods.
- Removed most extension-specific test registries. Use `EMPTY_REG`, `PRELUDE_REGISTRY`, or `STD_REG` instead.

#### Extensions

- Array scan and repeat ops get an additional type parameter specifying the extension requirements of their input functions. Furthermore, repeat is no longer part of `ArrayOpDef` but is instead specified via a new `ArrayScan` struct.
- `collections` extension renamed to `collections.list`
- Array type and operations have been moved out of prelude and into a new `collections.array` extension.

#### Passes

- `ConstantFoldPass` is no longer `UnwindSafe`, `RefUnwindSafe`, nor `Copy`.
- `fold_leaf_op` and `find_consts` have been removed.
- `ConstantFoldPass::new` function removed. Instead use `ConstantFoldPass::default`.
- Variant `ConstFoldError::SimpleReplacementError` was removed.

### Bug Fixes

- hierarchical simple replacement using insert_hugr (#1718)
- hugr-py not adding extension-reqs on custom ops (#1759)
- [**breaking**] Replace `LoadFunction::signature` with `LoadFunction::instantiation` (#1756)
- Resolve types in `Value`s and custom consts (#1779)
- allow disconnected outputs in SiblingSubgraph::from_node (#1769)

### Documentation

- Fix comment for scan op (#1751)

### New Features

- Dataflow analysis framework (#1476)
- *(hugr-passes)* [**breaking**] Rewrite constant_fold_pass using dataflow framework (#1603)
- Export/import of JSON metadata (#1622)
- Add `SiblingSubgraph::from_node` (#1655)
- [**breaking**] Replace GATs with `impl Iterator` returns (RPITIT) on `HugrView` (#1660)
- Emulate `TypeBound`s on parameters via constraints. (#1624)
- Add array `repeat` and `scan` ops (#1633)
- move unwrap builder to hugr core (#1674)
- Lists and extension sets with splicing (#1657)
- add HugrView::first_child and HugrMut::remove_subtree (#1721)
- Lower collections extension (#1720)
- [**breaking**] impl HugrView for any &(mut) to a HugrView (#1678)
- [**breaking**] Make array repeat and scan ops generic over extension reqs (#1716)
- Print []+[] as Bool and [] as Unit in user-facing messages (#1745)
- Add `PartialEq` impls for `FuncTypeBase` and `Cow<FuncTypeBase>` (#1762)
- [**breaking**] Rename `collections` extension to `collections.list` (#1764)
- add `is_` variant methods to `EdgeKind` (#1768)
- [**breaking**] Move arrays from prelude into new extension (#1770)
- Add `LoadNat` operation to enable loading generic `BoundedNat`s into runtime values (#1763)
- [**breaking**] Add `monomorphization` pass (#1733)
- Update extension pointers in customConsts (#1780)
- [**breaking**] Use registries of `Weak<Extension>`s when doing resolution  (#1781)
- [**breaking**] Resolve extension references inside the extension themselves (#1783)
- [**breaking**] Remove ExtensionRegistry args in UnwrapBuilder and ListOp (#1785)
- export llvm test utilities under llvm-test feature (#1677)
- [**breaking**] Share `Extension`s under `Arc`s (#1647)
- [**breaking**] OpDefs and TypeDefs keep a reference to their extension (#1719)
- [**breaking**] Have `CustomType`s reference their `Extension` definition (#1723)
- [**breaking**] Resolve OpaqueOps and CustomType extensions  (#1735)
- [**breaking**] `used_extensions` calls for both ops and signatures (#1739)
- [**breaking**] Hugrs now keep a `ExtensionRegistry` with their requirements (#1738)
- [**breaking**] rename `extension_reqs` to `runtime_reqs` (#1776)
- [**breaking**] Don't require explicit extension registers for validation (#1784)

### Performance

- Return `Cow<Signature>` where possible (#1743)
- Faster singleton SiblingSubgraph construction (#1654)

### Refactor

- avoid hugr clone in simple replace (#1724)
- [trivial] replace.rs: use HugrView::first_child  (#1737)

## [0.13.3](https://github.com/CQCL/hugr/compare/hugr-v0.13.2...hugr-v0.13.3) - 2024-11-06

### Bug Fixes

- Insert DFG directly as a funcdefn in `Package::from_hugr`  ([#1621](https://github.com/CQCL/hugr/pull/1621))

### New Features

- `HugrMut::remove_metadata` ([#1619](https://github.com/CQCL/hugr/pull/1619))
- Operation and constructor declarations in `hugr-model` ([#1605](https://github.com/CQCL/hugr/pull/1605))
- Add TailLoop::BREAK_TAG and CONTINUE_TAG ([#1626](https://github.com/CQCL/hugr/pull/1626))

## [0.13.2](https://github.com/CQCL/hugr/compare/hugr-v0.13.1...hugr-v0.13.2) - 2024-10-22

### Bug Fixes

- Allocate ports on root nodes ([#1585](https://github.com/CQCL/hugr/pull/1585))

### New Features

- Render function names in `mermaid`/`dot` ([#1583](https://github.com/CQCL/hugr/pull/1583))
- Add filter_edge_kind to PortIterator ([#1593](https://github.com/CQCL/hugr/pull/1593))
- make errors more readable with Display impls ([#1597](https://github.com/CQCL/hugr/pull/1597))
- Ensure packages always have modules at the root ([#1589](https://github.com/CQCL/hugr/pull/1589))
- Add `Package` definition on `hugr-core` ([#1587](https://github.com/CQCL/hugr/pull/1587))

## [0.13.1](https://github.com/CQCL/hugr/compare/hugr-v0.13.0...hugr-v0.13.1) - 2024-10-14

### New Features

- return replaced ops from lowering ([#1568](https://github.com/CQCL/hugr/pull/1568))
- Make `BuildHandle::num_value_outputs` public ([#1560](https://github.com/CQCL/hugr/pull/1560))
- `FunctionBuilder::add_{in,out}put` ([#1570](https://github.com/CQCL/hugr/pull/1570))
- Binary serialisation format for hugr-model based on capnproto. ([#1557](https://github.com/CQCL/hugr/pull/1557))

## 0.13.0 (2024-10-08)

### Bug Fixes

- [**breaking**] Make list length op give back the list ([#1547](https://github.com/CQCL/hugr/pull/1547))

### Features

- [**breaking**] Allow CustomConsts to (optionally) be hashable ([#1397](https://github.com/CQCL/hugr/pull/1397))
- Add an `OpLoadError` variant of `BuildError`. ([#1537](https://github.com/CQCL/hugr/pull/1537))
- [**breaking**] `HugrMut::remove_node` and `SimpleReplacement` return removed weights ([#1516](https://github.com/CQCL/hugr/pull/1516))
- Draft for `hugr-model` with export, import, parsing and pretty printing ([#1542](https://github.com/CQCL/hugr/pull/1542))


## 0.12.1 (2024-09-04)

### Bug Fixes

- `std.collections.insert` wrong output order ([#1513](https://github.com/CQCL/hugr/pull/1513))

### Features

- Op replacement and lowering functions ([#1509](https://github.com/CQCL/hugr/pull/1509))


## 0.12.0 (2024-08-30)

### Features

- [**breaking**] Disallow opaque ops during validation ([#1431](https://github.com/CQCL/hugr/pull/1431))
- [**breaking**] Add collections to serialized standard extensions ([#1452](https://github.com/CQCL/hugr/pull/1452))
- [**breaking**] Variadic logic ops now binary ([#1451](https://github.com/CQCL/hugr/pull/1451))
- [**breaking**] Int operations other than widen/narrow have only one width arg ([#1455](https://github.com/CQCL/hugr/pull/1455))
- Add a `FuncTypeBase::io` method ([#1458](https://github.com/CQCL/hugr/pull/1458))
- Add missing ops ([#1463](https://github.com/CQCL/hugr/pull/1463))
- [**breaking**] Move `Lift`, `MakeTuple`, `UnpackTuple` and `Lift` to prelude ([#1475](https://github.com/CQCL/hugr/pull/1475))
- `Option` / `Result` helpers ([#1481](https://github.com/CQCL/hugr/pull/1481))
- [**breaking**] Add more list operations ([#1474](https://github.com/CQCL/hugr/pull/1474))
- [**breaking**] Move int conversions to `conversions` ext, add to/from usize ([#1490](https://github.com/CQCL/hugr/pull/1490))
- Fill out array ops ([#1491](https://github.com/CQCL/hugr/pull/1491))

### Refactor

- [**breaking**] Bring the collections ext in line with other extension defs ([#1469](https://github.com/CQCL/hugr/pull/1469))
- [**breaking**] Make Either::Right the "success" case ([#1489](https://github.com/CQCL/hugr/pull/1489))
- [**breaking**] Flatten `CustomOp` in to `OpType` ([#1429](https://github.com/CQCL/hugr/pull/1429))

### Testing

- Add serialization benchmarks ([#1439](https://github.com/CQCL/hugr/pull/1439))


## 0.11.0 (2024-08-12)

### Bug Fixes

- [**breaking**] BasicBlockExits should not be `OpTag::DataflowParent` ([#1409](https://github.com/CQCL/hugr/pull/1409))

### Documentation

- Clarify CustomConst::equal_consts ([#1396](https://github.com/CQCL/hugr/pull/1396))

### Features

- [**breaking**] Serialised extensions ([#1371](https://github.com/CQCL/hugr/pull/1371))
- Serialised standard extensions ([#1377](https://github.com/CQCL/hugr/pull/1377))
- [**breaking**] Update remaining builder methods to "infer by default" ([#1386](https://github.com/CQCL/hugr/pull/1386))
- Add Eq op to logic extension ([#1398](https://github.com/CQCL/hugr/pull/1398))
- Improve error message on failed custom op validation ([#1416](https://github.com/CQCL/hugr/pull/1416))
- [**breaking**] `Extension` requires a version ([#1367](https://github.com/CQCL/hugr/pull/1367))


## 0.10.0 (2024-07-26)

### Bug Fixes

- [**breaking**] Bump serialisation version with no upgrade path ([#1352](https://github.com/CQCL/hugr/pull/1352))

### Features

- Add `nonlocal_edges` and `ensure_no_nonlocal_edges` ([#1345](https://github.com/CQCL/hugr/pull/1345))
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


## 0.9.1 (2024-07-25)

### Bug Fixes

- Dfg wrapper build handles incorrect output wire numbers ([#1332](https://github.com/CQCL/hugr/pull/1332))
- Sibling extension panics while computing signature with non-dataflow nodes ([#1350](https://github.com/CQCL/hugr/pull/1350))


## 0.9.0 (2024-07-19)

### Bug Fixes

- Add op's extension to signature check in `resolve_opaque_op` ([#1317](https://github.com/CQCL/hugr/pull/1317))
- Panic on `SimpleReplace` with multiports ([#1324](https://github.com/CQCL/hugr/pull/1324))

### Refactor

- [**breaking**] Separate Signature from FuncValueType by parametrizing Type(/Row)/etc. ([#1138](https://github.com/CQCL/hugr/pull/1138))

### Testing

- Verify order edges ([#1293](https://github.com/CQCL/hugr/pull/1293))
- Add failing test case for [#1315](https://github.com/CQCL/hugr/pull/1315) ([#1316](https://github.com/CQCL/hugr/pull/1316))


## 0.8.0 (2024-07-16)

### Bug Fixes

- [**breaking**] Force_order failing on Const nodes, add arg to rank. ([#1300](https://github.com/CQCL/hugr/pull/1300))
- NonConvex error on SiblingSubgraph::from_nodes with multiports ([#1295](https://github.com/CQCL/hugr/pull/1295))
- [**breaking**] Ops require their own extension ([#1226](https://github.com/CQCL/hugr/pull/1226))

### Documentation

- Attempt to correct force_order docs ([#1299](https://github.com/CQCL/hugr/pull/1299))

### Features

- Make `DataflowOpTrait` public ([#1283](https://github.com/CQCL/hugr/pull/1283))
- Make op members consistently public ([#1274](https://github.com/CQCL/hugr/pull/1274))

### Refactor

- [**breaking**] Rename builder helpers: ft1->endo_ft, ft2->inout_ft ([#1297](https://github.com/CQCL/hugr/pull/1297))


## 0.7.0 (2024-07-10)

### Bug Fixes

- Bring back input_extensions serialized field in rust NodeSer ([#1275](https://github.com/CQCL/hugr/pull/1275))
- [**breaking**] `ops::Module` now empty struct rather than unit struct ([#1271](https://github.com/CQCL/hugr/pull/1271))

### Features

- Add `force_order` pass. ([#1285](https://github.com/CQCL/hugr/pull/1285))
- [**breaking**] `MakeOpDef` has new `extension` method. ([#1266](https://github.com/CQCL/hugr/pull/1266))

### Refactor

- [**breaking**] Remove `Value::Tuple` ([#1255](https://github.com/CQCL/hugr/pull/1255))
- [**breaking**] Rename `HugrView` function type methods + simplify behaviour ([#1265](https://github.com/CQCL/hugr/pull/1265))

### Styling

- Change "serialise" etc to "serialize" etc. ([#1251](https://github.com/CQCL/hugr/pull/1251))

### Testing

- Add a test for [#1257](https://github.com/CQCL/hugr/pull/1257) ([#1260](https://github.com/CQCL/hugr/pull/1260))


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

- Serialization round trip testing for `OpDef` ([#999](https://github.com/CQCL/hugr/pull/999))

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
- Disable serialization tests when miri is active ([#977](https://github.com/CQCL/hugr/pull/977))
- [**breaking**] Serialization schema ([#968](https://github.com/CQCL/hugr/pull/968))
- Correct constant fold for `fne`. ([#995](https://github.com/CQCL/hugr/pull/995))
- [**breaking**] Serialization fixes ([#997](https://github.com/CQCL/hugr/pull/997))
- [**breaking**] OpDef serialization ([#1013](https://github.com/CQCL/hugr/pull/1013))
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
- [**breaking**] Update serialization schema, implement `CustomConst` serialization ([#1005](https://github.com/CQCL/hugr/pull/1005))
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
- Proptests for round trip serialization of `Type`s and `Op`s. ([#981](https://github.com/CQCL/hugr/pull/981))
- Add a test of instantiating an extension set ([#939](https://github.com/CQCL/hugr/pull/939))
- Ignore serialization tests when using miri ([#975](https://github.com/CQCL/hugr/pull/975))
- [**breaking**] Test roundtrip serialization against strict + lax schema ([#982](https://github.com/CQCL/hugr/pull/982))
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

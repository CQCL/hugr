# Changelog

## [0.12.2](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.1...hugr-py-v0.12.2) (2025-06-03)


### Bug Fixes

* use envelopes for `FixedHugr` encoding ([#2283](https://github.com/CQCL/hugr/issues/2283)) ([2c8cbb9](https://github.com/CQCL/hugr/commit/2c8cbb99bc74d5d43956b5f75c89f17748b5ee39)), closes [#2282](https://github.com/CQCL/hugr/issues/2282)


### Performance Improvements

* **py:** mutable `Node` to avoid linear update cost ([#2288](https://github.com/CQCL/hugr/issues/2288)) ([84fb200](https://github.com/CQCL/hugr/commit/84fb2002dc835f6b98ceb95bd80a7bcff9eecdd8))


### Documentation

* **py:** fix `TypeDef` example ([#2268](https://github.com/CQCL/hugr/issues/2268)) ([ede8e7b](https://github.com/CQCL/hugr/commit/ede8e7b087591303038ecc5b449bb85bf39c948b))

## [0.12.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.0...hugr-py-v0.12.1) (2025-05-20)


### Bug Fixes

* handle order port case for missing dataflow ops ([#2238](https://github.com/CQCL/hugr/issues/2238)) ([dc44b81](https://github.com/CQCL/hugr/commit/dc44b811b19537a763cfe789c595fc26ed69c34d))
* **py:** deprecate extensions sets in values ([#2233](https://github.com/CQCL/hugr/issues/2233)) ([fe98ba1](https://github.com/CQCL/hugr/commit/fe98ba10684b9cf38d96c0cf9cde89a736b27bf3))


### Documentation

* update hugr-py docs appearance, add HUGR logo ([#2222](https://github.com/CQCL/hugr/issues/2222)) ([fefa599](https://github.com/CQCL/hugr/commit/fefa599f12a0ddb3335b01abad87bb80ecf2ed36))

## [0.12.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.5...hugr-py-v0.12.0) (2025-05-16)


### ⚠ BREAKING CHANGES

* Hugrs now define an `entrypoint` in addition to a module root.
* `std.collections.array` is now a linear type, even if the contained elements are copyable. Use the new `std.collections.value_array` for an array with the previous copyable semantics.
* `std.collections.array.get` now also returns the passed array as an extra output.
* `ArrayOpBuilder` was moved from `hugr_core::std_extensions::collections::array::op_builder` to `hugr_core::std_extensions::collections::array`.
* Functions that manipulate runtime extension sets have been removed from the Rust and Python code. Extension set parameters were removed from operations.
* `values` field in `Extension` and `ExtensionValue` struct/class removed in rust and python. Use 0-input ops that return constant values.

### Features

* Entrypoints in `hugr-py` ([#2148](https://github.com/CQCL/hugr/issues/2148)) ([ef8ea5e](https://github.com/CQCL/hugr/commit/ef8ea5e0ac6f4ea8a3e4ba8f6d1a36e53227546e))
* **hugr-py:** Add `to/from_bytes/str` to Hugr, using envelopes ([#2228](https://github.com/CQCL/hugr/issues/2228)) ([9985143](https://github.com/CQCL/hugr/commit/9985143bfb1d751911e519dd55890d179868524f))
* Improved array lowering ([#2109](https://github.com/CQCL/hugr/issues/2109)) ([1bc91c1](https://github.com/CQCL/hugr/commit/1bc91c197519f4a81f5fff1bf9df5905a1d1559e))
* Remove description on opaque ops. ([#2197](https://github.com/CQCL/hugr/issues/2197)) ([f6163bf](https://github.com/CQCL/hugr/commit/f6163bf09f208047bfa8fcf4069f2991f0434101))
* remove ExtensionValue ([#2093](https://github.com/CQCL/hugr/issues/2093)) ([70881b7](https://github.com/CQCL/hugr/commit/70881b7c5a55613f0304f41ee7cae8236a8bd668))
* Removed runtime extension sets. ([#2145](https://github.com/CQCL/hugr/issues/2145)) ([cd7ef68](https://github.com/CQCL/hugr/commit/cd7ef68120b5b903b12ac2fcbbf5fae812e3e70f))

## [0.11.5](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.4...hugr-py-v0.11.5) (2025-04-16)


### Features

* **hugr-py:** move in result classes from guppylang ([#2084](https://github.com/CQCL/hugr/issues/2084)) ([b6efb03](https://github.com/CQCL/hugr/commit/b6efb03bde407740ba546fff72435cc9d70a380b))
* Packages in `hugr-model` and envelope support. ([#2026](https://github.com/CQCL/hugr/issues/2026)) ([a16389f](https://github.com/CQCL/hugr/commit/a16389fd6909e29ba1a7d93efea2fc75f810e6b8))
* Represent order edges in `hugr-model` as metadata. ([#2027](https://github.com/CQCL/hugr/issues/2027)) ([09de9e3](https://github.com/CQCL/hugr/commit/09de9e32712c2659cf9ce0ef0254273e2cc916b1))

## [0.11.4](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.3...hugr-py-v0.11.4) (2025-03-28)


### Features

* Python bindings for `hugr-model`. ([#1959](https://github.com/CQCL/hugr/issues/1959)) ([25df063](https://github.com/CQCL/hugr/commit/25df06380d9e14a4bde8f6353c70dc27ef58c3ef))
* Remove extension sets from `hugr-model`. ([#2031](https://github.com/CQCL/hugr/issues/2031)) ([5dd1f96](https://github.com/CQCL/hugr/commit/5dd1f9659f0954e937e0a3a41886e953db58628b))

## [0.11.3](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.2...hugr-py-v0.11.3) (2025-03-21)


### Bug Fixes

* Don't enable envelope compression by default (yet) ([#2014](https://github.com/CQCL/hugr/issues/2014)) ([c5423ed](https://github.com/CQCL/hugr/commit/c5423edf6a4650a6222df08b5c5fb13529b5ef9f))

## [0.11.2](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.1...hugr-py-v0.11.2) (2025-03-21)


### Features

* add exit operation to prelude ([#2008](https://github.com/CQCL/hugr/issues/2008)) ([6bd7665](https://github.com/CQCL/hugr/commit/6bd76659d1f3f3b100cef46f0d5f7ceec79699a9))
* Add llvm codegen for collections.static_array ([#2003](https://github.com/CQCL/hugr/issues/2003)) ([f3dd145](https://github.com/CQCL/hugr/commit/f3dd145963ce23152f29d2d46be7eaa9a78ef2c5))
* **hugr-py:** Add `StaticArray` to standard extensions ([#1985](https://github.com/CQCL/hugr/issues/1985)) ([cf860f3](https://github.com/CQCL/hugr/commit/cf860f34b132a26411787f80668d621b7273f2c9)), closes [#1984](https://github.com/CQCL/hugr/issues/1984)
* **hugr-py:** Support envelope compression ([#1994](https://github.com/CQCL/hugr/issues/1994)) ([434c563](https://github.com/CQCL/hugr/commit/434c563ae4134b34070c45dfd8d13865b613c49d))


### Bug Fixes

* StaticArrayValue serialisation ([#2009](https://github.com/CQCL/hugr/issues/2009)) ([3fe6bf8](https://github.com/CQCL/hugr/commit/3fe6bf82ad3ebed5689e3304e7df88f43b9128b1))


### Reverts

* Revert breaking change to StaticArrayValue ([33a2b49](https://github.com/CQCL/hugr/commit/33a2b49d2d343265415dab3c52631845b5cd53ce))

## [0.11.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.11.0...hugr-py-v0.11.1) (2025-03-17)


### Features

* Add default envelope config to `Package.to_str`/`to_bytes` ([#1980](https://github.com/CQCL/hugr/issues/1980)) ([44deda1](https://github.com/CQCL/hugr/commit/44deda11e9c335f7ff705eaf1b8bcc01b6c2e202))


### Bug Fixes

* Cyclic import in hugr.envelope ([#1981](https://github.com/CQCL/hugr/issues/1981)) ([cff0dba](https://github.com/CQCL/hugr/commit/cff0dba67c241f6a817eb029f73e7c9634b2d441))

## [0.11.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.10.3...hugr-py-v0.11.0) (2025-03-14)


### ⚠ BREAKING CHANGES

* Lift op in prelude replaced with a Barrier that behaves similarly except does not add any extensions.

### Features

* Add collections.static_array extension. ([#1964](https://github.com/CQCL/hugr/issues/1964)) ([fdcd48a](https://github.com/CQCL/hugr/commit/fdcd48af26c10ec92b63287681fd201dff0f281c))
* Add float &lt;--&gt; int bytecasting ops to conversions extension ([#1956](https://github.com/CQCL/hugr/issues/1956)) ([fa1bf86](https://github.com/CQCL/hugr/commit/fa1bf867289bbfce10316a7101ea123419d1893f))
* Generic HUGR serialization with envelopes ([6710e5f](https://github.com/CQCL/hugr/commit/6710e5fe3939a71b38018fca547c4d9cc421cbac))
* replace `Lift` with `Barrier` ([#1952](https://github.com/CQCL/hugr/issues/1952)) ([4e6b6d8](https://github.com/CQCL/hugr/commit/4e6b6d8df576f43bd5ced51b2eb4f1ed4d5c3b82))


### Bug Fixes

* **hugr-py:** output and delete node issues ([#1971](https://github.com/CQCL/hugr/issues/1971)) ([408517d](https://github.com/CQCL/hugr/commit/408517d620711c5573ea013e4d062499a62f55dd))

## [0.10.3](https://github.com/CQCL/hugr/compare/hugr-py-v0.10.2...hugr-py-v0.10.3) (2025-02-17)


### Features

* add xor to logic extension ([#1911](https://github.com/CQCL/hugr/issues/1911)) ([5e7c81e](https://github.com/CQCL/hugr/commit/5e7c81e2a6e939629feac8448f79ed30d986349a))
* **hugr-llvm:** Emit ipow ([#1839](https://github.com/CQCL/hugr/issues/1839)) ([89e671a](https://github.com/CQCL/hugr/commit/89e671a27501363994322a71c2a0d83f59ebebe4))

## [0.10.2](https://github.com/CQCL/hugr/compare/hugr-py-v0.10.1...hugr-py-v0.10.2) (2024-12-20)


### Features

* Add `StringVal` to hugr-py ([#1818](https://github.com/CQCL/hugr/issues/1818)) ([b05a419](https://github.com/CQCL/hugr/commit/b05a419e08dfcccf10fa081a22fc83af0d11502b))


### Bug Fixes

* **py:** Fix array/list value serialization ([#1827](https://github.com/CQCL/hugr/issues/1827)) ([7bf85b9](https://github.com/CQCL/hugr/commit/7bf85b94dfbeebddb91c261c155e8f47a6cd14ef))

## [0.10.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.10.0...hugr-py-v0.10.1) (2024-12-18)


### Features

* add ArrayValue to python, rust and lowering ([#1773](https://github.com/CQCL/hugr/issues/1773)) ([d429cff](https://github.com/CQCL/hugr/commit/d429cffc8a5a6a10af44b701aca772622c862eb6))
* add wrapper for tagging Some, Left, Right, Break, Continue ([#1814](https://github.com/CQCL/hugr/issues/1814)) ([f0385a0](https://github.com/CQCL/hugr/commit/f0385a0f3edc1490b3bb2e639b00379c5a556866)), closes [#1808](https://github.com/CQCL/hugr/issues/1808)

## [0.10.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.9.0...hugr-py-v0.10.0) (2024-12-16)


### ⚠ BREAKING CHANGES

* `extension_reqs` field in FunctionType and Extension renamed to `runtime_reqs`
* Array type and operations have been moved out of `prelude` and into a new `collections.array` extension. (py) `list_type` method replaced with `List` class. Removed `Array` type variant from the serialization format.
* `collections` extension renamed to `collections.list`


### Features

* Add `LoadNat` operation to enable loading generic `BoundedNat`s into runtime values ([#1763](https://github.com/CQCL/hugr/issues/1763)) ([6f035d6](https://github.com/CQCL/hugr/commit/6f035d68bd5c0444e4f1aedd254ee518e9c705ea)), closes [#1629](https://github.com/CQCL/hugr/issues/1629)
* Add array `repeat` and `scan` ops ([#1633](https://github.com/CQCL/hugr/issues/1633)) ([649589c](https://github.com/CQCL/hugr/commit/649589c9e3f1fbd9cfff53a2adb8e1f9649fbe87)), closes [#1627](https://github.com/CQCL/hugr/issues/1627)
* Make array repeat and scan ops generic over extension reqs ([#1716](https://github.com/CQCL/hugr/issues/1716)) ([4c1c6ee](https://github.com/CQCL/hugr/commit/4c1c6ee4c7d657c4bdb6b37c2237ae3f06b8d0be))
* Move arrays from prelude into new extension ([#1770](https://github.com/CQCL/hugr/issues/1770)) ([187ea8f](https://github.com/CQCL/hugr/commit/187ea8f59ee307c0ed5afe2b0faad7c6e90051f0))
* Rename `collections` extension to `collections.list` ([#1764](https://github.com/CQCL/hugr/issues/1764)) ([eef239f](https://github.com/CQCL/hugr/commit/eef239fa02019180f398444de4b9a45a1f2f3a3e))
* rename `extension_reqs` to `runtime_reqs` ([#1776](https://github.com/CQCL/hugr/issues/1776)) ([5f5bce4](https://github.com/CQCL/hugr/commit/5f5bce4805897d5b0fa70af69fddc039f7a8d8ab))


### Bug Fixes

* hugr-py not adding extension-reqs on custom ops ([#1759](https://github.com/CQCL/hugr/issues/1759)) ([97ba7f4](https://github.com/CQCL/hugr/commit/97ba7f4b26773598affb4dd8ac119e9e1d1444e2))
* allow conditional cases to be defined out of order ([#1599](https://github.com/CQCL/hugr/issues/1599)) ([583d21d](https://github.com/CQCL/hugr/commit/583d21d371320851f8608daa295ef8b723d31326))
* Update number of ports for PartialOps, and sanitize orderd edges ([#1635](https://github.com/CQCL/hugr/issues/1635)) ([81a1385](https://github.com/CQCL/hugr/commit/81a1385fd56a9a12b84153756b4c0bb046808c50)), closes [#1625](https://github.com/CQCL/hugr/issues/1625)

## [0.9.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.8.1...hugr-py-v0.9.0) (2024-10-14)


### ⚠ BREAKING CHANGES

* `Package` moved to new `hugr.package` module
* The `length` op in the std `collections` extensions now also returns the list.

### Features

* `instantiate` method for `OpDef` ([#1576](https://github.com/CQCL/hugr/issues/1576)) ([36548ab](https://github.com/CQCL/hugr/commit/36548ab4e377ca9074a80355fabd51693d89c649)), closes [#1512](https://github.com/CQCL/hugr/issues/1512)
* define wrappers around package that point to internals ([#1573](https://github.com/CQCL/hugr/issues/1573)) ([f74dbf3](https://github.com/CQCL/hugr/commit/f74dbf333fb69b2965cc38c513ad8dfbdfcb0e3c))
* to/from json for extension/package ([#1575](https://github.com/CQCL/hugr/issues/1575)) ([f8bf61a](https://github.com/CQCL/hugr/commit/f8bf61aa54dd2424c42ecba7d1ae41b1d35f7f9d)), closes [#1523](https://github.com/CQCL/hugr/issues/1523)


### Bug Fixes

* Make list length op give back the list ([#1547](https://github.com/CQCL/hugr/issues/1547)) ([cf31698](https://github.com/CQCL/hugr/commit/cf31698113ea02e2d13596638b1fe0f4f118a601))

## [0.8.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.8.0...hugr-py-v0.8.1) (2024-09-04)


### Features

* Export the collections extension ([#1506](https://github.com/CQCL/hugr/issues/1506)) ([70e0a64](https://github.com/CQCL/hugr/commit/70e0a64cf3aaa8d8be8c999684a6c173d7181663))


### Bug Fixes

* Export the float ops extension ([#1517](https://github.com/CQCL/hugr/issues/1517)) ([4cbe890](https://github.com/CQCL/hugr/commit/4cbe890ab4e72090708ff83592c0771caf2335df))
* IndexError on node slicing ([#1500](https://github.com/CQCL/hugr/issues/1500)) ([a32bd84](https://github.com/CQCL/hugr/commit/a32bd84139013f279ba62d42ffc71ef79340de52))
* Update collections extension ([#1518](https://github.com/CQCL/hugr/issues/1518)) ([60e1da0](https://github.com/CQCL/hugr/commit/60e1da0144c5080de7427d49d3700d6d8443609b))

## [0.8.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.7.0...hugr-py-v0.8.0) (2024-08-30)


### ⚠ BREAKING CHANGES

* Moved `itobool`/`ifrombool`/`itostring_{u,s}` to the conversions extension.
* Binary sums representing fallible values now use tag `1` for the successful variant
* Renamed `Custom.name` to `Custom.op_name` and `Func(Defn/Decl).name` to `f_name` to allow for new `name` method
* `ListOp::pop` now returns an option.
* Moved all builder definitions into the `hugr.build` module. Moved `node_port` and `render` into the `hugr.hugr` module.
* Moved `Lift`, `MakeTuple`, `UnpackTuple` and `Lift` from core operations to prelude. Rename `ops::leaf` module to `ops::sum`.
* `hugr.serialization` module and `Hugr.to_serial` methods are now internal only.
* Renamed `_DfBase` to `DfBase` and `_DefinitionBuilder` to `DefinitionBuilder`
* `idivmod_checked`, `idivmod`, `idiv`, `idiv_checked`, `imod`, `ishl`, `ishr`, `irotl`, `irotr` operations now only have one width argument for all inputs and outputs rather than two.
* HUGRs containing opaque operations that don't point to an extension in the registry will fail to validate. Use `Package` to pack extensions with HUGRs for serialisation.
* Removed `CustomOp`, `OpType` now contains `ExtensionOp` and `OpaqueOp` directly. `CustomOpError` renamed to`OpaqueOpError`.

### Features

* `Option` / `Result` helpers ([#1481](https://github.com/CQCL/hugr/issues/1481)) ([9698420](https://github.com/CQCL/hugr/commit/969842091e06d1482c8bc77965847865cb9f77a0))
* Add missing ops ([#1463](https://github.com/CQCL/hugr/issues/1463)) ([841f450](https://github.com/CQCL/hugr/commit/841f450dab3b50bb3fa7d0da75902608ff7165e7))
* Add more list operations ([#1474](https://github.com/CQCL/hugr/issues/1474)) ([037005f](https://github.com/CQCL/hugr/commit/037005f27520511401b4c116244435fbbdbe0b60))
* Bring in the pure-python renderer from guppy ([#1462](https://github.com/CQCL/hugr/issues/1462)) ([001e66a](https://github.com/CQCL/hugr/commit/001e66a49ae2cbd0b49a7c2ed0b73eae8ab07379))
* Disallow opaque ops during validation ([#1431](https://github.com/CQCL/hugr/issues/1431)) ([fbbb805](https://github.com/CQCL/hugr/commit/fbbb805b9d25d5219e1081d015c67422225d7f79))
* Fill out array ops ([#1491](https://github.com/CQCL/hugr/issues/1491)) ([26ec57a](https://github.com/CQCL/hugr/commit/26ec57ac006ab6c44902c68dbf354f8f8e0933f1))
* Pretty printing for ops and types ([#1482](https://github.com/CQCL/hugr/issues/1482)) ([aca403a](https://github.com/CQCL/hugr/commit/aca403a2f3eef5dd6a1fd614079d8eee1243fdde))
* Use serialized extensions in python ([#1459](https://github.com/CQCL/hugr/issues/1459)) ([a61f4df](https://github.com/CQCL/hugr/commit/a61f4df66cb6ce11b342103af705145441ea9b5c)), closes [#1450](https://github.com/CQCL/hugr/issues/1450)
* Int operations other than widen/narrow have only one width arg ([#1455](https://github.com/CQCL/hugr/issues/1455)) ([c39ed15](https://github.com/CQCL/hugr/commit/c39ed151f413284091f0d861f926541dfed8a1ef))
* Move `Lift`, `MakeTuple`, `UnpackTuple` and `Lift` to prelude ([#1475](https://github.com/CQCL/hugr/issues/1475)) ([b387505](https://github.com/CQCL/hugr/commit/b38750585c19b41cc486095186f72d70ff11980c))
* Move int conversions to `conversions` ext, add to/from usize ([#1490](https://github.com/CQCL/hugr/issues/1490)) ([88913f2](https://github.com/CQCL/hugr/commit/88913f29287efafc5303e91dd4677582348ee2f7))


### Bug Fixes

* Fixed deserialization panic while extracting node children ([#1480](https://github.com/CQCL/hugr/issues/1480)) ([331125a](https://github.com/CQCL/hugr/commit/331125a6ca9d05e58b30c8593257126d68a02bc7)), closes [#1479](https://github.com/CQCL/hugr/issues/1479)
* Fixed errors while indexing on a `ToNode` ([#1457](https://github.com/CQCL/hugr/issues/1457)) ([d6edcd7](https://github.com/CQCL/hugr/commit/d6edcd77e7679791ae5ab910d13ccccf9f8ca914))
* Fixed schema for array inner types ([#1494](https://github.com/CQCL/hugr/issues/1494)) ([d43cbb2](https://github.com/CQCL/hugr/commit/d43cbb2c3035f49e4ea7a7769fd1a51db31806ce)), closes [#1471](https://github.com/CQCL/hugr/issues/1471)
* Fixed `val.Sum` equality. Add unit tests ([#1484](https://github.com/CQCL/hugr/issues/1484)) ([a7b2718](https://github.com/CQCL/hugr/commit/a7b27180cbb85490c09f8e24f46eeb4d5fd5eb21))


### Code Refactoring

* Flattened `CustomOp` in to `OpType` ([#1429](https://github.com/CQCL/hugr/issues/1429)) ([8e8bba5](https://github.com/CQCL/hugr/commit/8e8bba55a5d2a0a421a835b80c8aea07eae28e65))
* Made serialization (module/methods) private ([#1477](https://github.com/CQCL/hugr/issues/1477)) ([49a5bad](https://github.com/CQCL/hugr/commit/49a5bad5399eef248aba0b74b14ac23546324b14))
* Made `_DfBase` and `_DefinitionBuilder` public ([#1461](https://github.com/CQCL/hugr/issues/1461)) ([ea9cca0](https://github.com/CQCL/hugr/commit/ea9cca001bff3fff41d84e861f8b2b7ee26645d1))
* Made Either::Right the "success" case ([#1489](https://github.com/CQCL/hugr/issues/1489)) ([8caa572](https://github.com/CQCL/hugr/commit/8caa572d01aac59715480827eaf568d8488ff542))
* Reorganised the python module structure ([#1460](https://github.com/CQCL/hugr/issues/1460)) ([3ca56f4](https://github.com/CQCL/hugr/commit/3ca56f43c9499c629307b0a52aabee6661e22c99))

## [0.7.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.6.0...hugr-py-v0.7.0) (2024-08-14)


### ⚠ BREAKING CHANGES

* `AsCustomOp` replaced with `AsExtOp`, so all such operations now need to be attached to an extension.

### Features

* user facing Extension class ([#1413](https://github.com/CQCL/hugr/issues/1413)) ([c6473c9](https://github.com/CQCL/hugr/commit/c6473c9de27f7371798c8fbb27d193329d3210f2))
* Add node metadata ([#1428](https://github.com/CQCL/hugr/issues/1428)) ([b229be6](https://github.com/CQCL/hugr/commit/b229be6b8f709d28c5b57c380db03cd21598c3c1)), closes [#1319](https://github.com/CQCL/hugr/issues/1319)


### Bug Fixes

* Equality check between `Sum` types ([#1422](https://github.com/CQCL/hugr/issues/1422)) ([8dfea09](https://github.com/CQCL/hugr/commit/8dfea09c61d9e4e0d85c8f9829e85db7285d99c1))
* Invalid serialization of float and int constants ([#1427](https://github.com/CQCL/hugr/issues/1427)) ([b89c08f](https://github.com/CQCL/hugr/commit/b89c08f597019dae30f415787f7054c1a79bcefa))

## [0.6.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.5.0...hugr-py-v0.6.0) (2024-08-12)


### ⚠ BREAKING CHANGES

* **hugr-py:** Moved `hugr.get_serialization_version` to `hugr.serialization.serial_hugr.serialization_version`
* **hugr-cli:** Cli validate command no longer has a mermaid option, use `mermaid` sub-command instead.
* `TypeDefBound` uses struct-variants for serialization. `SignatureFunc` now has variants for missing binary functions, and serializes in to a new format that indicates expected binaries.

### Features

* `Package` pydantic model for modules + extensions ([#1387](https://github.com/CQCL/hugr/issues/1387)) ([68cfac5](https://github.com/CQCL/hugr/commit/68cfac5f63b35e8d709c5738a8893048240d5706)), closes [#1358](https://github.com/CQCL/hugr/issues/1358)
* Define `Const` inline by default, and add a parameter to change the parent ([#1404](https://github.com/CQCL/hugr/issues/1404)) ([3609736](https://github.com/CQCL/hugr/commit/36097366df6bb3b79058b5f3bc49cb25362dc1b2))
* **hugr-cli:** move mermaid to own sub-command ([#1390](https://github.com/CQCL/hugr/issues/1390)) ([77795b9](https://github.com/CQCL/hugr/commit/77795b90f029d5c00a99037f4306724176e1dfbd))
* **hugr-py:** add type_bound method to `Type` ([#1410](https://github.com/CQCL/hugr/issues/1410)) ([bd5ba47](https://github.com/CQCL/hugr/commit/bd5ba478c6d6f821a5ba2fbcaa56bcb61c490b2f)), closes [#1365](https://github.com/CQCL/hugr/issues/1365)
* **hugr-py:** Allow defining functions, consts, and aliases inside DFGs ([#1394](https://github.com/CQCL/hugr/issues/1394)) ([d554072](https://github.com/CQCL/hugr/commit/d554072d0266a7a584ef1c03e6fd78c9d4167933))
* **hugr-py:** Reexport commonly used classes from the package root ([#1393](https://github.com/CQCL/hugr/issues/1393)) ([69925d0](https://github.com/CQCL/hugr/commit/69925d05b152205eb277e48605a6a8806d4fda24))
* **py:** `Hugr.to_json` and `.load_json` helpers ([#1403](https://github.com/CQCL/hugr/issues/1403)) ([e7f9f4c](https://github.com/CQCL/hugr/commit/e7f9f4c739e1b523fb7f2b7df0246baea2abf3df))
* **py:** Allow pre-declaring a `Function`'s output types ([#1417](https://github.com/CQCL/hugr/issues/1417)) ([fa0f5a4](https://github.com/CQCL/hugr/commit/fa0f5a4d4c13f54e5247ea16f43b928d4850c34b))
* **py:** implement `iter` on `ToNode` ([#1399](https://github.com/CQCL/hugr/issues/1399)) ([e88910b](https://github.com/CQCL/hugr/commit/e88910bc50f6459213e17d0ce5a8c1b87037b9a6))
* **py:** Parametric int type helper, and arbitrary width int constants ([#1406](https://github.com/CQCL/hugr/issues/1406)) ([abd70c9](https://github.com/CQCL/hugr/commit/abd70c99291f41fce57ed8d8b0692faa63117b74))
* Serialised extensions ([#1371](https://github.com/CQCL/hugr/issues/1371)) ([31be204](https://github.com/CQCL/hugr/commit/31be2047ffb5515e3ac4d7a4214a3164399f9b3c))


### Bug Fixes

* **py:** `Hugr.__iter__` returning `NodeData | None` instead of `Node`s ([#1401](https://github.com/CQCL/hugr/issues/1401)) ([c134584](https://github.com/CQCL/hugr/commit/c1345849a83e33571e1a51398f94348ea221d96b))
* **py:** Set output cont for Conditionals ([#1415](https://github.com/CQCL/hugr/issues/1415)) ([67bb8a0](https://github.com/CQCL/hugr/commit/67bb8a0bb31a93798606712695085422607e6c7c))


### Documentation

* **hugr-py:** expand toctree ([#1411](https://github.com/CQCL/hugr/issues/1411)) ([aa81c9a](https://github.com/CQCL/hugr/commit/aa81c9a79f1737c0e3dd77ff65c3af8716f26d21))
* **hugr-py:** remove multiversion + add justfile command ([#1381](https://github.com/CQCL/hugr/issues/1381)) ([dd1dc48](https://github.com/CQCL/hugr/commit/dd1dc484447cc2180058f4474594ddb51fa45cd6))

## [0.5.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.4.0...hugr-py-v0.5.0) (2024-07-29)


### ⚠ BREAKING CHANGES

* Eq type bound removed. References to `Eq` in serialized HUGRs will be treated as `Copyable`.
* **hugr-core:** All Hugrs serialised with earlier versions will fail to deserialise
* opaque type parameters replaced with string parameters.

### Features

* **hugr-py:** `AsCustomOp` protocol for user-defined custom op types. ([#1290](https://github.com/CQCL/hugr/issues/1290)) ([1db43eb](https://github.com/CQCL/hugr/commit/1db43eb57cb7455fc5b11e777aa299d7b10ce7c6))
* remove the `Eq` type bound. ([#1364](https://github.com/CQCL/hugr/issues/1364)) ([1218d21](https://github.com/CQCL/hugr/commit/1218d21cb6509c253fed011ca37c0dbbb566ae83))
* replace opaque type arguments with String ([#1328](https://github.com/CQCL/hugr/issues/1328)) ([24b2217](https://github.com/CQCL/hugr/commit/24b2217c45d6bfeef53573684787a7d57989ae75)), closes [#1308](https://github.com/CQCL/hugr/issues/1308)
* Serialization upgrade path ([#1327](https://github.com/CQCL/hugr/issues/1327)) ([d493139](https://github.com/CQCL/hugr/commit/d49313989b69f30072e7f36a380ecd538a3ac18e))


### Bug Fixes

* add op's extension to signature check in `resolve_opaque_op` ([#1317](https://github.com/CQCL/hugr/issues/1317)) ([01da7ba](https://github.com/CQCL/hugr/commit/01da7ba75e2c48604605c41cadae9360d567cf89))
* **hugr-core:** bump serialisation version with no upgrade path ([#1352](https://github.com/CQCL/hugr/issues/1352)) ([657cbb0](https://github.com/CQCL/hugr/commit/657cbb0399fbd9f151a494ef34d4745b30b047d2))
* **hugr-py:** ops require their own extensions ([#1303](https://github.com/CQCL/hugr/issues/1303)) ([026bfcb](https://github.com/CQCL/hugr/commit/026bfcb0008ce55b102fffd21fb31f24aefe4a69)), closes [#1301](https://github.com/CQCL/hugr/issues/1301)

## [0.4.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.3.1...hugr-py-v0.4.0) (2024-07-10)


### ⚠ BREAKING CHANGES

* **hugr-py:** dataclasses that derive from `ops.Custom` now have to be frozen, and instances cannot be mutated.

### Features

* **hugr-py:** `ops.Custom` is now a frozen dataclass ([94702d2](https://github.com/CQCL/hugr/commit/94702d2a9a9a6f4d311db6945ed2bee86b7bc46d))
* **hugr-py:** move std extension types/ops in to `std` module ([#1288](https://github.com/CQCL/hugr/issues/1288)) ([7d82245](https://github.com/CQCL/hugr/commit/7d8224530ec4e70a7749505b379d7a4fe65f1168))

## [0.3.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.3.0...hugr-py-v0.3.1) (2024-07-08)


### Features

* **hugr-py:** `TrackedDfg` builder for appending operations by index ([df9b4cc](https://github.com/CQCL/hugr/commit/df9b4cc9725529b0bd9f1cfde97fc7fa544296c9))
* **hugr-py:** context manager style nested building ([#1276](https://github.com/CQCL/hugr/issues/1276)) ([6b32734](https://github.com/CQCL/hugr/commit/6b32734c929a28ac1e8f5ee48362bce940b53d4a)), closes [#1243](https://github.com/CQCL/hugr/issues/1243)

## [0.3.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.2.1...hugr-py-v0.3.0) (2024-07-03)

### Features

* HUGR builders in hugr-py ([#486](https://github.com/CQCL/hugr/issues/486))


### Bug Fixes

* get rid of pydantic config deprecation warnings ([#1084](https://github.com/CQCL/hugr/issues/1084)) ([52fcb9d](https://github.com/CQCL/hugr/commit/52fcb9dc88e95e9660fc291181a37dc9d1802a3d))


### Documentation

* build and publish docs ([#1253](https://github.com/CQCL/hugr/issues/1253)) ([902fc14](https://github.com/CQCL/hugr/commit/902fc14069caad0af6af7b55cbf649134703f9b5))

## [0.2.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.2.0...hugr-py-v0.2.1) (2024-05-20)

### ⚠ BREAKING CHANGES

* New serialization schema
* rename `Const::const_type` and `Value::const_type` to `Const::get_type` and `Value::get_type`. These now match several other `get_type` functions ([#1005](https://github.com/CQCL/hugr/issues/1005))
* Many uses of `Const` now use `Value`.

### Features

* Add serialization schema for metadata ([#1038](https://github.com/CQCL/hugr/issues/1038)) ([19bac62](https://github.com/CQCL/hugr/commit/19bac6210aa8c495679bd7c783751e9cde744c61))
* Add LoadFunction node ([#947](https://github.com/CQCL/hugr/issues/947)) ([81e9602](https://github.com/CQCL/hugr/commit/81e9602a47eddadc1c11d74ca7bda3b194d24f00))
* Encoder metadata in serialized hugr ([#955](https://github.com/CQCL/hugr/issues/955)) ([0a44d48](https://github.com/CQCL/hugr/commit/0a44d487b73f58674eb5884c72479a03e924bef0))
* Implement `CustomConst` serialization ([#1005](https://github.com/CQCL/hugr/issues/1005)) ([c45e6fc](https://github.com/CQCL/hugr/commit/c45e6fc67334768ea55c4bd5223af0b7b0cc47ec))
* Revert the removal of `Value` ([#967](https://github.com/CQCL/hugr/issues/967)) ([0c354b6](https://github.com/CQCL/hugr/commit/0c354b6e07ae1aafee17e412fe54f7b3db321beb))
* Set default value for `Conditional.sum_rows` ([#934](https://github.com/CQCL/hugr/issues/934)) ([d69198e](https://github.com/CQCL/hugr/commit/d69198eb57bf77f32538e1ba8de1f308815a067d))


### Bug Fixes

* `OpDef` serialization ([#1013](https://github.com/CQCL/hugr/issues/1013)) ([3d8f6f6](https://github.com/CQCL/hugr/commit/3d8f6f6a655f8af7f8fc2929f9bd7d3031b403f5))
* input_port_types and other helper functions on pydantic schema ([#958](https://github.com/CQCL/hugr/issues/958)) ([8651839](https://github.com/CQCL/hugr/commit/86518390296bd93ca2fc65eccf158e21625b9073))
* Remove insert_port_types for LoadFunction ([#993](https://github.com/CQCL/hugr/issues/993)) ([acca7bf](https://github.com/CQCL/hugr/commit/acca7bfb4a074c7feb3b4b5758f589941632bc5a))
* Serialization for `Type`, `PolyFuncType`, and `Value` ([#968](https://github.com/CQCL/hugr/issues/968)) ([d913f40](https://github.com/CQCL/hugr/commit/d913f406478a9f884bffef2002a02d423796b4e9))
* Serialization for `Op`s ([#997](https://github.com/CQCL/hugr/issues/997)) ([9ce6e49](https://github.com/CQCL/hugr/commit/9ce6e49d1d0c8c200b9b78ebe35a0a3257009ca1))
* set `[build-system]` in `hugr-py/pyproject.toml` ([#1022](https://github.com/CQCL/hugr/issues/1022)) ([b9c3ee4](https://github.com/CQCL/hugr/commit/b9c3ee46abbc166fb82155c62c8583e575284578))


### Code Refactoring

* rename `Const::const_type` and `Value::const_type` to `Const::get_type` and `Value::get_type`. These now match several other `get_type` functions ([#1005](https://github.com/CQCL/hugr/issues/1005)) ([c45e6fc](https://github.com/CQCL/hugr/commit/c45e6fc67334768ea55c4bd5223af0b7b0cc47ec))

## 0.1.0 (2024-04-15)

This first release includes a pydantic model for the hugr serialization format version 1.

### Features

* Flatten `LeafOp` ([#922](https://github.com/CQCL/hugr/issues/922)) ([3598913](https://github.com/CQCL/hugr/commit/3598913002a361d487aa2f6ba899739d9a3c7f13))
* No polymorphic closures ([#906](https://github.com/CQCL/hugr/issues/906)) ([b05dd6b](https://github.com/CQCL/hugr/commit/b05dd6b1a15aefee277d4034ed07039a259261e0))
* **py:** Rename package to `hugr` ([#913](https://github.com/CQCL/hugr/issues/913)) ([9fe65db](https://github.com/CQCL/hugr/commit/9fe65db9dd7fd8eee28c13e6abe71fd5cf05f90a))

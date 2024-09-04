# Changelog

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

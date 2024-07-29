# Changelog

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

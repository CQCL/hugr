# Changelog

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

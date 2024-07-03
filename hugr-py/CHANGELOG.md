# Changelog

## [0.3.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.2.1...hugr-py-v0.3.0) (2024-07-03)


### ⚠ BREAKING CHANGES

* * `add_child_op`(`_with_parent`), etc., gone; use `add_child_node`(`_with_parent`) with an (impl Into-)OpType.
    * `get_nodetype` gone - use `get_optype`.
    * `NodeType` gone - use `OpType` directly. 
    * Various (Into<)Option<ExtensionSet> params removed from builder
    methods especially {cfg_,dfg_}builder.
    * `input_extensions` removed from serialization schema.
* the Signature class is gone, but it's not clear how or why you might have been using it...
* TailLoop node and associated builder functions now require specifying an ExtensionSet; extension/validate.rs deleted; some changes to Hugrs validated/rejected when the `extension_inference` feature flag is turned on
* Type::validate takes extra bool (allow_rowvars); renamed {FunctionType, PolyFuncType}::(validate=>validate_var_len).

### Features

* Allow "Row Variables" declared as List&lt;Type&gt; ([#804](https://github.com/CQCL/hugr/issues/804)) ([3ea4834](https://github.com/CQCL/hugr/commit/3ea4834dd00466e3c106917c1e09c0c5b74c5826))
* **hugr-py:** add builders for Conditional and TailLoop ([#1210](https://github.com/CQCL/hugr/issues/1210)) ([43569a4](https://github.com/CQCL/hugr/commit/43569a45575a005dd69808bb8d534b3ed55b2039))
* **hugr-py:** add CallIndirect, LoadFunction, Lift, Alias ([#1218](https://github.com/CQCL/hugr/issues/1218)) ([db09193](https://github.com/CQCL/hugr/commit/db0919312eb387cfef57f8a71bb2529c267445d1)), closes [#1213](https://github.com/CQCL/hugr/issues/1213)
* **hugr-py:** add values and constants ([#1203](https://github.com/CQCL/hugr/issues/1203)) ([f7ea178](https://github.com/CQCL/hugr/commit/f7ea17849dce860a84292ef5270a3ce2a65be870)), closes [#1202](https://github.com/CQCL/hugr/issues/1202)
* **hugr-py:** automatically add state order edges for inter-graph edges ([#1165](https://github.com/CQCL/hugr/issues/1165)) ([5da06e1](https://github.com/CQCL/hugr/commit/5da06e10581cbfed583bd466b27706241341ff14))
* **hugr-py:** builder for function definition/declaration and call ([#1212](https://github.com/CQCL/hugr/issues/1212)) ([af062ea](https://github.com/CQCL/hugr/commit/af062ea5a64636072bc3168b2301cbf12c96c8d5))
* **hugr-py:** builder ops separate from serialised ops ([#1140](https://github.com/CQCL/hugr/issues/1140)) ([342eda3](https://github.com/CQCL/hugr/commit/342eda34c1f3b4ea4423268e935af44af07c976f))
* **hugr-py:** CFG builder ([#1192](https://github.com/CQCL/hugr/issues/1192)) ([c5ea47f](https://github.com/CQCL/hugr/commit/c5ea47fd77cfbdda5f32d651618ed69b97740e2e)), closes [#1188](https://github.com/CQCL/hugr/issues/1188)
* **hugr-py:** define type hierarchy separate from serialized ([#1176](https://github.com/CQCL/hugr/issues/1176)) ([10f4c42](https://github.com/CQCL/hugr/commit/10f4c42cfe051381e50e6387af603253e941215b))
* **hugr-py:** only require input type annotations when building ([#1199](https://github.com/CQCL/hugr/issues/1199)) ([2bb079f](https://github.com/CQCL/hugr/commit/2bb079fd80fbec7a4f6fe4a5baeed3cf064d85a9))
* **hugr-py:** python hugr builder ([#1098](https://github.com/CQCL/hugr/issues/1098)) ([23408b5](https://github.com/CQCL/hugr/commit/23408b5bbb9666002a58bf88a2a33cca0a484b30))
* **hugr-py:** store children in node weight ([#1160](https://github.com/CQCL/hugr/issues/1160)) ([1cdaeed](https://github.com/CQCL/hugr/commit/1cdaeedde805fe3a9fd7c466ab9f2b34ac2d75c7)), closes [#1159](https://github.com/CQCL/hugr/issues/1159)
* **hugr-py:** ToNode interface to treat builders as nodes ([#1193](https://github.com/CQCL/hugr/issues/1193)) ([1da33e6](https://github.com/CQCL/hugr/commit/1da33e654df4a122c0af57e1c6db0ada7ca066df))
* Validate Extensions using hierarchy, ignore input_extensions, RIP inference ([#1142](https://github.com/CQCL/hugr/issues/1142)) ([8bec8e9](https://github.com/CQCL/hugr/commit/8bec8e93bcaa8917b00098837269da60e3312d6c))


### Bug Fixes

* Add some validation for const nodes ([#1222](https://github.com/CQCL/hugr/issues/1222)) ([c05edd3](https://github.com/CQCL/hugr/commit/c05edd3cbcb644556bb3b2b23b6d27a211fe7e4f))
* **hugr-py:** more ruff lints + fix some typos ([#1246](https://github.com/CQCL/hugr/issues/1246)) ([f158384](https://github.com/CQCL/hugr/commit/f158384c88787d1e436b634657dcfc12d531d68e))
* **py:** get rid of pydantic config deprecation warnings ([#1084](https://github.com/CQCL/hugr/issues/1084)) ([52fcb9d](https://github.com/CQCL/hugr/commit/52fcb9dc88e95e9660fc291181a37dc9d1802a3d))


### Documentation

* **hugr-py:** add docs link to README ([#1259](https://github.com/CQCL/hugr/issues/1259)) ([d2a9148](https://github.com/CQCL/hugr/commit/d2a9148ca67e3cd7517c897b3679294993a9526f))
* **hugr-py:** build and publish docs ([#1253](https://github.com/CQCL/hugr/issues/1253)) ([902fc14](https://github.com/CQCL/hugr/commit/902fc14069caad0af6af7b55cbf649134703f9b5))
* **hugr-py:** docstrings for builder ([#1231](https://github.com/CQCL/hugr/issues/1231)) ([3e4ac18](https://github.com/CQCL/hugr/commit/3e4ac18931f24a66028afb662019bf2c90304cdc))


### Code Refactoring

* Remove "Signature" from hugr-py ([#1186](https://github.com/CQCL/hugr/issues/1186)) ([65718f7](https://github.com/CQCL/hugr/commit/65718f7dbe70397eab7ab856965566f11b9322a5))
* Remove NodeType and input_extensions ([#1183](https://github.com/CQCL/hugr/issues/1183)) ([ea5213d](https://github.com/CQCL/hugr/commit/ea5213d4b3a42a86c637d709c48cad007eae1f9e))

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

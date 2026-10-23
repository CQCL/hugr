# Changelog

## [0.18.9](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.8...hugr-py-v0.18.9) (2026-10-23)


### Bug Fixes

* **python:** preserve CFG block shared outputs on model import ([#3300](https://github.com/Quantinuum/hugr/issues/3300)) ([924113b](https://github.com/Quantinuum/hugr/commit/924113b5df3279f65a6064ad94788e541618cc3d))


### Documentation

* Add examples for beginner HUGR guide ([#3305](https://github.com/Quantinuum/hugr/issues/3305)) ([9fcfd3b](https://github.com/Quantinuum/hugr/commit/9fcfd3bddd9f792a3fceaae0a3e9f1b62664f223))
* Add informal beginner HUGR construction guide ([#3272](https://github.com/Quantinuum/hugr/issues/3272)) ([7d2024a](https://github.com/Quantinuum/hugr/commit/7d2024a5f771c6944c8d503303f2f503566e9b83))

## [0.18.8](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.7...hugr-py-v0.18.8) (2026-10-05)


### Features

* Add `arithmetic.math` to standard extension registry ([#3291](https://github.com/Quantinuum/hugr/issues/3291)) ([9cf29ad](https://github.com/Quantinuum/hugr/commit/9cf29adcde3af6cc70a117bc797137e73e744f55))

## [0.18.7](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.6...hugr-py-v0.18.7) (2026-10-01)


### Features

* Add `math` extension with LLVM intrinsic lowering  ([#3277](https://github.com/Quantinuum/hugr/issues/3277)) ([ae203a3](https://github.com/Quantinuum/hugr/commit/ae203a3a542a1a794be3c6f8b3558e2fa859c28b))
* Improve extension resolution errors ([#3247](https://github.com/Quantinuum/hugr/issues/3247)) ([3c7f736](https://github.com/Quantinuum/hugr/commit/3c7f7360b0ef220222822f4fa7a73ff8c6aa54bf))


### Documentation

* Use Oxford spelling consistently in public documentation ([#3273](https://github.com/Quantinuum/hugr/issues/3273)) ([c4740ba](https://github.com/Quantinuum/hugr/commit/c4740ba0bee8f9cc1034e4dc4d0b5646e96e2d82))

## [0.18.6](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.5...hugr-py-v0.18.6) (2026-09-04)


### Features

* Add froundeven float op ([#3122](https://github.com/Quantinuum/hugr/issues/3122)) ([98a5122](https://github.com/Quantinuum/hugr/commit/98a5122bd1abbcdfc8a207cac2c877aada3a6707))
* **cli:** include edge counts in describe output ([#3209](https://github.com/Quantinuum/hugr/issues/3209)) ([caa558d](https://github.com/Quantinuum/hugr/commit/caa558dc2c39ce7f6a8ddd994a96023d7cd75c58))
* Read ext version as optional in ExtensionDesc metadata ([#3231](https://github.com/Quantinuum/hugr/issues/3231)) ([8fd517b](https://github.com/Quantinuum/hugr/commit/8fd517bd5dfe2b0df3da4cd86ca6b390962d2f57))


### Bug Fixes

* **hugr-py:** Return newest compatible version when registering an extension in a registry ([#3245](https://github.com/Quantinuum/hugr/issues/3245)) ([a793e38](https://github.com/Quantinuum/hugr/commit/a793e38511f03639bbe0aa48fa748169d49e05c6)), closes [#3244](https://github.com/Quantinuum/hugr/issues/3244)

## [0.18.5](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.4...hugr-py-v0.18.5) (2026-08-19)


### Features

* **hugr-py:** Render extension version with render dot ([#3199](https://github.com/Quantinuum/hugr/issues/3199)) ([e040538](https://github.com/Quantinuum/hugr/commit/e04053817e9c8b512a29aec8c3bf045063c3469c))


### Performance Improvements

* Avoid computing unused sub offsets during link establishment ([#3208](https://github.com/Quantinuum/hugr/issues/3208)) ([40986a8](https://github.com/Quantinuum/hugr/commit/40986a80bde7996e4820819d6623a7d6e116e7ca))

## [0.18.4](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.3...hugr-py-v0.18.4) (2026-08-12)


### Bug Fixes

* Docstring formatting ([#3190](https://github.com/Quantinuum/hugr/issues/3190)) ([b1f37c3](https://github.com/Quantinuum/hugr/commit/b1f37c35b9d87c5251d593a321e38f4d849bd158))
* Fix pointer to hugr cli function ([#3194](https://github.com/Quantinuum/hugr/issues/3194)) ([2f47988](https://github.com/Quantinuum/hugr/commit/2f47988e1d562ceefdd0c24d6fc5f7d3bb45d0f9))

## [0.18.3](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.2...hugr-py-v0.18.3) (2026-07-27)


### Bug Fixes

* Narrow the type definition for hugr.qsystem.result.DataValue ([#3181](https://github.com/Quantinuum/hugr/issues/3181)) ([d39e670](https://github.com/Quantinuum/hugr/commit/d39e670afe24d9bc5366d50ae4cdd961ccd68b5f))

## [0.18.2](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.1...hugr-py-v0.18.2) (2026-07-15)


### Features

* Standalone extension resolution on extension load ([#3160](https://github.com/Quantinuum/hugr/issues/3160)) ([6f96ce8](https://github.com/Quantinuum/hugr/commit/6f96ce8c270032f17db95333f21e3d3c589f9835))


### Bug Fixes

* Avoid breaking change to ExtensionValue present in hugr 0.18.1 ([#3163](https://github.com/Quantinuum/hugr/issues/3163)) ([eeff49d](https://github.com/Quantinuum/hugr/commit/eeff49d7201c4ffc4cf7e6f2c5ca15fc5ed36654))

## [0.18.1](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.18.0...hugr-py-v0.18.1) (2026-07-13)


### Features

* Overrride `__repr__` for `RegionKind` ([#3144](https://github.com/Quantinuum/hugr/issues/3144)) ([2c268a3](https://github.com/Quantinuum/hugr/commit/2c268a3fa6869865d0dc332fbf0665d89be86303))


### Performance Improvements

* Register used extensions through a special resolver and lazy coalesce versions ([#3152](https://github.com/Quantinuum/hugr/issues/3152)) ([adc55c0](https://github.com/Quantinuum/hugr/commit/adc55c0c8298cf53f6225980861240dfa9e8364a))

## [0.18.0](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.17.1...hugr-py-v0.18.0) (2026-06-29)


### ⚠ BREAKING CHANGES

* Removed `ToNode.metadata` method. Use `Hugr[node].metadata` instead.

### Features

* Remove metadata object from python Nodes ([#3128](https://github.com/Quantinuum/hugr/issues/3128)) ([e274098](https://github.com/Quantinuum/hugr/commit/e27409887716dc7c8d07e41c499af91c43175057))

## [0.17.1](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.17.0...hugr-py-v0.17.1) (2026-06-10)


### Bug Fixes

* Don't read partial symbol definitions ([#3104](https://github.com/Quantinuum/hugr/issues/3104)) ([697fbce](https://github.com/Quantinuum/hugr/commit/697fbcea26543d4fd2c2ea1fea58b845e1eb7958))
* Quote non-parsable Symbol names in S-expressions ([#3101](https://github.com/Quantinuum/hugr/issues/3101)) ([84f2448](https://github.com/Quantinuum/hugr/commit/84f244812624cce82e1321291e5ae18fa67b0577))

## [0.17.0](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.16.0...hugr-py-v0.17.0) (2026-06-05)

This release includes some improvements to the hugr serialization, adds a fast
hugr linking utility, and allows aliases for metadata keys.

The `hugr-model` serialization now tracks granular information about
extensions' versions requirements. This improves error reporting on missing
extensions and opens the way for extension version migrations in the future.

Support for bare `.json` HUGR files has been removed. JSON-encoded envelopes are
still supported, but will be removed in the future.
To convert old JSON files, add the following header and footer to the file:
```python
PREPEND = b'"HUGRiHJv?@{"modules": ["'
APPEND = b'"],"extensions": []}"'
```

### ⚠ BREAKING CHANGES

* Removed pure HUGR JSON store/decode APIs
* The capnp import/export code has been updated to the format version `v1.1.0`.
* **hugr-llvm:** `hugr-llvm::emit::EmitHugr::emit_hugr::emit_module` takes two new args: `emit_debug: bool` and `ptr_size: u32`. In addition, `hugr_core::views::render::NodeLabel` has a new variant `MetadataValues` which combines a numeric label and JSON metadata values.
* Hugrs are no longer guaranteed to have an Order edge representing the ordering constraint from each nonlocal `Ext` edge; require HugrView impls to provide scheduling_graph; remove deprecated region_portgraph(), as_petgraph(), into_region_portgraph(),

### Features

* Add extension version to custom ops and types ([#3063](https://github.com/Quantinuum/hugr/issues/3063)) ([11e68b7](https://github.com/Quantinuum/hugr/commit/11e68b72fa1067156ab05ca1edf4c80272db175e))
* Add extension versions to hugr-model elements ([#3080](https://github.com/Quantinuum/hugr/issues/3080)) ([39f0fb7](https://github.com/Quantinuum/hugr/commit/39f0fb7779ea105d894a733cc8f7440eaf5546d5))
* do not require an Order edge for each nonlocal `Ext` edge; compute on demand in scheduling_graph() ([#2951](https://github.com/Quantinuum/hugr/issues/2951)) ([deadaa6](https://github.com/Quantinuum/hugr/commit/deadaa6c1f972b6e0208f55f36e8ed3ee48ccfba))
* **hugr-llvm:** Add support for emitting debug locations ([#3026](https://github.com/Quantinuum/hugr/issues/3026)) ([28ab1a7](https://github.com/Quantinuum/hugr/commit/28ab1a7b7c6c469485b9400b8b4576be4b68384e))
* Keep multiple incompatible extension versions in ExtensionRegistry ([#3064](https://github.com/Quantinuum/hugr/issues/3064)) ([dc03683](https://github.com/Quantinuum/hugr/commit/dc03683fd144781e3e697266d4579373ea74434a))
* Metadata key aliases for non-breaking migrations ([#3057](https://github.com/Quantinuum/hugr/issues/3057)) ([e588703](https://github.com/Quantinuum/hugr/commit/e588703a281fa7f82c728f3704df7696e430b31e))
* **py:** Linking packages serialised as bytes ([#3070](https://github.com/Quantinuum/hugr/issues/3070)) ([e2399c2](https://github.com/Quantinuum/hugr/commit/e2399c2cecf1fe3ee34c85b0740de5da91aa55b0))
* Remove deprecated pure json HUGR APIs ([#3089](https://github.com/Quantinuum/hugr/issues/3089)) ([60dca49](https://github.com/Quantinuum/hugr/commit/60dca4904fdcfbb2420838f65d36f4ed4a20f0b0))


### Bug Fixes

* Export dataflow blocks with CFG node link merged into input port when possible ([#3060](https://github.com/Quantinuum/hugr/issues/3060)) ([6742c53](https://github.com/Quantinuum/hugr/commit/6742c53505b2e13c3234dd847a6201759e73b757)), closes [#3059](https://github.com/Quantinuum/hugr/issues/3059)


### Documentation

* Cleanup the python README ([#3099](https://github.com/Quantinuum/hugr/issues/3099)) ([3db96df](https://github.com/Quantinuum/hugr/commit/3db96df0991c70bf5f2e54ac677dab5e3366b687))
* Simplify package description. ([#3040](https://github.com/Quantinuum/hugr/issues/3040)) ([2868f4a](https://github.com/Quantinuum/hugr/commit/2868f4a5aae74f969ea47caef6a13cec7e8c898e))

## [0.16.0](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.4...hugr-py-v0.16.0) (2026-04-01)

This release changes the default text serialization format from JSON to `MODEL_WITH_EXTS` (the JSON format is now deprecated).
It also adds the ability to link packages and modules from Python, renders operation type arguments in HUGR visualizations, introduces a new `debug_info` module for attaching source-level debug metadata to nodes,
and introduces multi-version support in `ExtensionRegistry`, allowing different versions of the same extension to coexist in the same registry.


### ⚠ BREAKING CHANGES

* `ExtensionRegistry` may now contain multiple versions of the same extension. `.extensions` is now an iterator of the latest versions instead of a dictionary. `.register_updated` and `.add_extension` have been replaced with a singular `.register`.
* Default `EnvelopeConfig` changed from `EnvelopeFormat.JSON` to `EnvelopeFormat.MODEL_WITH_EXTS`.
* Substituted `hugr-py/src/hugr/tys.py::_type_str` with `hugr-py/src/hugr/utils.py::name_w_args`.

### Features

* Add debug info metadata specification in `hugr-py` ([#2971](https://github.com/Quantinuum/hugr/issues/2971)) ([a4da8ef](https://github.com/Quantinuum/hugr/commit/a4da8ef4c53eddf443b0b8d854c961451c881e3c))
* Allow multiple versions of an extension in an ExtensionRegistry ([#3005](https://github.com/Quantinuum/hugr/issues/3005)) ([ee90cd1](https://github.com/Quantinuum/hugr/commit/ee90cd1f4f34ceb9772990aadb4f4064dddc4b1e))
* Deprecate JSON serialization format ([#2991](https://github.com/Quantinuum/hugr/issues/2991)) ([0f7a880](https://github.com/Quantinuum/hugr/commit/0f7a88036b5253c60bd36e78d2d06cf2869a981a))
* Operation parameter rendering in HUGR visualizations ([#2995](https://github.com/Quantinuum/hugr/issues/2995)) ([d619148](https://github.com/Quantinuum/hugr/commit/d61914880eaa94fbd198a087df1be95425633ce8))
* Allow linking packages and modules from Python ([#2947](https://github.com/Quantinuum/hugr/issues/2947)) ([329c243](https://github.com/Quantinuum/hugr/commit/329c243fffff0d6c4437664a012361619a0a425e))
* Make `_composable_pass` and `_scope` public modules ([#2962](https://github.com/Quantinuum/hugr/issues/2962)) ([1d6af9a](https://github.com/Quantinuum/hugr/commit/1d6af9a331b1dc68eddd3a0d5283e7b9dab6aa1c)), closes [#2961](https://github.com/Quantinuum/hugr/issues/2961)
* Allow missing ext versions ExtensionDesc metadata ([#2979](https://github.com/Quantinuum/hugr/issues/2979)) ([bc1c445](https://github.com/Quantinuum/hugr/commit/bc1c4453b46ab24c52ca7b3c679730645da691bc))
* Allow dfg conversion in to TrackedDfg ([#2993](https://github.com/Quantinuum/hugr/issues/2993)) ([3bd7a88](https://github.com/Quantinuum/hugr/commit/3bd7a887a20289b6fcd6dd3f9c62886f31663518))


### Bug Fixes

* Allow both _composable_pass and composable_pass imports ([#2965](https://github.com/Quantinuum/hugr/issues/2965)) ([0a5664d](https://github.com/Quantinuum/hugr/commit/0a5664d2374d1bd18aedb4e7619d26d2f564b412))
* Allow both _scope and scope imports ([#3003](https://github.com/Quantinuum/hugr/issues/3003)) ([de0458c](https://github.com/Quantinuum/hugr/commit/de0458c1620c95d045df1e6660d6ae5934be6e66))
* Always do extension resolution when loading packages and hugrs ([#2953](https://github.com/Quantinuum/hugr/issues/2953)) ([0086d2a](https://github.com/Quantinuum/hugr/commit/0086d2a9483283216858ca5f4e86691541d3530a))
* Correct pyo3 module definitions and add missing exception class export ([#2964](https://github.com/Quantinuum/hugr/issues/2964)) ([3df73f4](https://github.com/Quantinuum/hugr/commit/3df73f4d3d0520fcbf278b4072a15e3b24c6edab))
* Include extensions during link serialization ([#2992](https://github.com/Quantinuum/hugr/issues/2992)) ([9236e89](https://github.com/Quantinuum/hugr/commit/9236e8972acb0682ff1493d7ba6df8f9a3f1578d))
* Prefix names with underscores during serialization ([#2989](https://github.com/Quantinuum/hugr/issues/2989)) ([8aab636](https://github.com/Quantinuum/hugr/commit/8aab6366134db1ffdb7e65d70c9bcd04d661851d))
* Use valid identifiers when constructing AST model ([#2973](https://github.com/Quantinuum/hugr/issues/2973)) ([8750a6b](https://github.com/Quantinuum/hugr/commit/8750a6b29ad93aaa72ff3fc48804e4f03b878d6c))


## [0.15.5](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.4...hugr-py-v0.15.5) (2026-03-16)


### Features

* Define pass application scopes ([#2772](https://github.com/Quantinuum/hugr/issues/2772)) ([847b864](https://github.com/Quantinuum/hugr/commit/847b8640582f229d45577439bd58c587242b52fc))
* deprecate Function (Value) in hugr-py too ([#2882](https://github.com/Quantinuum/hugr/issues/2882)) ([043fec2](https://github.com/Quantinuum/hugr/commit/043fec20d9ef973bf1162d250b667c1a57fa3de2))
* Improve rendering of tags and cases ([#2943](https://github.com/Quantinuum/hugr/issues/2943)) ([6ba9e45](https://github.com/Quantinuum/hugr/commit/6ba9e45d70e0a6a6255474c1c78b5ed7b91c705b))


### Bug Fixes

* **py:** Load and resolve lower_funcs in extensions ([#2924](https://github.com/Quantinuum/hugr/issues/2924)) ([6dfc8e3](https://github.com/Quantinuum/hugr/commit/6dfc8e37bc4c4b98881e3b5abe1ffa50df34adc2))


### Documentation

* Add specification to sphinx docs ([#2907](https://github.com/Quantinuum/hugr/issues/2907)) ([46ea04b](https://github.com/Quantinuum/hugr/commit/46ea04b17e176cf54f1ddbc63efd72aa4dcc31f0))

## [0.15.4](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.3...hugr-py-v0.15.4) (2026-02-20)


### Features

* **hugr-py:** Define typed Metadata protocol ([#2765](https://github.com/Quantinuum/hugr/issues/2765)) ([4390230](https://github.com/Quantinuum/hugr/commit/4390230f0b17b5a5ab12656853c40be195514ef5))


### Bug Fixes

* Add truncation options for node and edge labels in rendering ([#2885](https://github.com/Quantinuum/hugr/issues/2885)) ([25c625d](https://github.com/Quantinuum/hugr/commit/25c625d27c7e5a594ad5a0f7e5bc6c044125e590))
* used_extensions should include transitive requirements ([#2891](https://github.com/Quantinuum/hugr/issues/2891)) ([18e78e4](https://github.com/Quantinuum/hugr/commit/18e78e4f0efa08cb15f6474b875fafa2eee8300a))

## [0.15.3](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.2...hugr-py-v0.15.3) (2026-02-09)


### Bug Fixes

* Mark SExp formats as ascii printable ([#2866](https://github.com/Quantinuum/hugr/issues/2866)) ([a4bc253](https://github.com/Quantinuum/hugr/commit/a4bc25313ef5d123f014299315f5ca36f0807983))

## [0.15.2](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.1...hugr-py-v0.15.2) (2026-02-06)


### Features

* Add ptr extension and cleanup `hugr.std` module ([#2842](https://github.com/Quantinuum/hugr/issues/2842)) ([0871b82](https://github.com/Quantinuum/hugr/commit/0871b8268e391886df48a9a5a0b3152ef8156aa0))
* Add s expression format to envelope formats ([#2864](https://github.com/Quantinuum/hugr/issues/2864)) ([2da5e8e](https://github.com/Quantinuum/hugr/commit/2da5e8ef071245b4032a1c2fddf5b16a995f9610))
* Auto-embed extensions in stored hugrs ([#2843](https://github.com/Quantinuum/hugr/issues/2843)) ([36408d4](https://github.com/Quantinuum/hugr/commit/36408d4bb83a752e8a4ac986e2b0857c8b1df5c1))
* Replace pyzstd dependency with rust bindings ([#2846](https://github.com/Quantinuum/hugr/issues/2846)) ([5047c81](https://github.com/Quantinuum/hugr/commit/5047c81926eb365be7306631326d74e237dc3aae))

## [0.15.1](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.15.0...hugr-py-v0.15.1) (2026-01-28)


### Features

* Add used_extensions method to Hugr ([#2817](https://github.com/Quantinuum/hugr/issues/2817)) ([0d88a43](https://github.com/Quantinuum/hugr/commit/0d88a430bdb8a6af9c7ec7c45ef6c853a7c79041)), closes [#2614](https://github.com/Quantinuum/hugr/issues/2614)
* Do resolution while computing used_extensions ([#2834](https://github.com/Quantinuum/hugr/issues/2834)) ([7aff29e](https://github.com/Quantinuum/hugr/commit/7aff29e2f53bbb0c7c63f962ae48fd27fa365644))
* **py:** add qualified_name method to TypeDef ([#2804](https://github.com/Quantinuum/hugr/issues/2804)) ([b2231b3](https://github.com/Quantinuum/hugr/commit/b2231b377d2c655d14b7528640a66c9fe02a2c1e)), closes [#2426](https://github.com/Quantinuum/hugr/issues/2426)


### Bug Fixes

* add register_updated function for registry for self-referential ops ([#2830](https://github.com/Quantinuum/hugr/issues/2830)) ([afba54f](https://github.com/Quantinuum/hugr/commit/afba54fda2d348161baf792f1cba2128369be09d)), closes [#2829](https://github.com/Quantinuum/hugr/issues/2829)

## [0.15.0](https://github.com/Quantinuum/hugr/compare/hugr-py-v0.14.2...hugr-py-v0.15.0) (2026-01-02)


### ⚠ BREAKING CHANGES

* ValueArray is gone, as is LinearizeArrayPass
* **hugr-py:** Removes `extensions` field of `val.Extension`.

### Features

* `collated_digitstring_counts` to extend bitstring collation to digits ([#2788](https://github.com/Quantinuum/hugr/issues/2788)) ([191c473](https://github.com/Quantinuum/hugr/commit/191c4736b5fbe869f8d797703bbf1c29f33b3a92))
* **hugr-py:** Remove `extensions` field of `val.Extension`. ([#2686](https://github.com/Quantinuum/hugr/issues/2686)) ([911c089](https://github.com/Quantinuum/hugr/commit/911c08921b9057f3854dd4431e4a7854b8b6991e))
* Make `ComposablePass.__call__` return a Hugr ([#2697](https://github.com/Quantinuum/hugr/issues/2697)) ([dbf8c8e](https://github.com/Quantinuum/hugr/commit/dbf8c8e6fe72febd709921c10283fddc1f257983))
* Result type for ComposablePasses ([#2703](https://github.com/Quantinuum/hugr/issues/2703)) ([b8df28e](https://github.com/Quantinuum/hugr/commit/b8df28e5b89eb3130d12e5cf429d642be3514e63))


### Bug Fixes

* **hugr-py:** solved graph rendering with `Const` nodes after applying`NormalizeGuppy` ([#2744](https://github.com/Quantinuum/hugr/issues/2744)) ([d996690](https://github.com/Quantinuum/hugr/commit/d99669009b5d98a0e99201f7bfa8621481741d54))
* set hugr field of `PassResult` correctly ([#2715](https://github.com/Quantinuum/hugr/issues/2715)) ([d860722](https://github.com/Quantinuum/hugr/commit/d8607227ce6e2f79a5bf339100f303c750c7fdab))


### Code Refactoring

* Delete ValueArray ([#2760](https://github.com/Quantinuum/hugr/issues/2760)) ([b3cdc4e](https://github.com/Quantinuum/hugr/commit/b3cdc4efd2fdb4632628a641eb84f86e14f5119a))

## [0.14.4](https://github.com/CQCL/hugr/compare/hugr-py-v0.14.3...hugr-py-v0.14.4) (2025-11-26)


### Bug Fixes

* set hugr field of `PassResult` correctly ([#2715](https://github.com/CQCL/hugr/issues/2715)) ([4abedab](https://github.com/CQCL/hugr/commit/4abedabfd9015e79cfbe84ca2d369ec2175f9492))

## [0.14.3](https://github.com/CQCL/hugr/compare/hugr-py-v0.14.2...hugr-py-v0.14.3) (2025-11-24)


### Features

* Make `ComposablePass.__call__` return a Hugr ([#2697](https://github.com/CQCL/hugr/issues/2697)) ([dbf8c8e](https://github.com/CQCL/hugr/commit/dbf8c8e6fe72febd709921c10283fddc1f257983))
* Result type for ComposablePasses ([#2703](https://github.com/CQCL/hugr/issues/2703)) ([b8df28e](https://github.com/CQCL/hugr/commit/b8df28e5b89eb3130d12e5cf429d642be3514e63))

## [0.14.2](https://github.com/CQCL/hugr/compare/hugr-py-v0.14.1...hugr-py-v0.14.2) (2025-11-13)


### Features

* **cli, python:** programmatic interface to cli with python bindings ([#2677](https://github.com/CQCL/hugr/issues/2677)) ([0fd0332](https://github.com/CQCL/hugr/commit/0fd0332f6629abd75e3fd5a5681eeefb72fe6766))
* ComposablePass protocol and ComposedPass for hugr-py (unstable) ([[#2636](https://github.com/CQCL/hugr/issues/2636)](https://github.com/CQCL/hugr/pull/2636)) ([45dc3fc](https://github.com/CQCL/hugr/commit/45dc3fc20623ceb2734e7e3be2b80c47611e9ac6))
* return description output to python on error ([#2681](https://github.com/CQCL/hugr/issues/2681)) ([f483146](https://github.com/CQCL/hugr/commit/f48314684a1d764c5856dbf09dc384eb3fad719d))
* track package descriptions when loading ([#2639](https://github.com/CQCL/hugr/issues/2639)) ([349dd61](https://github.com/CQCL/hugr/commit/349dd61784906d433312fc93603f337c6d763e99))


### Documentation

* Fix typo in docstring. ([#2656](https://github.com/CQCL/hugr/issues/2656)) ([a1ce622](https://github.com/CQCL/hugr/commit/a1ce622642e2cee091d957fa55612ceebc98b2f3))

## [0.14.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.14.0...hugr-py-v0.14.1) (2025-10-27)


### Bug Fixes

* **py:** loosen dependencies to allow for 3.14 support ([#2634](https://github.com/CQCL/hugr/issues/2634)) ([35b5b7b](https://github.com/CQCL/hugr/commit/35b5b7b3607e8d92b49b3ff4520ef8972e517308))

## [0.14.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.13.1...hugr-py-v0.14.0) (2025-10-13)


### ⚠ BREAKING CHANGES

* BorrowArray::{borrow, is_borrowed} return types have been swapped such that the array is first.

### Features

* Add more options to `DotRenderer` config ([#2540](https://github.com/CQCL/hugr/issues/2540)) ([45f7573](https://github.com/CQCL/hugr/commit/45f7573fc0a863ea26d9e854a20482078bfdbd0c))
* Allow importing from model data to python. ([#2581](https://github.com/CQCL/hugr/issues/2581)) ([4fb0a5e](https://github.com/CQCL/hugr/commit/4fb0a5e04b1f8f3ee2f062d505307a2698584534))
* **py, core, llvm:** add `is_borrowed` op for BorrowArray ([#2610](https://github.com/CQCL/hugr/issues/2610)) ([1cd08ef](https://github.com/CQCL/hugr/commit/1cd08ef7538091b81c6442d014d5aa9a469eea38)), closes [#2569](https://github.com/CQCL/hugr/issues/2569)


### Bug Fixes

* Preserve offset for CFG edges when serializing to JSON ([#2606](https://github.com/CQCL/hugr/issues/2606)) ([69a126d](https://github.com/CQCL/hugr/commit/69a126d8b24e1f03003417ae47fccc3dc9ae4f94))


### Code Refactoring

* consistent inout order in borrow array ([#2621](https://github.com/CQCL/hugr/issues/2621)) ([8fc59f3](https://github.com/CQCL/hugr/commit/8fc59f3ea36ed0c707cd21b4c936d79e1d6baae3))

## [0.13.1](https://github.com/CQCL/hugr/compare/hugr-py-v0.13.0...hugr-py-v0.13.1) (2025-08-18)


### Features

* Add option to render a subgraph of a hugr. ([#2497](https://github.com/CQCL/hugr/issues/2497)) ([c3370fe](https://github.com/CQCL/hugr/commit/c3370fef70918ba30b0a7aeacd7737739f15b96e))
* **py:** implement `Sequence` protocol for qsys results/shots ([#2524](https://github.com/CQCL/hugr/issues/2524)) ([b42e9df](https://github.com/CQCL/hugr/commit/b42e9dfebda0a6283bcad5cf51c16612a5487a54))
* Type of constants in `core` `Term`s. ([#2411](https://github.com/CQCL/hugr/issues/2411)) ([2ba5764](https://github.com/CQCL/hugr/commit/2ba576495195be8c73d68dd4feb73fcc74719118))

## [0.13.0](https://github.com/CQCL/hugr/compare/hugr-py-v0.13.0rc1...hugr-py-v0.13.0) (2025-07-29)


### Bug Fixes

* **py:** Python builder for function declarations did not create static output port ([#2485](https://github.com/CQCL/hugr/issues/2485)) ([06bc080](https://github.com/CQCL/hugr/commit/06bc080713398450f8c64d94bddc70f4e31749ee))


### Documentation

* **py:** add docstrings for result fields ([#2480](https://github.com/CQCL/hugr/issues/2480)) ([20dd25f](https://github.com/CQCL/hugr/commit/20dd25f1fcc911f73fcc92a58ec8e781f48db4ec))

## [0.13.0rc1](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.5...hugr-py-v0.13.0rc1) (2025-07-24)


### ⚠ BREAKING CHANGES

* Lowering functions in extension operations are now encoded as binary envelopes. Older hugr versions will error out when trying to load them.
* **py:** `EnvelopeConfig::BINARY` now uses the model binary encoding. `EnvelopeFormat.MODULE` is now `EnvelopeFormat.MODEL`. `EnvelopeFormat.MODULE_WITH_EXTS` is now `EnvelopeFormat.MODEL_WITH_EXTS`
* hugr-model: Symbol has an extra field
* Renamed the `Any` type bound to `Linear`
* The model CFG signature types were changed.
* Added `TypeParam`s and `TypeArg`s corresponding to floats and bytes.
* `TypeArg::Sequence` needs to be replaced with
* FuncDefns must be moved to beneath Module. `Container::define_function` is gone, use `HugrBuilder::module_root_builder`; similarly in hugr-py `DefinitionBuilder` (`define_function` -> `module_root_builder().define_function`). In hugr-llvm, some uses of

### Features

* Add `BorrowArray` extension ([#2395](https://github.com/CQCL/hugr/issues/2395)) ([782687e](https://github.com/CQCL/hugr/commit/782687ed917c3e4295c2c3c59a17d784fc6f932d))
* Add `MakeError` op ([#2377](https://github.com/CQCL/hugr/issues/2377)) ([909a794](https://github.com/CQCL/hugr/commit/909a7948c1465aab5528895bdee0e49958a416b6)), closes [#1863](https://github.com/CQCL/hugr/issues/1863)
* add toposort to HUGR-py ([#2367](https://github.com/CQCL/hugr/issues/2367)) ([34eed34](https://github.com/CQCL/hugr/commit/34eed3422c9aa34bd6b8ad868dcbab733eb5d14c))
* Add Visibility to FuncDefn/FuncDecl. ([#2143](https://github.com/CQCL/hugr/issues/2143)) ([5bbe0cd](https://github.com/CQCL/hugr/commit/5bbe0cdc60625b4047f0cddc9598d6652ed6f736))
* Added float and bytes literal to core and python bindings. ([#2289](https://github.com/CQCL/hugr/issues/2289)) ([e9c5e91](https://github.com/CQCL/hugr/commit/e9c5e914d4fd9ee270dee8e43875d8a413b02926))
* **core, llvm:** add array unpack operations ([#2339](https://github.com/CQCL/hugr/issues/2339)) ([a1a70f1](https://github.com/CQCL/hugr/commit/a1a70f1afb5d8d57082269d167816c7a90497dcf)), closes [#1947](https://github.com/CQCL/hugr/issues/1947)
* Detect and fail on unrecognised envelope flags ([#2453](https://github.com/CQCL/hugr/issues/2453)) ([5e36770](https://github.com/CQCL/hugr/commit/5e36770895b79e878c1bbdf22e67e8cbff6513b6))
* Export entrypoint metadata in Python and fix bug in import ([#2434](https://github.com/CQCL/hugr/issues/2434)) ([d17b245](https://github.com/CQCL/hugr/commit/d17b245c41d943da1c338094c31a75b55efe4061))
* Expose `BorrowArray` in `hugr-py` ([#2425](https://github.com/CQCL/hugr/issues/2425)) ([fdb675f](https://github.com/CQCL/hugr/commit/fdb675f1473a9bf349fce0824c56539e239c11f3)), closes [#2406](https://github.com/CQCL/hugr/issues/2406)
* include generator metatada in model import and cli validate errors ([#2452](https://github.com/CQCL/hugr/issues/2452)) ([f7cedb4](https://github.com/CQCL/hugr/commit/f7cedb4f39b67a77b4c6a55ec00b624b54668eaa))
* Names of private functions become `core.title` metadata. ([#2448](https://github.com/CQCL/hugr/issues/2448)) ([4bc7f65](https://github.com/CQCL/hugr/commit/4bc7f65338d9a8b37d3a5625aeaf093970d97926))
* No nested FuncDefns (or AliasDefns) ([#2256](https://github.com/CQCL/hugr/issues/2256)) ([214b8df](https://github.com/CQCL/hugr/commit/214b8df837537b8ac15c3b60845350c3818a6ac7))
* Non-region entrypoints in `hugr-model`. ([#2467](https://github.com/CQCL/hugr/issues/2467)) ([7b42da6](https://github.com/CQCL/hugr/commit/7b42da6f62de9fe36187512dba428fe3db8d6120))
* Open lists and tuples in `Term` ([#2360](https://github.com/CQCL/hugr/issues/2360)) ([292af80](https://github.com/CQCL/hugr/commit/292af8010dba6b4c2ea5bb69edae31cbf1e0cb6a))
* **py:** enable Model as default BINARY envelope format ([#2317](https://github.com/CQCL/hugr/issues/2317)) ([f089931](https://github.com/CQCL/hugr/commit/f08993124e48093c2328096a93cec8a9ad67a41c))
* **py:** Helper methods to get the neighbours of a node ([#2370](https://github.com/CQCL/hugr/issues/2370)) ([bb6fa50](https://github.com/CQCL/hugr/commit/bb6fa50957ac5121bebc78a06335262a6559e695))
* **py:** Use SumValue serialization for tuples ([#2466](https://github.com/CQCL/hugr/issues/2466)) ([f615037](https://github.com/CQCL/hugr/commit/f615037621aa0eeb37de8f1126fa9020705cb565))
* Rename 'Any' type bound to 'Linear' ([#2421](https://github.com/CQCL/hugr/issues/2421)) ([c2f8b30](https://github.com/CQCL/hugr/commit/c2f8b30afd3a1b75f6babe77a90b13211e45e3a7))
* Split `TypeArg::Sequence` into tuples and lists. ([#2140](https://github.com/CQCL/hugr/issues/2140)) ([cc4997f](https://github.com/CQCL/hugr/commit/cc4997f12dad4dfecc37be564712cae18dfce159))
* Standarize the string formating of sum types and values ([#2432](https://github.com/CQCL/hugr/issues/2432)) ([ec207e7](https://github.com/CQCL/hugr/commit/ec207e7dbe6dbaa9f40421eb0836c9de7e3ea240))
* Use binary envelopes for operation lower_func encoding  ([#2447](https://github.com/CQCL/hugr/issues/2447)) ([2c16a77](https://github.com/CQCL/hugr/commit/2c16a7797a3b5800c5540d1e6a767dd38ad8ca6b))


### Bug Fixes

* Ensure SumTypes have the same json encoding in -rs and -py ([#2465](https://github.com/CQCL/hugr/issues/2465)) ([7f97e6f](https://github.com/CQCL/hugr/commit/7f97e6f84f0bb2b441fe3e2589e91f19de50198e))
* Escape html-like labels in DotRenderer ([#2383](https://github.com/CQCL/hugr/issues/2383)) ([eaa7dfe](https://github.com/CQCL/hugr/commit/eaa7dfe35eb08dbd20d5f5353e92b58850e0f31f))
* Export metadata in Python ([#2342](https://github.com/CQCL/hugr/issues/2342)) ([7be52db](https://github.com/CQCL/hugr/commit/7be52db4f63d7ce8556a5ba0d8d245ebb567e7ed))
* Fix model export of `Opaque` types. ([#2446](https://github.com/CQCL/hugr/issues/2446)) ([3943499](https://github.com/CQCL/hugr/commit/39434996ba18db83a50455fda90c60aea11a8387))
* Fixed bug in python model export name mangling. ([#2323](https://github.com/CQCL/hugr/issues/2323)) ([041342f](https://github.com/CQCL/hugr/commit/041342f58a3dcd9f73dbbaab102221c5d9ff5f61))
* Fixed bugs in model CFG handling and improved CFG signatures ([#2334](https://github.com/CQCL/hugr/issues/2334)) ([ccd2eb2](https://github.com/CQCL/hugr/commit/ccd2eb226358b44aede7dd9e9217448c7e6c0f3a))
* Fixed export of `Call` and `LoadConst` nodes in `hugr-py`. ([#2429](https://github.com/CQCL/hugr/issues/2429)) ([6a0e270](https://github.com/CQCL/hugr/commit/6a0e270e7edbea4cc08e2948d3f8a16b9e763af7))
* Fixed invalid extension name in test. ([#2319](https://github.com/CQCL/hugr/issues/2319)) ([c58ddbf](https://github.com/CQCL/hugr/commit/c58ddbfcc0a557a1644fc8094370e6c62a7ce129))
* Fixed two bugs in import/export of function operations ([#2324](https://github.com/CQCL/hugr/issues/2324)) ([1ad450f](https://github.com/CQCL/hugr/commit/1ad450f807485f7ef6083270aaa4523cb95b2490))
* map IntValue to unsigned repr when serializing ([#2413](https://github.com/CQCL/hugr/issues/2413)) ([26d426e](https://github.com/CQCL/hugr/commit/26d426ee7ffdc38063a337e66458b8d797131bca)), closes [#2409](https://github.com/CQCL/hugr/issues/2409)
* Order hints on input and output nodes. ([#2422](https://github.com/CQCL/hugr/issues/2422)) ([a31ccbc](https://github.com/CQCL/hugr/commit/a31ccbcaaa7561f8d221269262cd9ca9e89ad67b))
* **py:** correct ConstString JSON encoding ([#2325](https://github.com/CQCL/hugr/issues/2325)) ([9649a48](https://github.com/CQCL/hugr/commit/9649a48d376aff27e475c70072aecd55ae7a4ccb))
* StaticArrayVal payload encoding, improve roundtrip checker ([#2444](https://github.com/CQCL/hugr/issues/2444)) ([1a301eb](https://github.com/CQCL/hugr/commit/1a301eb818401c314d4d7bac40698ec2e73babe7))
* stringify metadata before escaping in renderer ([#2405](https://github.com/CQCL/hugr/issues/2405)) ([8d67420](https://github.com/CQCL/hugr/commit/8d67420e8fd2e979256ff64bcf0b2813ed19ac00))

## [0.12.5](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.4...hugr-py-v0.12.5) (2025-07-08)


### Bug Fixes

* map IntValue to unsigned repr when serializing ([#2413](https://github.com/CQCL/hugr/issues/2413)) ([4ad1d4e](https://github.com/CQCL/hugr/commit/4ad1d4e010eca07207306320b3cf74396f1f8181)), closes [#2409](https://github.com/CQCL/hugr/issues/2409)

## [0.12.4](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.3...hugr-py-v0.12.4) (2025-07-03)


### Bug Fixes

* stringify metadata before escaping in renderer ([#2405](https://github.com/CQCL/hugr/issues/2405)) ([1f01e97](https://github.com/CQCL/hugr/commit/1f01e97696afe02b46eedb2c6e3e2f2369a4ac7b))

## [0.12.3](https://github.com/CQCL/hugr/compare/hugr-py-v0.12.2...hugr-py-v0.12.3) (2025-07-03)


### Features

* add toposort to HUGR-py ([#2367](https://github.com/CQCL/hugr/issues/2367)) ([ba8988e](https://github.com/CQCL/hugr/commit/ba8988e87c2a3d64953838e9a1cff4989740cf05))
* **core, llvm:** add array unpack operations ([#2339](https://github.com/CQCL/hugr/issues/2339)) ([74b25aa](https://github.com/CQCL/hugr/commit/74b25aa3a704c082f84a0c34fad2654e3392ff50)), closes [#1947](https://github.com/CQCL/hugr/issues/1947)
* **py:** Helper methods to get the neighbours of a node ([#2370](https://github.com/CQCL/hugr/issues/2370)) ([1ed6440](https://github.com/CQCL/hugr/commit/1ed64409aaf7e8f26fb5928051245e560881a621))


### Bug Fixes

* Escape html-like labels in DotRenderer ([#2383](https://github.com/CQCL/hugr/issues/2383)) ([c7a43a6](https://github.com/CQCL/hugr/commit/c7a43a69878e1271251b570070f192ebf57aaadd))
* Fixed invalid extension name in test. ([#2319](https://github.com/CQCL/hugr/issues/2319)) ([fbe1d9c](https://github.com/CQCL/hugr/commit/fbe1d9c061768360144f5463dcf357fb59ac736f))
* **py:** correct ConstString JSON encoding ([#2325](https://github.com/CQCL/hugr/issues/2325)) ([325168b](https://github.com/CQCL/hugr/commit/325168b50b5e40e884127ad89d7acb5ab3a412f8))

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

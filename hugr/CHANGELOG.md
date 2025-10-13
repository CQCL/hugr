# Changelog

## [0.24.0](https://github.com/CQCL/hugr/compare/hugr-v0.23.0...hugr-v0.24.0) - 2025-10-13

### Bug Fixes

- Preserve offset for CFG edges when serializing to JSON ([#2606](https://github.com/CQCL/hugr/pull/2606))

### New Features

- Add handler for copying / discarding borrow arrays to default lineariser ([#2602](https://github.com/CQCL/hugr/pull/2602))
- LLVM lowering for borrow arrays using bitmasks ([#2574](https://github.com/CQCL/hugr/pull/2574))
- *(py, core, llvm)* add `is_borrowed` op for BorrowArray ([#2610](https://github.com/CQCL/hugr/pull/2610))

### Refactor

- [**breaking**] consistent inout order in borrow array ([#2621](https://github.com/CQCL/hugr/pull/2621))

## [0.23.0](https://github.com/CQCL/hugr/compare/hugr-v0.22.4...hugr-v0.23.0) - 2025-09-30

### Bug Fixes

- DeadCodeElim keeps consumers of linear outputs ([#2560](https://github.com/CQCL/hugr/pull/2560))
- [**breaking**] Appease `cargo-audit` by replacing unmaintained dependencies ([#2572](https://github.com/CQCL/hugr/pull/2572))
- *(core)* check extension versions on model import ([#2580](https://github.com/CQCL/hugr/pull/2580))
- [**breaking**] test extension version compatibility on ModelWithExts ([#2587](https://github.com/CQCL/hugr/pull/2587))
- *(core)* check used extension versions against resolved extensions ([#2588](https://github.com/CQCL/hugr/pull/2588))
- [**breaking**] model import loads Package extensions ([#2590](https://github.com/CQCL/hugr/pull/2590))

### Miscellaneous Tasks

- [**breaking**] Cleanup deprecated definitions ([#2594](https://github.com/CQCL/hugr/pull/2594))

### New Features

- [**breaking**] DeadCodeElimPass reports error on non-existent entry_points ([#2566](https://github.com/CQCL/hugr/pull/2566))
- add trait+funcs for linking Hugrs explicitly by Node ([#2521](https://github.com/CQCL/hugr/pull/2521))
- Documentation and error hints ([#2523](https://github.com/CQCL/hugr/pull/2523))
- Allow creating DFG builders from existing hugrs ([#2562](https://github.com/CQCL/hugr/pull/2562))
- add_input/output for arbitrary DFGBuilders ([#2564](https://github.com/CQCL/hugr/pull/2564))
- [**breaking**] Return error instead of panicking in DFGWrapper::add_{in,out}put ([#2571](https://github.com/CQCL/hugr/pull/2571))
- *(core)* inner acccesors for WithGenerator error ([#2583](https://github.com/CQCL/hugr/pull/2583))
- Normalize CFGs ([#2591](https://github.com/CQCL/hugr/pull/2591))

### Refactor

- [**breaking**] Replace lazy_static with std::sync::LazyLock ([#2567](https://github.com/CQCL/hugr/pull/2567))

## [0.22.4](https://github.com/CQCL/hugr/compare/hugr-v0.22.3...hugr-v0.22.4) - 2025-09-24

### Bug Fixes

- DeadCodeElim keeps consumers of linear outputs ([#2560](https://github.com/CQCL/hugr/pull/2560))
- *(core)* check extension versions on model import ([#2580](https://github.com/CQCL/hugr/pull/2580))

### New Features

- Documentation and error hints ([#2523](https://github.com/CQCL/hugr/pull/2523))
- Allow creating DFG builders from existing hugrs ([#2562](https://github.com/CQCL/hugr/pull/2562))
- add_input/output for arbitrary DFGBuilders ([#2564](https://github.com/CQCL/hugr/pull/2564))
- *(core)* inner acccesors for WithGenerator error ([#2583](https://github.com/CQCL/hugr/pull/2583))

## [0.22.3](https://github.com/CQCL/hugr/compare/hugr-v0.22.2...hugr-v0.22.3) - 2025-09-11

### Bug Fixes

- SiblingSubgraph::try_from_nodes not including disconnected components ([#2549](https://github.com/CQCL/hugr/pull/2549))

### Documentation

- Clarify docs for SiblingSubgraph::{inputs, outputs} ([#2508](https://github.com/CQCL/hugr/pull/2508))

### New Features

- SiblingSubgraph supports function calls ([#2528](https://github.com/CQCL/hugr/pull/2528))
- Add unchecked constructor for SiblingSubgraph ([#2526](https://github.com/CQCL/hugr/pull/2526))
- Add extend_inputs function for DFGs ([#2536](https://github.com/CQCL/hugr/pull/2536))
- Loosen bound on Patch trait ([#2545](https://github.com/CQCL/hugr/pull/2545))
- Add HugrMut::insert(_view)_forest ([#2518](https://github.com/CQCL/hugr/pull/2518))

## [0.22.2](https://github.com/CQCL/hugr/compare/hugr-v0.22.1...hugr-v0.22.2) - 2025-08-06

### New Features

- Type of constants in `core` `Term`s. ([#2411](https://github.com/CQCL/hugr/pull/2411))
- Support LineConvexChecker ([#2487](https://github.com/CQCL/hugr/pull/2487))

## [0.22.1](https://github.com/CQCL/hugr/compare/hugr-v0.22.0...hugr-v0.22.1) - 2025-07-28

### New Features

- Include copy_discard_array in DelegatingLinearizer::default ([#2479](https://github.com/CQCL/hugr/pull/2479))
- Inline calls to functions not on cycles in the call graph ([#2450](https://github.com/CQCL/hugr/pull/2450))

## [0.22.0](https://github.com/CQCL/hugr/compare/hugr-v0.21.0...hugr-v0.22.0) - 2025-07-24

This release fixes multiple inconsistencies between the serialization formats
and improves the error messages when loading unsupported envelopes.

We now also support nodes with up to `2^32` connections to the same port (up from `2^16`).

### Bug Fixes

- Ensure SumTypes have the same json encoding in -rs and -py ([#2465](https://github.com/CQCL/hugr/pull/2465))

### New Features

- ReplaceTypes allows linearizing inside Op replacements ([#2435](https://github.com/CQCL/hugr/pull/2435))
- Add pass for DFG inlining ([#2460](https://github.com/CQCL/hugr/pull/2460))
- Export entrypoint metadata in Python and fix bug in import ([#2434](https://github.com/CQCL/hugr/pull/2434))
- Names of private functions become `core.title` metadata. ([#2448](https://github.com/CQCL/hugr/pull/2448))
- [**breaking**] Use binary envelopes for operation lower_func encoding  ([#2447](https://github.com/CQCL/hugr/pull/2447))
- [**breaking**] Update portgraph dependency to 0.15 ([#2455](https://github.com/CQCL/hugr/pull/2455))
- Detect and fail on unrecognised envelope flags ([#2453](https://github.com/CQCL/hugr/pull/2453))
- include generator metatada in model import and cli validate errors ([#2452](https://github.com/CQCL/hugr/pull/2452))
- [**breaking**] Add `insert_region` to HugrMut ([#2463](https://github.com/CQCL/hugr/pull/2463))
- Non-region entrypoints in `hugr-model`. ([#2467](https://github.com/CQCL/hugr/pull/2467))

## [0.21.0](https://github.com/CQCL/hugr/compare/hugr-v0.20.2...hugr-v0.21.0) - 2025-07-09


This release includes a long list of changes:

- The HUGR model serialization format is now stable, and should be preferred over the old JSON format.
- Type parameters and type arguments are now unified into a single `Term` type.
- Function definitions can no longer be nested inside dataflow regions. Now they must be defined at the top level module.
- Function definitions and declarations now have a `Visibility` field, which define whether they are visible in the public API of the module.
- And many more fixes and improvements.

### Bug Fixes

- DeadFuncElimPass+CallGraph w/ non-module-child entrypoint ([#2390](https://github.com/CQCL/hugr/pull/2390))
- Fixed two bugs in import/export of function operations ([#2324](https://github.com/CQCL/hugr/pull/2324))
- Model import should perform extension resolution ([#2326](https://github.com/CQCL/hugr/pull/2326))
- [**breaking**] Fixed bugs in model CFG handling and improved CFG signatures ([#2334](https://github.com/CQCL/hugr/pull/2334))
- Use List instead of Tuple in conversions for TypeArg/TypeRow ([#2378](https://github.com/CQCL/hugr/pull/2378))
- Do extension resolution on loaded extensions from the model format ([#2389](https://github.com/CQCL/hugr/pull/2389))
- Make JSON Schema checks actually work again ([#2412](https://github.com/CQCL/hugr/pull/2412))
- Order hints on input and output nodes. ([#2422](https://github.com/CQCL/hugr/pull/2422))

### Documentation

- Hide hugr-persistent docs ([#2357](https://github.com/CQCL/hugr/pull/2357))

### New Features

- [**breaking**] Split `TypeArg::Sequence` into tuples and lists. ([#2140](https://github.com/CQCL/hugr/pull/2140))
- [**breaking**] Added float and bytes literal to core and python bindings. ([#2289](https://github.com/CQCL/hugr/pull/2289))
- [**breaking**] More helpful error messages in model import ([#2272](https://github.com/CQCL/hugr/pull/2272))
- [**breaking**] Better error reporting in `hugr-cli`. ([#2318](https://github.com/CQCL/hugr/pull/2318))
- [**breaking**] Merge `TypeParam` and `TypeArg` into one `Term` type in Rust ([#2309](https://github.com/CQCL/hugr/pull/2309))
- *(persistent)* Add serialisation for CommitStateSpace ([#2344](https://github.com/CQCL/hugr/pull/2344))
- add TryFrom impls for TypeArg/TypeRow ([#2366](https://github.com/CQCL/hugr/pull/2366))
- Add `MakeError` op ([#2377](https://github.com/CQCL/hugr/pull/2377))
- Open lists and tuples in `Term` ([#2360](https://github.com/CQCL/hugr/pull/2360))
- Call `FunctionBuilder::add_{in,out}put` for any AsMut<Hugr> ([#2376](https://github.com/CQCL/hugr/pull/2376))
- Add Root checked methods to DataflowParentID ([#2382](https://github.com/CQCL/hugr/pull/2382))
- Add PersistentWire type ([#2361](https://github.com/CQCL/hugr/pull/2361))
- Add `BorrowArray` extension ([#2395](https://github.com/CQCL/hugr/pull/2395))
- [**breaking**] Add Visibility to FuncDefn/FuncDecl. ([#2143](https://github.com/CQCL/hugr/pull/2143))
- *(per)* [**breaking**] Support empty wires in commits ([#2349](https://github.com/CQCL/hugr/pull/2349))
- [**breaking**] hugr-model use explicit Option<Visibility>, with ::Unspecified in capnp ([#2424](https://github.com/CQCL/hugr/pull/2424))
- [**breaking**] No nested FuncDefns (or AliasDefns) ([#2256](https://github.com/CQCL/hugr/pull/2256))
- [**breaking**] Rename 'Any' type bound to 'Linear' ([#2421](https://github.com/CQCL/hugr/pull/2421))

### Refactor

- [**breaking**] remove deprecated runtime extension errors ([#2369](https://github.com/CQCL/hugr/pull/2369))
- [**breaking**] Reduce error type sizes ([#2420](https://github.com/CQCL/hugr/pull/2420))
- [**breaking**] move PersistentHugr into separate crate ([#2277](https://github.com/CQCL/hugr/pull/2277))

### Testing

- Check hugr json serializations against the schema (again) ([#2216](https://github.com/CQCL/hugr/pull/2216))

## [0.20.2](https://github.com/CQCL/hugr/compare/hugr-v0.20.1...hugr-v0.20.2) - 2025-06-25

### Bug Fixes

- update CallGraph and remove_dead_funcs for module-only FuncDefns ([#2336](https://github.com/CQCL/hugr/pull/2336))

### Documentation

- fix doc links in persistent

### New Features

- Add serial data types for SimpleReplacement and PersistentHugr ([#2300](https://github.com/CQCL/hugr/pull/2300))
- Add MermaidFormatter to replace RenderConfig ([#2275](https://github.com/CQCL/hugr/pull/2275))
- *(core, llvm)* add array unpack operations ([#2339](https://github.com/CQCL/hugr/pull/2339))
- Deprecate invalidation_set, add invalidated_nodes and SimpleReplacement::invalidation_set ([#2358](https://github.com/CQCL/hugr/pull/2358))
- Rewrite for peeling a TailLoop ([#2290](https://github.com/CQCL/hugr/pull/2290))
- Create Module/FunctionBuilders from existing Hugrs ([#2359](https://github.com/CQCL/hugr/pull/2359))
- better errors using metadata from generator ([#2368](https://github.com/CQCL/hugr/pull/2368))
- use `core.` prefixes for generator metadata keys ([#2371](https://github.com/CQCL/hugr/pull/2371))
- *(core)* builder pattern for EnvelopeConfig ([#2330](https://github.com/CQCL/hugr/pull/2330))

### Refactor

- *(types.rs)* rm incorrect comment and unnecessary allow-unused ([#2340](https://github.com/CQCL/hugr/pull/2340))

## [0.20.1](https://github.com/CQCL/hugr/compare/hugr-v0.20.0...hugr-v0.20.1) - 2025-06-03

### Bug Fixes

- Dataflow analysis produces unsound results on Hugrs with entrypoint ([#2255](https://github.com/CQCL/hugr/pull/2255))
- check well-definedness of DFG wires in validate ([#2221](https://github.com/CQCL/hugr/pull/2221))
- Check for order edges in SiblingSubgraph::from_node ([#2223](https://github.com/CQCL/hugr/pull/2223))
- Make SumType::Unit(N) equal to SumType::General([(); N]) ([#2250](https://github.com/CQCL/hugr/pull/2250))
- canonicalize_nodes sometimes mangles the entrypoint ([#2263](https://github.com/CQCL/hugr/pull/2263))

### New Features

- LocalizeEdges pass ([#2237](https://github.com/CQCL/hugr/pull/2237))
- Add PersistentHugr ([#2080](https://github.com/CQCL/hugr/pull/2080))
- Add `Type::used_extensions` ([#2224](https://github.com/CQCL/hugr/pull/2224))
- Add boundary edge traversal in SimpleReplacement ([#2231](https://github.com/CQCL/hugr/pull/2231))
- Add signature map function for DFGs ([#2239](https://github.com/CQCL/hugr/pull/2239))
- PersistentHugr implements HugrView ([#2202](https://github.com/CQCL/hugr/pull/2202))
- PersistentHugr Walker API ([#2168](https://github.com/CQCL/hugr/pull/2168))
- Hugr::store_with_exts and auto-include in serde_as ([#2280](https://github.com/CQCL/hugr/pull/2280))

### Refactor

- tidies/readability improvements to PersistentHugr ([#2251](https://github.com/CQCL/hugr/pull/2251))
- Deprecate ValidationError::ExtensionError ([#2281](https://github.com/CQCL/hugr/pull/2281))

### Testing

- Ignore miri errors in tests involving `assert_snapshot` ([#2261](https://github.com/CQCL/hugr/pull/2261))

## [0.20.0](https://github.com/CQCL/hugr/compare/hugr-v0.15.4...hugr-v0.20.0) - 2025-05-14

This release contains a big list of changes reworking multiple core definitions of HUGR.
Some of these changes include:

- 🧹 Cleanup of the internal hugr traits, making it possible to create virtual
  Hugrs that are not backed by a portgraph or use different node index types.
- 🚪 The functional root of a Hugr is now called the *entrypoint*. This node may
  exist somewhere inside the region hierarchy, so we can now model edges that
  connect beyond the descendants of the entrypoint.
- 🌲 The hugr region hierarchy is now always rooted by a `Module` operation.
- ✂ Made `SimpleReplacement` even simpler and easy to use.
- 🔧 The `Rewrite` trait has been split into `VerifyPatch` and `Patch` parts.
- ✉️ Hugr and package serialization is now done entirely using Envelopes. We
  provide some backward-compatibility layers that will be removed in the future.
- 🦀 The Minimum Supported Rust Version (MSRV) has been updated to `1.85`.

### Bug Fixes

- [**breaking**] Don't expose `HugrMutInternals` ([#2071](https://github.com/CQCL/hugr/pull/2071))
- `as_unary_option` indexing bug ([#2163](https://github.com/CQCL/hugr/pull/2163))
- Skip phantom data for array value serialisation ([#2166](https://github.com/CQCL/hugr/pull/2166))
- Remove deleted nodes from node_map in `SimpleReplacement` ([#2176](https://github.com/CQCL/hugr/pull/2176))
- [**breaking**] Use unique region_portgraph in convexity check ([#2192](https://github.com/CQCL/hugr/pull/2192))
- Panic when an extension name is too long. ([#2198](https://github.com/CQCL/hugr/pull/2198))
- Respect type bounds on local variables when importing. ([#2206](https://github.com/CQCL/hugr/pull/2206))
- Import and export JSON metadata on module roots. ([#2207](https://github.com/CQCL/hugr/pull/2207))

### New Features

- [**breaking**] Handle CallIndirect in Dataflow Analysis ([#2059](https://github.com/CQCL/hugr/pull/2059))
- [**breaking**] ComposablePass trait allowing sequencing and validation ([#1895](https://github.com/CQCL/hugr/pull/1895))
- [**breaking**] ReplaceTypes: allow lowering ops into a Call to a function already in the Hugr ([#2094](https://github.com/CQCL/hugr/pull/2094))
- export mangle name function ([#2152](https://github.com/CQCL/hugr/pull/2152))
- [**breaking**] Explicit hugr type param to ComposablePass ([#2179](https://github.com/CQCL/hugr/pull/2179))
- [**breaking**] Allow generic Nodes in HugrMut insert operations ([#2075](https://github.com/CQCL/hugr/pull/2075))
- [**breaking**] Mark all Error enums as non_exhaustive ([#2056](https://github.com/CQCL/hugr/pull/2056))
- Make NodeHandle generic ([#2092](https://github.com/CQCL/hugr/pull/2092))
- [**breaking**] remove ExtensionValue ([#2093](https://github.com/CQCL/hugr/pull/2093))
- [**breaking**] Hugrmut on generic nodes ([#2111](https://github.com/CQCL/hugr/pull/2111))
- [**breaking**] Cleanup core trait definitions ([#2126](https://github.com/CQCL/hugr/pull/2126))
- [**breaking**] Accept outgoing ports in SimpleReplacement nu_out ([#2151](https://github.com/CQCL/hugr/pull/2151))
- [**breaking**] Improved array lowering ([#2109](https://github.com/CQCL/hugr/pull/2109))
- [**breaking**] Make `NamedOp` private. Add `MakeExtensionOp::name` and `MakeOpDef::opdef_name` ([#2138](https://github.com/CQCL/hugr/pull/2138))
- InsertCut patch for inserting HUGR across edges. ([#2153](https://github.com/CQCL/hugr/pull/2153))
- [**breaking**] Add Hugr entrypoints ([#2147](https://github.com/CQCL/hugr/pull/2147))
- [**breaking**] Return a node mapping in HugrInternals::region_portgraph ([#2164](https://github.com/CQCL/hugr/pull/2164))
- [**breaking**] Validate any HugrView, make errors generic ([#2155](https://github.com/CQCL/hugr/pull/2155))
- Reimplement `insert_hugr` using only `HugrView` ([#2174](https://github.com/CQCL/hugr/pull/2174))
- Only allow region containers as entrypoints ([#2173](https://github.com/CQCL/hugr/pull/2173))
- Export and import entrypoints via metadata in `hugr-model`. ([#2172](https://github.com/CQCL/hugr/pull/2172))
- [**breaking**] Removed model_unstable feature flag ([#2120](https://github.com/CQCL/hugr/pull/2120))
- [**breaking**] Remove `RootTagged` from the hugr view trait hierarchy ([#2122](https://github.com/CQCL/hugr/pull/2122))
- [**breaking**] Split Rewrite trait into VerifyPatch and ApplyPatch ([#2070](https://github.com/CQCL/hugr/pull/2070))
- [**breaking**] Bump MSRV to 1.85 ([#2136](https://github.com/CQCL/hugr/pull/2136))
- [**breaking**] Removed runtime extension sets. ([#2145](https://github.com/CQCL/hugr/pull/2145))
- [**breaking**] Only expose envelope serialization of hugrs and packages ([#2167](https://github.com/CQCL/hugr/pull/2167))
- Packages do not include the hugr extensions by default ([#2187](https://github.com/CQCL/hugr/pull/2187))
- Define text-model envelope formats ([#2188](https://github.com/CQCL/hugr/pull/2188))
- [**breaking**] Remove description on opaque ops. ([#2197](https://github.com/CQCL/hugr/pull/2197))
- Import CFG regions without adding an entry block. ([#2200](https://github.com/CQCL/hugr/pull/2200))
- [**breaking**] Remove boundary map in SimpleReplacement ([#2208](https://github.com/CQCL/hugr/pull/2208))
- Export macro for hugr serde wrappers with custom extensions ([#2209](https://github.com/CQCL/hugr/pull/2209))
- Allow any dataflow parent as SiblingSubgraph replacement ([#2210](https://github.com/CQCL/hugr/pull/2210))
- [**breaking**] Hide FuncDefn/cl fields, add accessors and ::new(...) ([#2213](https://github.com/CQCL/hugr/pull/2213))
- Add SiblingSubgraph::set_outgoing_ports ([#2217](https://github.com/CQCL/hugr/pull/2217))
- Symbol applications can leave out prefixes of wildcards. ([#2201](https://github.com/CQCL/hugr/pull/2201))

### Refactor

- do not use .portgraph in mermaid/graphviz ([#2177](https://github.com/CQCL/hugr/pull/2177))
- [**breaking**] Removed global portgraph-related methods from `HugrInternals` ([#2180](https://github.com/CQCL/hugr/pull/2180))

### Testing

- Check envelope roundtrips rather than json in `HugrView::verify` ([#2186](https://github.com/CQCL/hugr/pull/2186))

## [0.15.4](https://github.com/CQCL/hugr/compare/hugr-v0.15.3...hugr-v0.15.4) - 2025-05-07

### New Features

- Export the portgraph hierarchy in HugrInternals ([#2057](https://github.com/CQCL/hugr/pull/2057))
- Implement Debug for generic Wire<N>s ([#2068](https://github.com/CQCL/hugr/pull/2068))
- Add ExtensionOp helpers ([#2072](https://github.com/CQCL/hugr/pull/2072))
- ReplaceTypes: handlers for array constants + linearization ([#2023](https://github.com/CQCL/hugr/pull/2023))
- move `ArrayOpBuilder` to hugr-core ([#2115](https://github.com/CQCL/hugr/pull/2115))

### Testing

- Disable IO-dependent tests when running miri ([#2123](https://github.com/CQCL/hugr/pull/2123))

## [0.15.3](https://github.com/CQCL/hugr/compare/hugr-v0.15.2...hugr-v0.15.3) - 2025-04-02

### Documentation

- Provide docs for array ops, fix bad doc for HugrView::poly_func_type ([#2021](https://github.com/CQCL/hugr/pull/2021))

### New Features

- MakeTuple->UnpackTuple elision pass ([#2012](https://github.com/CQCL/hugr/pull/2012))
- Extend LowerTypes pass to linearize by inserting copy/discard ([#2018](https://github.com/CQCL/hugr/pull/2018))
- Expand SimpleReplacement API ([#1920](https://github.com/CQCL/hugr/pull/1920))
- Python bindings for `hugr-model`. ([#1959](https://github.com/CQCL/hugr/pull/1959))
- ReplaceTypes pass allows replacing extension types and ops ([#1989](https://github.com/CQCL/hugr/pull/1989))
- Remove extension sets from `hugr-model`. ([#2031](https://github.com/CQCL/hugr/pull/2031))
- Represent order edges in `hugr-model` as metadata. ([#2027](https://github.com/CQCL/hugr/pull/2027))
- add `build_expect_sum` to allow specific error messages ([#2032](https://github.com/CQCL/hugr/pull/2032))
- Packages in `hugr-model` and envelope support. ([#2026](https://github.com/CQCL/hugr/pull/2026))

## [0.15.2](https://github.com/CQCL/hugr/compare/hugr-v0.15.1...hugr-v0.15.2) - 2025-03-21

### Bug Fixes

- Don't enable envelope compression by default (yet) ([#2014](https://github.com/CQCL/hugr/pull/2014))
- Inconsistent behaviour in `SiblingSubgraph::from_nodes` ([#2011](https://github.com/CQCL/hugr/pull/2011))

## [0.15.1](https://github.com/CQCL/hugr/compare/hugr-v0.15.0...hugr-v0.15.1) - 2025-03-21

### Bug Fixes

- correct `CallIndirect` tag from `FnCall` to `DataflowChild` ([#2006](https://github.com/CQCL/hugr/pull/2006))
- StaticArrayValue serialisation ([#2009](https://github.com/CQCL/hugr/pull/2009))

### New Features

- traits for transforming Types/TypeArgs/etc. ([#1991](https://github.com/CQCL/hugr/pull/1991))
- add exit operation to prelude ([#2008](https://github.com/CQCL/hugr/pull/2008))
- Add llvm codegen for collections.static_array ([#2003](https://github.com/CQCL/hugr/pull/2003))
- *(hugr-py)* Support envelope compression ([#1994](https://github.com/CQCL/hugr/pull/1994))

## [0.15.0](https://github.com/CQCL/hugr/compare/hugr-v0.14.4...hugr-v0.15.0) - 2025-03-14

This release adds a new `Node` parameter to the `HugrView` trait, so it is now possible to have more complex structures behave like hugrs.

We also added a new text/binary serialization format for hugrs that will let us keep interoperability between different versions.

Finally, we added multiple new operations and improvements to the rewrites and passes.

### New Features

- Add separate DCE pass ([#1902](https://github.com/CQCL/hugr/pull/1902))
- [**breaking**] Don't assume "main"-function in dataflow + constant folding ([#1896](https://github.com/CQCL/hugr/pull/1896))
- [**breaking**] Add associated type Node to HugrView ([#1932](https://github.com/CQCL/hugr/pull/1932))
- Rewrite for inlining a single Call ([#1934](https://github.com/CQCL/hugr/pull/1934))
- [**breaking**] replace `Lift` with `Barrier` ([#1952](https://github.com/CQCL/hugr/pull/1952))
- `hugr-model` AST ([#1953](https://github.com/CQCL/hugr/pull/1953))
- Add float <--> int bytecasting ops to conversions extension ([#1956](https://github.com/CQCL/hugr/pull/1956))
- Add collections.static_array extension. ([#1964](https://github.com/CQCL/hugr/pull/1964))
- [**breaking**] Generic HUGR serialization with envelopes ([#1958](https://github.com/CQCL/hugr/pull/1958))

### Refactor

- *(hugr-model)* Reexport `bumpalo` from `hugr-model` ([#1870](https://github.com/CQCL/hugr/pull/1870))
- [**breaking**] remove unused dependencies ([#1935](https://github.com/CQCL/hugr/pull/1935))

## [0.14.4](https://github.com/CQCL/hugr/compare/hugr-v0.14.3...hugr-v0.14.4) - 2025-02-24

### Bug Fixes

- delegate default impls in HugrView (#1921)

### New Features

- add xor to logic extension (#1911)
- Add `Type::as_sum` and `SumType::variants`. (#1914)
- Add `HugrMutInternals::insert_ports` (#1915)

## [0.14.3](https://github.com/CQCL/hugr/compare/hugr-v0.14.2...hugr-v0.14.3) - 2025-02-05

### Bug Fixes

- Export `RemoveDeadFuncsError` (#1883)
- const-folding Module keeps at least "main" (#1901)
- determine correct bounds of custom types (#1888)
- Exporting converging control flow edges (#1890)

### Documentation

- Fix deprecation warning messages (#1891)
- Explain why `ConstF64` is not PartialEq (#1829)

### New Features

- Special cased array, float and int constants in hugr-model export (#1857)
- Simplify hugr-model (#1893)

## [0.14.2](https://github.com/CQCL/hugr/compare/hugr-v0.14.1...hugr-v0.14.2) - 2025-01-20

### Bug Fixes

- Three bugfixes in model import and export. (#1844)

### Documentation

- Fix typo in `DataflowParent` doc (#1865)

### New Features

- Add CallGraph struct, and dead-function-removal pass (#1796)
- `Value::some`, `::none`, and `SumType::new_option` helpers (#1828)
- Constant values in `hugr-model` (#1838)
- *(hugr-llvm)* Emit ipow (#1839)
- Bytes literal in hugr-model. (#1845)
- Improved representation for metadata in `hugr-model` (#1849)

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

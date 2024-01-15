# Changelog

## 0.1.0 (2024-01-15)

### Bug Fixes

- Subgraph boundaries with copies ([#440](https://github.com/CQCL/hugr/pull/440))
- [**breaking**] Use internal tag for SumType enum serialisation ([#462](https://github.com/CQCL/hugr/pull/462))
- Check kind before unwrap in insert_identity ([#475](https://github.com/CQCL/hugr/pull/475))
- Allow for variables to get solved in inference ([#478](https://github.com/CQCL/hugr/pull/478))
- In IdentityInsertion add noop to correct parent ([#477](https://github.com/CQCL/hugr/pull/477))
- Failing release tests ([#485](https://github.com/CQCL/hugr/pull/485))
- [**breaking**] Serialise `Value`, `PrimValue`, and `TypeArg` with internal tags ([#496](https://github.com/CQCL/hugr/pull/496))
- Serialise custom constants with internal tag ([#502](https://github.com/CQCL/hugr/pull/502))
- [**breaking**] Reduce max int width in arithmetic extension to 64 ([#504](https://github.com/CQCL/hugr/pull/504))
- HugrView::get_function_type ([#507](https://github.com/CQCL/hugr/pull/507))
- TODO in resolve_extension_ops: copy across input_extensions ([#510](https://github.com/CQCL/hugr/pull/510))
- Use given input extensions in `define_function` ([#524](https://github.com/CQCL/hugr/pull/524))
- Lessen requirements for hugrs in outline_cfg ([#528](https://github.com/CQCL/hugr/pull/528))
- Make unification logic less strict ([#538](https://github.com/CQCL/hugr/pull/538))
- Simple replace incorrectly copying metadata ([#545](https://github.com/CQCL/hugr/pull/545))
- Account for self-referencial constraints ([#551](https://github.com/CQCL/hugr/pull/551))
- Consider shunted metas when comparing equality ([#555](https://github.com/CQCL/hugr/pull/555))
- Join labels in issue workflow ([#563](https://github.com/CQCL/hugr/pull/563))
- Comment out broken priority code ([#562](https://github.com/CQCL/hugr/pull/562))
- Handling of issues with no priority label ([#573](https://github.com/CQCL/hugr/pull/573))
- Don't insert temporary wires when extracting a subgraph ([#582](https://github.com/CQCL/hugr/pull/582))
- Improve convexity checking and fix test ([#585](https://github.com/CQCL/hugr/pull/585))
- Ignore input->output links in SiblingSubgraph::try_new_dataflow_subgraph ([#589](https://github.com/CQCL/hugr/pull/589))
- Enforce covariance of SiblingMut::RootHandle ([#594](https://github.com/CQCL/hugr/pull/594))
- Erratic stack overflow in infer.rs (live_var) ([#638](https://github.com/CQCL/hugr/pull/638))
- Work harder in variable instantiation ([#591](https://github.com/CQCL/hugr/pull/591))
- Actually add the error type to prelude ([#672](https://github.com/CQCL/hugr/pull/672))
- Serialise dynamically computed opaqueOp signatures ([#690](https://github.com/CQCL/hugr/pull/690))
- FuncDefns don't require that their extensions match their children ([#688](https://github.com/CQCL/hugr/pull/688))
- Binary compute_signature returning a PolyFuncType with binders ([#710](https://github.com/CQCL/hugr/pull/710))
- Use correct number of args for int ops ([#723](https://github.com/CQCL/hugr/pull/723))
- [**breaking**] Add serde tag to TypeParam enum ([#722](https://github.com/CQCL/hugr/pull/722))
- Allow widening and narrowing to same width. ([#735](https://github.com/CQCL/hugr/pull/735))
- Case node should not have an external signature ([#749](https://github.com/CQCL/hugr/pull/749))
- [**breaking**] Normalize input/output value/static/other ports in `OpType` ([#783](https://github.com/CQCL/hugr/pull/783))
- No dataflow_signature for block types ([#792](https://github.com/CQCL/hugr/pull/792))
- Ignore unsupported test in miri ([#794](https://github.com/CQCL/hugr/pull/794))
- Include schema rather than read file ([#807](https://github.com/CQCL/hugr/pull/807))

### Documentation

- Add operation constraint to quantum extension ([#543](https://github.com/CQCL/hugr/pull/543))
- Coverage check section in DEVELOPMENT.md ([#648](https://github.com/CQCL/hugr/pull/648))
- Remove "quantum extension" from HUGR spec. ([#694](https://github.com/CQCL/hugr/pull/694))
- Improve crate-level docs, including example code. ([#698](https://github.com/CQCL/hugr/pull/698))
- Spec cleanups and clarifications ([#742](https://github.com/CQCL/hugr/pull/742))
- Spec clarifications ([#738](https://github.com/CQCL/hugr/pull/738))
- Spec updates ([#741](https://github.com/CQCL/hugr/pull/741))
- [spec] Remove references to causal cone and Order edges from Input ([#762](https://github.com/CQCL/hugr/pull/762))
- Mention experimental inference in readme ([#800](https://github.com/CQCL/hugr/pull/800))
- Collection of spec updates for 0.1 ([#801](https://github.com/CQCL/hugr/pull/801))
- Add schema v0 ([#805](https://github.com/CQCL/hugr/pull/805))
- Update spec wrt. polymorphism ([#791](https://github.com/CQCL/hugr/pull/791))

### Features

- Derive things for builder structs ([#229](https://github.com/CQCL/hugr/pull/229))
- Return a slice of types from the signature ([#238](https://github.com/CQCL/hugr/pull/238))
- Move `dot_string` to `HugrView` ([#271](https://github.com/CQCL/hugr/pull/271))
- [**breaking**] Change `TypeParam::USize` to `TypeParam::BoundedNat` and use in int extensions ([#445](https://github.com/CQCL/hugr/pull/445))
- TKET2 compatibility requirements  ([#450](https://github.com/CQCL/hugr/pull/450))
- Split methods between `HugrMut` and `HugrMutInternals` ([#441](https://github.com/CQCL/hugr/pull/441))
- Add `HugrView::node_connections` to get all links between nodes ([#460](https://github.com/CQCL/hugr/pull/460))
- Location information in extension inference error ([#464](https://github.com/CQCL/hugr/pull/464))
- Add T, Tdg, X gates ([#466](https://github.com/CQCL/hugr/pull/466))
- [**breaking**] Add `ApplyResult` associated type to `Rewrite` ([#472](https://github.com/CQCL/hugr/pull/472))
- Implement rewrite `IdentityInsertion` ([#474](https://github.com/CQCL/hugr/pull/474))
- Instantiate inferred extensions ([#461](https://github.com/CQCL/hugr/pull/461))
- Default DFG builders to open extension sets ([#473](https://github.com/CQCL/hugr/pull/473))
- Some helper methods ([#482](https://github.com/CQCL/hugr/pull/482))
- Extension inference for conditional nodes ([#465](https://github.com/CQCL/hugr/pull/465))
- Add ConvexChecker ([#487](https://github.com/CQCL/hugr/pull/487))
- Add clippy lint for mut calls in a debug_assert ([#488](https://github.com/CQCL/hugr/pull/488))
- Default more builder methods to open extension sets ([#492](https://github.com/CQCL/hugr/pull/492))
- Port is serializable ([#489](https://github.com/CQCL/hugr/pull/489))
- More general portgraph references ([#494](https://github.com/CQCL/hugr/pull/494))
- Add extension deltas to CFG ops ([#503](https://github.com/CQCL/hugr/pull/503))
- Implement petgraph traits on Hugr ([#498](https://github.com/CQCL/hugr/pull/498))
- Make extension delta a parameter of CFG builders ([#514](https://github.com/CQCL/hugr/pull/514))
- [**breaking**] SiblingSubgraph does not borrow Hugr ([#515](https://github.com/CQCL/hugr/pull/515))
- Validate TypeArgs to ExtensionOp ([#509](https://github.com/CQCL/hugr/pull/509))
- Derive Debug & Clone for `ExtensionRegistry`. ([#530](https://github.com/CQCL/hugr/pull/530))
- Const nodes are built with some extension requirements ([#527](https://github.com/CQCL/hugr/pull/527))
- Some python errors and bindings ([#533](https://github.com/CQCL/hugr/pull/533))
- Insert_hugr/insert_view return node map ([#535](https://github.com/CQCL/hugr/pull/535))
- Add `SiblingSubgraph::try_from_nodes_with_checker` ([#547](https://github.com/CQCL/hugr/pull/547))
- PortIndex trait for undirected port parameters ([#553](https://github.com/CQCL/hugr/pull/553))
- Insert/extract subgraphs from a HugrView ([#552](https://github.com/CQCL/hugr/pull/552))
- Add `new_array` operation to prelude ([#544](https://github.com/CQCL/hugr/pull/544))
- Add a `DataflowParentID` node handle ([#559](https://github.com/CQCL/hugr/pull/559))
- Make TypeEnum and it's contents public ([#558](https://github.com/CQCL/hugr/pull/558))
- Optional direction check when querying a port index ([#566](https://github.com/CQCL/hugr/pull/566))
- Extension inference for CFGs ([#529](https://github.com/CQCL/hugr/pull/529))
- Better subgraph verification errors ([#587](https://github.com/CQCL/hugr/pull/587))
- Compute affected nodes for `SimpleReplacement` ([#600](https://github.com/CQCL/hugr/pull/600))
- Move `SimpleReplace::invalidation_set` to the `Rewrite` trait ([#602](https://github.com/CQCL/hugr/pull/602))
- [**breaking**] Resolve extension ops (mutating Hugr) in (infer_and_->)update_validate ([#603](https://github.com/CQCL/hugr/pull/603))
- Add accessors for node index and const values ([#605](https://github.com/CQCL/hugr/pull/605))
- [**breaking**] Expose the value of ConstUsize ([#621](https://github.com/CQCL/hugr/pull/621))
- Nicer debug and display for core types ([#628](https://github.com/CQCL/hugr/pull/628))
- [**breaking**] Static checking of Port direction ([#614](https://github.com/CQCL/hugr/pull/614))
- Builder and HugrMut add_op_xxx default to open extensions ([#622](https://github.com/CQCL/hugr/pull/622))
- [**breaking**] Add panicking integer division ops ([#625](https://github.com/CQCL/hugr/pull/625))
- Add hashable `Angle` type to Quantum extension ([#608](https://github.com/CQCL/hugr/pull/608))
- [**breaking**] Remove "rotations" extension. ([#645](https://github.com/CQCL/hugr/pull/645))
- Port.as_directed to match on either direction ([#647](https://github.com/CQCL/hugr/pull/647))
- Impl GraphRef for PetgraphWrapper ([#651](https://github.com/CQCL/hugr/pull/651))
- Provide+implement Replace API ([#613](https://github.com/CQCL/hugr/pull/613))
- Require the node's metadata to always be a Map ([#661](https://github.com/CQCL/hugr/pull/661))
- [**breaking**] Polymorphic function types (inc OpDefs) using dyn trait ([#630](https://github.com/CQCL/hugr/pull/630))
- Make prelude error type public ([#669](https://github.com/CQCL/hugr/pull/669))
- Shorthand for retrieving custom constants from `Const`, `Value` ([#679](https://github.com/CQCL/hugr/pull/679))
- [**breaking**] HugrView API improvements ([#680](https://github.com/CQCL/hugr/pull/680))
- Make FuncDecl/FuncDefn polymorphic ([#692](https://github.com/CQCL/hugr/pull/692))
- [**breaking**] Simplify `SignatureFunc` and add custom arg validation. ([#706](https://github.com/CQCL/hugr/pull/706))
- [**breaking**] Drop the `pyo3` feature ([#717](https://github.com/CQCL/hugr/pull/717))
- [**breaking**] `OpEnum` trait for common opdef functionality ([#721](https://github.com/CQCL/hugr/pull/721))
- MakeRegisteredOp trait for easier registration ([#726](https://github.com/CQCL/hugr/pull/726))
- Getter for `PolyFuncType::body` ([#727](https://github.com/CQCL/hugr/pull/727))
- `Into<OpType>` for custom ops ([#731](https://github.com/CQCL/hugr/pull/731))
- Always require a signature in `OpaqueOp` ([#732](https://github.com/CQCL/hugr/pull/732))
- Values (and hence Consts) know their extensions ([#733](https://github.com/CQCL/hugr/pull/733))
- [**breaking**] Use ConvexChecker trait ([#740](https://github.com/CQCL/hugr/pull/740))
- Custom const for ERROR_TYPE ([#756](https://github.com/CQCL/hugr/pull/756))
- Implement RemoveConst and RemoveConstIgnore ([#757](https://github.com/CQCL/hugr/pull/757))
- Constant folding implemented for core and float extension ([#769](https://github.com/CQCL/hugr/pull/769))
- Constant folding for arithmetic conversion operations ([#720](https://github.com/CQCL/hugr/pull/720))
- DataflowParent trait for getting inner signatures ([#782](https://github.com/CQCL/hugr/pull/782))
- Constant folding for logic extension ([#793](https://github.com/CQCL/hugr/pull/793))
- Constant folding for list operations ([#795](https://github.com/CQCL/hugr/pull/795))
- Add panic op to prelude ([#802](https://github.com/CQCL/hugr/pull/802))
- Const::from_bool function ([#803](https://github.com/CQCL/hugr/pull/803))

### HugrMut

- Validate nodes for set_metadata/get_metadata_mut, too ([#531](https://github.com/CQCL/hugr/pull/531))

### HugrView

- Validate nodes, and remove Base ([#523](https://github.com/CQCL/hugr/pull/523))

### Miscellaneous Tasks

- [**breaking**] Update portgraph 0.10 and pyo3 0.20 ([#612](https://github.com/CQCL/hugr/pull/612))
- [**breaking**] Hike MSRV to 1.75 ([#761](https://github.com/CQCL/hugr/pull/761))

### Performance

- Use lazy static definittion for prelude registry ([#481](https://github.com/CQCL/hugr/pull/481))

### Refactor

- Move `rewrite` inside `hugr`, `Rewrite` -> `Replace` implementing new 'Rewrite' trait ([#119](https://github.com/CQCL/hugr/pull/119))
- Use an excluded upper bound instead of max log width. ([#451](https://github.com/CQCL/hugr/pull/451))
- Add extension info to `Conditional` and `Case` ([#463](https://github.com/CQCL/hugr/pull/463))
- `ExtensionSolution` only consists of input extensions ([#480](https://github.com/CQCL/hugr/pull/480))
- Remove builder from more DFG tests ([#490](https://github.com/CQCL/hugr/pull/490))
- Separate hierarchy views ([#500](https://github.com/CQCL/hugr/pull/500))
- [**breaking**] Use named struct for float constants ([#505](https://github.com/CQCL/hugr/pull/505))
- Allow NodeType::new to take any Into<Option<ExtensionSet>> ([#511](https://github.com/CQCL/hugr/pull/511))
- Move apply_rewrite from Hugr to HugrMut ([#519](https://github.com/CQCL/hugr/pull/519))
- Use SiblingSubgraph in SimpleReplacement ([#517](https://github.com/CQCL/hugr/pull/517))
- CFG takes a FunctionType ([#532](https://github.com/CQCL/hugr/pull/532))
- Remove check_custom_impl by inlining into check_custom ([#604](https://github.com/CQCL/hugr/pull/604))
- Insert_subgraph just return HashMap, make InsertionResult new_root compulsory ([#609](https://github.com/CQCL/hugr/pull/609))
- [**breaking**] Rename predicate to TupleSum/UnitSum ([#557](https://github.com/CQCL/hugr/pull/557))
- Simplify infer.rs/report_mismatch using early return ([#615](https://github.com/CQCL/hugr/pull/615))
- Move the core types to their own module ([#627](https://github.com/CQCL/hugr/pull/627))
- Change &NodeType->&OpType conversion into op() accessor ([#623](https://github.com/CQCL/hugr/pull/623))
- Infer.rs 'fn results' ([#631](https://github.com/CQCL/hugr/pull/631))
- Be safe ([#637](https://github.com/CQCL/hugr/pull/637))
- NodeType constructors, adding new_auto ([#635](https://github.com/CQCL/hugr/pull/635))
- Constraint::Plus stores an ExtensionSet, which is a BTreeSet ([#636](https://github.com/CQCL/hugr/pull/636))
- [**breaking**] Remove `SignatureDescription` ([#644](https://github.com/CQCL/hugr/pull/644))
- [**breaking**] Remove add_op_<posn> by generalizing add_node_<posn> with "impl Into" ([#642](https://github.com/CQCL/hugr/pull/642))
- Rename accidentally-changed Extension::add_node_xxx back to add_op ([#659](https://github.com/CQCL/hugr/pull/659))
- [**breaking**] Remove quantum extension ([#670](https://github.com/CQCL/hugr/pull/670))
- Use type schemes in extension definitions wherever possible ([#678](https://github.com/CQCL/hugr/pull/678))
- [**breaking**] Flatten `Prim(Type/Value)` in to parent enum ([#685](https://github.com/CQCL/hugr/pull/685))
- [**breaking**] Rename `new_linear()` to `new_endo()`. ([#697](https://github.com/CQCL/hugr/pull/697))
- Replace NodeType::signature() with io_extensions() ([#700](https://github.com/CQCL/hugr/pull/700))
- Validate ExtensionRegistry when built, not as we build it ([#701](https://github.com/CQCL/hugr/pull/701))
- [**breaking**] One way to add_op to extension ([#704](https://github.com/CQCL/hugr/pull/704))
- Remove Signature struct ([#714](https://github.com/CQCL/hugr/pull/714))
- Use `MakeOpDef` for int_ops ([#724](https://github.com/CQCL/hugr/pull/724))
- [**breaking**] Use enum op traits for floats + conversions ([#755](https://github.com/CQCL/hugr/pull/755))
- Avoid dynamic dispatch for non-folding operations ([#770](https://github.com/CQCL/hugr/pull/770))
- Simplify removeconstignore verify ([#768](https://github.com/CQCL/hugr/pull/768))
- [**breaking**] Unwrap BasicBlock enum  ([#781](https://github.com/CQCL/hugr/pull/781))
- Make clear const folding only for leaf ops ([#785](https://github.com/CQCL/hugr/pull/785))
- [**breaking**] `s/RemoveConstIgnore/RemoveLoadConstant` ([#789](https://github.com/CQCL/hugr/pull/789))
- Put extension inference behind a feature gate ([#786](https://github.com/CQCL/hugr/pull/786))

### SerSimpleType

- Use Vec not TypeRow ([#381](https://github.com/CQCL/hugr/pull/381))

### SimpleReplace+OutlineCfg

- Use HugrMut methods rather than .hierarchy/.graph ([#280](https://github.com/CQCL/hugr/pull/280))

### Testing

- Update inference test to not use DFG builder ([#550](https://github.com/CQCL/hugr/pull/550))
- Strengthen "failing_sccs_test", rename to "sccs" as it's not failing! ([#660](https://github.com/CQCL/hugr/pull/660))
- [**breaking**] Improve coverage in signature and validate ([#643](https://github.com/CQCL/hugr/pull/643))
- Use insta snapshots to add dot_string coverage ([#682](https://github.com/CQCL/hugr/pull/682))
- Miri ignore file-opening test ([#684](https://github.com/CQCL/hugr/pull/684))
- Unify the serialisation tests ([#730](https://github.com/CQCL/hugr/pull/730))
- Add schema validation to roundtrips ([#806](https://github.com/CQCL/hugr/pull/806))

### `ConstValue

- :F64` and `OpaqueOp::new` ([#206](https://github.com/CQCL/hugr/pull/206))

### Cleanup

- Remove outdated comment ([#536](https://github.com/CQCL/hugr/pull/536))

### Cosmetic

- Format + remove stray TODO ([#444](https://github.com/CQCL/hugr/pull/444))

### Doc

- Crate name as README title + add reexport ([#199](https://github.com/CQCL/hugr/pull/199))

### S/EdgeKind

- :Const/EdgeKind::Static/ ([#201](https://github.com/CQCL/hugr/pull/201))

### Simple_replace.rs

- Use HugrMut::remove_node, includes clearing op_types ([#242](https://github.com/CQCL/hugr/pull/242))

### Spec

- Remove "Draft 3" from title of spec document. ([#590](https://github.com/CQCL/hugr/pull/590))
- Rephrase confusing paragraph about TailLoop inputs/outputs ([#567](https://github.com/CQCL/hugr/pull/567))

### Src/ops/validate.rs

- Common-up some type row calculations ([#254](https://github.com/CQCL/hugr/pull/254))


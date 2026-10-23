//! Tests for extension resolution.

use core::{f64, panic};
use std::io::BufReader;
use std::sync::{Arc, LazyLock};

use itertools::Itertools;
use rstest::rstest;

use crate::builder::{
    DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::envelope::{EnvelopeConfig, EnvelopeFormat};
use crate::extension::prelude::{ConstUsize, bool_t, usize_custom_t, usize_t};
use crate::extension::resolution::{
    ExtensionResolutionError, ExtensionResolutionErrorDescription, WeakExtensionRegistry,
};
use crate::extension::resolution::{
    resolve_op_extensions, resolve_type_extensions as resolve_type_extension_refs,
};
use crate::extension::{
    ExtensionId, ExtensionRegistry, ExtensionSet, PRELUDE, PRELUDE_REGISTRY, TypeDefBound, Version,
};
use crate::hugr::HugrMut;
use crate::ops::constant::CustomConst;
use crate::ops::constant::test::CustomTestValue;
use crate::ops::dataflow::IOTrait;
use crate::ops::{CallIndirect, ExtensionOp, Input, NamedOp, OpType, OpaqueOp, Output, Tag, Value};
use crate::package::Package;
use crate::std_extensions::arithmetic::conversions::{self, ConvertOpDef};
use crate::std_extensions::arithmetic::float_types::{self, ConstF64, float64_type};
use crate::std_extensions::arithmetic::int_ops;
use crate::std_extensions::arithmetic::int_types::{self, int_type};
use crate::std_extensions::collections::list::ListValue;
use crate::std_extensions::std_reg;
use crate::types::type_param::TypeParam;
use crate::types::{CustomType, PolyFuncType, Signature, Term, Type, TypeBound};
use crate::{Extension, Hugr, HugrView, type_row};

use super::types_mut::resolve_op_types_extensions;

#[rstest]
#[case::empty(Input { types: type_row![]}, ExtensionRegistry::default())]
// A type with extra extensions in its instantiated type arguments.
#[case::parametric_op(int_ops::IntOpDef::ieq.with_log_width(4),
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned(), PRELUDE.to_owned()]
))]
fn collect_type_extensions(#[case] op: impl Into<OpType>, #[case] extensions: ExtensionRegistry) {
    let op = op.into();
    let resolved = op.used_extensions().unwrap();
    assert_eq!(resolved, extensions);
}

#[rstest]
#[case::empty(Input { types: type_row![]}, ExtensionRegistry::default())]
// A type with extra extensions in its instantiated type arguments.
#[case::parametric_op(int_ops::IntOpDef::ieq.with_log_width(4),
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned(), PRELUDE.to_owned()]
))]
fn resolve_type_extensions(#[case] op: impl Into<OpType>, #[case] extensions: ExtensionRegistry) {
    let op = op.into();

    // Ensure that all the `Weak` pointers get invalidated by round-tripping via serialization.
    let ser = serde_json::to_string(&op).unwrap();
    let mut deser_op: OpType = serde_json::from_str(&ser).unwrap();

    let dummy_node = portgraph::NodeIndex::new(0).into();

    let weak_extensions: WeakExtensionRegistry = (&extensions).into();
    resolve_op_types_extensions(Some(dummy_node), &mut deser_op, &weak_extensions)
        .unwrap()
        .for_each(|_| ());

    resolve_op_extensions(dummy_node, &mut deser_op, &extensions).unwrap();

    let deser_extensions = deser_op.used_extensions().unwrap();

    assert_eq!(
        deser_extensions, extensions,
        "{deser_extensions} != {extensions}"
    );
}

fn make_versioned_extension(id: &ExtensionId, version: Version) -> Arc<Extension> {
    Extension::new_arc(id.clone(), version, |ext, extension_ref| {
        ext.add_type(
            "MyType".into(),
            vec![],
            String::new(),
            TypeDefBound::copyable(),
            extension_ref,
        )
        .unwrap();
        ext.add_op(
            "my_op".into(),
            String::new(),
            Signature::new_endo([bool_t()]),
            extension_ref,
        )
        .unwrap();
    })
}

#[test]
fn resolve_opaque_op_uses_highest_compatible_extension() {
    let ext_id = ExtensionId::new_unchecked("versioned_ext");
    let ext_0_2_5 = make_versioned_extension(&ext_id, Version::new(0, 2, 5));
    let ext_0_3_0 = make_versioned_extension(&ext_id, Version::new(0, 3, 0));
    let registry = ExtensionRegistry::new([ext_0_2_5, ext_0_3_0]);

    let mut op: OpType = OpaqueOp::new(
        ext_id.clone(),
        Version::new(0, 2, 3),
        "my_op",
        [],
        Signature::new_endo([bool_t()]),
    )
    .into();

    let dummy_node = portgraph::NodeIndex::new(0).into();
    resolve_op_extensions(dummy_node, &mut op, &registry).unwrap();

    let ext_op = op.as_extension_op().unwrap();
    assert_eq!(ext_op.extension_version(), Version::new(0, 2, 5));
}

#[test]
fn resolve_custom_type_uses_highest_compatible_extension() {
    let ext_id = ExtensionId::new_unchecked("versioned_type_ext");
    let ext_0_2_5 = make_versioned_extension(&ext_id, Version::new(0, 2, 5));
    let ext_0_3_0 = make_versioned_extension(&ext_id, Version::new(0, 3, 0));
    let registry = ExtensionRegistry::new([ext_0_2_5, ext_0_3_0]);
    let weak_registry = WeakExtensionRegistry::from(&registry);
    let mut op = OpType::Input(Input::new(vec![Type::new_extension(CustomType::new(
        "MyType",
        [],
        ext_id,
        Version::new(0, 2, 3),
        TypeBound::Copyable,
        &Default::default(),
    ))]));

    let dummy_node = portgraph::NodeIndex::new(0).into();
    resolve_op_types_extensions(Some(dummy_node), &mut op, &weak_registry)
        .unwrap()
        .for_each(|_| ());

    let OpType::Input(input) = op else {
        panic!("expected input op");
    };
    let Term::ExtensionType(custom) = &*input.types[0] else {
        panic!("expected custom type");
    };
    assert_eq!(custom.extension_version(), Some(&Version::new(0, 2, 5)));
}

#[rstest]
#[case::too_new(Version::new(0, 2, 6), "the available version 0.2.5 is too old")]
#[case::incompatible(
    Version::new(0, 4, 0),
    "the available versions [0.2.5, 0.3.0] are incompatible"
)]
fn opaque_op_reports_version_mismatch(
    #[case] required_version: Version,
    #[case] expected_reason: &str,
) {
    let ext_id = ExtensionId::new_unchecked("versioned_ext");
    let other_id = ExtensionId::new_unchecked("other_ext");
    let registry = ExtensionRegistry::new([
        make_versioned_extension(&ext_id, Version::new(0, 2, 5)),
        make_versioned_extension(&ext_id, Version::new(0, 3, 0)),
        make_versioned_extension(&other_id, Version::new(9, 0, 0)),
    ]);
    let mut op: OpType = OpaqueOp::new(
        ext_id.clone(),
        required_version.clone(),
        "my_op",
        [],
        Signature::new_endo([bool_t()]),
    )
    .into();

    let node = portgraph::NodeIndex::new(0).into();
    let error = resolve_op_extensions(node, &mut op, &registry).unwrap_err();
    let ExtensionResolutionError::UnresolvedOpExtension { description, .. } = &error else {
        panic!("expected an operation extension version mismatch, got {error:?}");
    };
    assert_eq!(
        description.as_ref(),
        &ExtensionResolutionErrorDescription {
            required_extension: ext_id.clone(),
            required_version: required_version.clone(),
            available_extensions: vec![
                (other_id.clone(), Version::new(9, 0, 0)),
                (ext_id.clone(), Version::new(0, 2, 5)),
                (ext_id.clone(), Version::new(0, 3, 0)),
            ],
        }
    );
    assert!(
        error
            .to_string()
            .contains(&format!("{ext_id}@{required_version}"))
    );
    assert!(error.to_string().contains(expected_reason));
    assert!(!error.to_string().contains(&format!("{other_id}@9.0.0")));
}

#[rstest]
#[case::too_new(Version::new(0, 2, 6), "the available version 0.2.5 is too old")]
#[case::incompatible(
    Version::new(0, 4, 0),
    "the available versions [0.2.5, 0.3.0] are incompatible"
)]
fn custom_type_reports_version_mismatch(
    #[case] required_version: Version,
    #[case] expected_reason: &str,
) {
    let ext_id = ExtensionId::new_unchecked("versioned_type_ext");
    let registry = ExtensionRegistry::new([
        make_versioned_extension(&ext_id, Version::new(0, 2, 5)),
        make_versioned_extension(&ext_id, Version::new(0, 3, 0)),
    ]);
    let weak_registry = WeakExtensionRegistry::from(&registry);
    let custom = CustomType::new(
        "MyType",
        [],
        ext_id.clone(),
        required_version.clone(),
        TypeBound::Copyable,
        &Default::default(),
    );

    let error =
        resolve_type_extension_refs(&mut Type::new_extension(custom.clone()), &weak_registry)
            .unwrap_err();
    let ExtensionResolutionError::UnresolvedTypeExtension { description, .. } = &error else {
        panic!("expected a type extension version mismatch, got {error:?}");
    };
    assert_eq!(
        description.as_ref(),
        &ExtensionResolutionErrorDescription {
            required_extension: ext_id.clone(),
            required_version: required_version.clone(),
            available_extensions: vec![
                (ext_id.clone(), Version::new(0, 2, 5)),
                (ext_id.clone(), Version::new(0, 3, 0)),
            ],
        }
    );
    assert!(
        error
            .to_string()
            .contains(&format!("{ext_id}@{required_version}"))
    );
    assert!(error.to_string().contains(expected_reason));
}

#[test]
fn opaque_op_reports_versioned_missing_extension() {
    let ext_id = ExtensionId::new_unchecked("missing_ext");
    let required_version = Version::new(1, 2, 3);
    let registry = ExtensionRegistry::new([make_versioned_extension(
        &ExtensionId::new_unchecked("available_ext"),
        Version::new(2, 0, 0),
    )]);
    let mut op: OpType = OpaqueOp::new(
        ext_id.clone(),
        required_version.clone(),
        "my_op",
        [],
        Signature::new_endo([bool_t()]),
    )
    .into();

    let node = portgraph::NodeIndex::new(0).into();
    let error = resolve_op_extensions(node, &mut op, &registry).unwrap_err();
    assert!(matches!(
        error,
        ExtensionResolutionError::UnresolvedOpExtension { .. }
    ));
    assert!(error.to_string().contains("missing_ext@1.2.3"));
    assert!(error.to_string().contains("available_ext@2.0.0"));
}

#[test]
fn custom_type_reports_versioned_missing_extension() {
    let ext_id = ExtensionId::new_unchecked("missing_type_ext");
    let required_version = Version::new(1, 2, 3);
    let registry = ExtensionRegistry::new([make_versioned_extension(
        &ExtensionId::new_unchecked("available_ext"),
        Version::new(2, 0, 0),
    )]);
    let weak_registry = WeakExtensionRegistry::from(&registry);
    let mut typ = Type::new_extension(CustomType::new(
        "MyType",
        [],
        ext_id,
        required_version,
        TypeBound::Copyable,
        &Default::default(),
    ));

    let error = resolve_type_extension_refs(&mut typ, &weak_registry).unwrap_err();
    assert!(matches!(
        error,
        ExtensionResolutionError::UnresolvedTypeExtension { .. }
    ));
    assert!(error.to_string().contains("missing_type_ext@1.2.3"));
    assert!(error.to_string().contains("available_ext@2.0.0"));
}

/// Resolving an already-resolved type should preserve its shared storage.
#[test]
fn resolving_current_type_extensions_preserves_shared_storage() {
    let registry = std_reg();
    let weak_registry = WeakExtensionRegistry::from(&registry);
    let original = usize_t();
    let mut resolved = original.clone();
    assert!(original.shares_storage_with(&resolved));

    resolve_type_extension_refs(&mut resolved, &weak_registry).unwrap();

    assert!(original.shares_storage_with(&resolved));
}

/// Shared stale types should be resolved once and reuse the resolved storage.
#[test]
fn resolving_shared_stale_types_reuses_resolved_storage() {
    let ext_id = ExtensionId::new_unchecked("shared_type_ext");
    let version = Version::new(1, 0, 0);
    let stale_extension = make_versioned_extension(&ext_id, version.clone());
    let shared_type: Type = stale_extension
        .get_type("MyType")
        .unwrap()
        .instantiate([])
        .unwrap()
        .into();

    let mut hugr = Hugr::new();
    let root = hugr.module_root();
    let input = hugr.add_node_with_parent(root, Input::new([shared_type.clone()]));
    let output = hugr.add_node_with_parent(root, Output::new([shared_type]));
    drop(stale_extension);

    let current_extension = make_versioned_extension(&ext_id, version);
    let registry = ExtensionRegistry::new([current_extension.clone()]);
    hugr.resolve_extension_defs(&registry).unwrap();

    let OpType::Input(input_op) = hugr.get_optype(input) else {
        panic!("expected input op");
    };
    let OpType::Output(output_op) = hugr.get_optype(output) else {
        panic!("expected output op");
    };
    let input_type = &input_op.types[0];
    let output_type = &output_op.types[0];
    assert!(input_type.shares_storage_with(output_type));

    let Term::ExtensionType(custom) = &**input_type else {
        panic!("expected custom type");
    };
    let resolved_extension = custom.extension_ref().upgrade().unwrap();
    assert!(Arc::ptr_eq(&resolved_extension, &current_extension));
}

/// Create a new test extension with a single operation.
///
/// Returns an instance of the defined op.
fn make_extension(name: &str, op_name: &str) -> (Arc<Extension>, OpType) {
    let ext = Extension::new_test_arc(ExtensionId::new_unchecked(name), |ext, extension_ref| {
        ext.add_op(
            op_name.into(),
            String::new(),
            Signature::new_endo([bool_t()]),
            extension_ref,
        )
        .unwrap();
    });
    let op_def = ext.get_op(op_name).unwrap();
    let op = ExtensionOp::new(op_def.clone(), vec![]).unwrap();
    (ext, op.into())
}

/// Create a new test extension with a type and an op using that type
///
/// Returns the defined extension.
fn make_extension_self_referencing(name: &str, op_name: &str, type_name: &str) -> Arc<Extension> {
    Extension::new_test_arc(ExtensionId::new_unchecked(name), |ext, extension_ref| {
        let type_def = ext
            .add_type(
                type_name.into(),
                vec![],
                String::new(),
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();
        let typ = type_def.instantiate([]).unwrap();

        ext.add_op(
            op_name.into(),
            String::new(),
            Signature::new(vec![typ.into()], vec![usize_t()]),
            extension_ref,
        )
        .unwrap();
    })
}

/// Create two extensions where one operation's signature uses the other's type.
fn make_dependent_extensions() -> (Arc<Extension>, Arc<Extension>) {
    let dep_ext =
        Extension::new_test_arc(ExtensionId::new_unchecked("dummy.dep"), |ext, ext_ref| {
            ext.add_type(
                "dep_type".into(),
                vec![],
                String::new(),
                TypeDefBound::copyable(),
                ext_ref,
            )
            .unwrap();
        });
    let dep_ty = dep_ext
        .get_type("dep_type")
        .unwrap()
        .instantiate([])
        .unwrap();
    let op_ext = Extension::new_test_arc(ExtensionId::new_unchecked("dummy.op"), |ext, ext_ref| {
        ext.add_op(
            "uses_dep".into(),
            String::new(),
            Signature::new(vec![dep_ty.into()], vec![]),
            ext_ref,
        )
        .unwrap();
    });
    (dep_ext, op_ext)
}

/// Check that resolved extension operations are relinked to OpDefs from the
/// registry when doing resolution, and that the resolved extensions are correct
/// even when there are multiple compatible versions in the registry.
#[test]
fn ext_resolution_relinks_opdefs() {
    let (original_dep, original_op_ext) = make_dependent_extensions();
    let dep_ty = original_dep
        .get_type("dep_type")
        .unwrap()
        .instantiate([])
        .unwrap();
    let op = original_op_ext
        .instantiate_extension_op("uses_dep", [])
        .unwrap();

    let mut build = DFGBuilder::new(Signature::new(vec![dep_ty.into()], vec![])).unwrap();
    let [input] = build.input_wires_arr();
    build.add_dataflow_op(op, [input]).unwrap();
    let mut hugr = build.finish_hugr_with_outputs([]).unwrap();

    let (canonical_dep, canonical_op_ext) = make_dependent_extensions();
    let resolution_extensions =
        ExtensionRegistry::new([canonical_dep.clone(), canonical_op_ext.clone()]);

    // After dropping the extensions here and running extension resolution (so the hugr's internal register also drops the old references),
    // the old OpDefs in the ExtensionOp should lose its reference.
    drop(original_dep);
    drop(original_op_ext);
    hugr.resolve_extension_defs(&resolution_extensions).unwrap();

    // Check that the ExtensionOp was relinked to the new OpDef from the registry.
    let op = hugr
        .nodes()
        .map(|node| hugr.get_optype(node))
        .find_map(OpType::as_extension_op)
        .unwrap();
    let canonical_def = canonical_op_ext.get_op("uses_dep").unwrap();
    assert!(Arc::ptr_eq(op.def_arc(), canonical_def));

    let rebuilt_op = ExtensionOp::new(op.def_arc().clone(), []).unwrap();
    let rebuilt_op: OpType = rebuilt_op.into();
    let rebuilt_extensions = rebuilt_op.used_extensions().unwrap();
    assert!(rebuilt_extensions.contains(canonical_dep.name()));
}

/// Check that the extensions added during building coincide with read-only collected extensions
/// and that they survive a serialization roundtrip.
fn check_extension_resolution(mut hugr: Hugr) {
    // Extensions used by the hugr, used to check that the roundtrip preserves them.
    let build_extensions = hugr.extensions().clone();

    // Extensions used for resolution.
    let mut resolution_extensions = std_reg();
    resolution_extensions.extend(&build_extensions);

    // Check that the read-only methods collect the same extensions.
    let mut collected_exts = ExtensionRegistry::default();
    for node in hugr.nodes() {
        collected_exts.extend(hugr.get_optype(node).used_extensions().unwrap_or_default());
    }
    assert_eq!(
        collected_exts, build_extensions,
        "{collected_exts} != {build_extensions}"
    );

    // Check that the mutable methods collect the same extensions.
    hugr.resolve_extension_defs(&resolution_extensions).unwrap();
    assert_eq!(
        hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        hugr.extensions()
    );

    // Roundtrip serialize so all weak references are dropped.
    let ser = hugr.store_str(EnvelopeConfig::text()).unwrap();

    let deser_hugr = Hugr::load_str(&ser, Some(&resolution_extensions)).unwrap();

    assert_eq!(
        deser_hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        deser_hugr.extensions()
    );
}

/// Build a small hugr using the float types extension and check that the extensions are resolved.
#[rstest]
fn resolve_hugr_extensions_simple() {
    let mut build = DFGBuilder::new(Signature::new(vec![], vec![float64_type()])).unwrap();

    // A constant op using a non-prelude extension.
    let f_const = build.add_load_const(Value::extension(ConstF64::new(f64::consts::PI)));

    let mut hugr = build
        .finish_hugr_with_outputs([f_const])
        .unwrap_or_else(|e| panic!("{e}"));

    let build_extensions = hugr.extensions().clone();

    // Check that the read-only methods collect the same extensions.
    let mut collected_exts = ExtensionRegistry::default();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        collected_exts.extend(op.used_extensions().unwrap());
    }
    assert_eq!(
        collected_exts, build_extensions,
        "{collected_exts} != {build_extensions}"
    );

    // Check that the mutable methods collect the same extensions.
    hugr.resolve_extension_defs(&build_extensions).unwrap();
    assert_eq!(
        hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        hugr.extensions()
    );

    // Serialization roundtrip to drop the weak pointers.
    let ser = hugr.store_str(EnvelopeConfig::text()).unwrap();
    let deser_hugr = Hugr::load_str(&ser, Some(&build_extensions)).unwrap();

    assert_eq!(
        deser_hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        hugr.extensions()
    );
}

/// Build a hugr with all possible op nodes and resolve the extensions.
#[rstest]
fn resolve_hugr_extensions() {
    let (ext_a, op_a) = make_extension("dummy.a", "op_a");
    let (ext_b, op_b) = make_extension("dummy.b", "op_b");
    let (ext_c, op_c) = make_extension("dummy.c", "op_c");
    let (ext_d, op_d) = make_extension("dummy.d", "op_d");
    let (_ext_e, op_e) = make_extension("dummy.e", "op_e");

    let mut module = ModuleBuilder::new();

    // A function declaration using the floats extension in its signature.
    let decl = module
        .declare(
            "dummy_declaration",
            Signature::new_endo([float64_type()]).into(),
        )
        .unwrap();

    // A function definition using the int_types and float_types extension in its body.
    let mut func = module
        .define_function(
            "dummy_fn",
            Signature::new(vec![float64_type(), bool_t()], vec![]),
        )
        .unwrap();
    let [func_i0, func_i1] = func.input_wires_arr();

    // Call the function declaration directly, and load & call indirectly.
    func.call(&decl, &[], vec![func_i0]).unwrap();
    let loaded_func = func.load_func(&decl, &[]).unwrap();
    func.add_dataflow_op(
        CallIndirect {
            signature: Signature::new_endo([float64_type()]),
        },
        vec![loaded_func, func_i0],
    )
    .unwrap();

    // Add one of the custom ops.
    func.add_dataflow_op(op_a, vec![func_i1]).unwrap();

    // A nested dataflow region.
    let mut dfg = func.dfg_builder_endo([(bool_t(), func_i1)]).unwrap();
    let dfg_inputs = dfg.input_wires().collect_vec();
    dfg.add_dataflow_op(op_b, dfg_inputs.clone()).unwrap();
    dfg.finish_with_outputs(dfg_inputs).unwrap();

    // A tag
    func.add_dataflow_op(
        Tag::new(0, vec![vec![bool_t()].into(), vec![int_type(4)].into()]),
        vec![func_i1],
    )
    .unwrap();

    // Dfg control flow: Tail loop
    let mut tail_loop = func
        .tail_loop_builder([(bool_t(), func_i1)], [], vec![].into())
        .unwrap();
    let tl_inputs = tail_loop.input_wires().collect_vec();
    tail_loop.add_dataflow_op(op_c, tl_inputs).unwrap();
    let tl_tag = tail_loop.add_load_const(Value::true_val());
    let tl_tag = tail_loop
        .add_dataflow_op(
            Tag::new(0, vec![vec![Type::new_unit_sum(2)].into(), vec![].into()]),
            vec![tl_tag],
        )
        .unwrap()
        .out_wire(0);
    tail_loop.finish_with_outputs(tl_tag, vec![]).unwrap();

    // Dfg control flow: Conditionals
    let cond_tag = func.add_load_const(Value::unary_unit_sum());
    let mut cond = func
        .conditional_builder(([type_row![]], cond_tag), [], type_row![])
        .unwrap();
    let mut case = cond.case_builder(0).unwrap();
    case.add_dataflow_op(op_e, [func_i1]).unwrap();
    case.finish_with_outputs([]).unwrap();

    // Cfg control flow.
    let mut cfg = func
        .cfg_builder([(bool_t(), func_i1)], vec![].into())
        .unwrap();
    let mut cfg_entry = cfg.entry_builder([type_row![]], type_row![]).unwrap();
    let [cfg_i0] = cfg_entry.input_wires_arr();
    cfg_entry.add_dataflow_op(op_d, [cfg_i0]).unwrap();
    let cfg_tag = cfg_entry.add_load_const(Value::unary_unit_sum());
    let cfg_entry_wire = cfg_entry.finish_with_outputs(cfg_tag, []).unwrap();
    let cfg_exit = cfg.exit_block();
    cfg.branch(&cfg_entry_wire, 0, &cfg_exit).unwrap();

    // --------------------------------------------------

    // Finally, finish the hugr and ensure it's using the right extensions.
    func.finish_with_outputs(vec![]).unwrap();
    let hugr = module.finish_hugr().unwrap_or_else(|e| panic!("{e}"));

    let build_extensions = hugr.extensions().clone();
    assert!(build_extensions.contains(ext_a.name()));
    assert!(build_extensions.contains(ext_b.name()));
    assert!(build_extensions.contains(ext_c.name()));
    assert!(build_extensions.contains(ext_d.name()));

    check_extension_resolution(hugr);
}

/// Test resolution of a custom constants.
#[rstest]
#[case::usize(ConstUsize::new(42))]
#[case::list(ListValue::new(
        float64_type(),
        [ConstF64::new(f64::consts::PI).into()],
))]
#[case::custom(CustomTestValue(usize_custom_t(
        &Arc::downgrade(&PRELUDE),
)))]
fn resolve_custom_const(#[case] custom_const: impl CustomConst) {
    // Return the loaded value so linear constants, including lists, are consumed.
    let ty = custom_const.get_type();
    let mut dfg_builder = DFGBuilder::new(Signature::new(vec![], vec![ty])).unwrap();
    let value = dfg_builder.add_load_const(Value::extension(custom_const));
    let hugr = dfg_builder
        .finish_hugr_with_outputs([value])
        .unwrap_or_else(|e| panic!("{e}"));

    check_extension_resolution(hugr);
}

/// A bare HUGR must retain the packaged allocation referenced by its types,
/// even when a custom constant reports the same type from a static extension.
#[rstest]
fn packaged_extension_lifetime() {
    const PACKAGED_TYPE_EXTENSION_ID: ExtensionId =
        ExtensionId::new_unchecked("test.packaged_type");

    static PACKAGED_TYPE_EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
        Extension::new_arc(
            PACKAGED_TYPE_EXTENSION_ID,
            Version::new(1, 0, 0),
            |extension, extension_ref| {
                extension
                    .add_type(
                        "value".into(),
                        vec![],
                        "A type returned by a stateless custom constant.".into(),
                        TypeDefBound::copyable(),
                        extension_ref,
                    )
                    .unwrap();
            },
        )
    });

    /// A stateless constant whose type comes from a process-wide extension.
    #[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
    struct PackagedValue;

    #[typetag::serde]
    impl CustomConst for PackagedValue {
        fn name(&self) -> crate::ops::constant::ValueName {
            "PackagedValue".into()
        }

        fn get_type(&self) -> Type {
            CustomType::new(
                "value",
                [],
                PACKAGED_TYPE_EXTENSION_ID,
                Version::new(1, 0, 0),
                TypeBound::Copyable,
                &Arc::downgrade(&PACKAGED_TYPE_EXTENSION),
            )
            .into()
        }
    }

    let mut builder = DFGBuilder::new(Signature::new([], [PackagedValue.get_type()])).unwrap();
    let value = builder.add_load_value(Value::extension(PackagedValue));
    let hugr = builder.finish_hugr_with_outputs([value]).unwrap();

    let mut package = Package::from_hugr(hugr);
    package.extensions.register(PACKAGED_TYPE_EXTENSION.clone());
    let mut encoded = Vec::new();
    package
        .store(
            &mut encoded,
            EnvelopeConfig::new(EnvelopeFormat::ModelWithExtensions),
        )
        .unwrap();

    let hugr = Hugr::load(encoded.as_slice(), None).unwrap();
    hugr.validate().unwrap();

    let datatype = hugr
        .nodes()
        .find_map(|node| match hugr.get_optype(node) {
            OpType::LoadConstant(load) => Some(&load.datatype),
            _ => None,
        })
        .unwrap();
    let referenced = datatype
        .used_extensions()
        .unwrap()
        .get("test.packaged_type")
        .unwrap()
        .clone();
    let retained = hugr.extensions().get("test.packaged_type").unwrap();
    assert!(Arc::ptr_eq(&referenced, retained));
}

/// Test resolution of function call with type arguments.
#[rstest]
fn resolve_call() {
    let dummy_fn_sig = PolyFuncType::new(
        vec![TypeParam::TypeKind(TypeBound::Linear)],
        Signature::new(vec![], vec![bool_t()]),
    );

    let generic_type_1 = float64_type().into();
    let generic_type_2 = int_type(6).into();
    let expected_exts = [
        float_types::EXTENSION_ID.clone(),
        int_types::EXTENSION_ID.clone(),
    ]
    .into_iter()
    .collect::<ExtensionSet>();

    let mut module = ModuleBuilder::new();
    let dummy_fn = module.declare("called_fn", dummy_fn_sig).unwrap();

    let mut func = module
        .define_function("caller_fn", Signature::new(vec![], vec![bool_t()]))
        .unwrap();
    let _load_func = func.load_func(&dummy_fn, &[generic_type_1]).unwrap();
    let call = func.call(&dummy_fn, &[generic_type_2], vec![]).unwrap();
    func.finish_with_outputs(call.outputs()).unwrap();

    let hugr = module.finish_hugr().unwrap();

    for ext in expected_exts {
        assert!(hugr.extensions().contains(&ext));
    }

    check_extension_resolution(hugr);
}

/// Test that extension resolution is transitive across extension dependencies.
///
/// `arithmetic.conversions` depends on `arithmetic.int_types` and
/// `arithmetic.float_types`, so using an op from the former should cause all
/// three extensions to be resolved even if the operation itself doesn't
/// directly use floats.
#[rstest]
fn resolve_transitive_extension_deps() {
    let mut build = DFGBuilder::new(Signature::new(vec![int_type(6)], vec![usize_t()])).unwrap();
    let [input] = build.input_wires_arr();

    let out = build
        .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [input])
        .unwrap();

    let hugr = build
        .finish_hugr_with_outputs(out.outputs())
        .unwrap_or_else(|e| panic!("{e}"));

    assert!(hugr.extensions().contains(&conversions::EXTENSION_ID));
    assert!(hugr.extensions().contains(&float_types::EXTENSION_ID));

    check_extension_resolution(hugr);
}

/// Test that extensions in `lower_funcs` are properly resolved when loading a package.
///
/// <https://github.com/Quantinuum/hugr/issues/2520>
#[rstest]
fn resolve_lower_func_extensions() {
    // Create an inner extension with a simple op.
    let (inner_ext, inner_op) = make_extension("inner.extension", "InnerOp");

    // Build a HUGR that uses the inner op as its lowering body.
    let mut dfg = DFGBuilder::new(Signature::new_endo(vec![bool_t()])).unwrap();
    let [input] = dfg.input_wires_arr();
    let inner_result = dfg.add_dataflow_op(inner_op.clone(), [input]).unwrap();
    let lower_hugr = dfg
        .finish_hugr_with_outputs(inner_result.outputs())
        .unwrap();

    // The lower func is stored as a Package so its extension definitions travel with it.
    let mut lower_pkg = Package::from_hugr(lower_hugr);
    lower_pkg.extensions.register(inner_ext.clone());

    // Create an outer extension whose op has the lower func.
    let inner_ext_name = inner_ext.name().clone();
    let outer_ext = Extension::new_test_arc(
        ExtensionId::new_unchecked("outer.extension"),
        |ext, ext_ref| {
            let op_def = ext
                .add_op(
                    "OuterOp".into(),
                    String::new(),
                    Signature::new_endo(vec![bool_t()]),
                    ext_ref,
                )
                .unwrap();
            op_def.add_lower_func(crate::extension::op_def::LowerFunc::FixedHugr {
                extensions: ExtensionSet::singleton(inner_ext_name),
                pkg: Box::new(lower_pkg),
            });
        },
    );

    // Build a package containing only the outer extension.
    // The inner extension is embedded inside the lower func's Package.
    let mut outer_pkg = Package::default();
    outer_pkg.extensions.register(outer_ext.clone());

    // Serialize and reload using only the standard extensions as the external registry.
    // The inner extension must be recovered from the embedded lower func package.
    let mut buffer = Vec::new();
    outer_pkg
        .store(&mut buffer, EnvelopeConfig::binary())
        .expect("serialize outer package");
    let loaded_pkg =
        Package::load(BufReader::new(buffer.as_slice()), None).expect("load outer package");

    // The outer extension should be present and its op should have a lower func.
    let loaded_outer_ext = loaded_pkg
        .extensions
        .get(outer_ext.name())
        .expect("outer extension missing after load");
    let loaded_op_def = loaded_outer_ext
        .get_op("OuterOp")
        .expect("OuterOp missing after load");
    assert_eq!(
        loaded_op_def.lower_funcs.len(),
        1,
        "expected 1 lower func after load"
    );

    // Calling try_lower should succeed and return a properly resolved HUGR
    // (all inner ops must be ExtensionOps, not OpaqueOps).
    let available = ExtensionSet::singleton(inner_ext.name().clone());
    let lower_hugr = loaded_op_def
        .try_lower(&[], &available)
        .expect("try_lower should succeed when inner extension is available");

    // No unresolved OpaqueOps after loading.
    for node in lower_hugr.nodes() {
        let op = lower_hugr.get_optype(node);
        assert!(
            !matches!(op, crate::ops::OpType::OpaqueOp(_)),
            "Op {op:?} on {node} is an unresolved OpaqueOp after loading"
        );
    }

    // There is some inner_ext operation in the lower hugr.
    lower_hugr.nodes().any(|node| {
        let op = lower_hugr.get_optype(node);
        let crate::ops::OpType::ExtensionOp(op) = op else {
            return false;
        };
        op.extension_id() == inner_ext.name() && op.name() == inner_op.name()
    });
}

/// Test the [`ExtensionRegistry::new_cyclic`] and [`ExtensionRegistry::new_with_extension_resolution`] methods.
#[test]
fn register_new_cyclic() {
    let ext_id = ExtensionId::new("ext").unwrap();
    let ext = make_extension_self_referencing(&ext_id, "my_op", "my_type");

    let reg = ExtensionRegistry::new([ext]);

    // Roundtrip serialization drops all the weak pointers,
    // and causes both initialization methods to be called.
    let ser = serde_json::to_string(&reg).unwrap();
    let new_reg = ExtensionRegistry::load_json(ser.as_bytes(), &PRELUDE_REGISTRY).unwrap();

    assert!(new_reg.contains(&ext_id));
    new_reg.validate().unwrap();
}

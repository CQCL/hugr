//! Tests for extension resolution.

use core::{f64, panic};
use std::sync::Arc;

use itertools::Itertools;
use rstest::rstest;

use crate::builder::{
    Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::envelope::EnvelopeConfig;
use crate::extension::prelude::{ConstUsize, bool_t, usize_custom_t, usize_t};
use crate::extension::resolution::WeakExtensionRegistry;
use crate::extension::resolution::{resolve_op_extensions, resolve_op_types_extensions};
use crate::extension::{
    ExtensionId, ExtensionRegistry, ExtensionSet, PRELUDE, PRELUDE_REGISTRY, TypeDefBound,
};
use crate::ops::constant::CustomConst;
use crate::ops::constant::test::CustomTestValue;
use crate::ops::{CallIndirect, ExtensionOp, Input, OpType, Tag, Value};
use crate::std_extensions::arithmetic::float_types::{self, ConstF64, float64_type};
use crate::std_extensions::arithmetic::int_ops;
use crate::std_extensions::arithmetic::int_types::{self, int_type};
use crate::std_extensions::collections::list::ListValue;
use crate::std_extensions::std_reg;
use crate::types::type_param::TypeParam;
use crate::types::{PolyFuncType, Signature, Type, TypeBound};
use crate::{Extension, Hugr, HugrView, type_row};

#[rstest]
#[case::empty(Input { types: type_row![]}, ExtensionRegistry::default())]
// A type with extra extensions in its instantiated type arguments.
#[case::parametric_op(int_ops::IntOpDef::ieq.with_log_width(4),
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned()]
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
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned()]
))]
fn resolve_type_extensions(#[case] op: impl Into<OpType>, #[case] extensions: ExtensionRegistry) {
    let op = op.into();

    // Ensure that all the `Weak` pointers get invalidated by round-tripping via serialization.
    let ser = serde_json::to_string(&op).unwrap();
    let mut deser_op: OpType = serde_json::from_str(&ser).unwrap();

    let dummy_node = portgraph::NodeIndex::new(0).into();

    resolve_op_extensions(dummy_node, &mut deser_op, &extensions).unwrap();

    let weak_extensions: WeakExtensionRegistry = (&extensions).into();
    resolve_op_types_extensions(Some(dummy_node), &mut deser_op, &weak_extensions)
        .unwrap()
        .for_each(|_| ());

    let deser_extensions = deser_op.used_extensions().unwrap();

    assert_eq!(
        deser_extensions, extensions,
        "{deser_extensions} != {extensions}"
    );
}

/// Create a new test extension with a single operation.
///
/// Returns an instance of the defined op.
fn make_extension(name: &str, op_name: &str) -> (Arc<Extension>, OpType) {
    let ext = Extension::new_test_arc(ExtensionId::new_unchecked(name), |ext, extension_ref| {
        ext.add_op(
            op_name.into(),
            String::new(),
            Signature::new_endo(vec![bool_t()]),
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

/// Check that the extensions added during building coincide with read-only collected extensions
/// and that they survive a serialization roundtrip.
fn check_extension_resolution(mut hugr: Hugr) {
    // Extensions used by the hugr, used to check that the roundtrip preserves them.
    let build_extensions = hugr.extensions().clone();

    // Extensions used for resolution.
    let mut resolution_extensions = std_reg();
    resolution_extensions.extend(&build_extensions);

    // Check that the read-only methods collect the same extensions.
    let collected_exts = ExtensionRegistry::new(hugr.nodes().flat_map(|node| {
        hugr.get_optype(node)
            .used_extensions()
            .unwrap_or_default()
            .into_iter()
    }));
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
            Signature::new_endo(vec![float64_type()]).into(),
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
            signature: Signature::new_endo(vec![float64_type()]),
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
    let mut module = ModuleBuilder::new();
    module.add_constant(Value::extension(custom_const));
    let hugr = module.finish_hugr().unwrap_or_else(|e| panic!("{e}"));

    check_extension_resolution(hugr);
}

/// Test resolution of function call with type arguments.
#[rstest]
fn resolve_call() {
    let dummy_fn_sig = PolyFuncType::new(
        vec![TypeParam::RuntimeType(TypeBound::Linear)],
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

//! Tests for extension resolution.

use core::{f64, panic};
use std::sync::Arc;

use cool_asserts::assert_matches;
use itertools::Itertools;
use rstest::rstest;

use crate::builder::{
    Container, Dataflow, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::{bool_t, usize_custom_t, ConstUsize};
use crate::extension::resolution::WeakExtensionRegistry;
use crate::extension::resolution::{
    resolve_op_extensions, resolve_op_types_extensions, ExtensionCollectionError,
};
use crate::extension::{ExtensionId, ExtensionRegistry, ExtensionSet, TypeDefBound, PRELUDE};
use crate::ops::constant::test::CustomTestValue;
use crate::ops::constant::CustomConst;
use crate::ops::{CallIndirect, ExtensionOp, Input, OpTrait, OpType, Tag, Value};
use crate::std_extensions::arithmetic::float_types::{float64_type, ConstF64};
use crate::std_extensions::arithmetic::int_ops;
use crate::std_extensions::arithmetic::int_types::{self, int_type};
use crate::std_extensions::collections::list::ListValue;
use crate::types::{Signature, Type};
use crate::{type_row, Extension, Hugr, HugrView};

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
            "".to_string(),
            Signature::new_endo(vec![bool_t()]),
            extension_ref,
        )
        .unwrap();
    });
    let op_def = ext.get_op(op_name).unwrap();
    let op = ExtensionOp::new(op_def.clone(), vec![], &ExtensionRegistry::default()).unwrap();
    (ext, op.into())
}

/// Create a new test extension with a type and an op using that type
///
/// Returns an instance of the defined op.
fn make_extension_self_referencing(name: &str, op_name: &str, type_name: &str) -> Arc<Extension> {
    let ext = Extension::new_test_arc(ExtensionId::new_unchecked(name), |ext, extension_ref| {
        let type_def = ext
            .add_type(
                type_name.into(),
                vec![],
                "".to_string(),
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();
        let typ = type_def.instantiate([]).unwrap();

        ext.add_op(
            op_name.into(),
            "".to_string(),
            Signature::new_endo(vec![typ.into()]),
            extension_ref,
        )
        .unwrap();
    });
    ext
}

/// Check that the extensions added during building coincide with read-only collected extensions
/// and that they survive a serialization roundtrip.
fn check_extension_resolution(mut hugr: Hugr) {
    let build_extensions = hugr.extensions().clone();

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
    hugr.resolve_extension_defs(&build_extensions).unwrap();
    assert_eq!(
        hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        hugr.extensions()
    );

    // Roundtrip serialize so all weak references are dropped.
    let ser = serde_json::to_string(&hugr).unwrap();
    let deser_hugr = Hugr::load_json(ser.as_bytes(), &build_extensions).unwrap();

    assert_eq!(
        deser_hugr.extensions(),
        &build_extensions,
        "{} != {build_extensions}",
        deser_hugr.extensions()
    );
}

/// Build a hugr with all possible op nodes and resolve the extensions.
#[rstest]
fn resolve_hugr_extensions() {
    let (ext_a, op_a) = make_extension("dummy.a", "op_a");
    let (ext_b, op_b) = make_extension("dummy.b", "op_b");
    let (ext_c, op_c) = make_extension("dummy.c", "op_c");
    let (ext_d, op_d) = make_extension("dummy.d", "op_d");
    let (ext_e, op_e) = make_extension("dummy.e", "op_e");

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
            Signature::new(vec![float64_type(), bool_t()], vec![]).with_extension_delta(
                [
                    ext_a.name(),
                    ext_b.name(),
                    ext_c.name(),
                    ext_d.name(),
                    ext_e.name(),
                ]
                .into_iter()
                .cloned()
                .collect::<ExtensionSet>(),
            ),
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

/// Fail when collecting extensions but the weak pointers are not resolved.
#[rstest]
fn dropped_weak_extensions() {
    let (ext_a, op_a) = make_extension("dummy.a", "op_a");
    let mut func = FunctionBuilder::new(
        "dummy_fn",
        Signature::new(vec![float64_type(), bool_t()], vec![]).with_extension_delta(
            [ext_a.name()]
                .into_iter()
                .cloned()
                .collect::<ExtensionSet>(),
        ),
    )
    .unwrap();
    let [_func_i0, func_i1] = func.input_wires_arr();
    func.add_dataflow_op(op_a, vec![func_i1]).unwrap();

    let hugr = func.finish_hugr().unwrap();

    // Do a serialization roundtrip to drop the references.
    let ser = serde_json::to_string(&hugr).unwrap();
    let hugr: Hugr = serde_json::from_str(&ser).unwrap();

    let op_collection = hugr
        .nodes()
        .try_for_each(|node| hugr.get_optype(node).used_extensions().map(|_| ()));
    assert_matches!(
        op_collection,
        Err(ExtensionCollectionError::DroppedOpExtensions { .. })
    );

    let op_collection = hugr.nodes().try_for_each(|node| {
        let op = hugr.get_optype(node);
        if let Some(sig) = op.dataflow_signature() {
            sig.used_extensions()?;
        }
        Ok(())
    });
    assert_matches!(
        op_collection,
        Err(ExtensionCollectionError::DroppedSignatureExtensions { .. })
    );
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
    let new_reg: ExtensionRegistry = serde_json::from_str(&ser).unwrap();

    assert!(new_reg.contains(&ext_id));
    new_reg.validate().unwrap();
}

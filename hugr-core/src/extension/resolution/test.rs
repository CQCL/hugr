//! Tests for extension resolution.

use core::panic;
use std::sync::Arc;

use itertools::Itertools;
use rstest::rstest;

use crate::builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
use crate::extension::prelude::{bool_t, ConstUsize};
use crate::extension::resolution::{
    collect_op_extensions, collect_op_types_extensions, update_op_extensions,
    update_op_types_extensions,
};
use crate::extension::{ExtensionId, ExtensionRegistry, ExtensionSet, PRELUDE};
use crate::ops::{CallIndirect, ExtensionOp, Input, OpType, Tag, Value};
use crate::std_extensions::arithmetic::float_types::{self, float64_type};
use crate::std_extensions::arithmetic::int_ops;
use crate::std_extensions::arithmetic::int_types::{self, int_type};
use crate::types::{Signature, Type};
use crate::{type_row, Extension, HugrView};

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

    let mut used_exts = ExtensionRegistry::default();
    update_op_extensions(dummy_node, &mut deser_op, &extensions).unwrap();
    update_op_types_extensions(dummy_node, &mut deser_op, &extensions, &mut used_exts).unwrap();

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

/// Build a hugr with all possible op nodes and resolve the extensions.
#[rstest]
fn resolve_hugr_extensions() {
    let (ext_a, op_a) = make_extension("dummy.a", "op_a");
    let (ext_b, op_b) = make_extension("dummy.b", "op_b");
    let (ext_c, op_c) = make_extension("dummy.c", "op_c");
    let (ext_d, op_d) = make_extension("dummy.d", "op_d");

    let build_extensions = ExtensionRegistry::new([
        PRELUDE.to_owned(),
        ext_a.clone(),
        ext_b.clone(),
        ext_c.clone(),
        ext_d.clone(),
        float_types::EXTENSION.to_owned(),
        int_types::EXTENSION.to_owned(),
    ]);

    let mut module = ModuleBuilder::new();

    // A constant op using the prelude extension.
    module.add_constant(Value::extension(ConstUsize::new(42)));

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
                [ext_a.name(), ext_b.name(), ext_c.name(), ext_d.name()]
                    .into_iter()
                    .cloned()
                    .collect::<ExtensionSet>(),
            ),
        )
        .unwrap();
    let [func_i0, func_i1] = func.input_wires_arr();

    // Call the function declaration directly, and load & call indirectly.
    func.call(
        &decl,
        &[],
        vec![func_i0],
        &ExtensionRegistry::new([float_types::EXTENSION.to_owned()]),
    )
    .unwrap();
    let loaded_func = func
        .load_func(
            &decl,
            &[],
            &ExtensionRegistry::new([float_types::EXTENSION.to_owned()]),
        )
        .unwrap();
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

    // Dfg control flow.
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
    let mut hugr = module
        .finish_hugr(&build_extensions)
        .unwrap_or_else(|e| panic!("{e}"));

    // Check that the read-only methods collect the same extensions.
    let mut collected_exts = ExtensionRegistry::default();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        collected_exts.extend(collect_op_extensions(Some(node), op).unwrap());
        collected_exts.extend(collect_op_types_extensions(Some(node), op).unwrap());
    }
    assert_eq!(
        collected_exts, build_extensions,
        "{collected_exts} != {build_extensions}"
    );

    // Check that the mutable methods collect the same extensions.
    let resolved = hugr.resolve_extension_defs(&build_extensions).unwrap();
    assert_eq!(
        &resolved, &build_extensions,
        "{resolved} != {build_extensions}"
    );
}

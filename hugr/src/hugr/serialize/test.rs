use super::*;
use crate::builder::{
    test::closed_dfg_root_hugr, Container, DFGBuilder, Dataflow, DataflowHugr,
    DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::{BOOL_T, USIZE_T};
use crate::extension::simple_op::MakeRegisteredOp;
use crate::extension::{EMPTY_REG, PRELUDE_REGISTRY};
use crate::hugr::hugrmut::sealed::HugrMutInternals;
use crate::hugr::NodeType;
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::ops::{dataflow::IOTrait, Input, Module, Noop, Output, DFG};
use crate::std_extensions::arithmetic::float_ops::FLOAT_OPS_REGISTRY;
use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};

use crate::std_extensions::logic::NotOp;

use crate::types::{FunctionType, Type};
use crate::{type_row, OutgoingPort};
use itertools::Itertools;
use jsonschema::{Draft, JSONSchema};
use lazy_static::lazy_static;
use portgraph::LinkView;
use portgraph::{
    multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, PortView, UnmanagedDenseMap,
};

const NAT: Type = crate::extension::prelude::USIZE_T;
const QB: Type = crate::extension::prelude::QB_T;

type TestingModel = SerTestingV1;

macro_rules! include_schema {
    ($name:ident, $path:literal) => {
        lazy_static! {
            static ref $name: JSONSchema = {
                let schema_val: serde_json::Value =
                    serde_json::from_str(include_str!($path)).unwrap();
                JSONSchema::options()
                    .with_draft(Draft::Draft7)
                    .compile(&schema_val)
                    .expect("Schema is invalid.")
            };
        }
    };
}

include_schema!(
    SCHEMA,
    "../../../../specification/schema/hugr_schema_v1.json"
);
include_schema!(
    SCHEMA_STRICT,
    "../../../../specification/schema/hugr_schema_strict_v1.json"
);
include_schema!(
    TESTING_SCHEMA,
    "../../../../specification/schema/testing_hugr_schema_v1.json"
);
include_schema!(
    TESTING_SCHEMA_STRICT,
    "../../../../specification/schema/testing_hugr_schema_strict_v1.json"
);

macro_rules! impl_sertesting_from {
    ($typ:ty, $field:ident) => {
        #[cfg(test)]
        impl From<$typ> for TestingModel {
            fn from(v: $typ) -> Self {
                let mut r: Self = Default::default();
                r.$field = Some(v);
                r
            }
        }
    };
}

impl_sertesting_from!(crate::types::Type, typ);
impl_sertesting_from!(crate::types::SumType, sum_type);
impl_sertesting_from!(crate::types::PolyFuncType, poly_func_type);
impl_sertesting_from!(crate::ops::Value, value);
impl_sertesting_from!(NodeSer, optype);

#[test]
fn empty_hugr_serialize() {
    let hg = Hugr::default();
    assert_eq!(ser_roundtrip(&hg), hg);
}

/// Serialize and deserialize a value.
pub fn ser_roundtrip<T: Serialize + serde::de::DeserializeOwned>(g: &T) -> T {
    ser_roundtrip_validate(g, None)
}

/// Serialize and deserialize a value, optionally validating against a schema.
pub fn ser_roundtrip_validate<T: Serialize + serde::de::DeserializeOwned>(
    g: &T,
    schema: Option<&JSONSchema>,
) -> T {
    let s = serde_json::to_string(g).unwrap();
    let val: serde_json::Value = serde_json::from_str(&s).unwrap();

    if let Some(schema) = schema {
        let validate = schema.validate(&val);

        if let Err(errors) = validate {
            // errors don't necessarily implement Debug
            for error in errors {
                println!("Validation error: {}", error);
                println!("Instance path: {}", error.instance_path);
            }
            panic!("Serialization test failed.");
        }
    }
    serde_json::from_value(val).unwrap()
}

/// Serialize and deserialize a HUGR, and check that the result is the same as the original.
/// Checks the serialized json against the in-tree schema.
///
/// Returns the deserialized HUGR.
pub fn check_hugr_schema_roundtrip(hugr: &Hugr) -> Hugr {
    check_hugr_roundtrip(hugr, true)
}

/// Serialize and deserialize a HUGR, and check that the result is the same as the original.
///
/// If `check_schema` is true, checks the serialized json against the in-tree schema.
///
/// Returns the deserialized HUGR.
pub fn check_hugr_roundtrip(hugr: &Hugr, check_schema: bool) -> Hugr {
    let new_hugr: Hugr = ser_roundtrip_validate(hugr, check_schema.then_some(&SCHEMA));
    let new_hugr_strict: Hugr =
        ser_roundtrip_validate(hugr, check_schema.then_some(&SCHEMA_STRICT));
    assert_eq!(new_hugr, new_hugr_strict);

    // Original HUGR, with canonicalized node indices
    //
    // The internal port indices may still be different.
    let mut h_canon = hugr.clone();
    h_canon.canonicalize_nodes(|_, _| {});

    assert_eq!(new_hugr.root, h_canon.root);
    assert_eq!(new_hugr.hierarchy, h_canon.hierarchy);
    assert_eq!(new_hugr.metadata, h_canon.metadata);

    // Extension operations may have been downgraded to opaque operations.
    for node in new_hugr.nodes() {
        let new_op = new_hugr.get_optype(node);
        let old_op = h_canon.get_optype(node);
        assert_eq!(new_op, old_op);
    }

    // Check that the graphs are equivalent up to port renumbering.
    let new_graph = &new_hugr.graph;
    let old_graph = &h_canon.graph;
    assert_eq!(new_graph.node_count(), old_graph.node_count());
    assert_eq!(new_graph.port_count(), old_graph.port_count());
    assert_eq!(new_graph.link_count(), old_graph.link_count());
    for n in old_graph.nodes_iter() {
        assert_eq!(new_graph.num_inputs(n), old_graph.num_inputs(n));
        assert_eq!(new_graph.num_outputs(n), old_graph.num_outputs(n));
        assert_eq!(
            new_graph.output_neighbours(n).collect_vec(),
            old_graph.output_neighbours(n).collect_vec()
        );
    }

    new_hugr
}

#[allow(unused)]
fn check_testing_roundtrip(t: impl Into<TestingModel>) {
    let before = Versioned::new(t.into());
    let after_strict = ser_roundtrip_validate(&before, Some(&TESTING_SCHEMA_STRICT));
    let after = ser_roundtrip_validate(&before, Some(&TESTING_SCHEMA));
    assert_eq!(before, after);
    assert_eq!(after, after_strict);
}

/// Generate an optype for a node with a matching amount of inputs and outputs.
fn gen_optype(g: &MultiPortGraph, node: portgraph::NodeIndex) -> OpType {
    let inputs = g.num_inputs(node);
    let outputs = g.num_outputs(node);
    match (inputs == 0, outputs == 0) {
        (false, false) => DFG {
            signature: FunctionType::new(vec![NAT; inputs - 1], vec![NAT; outputs - 1]),
        }
        .into(),
        (true, false) => Input::new(vec![NAT; outputs - 1]).into(),
        (false, true) => Output::new(vec![NAT; inputs - 1]).into(),
        (true, true) => Module.into(),
    }
}

#[test]
fn simpleser() {
    let mut g = MultiPortGraph::new();

    let root = g.add_node(0, 0);
    let a = g.add_node(1, 1);
    let b = g.add_node(3, 2);
    let c = g.add_node(1, 1);

    g.link_nodes(a, 0, b, 0).unwrap();
    g.link_nodes(a, 0, b, 0).unwrap();
    g.link_nodes(b, 0, b, 1).unwrap();
    g.link_nodes(b, 1, c, 0).unwrap();
    g.link_nodes(b, 1, a, 0).unwrap();
    g.link_nodes(c, 0, a, 0).unwrap();

    let mut h = Hierarchy::new();
    let mut op_types = UnmanagedDenseMap::new();

    op_types[root] = NodeType::new_open(gen_optype(&g, root));

    for n in [a, b, c] {
        h.push_child(n, root).unwrap();
        op_types[n] = NodeType::new_pure(gen_optype(&g, n));
    }

    let hugr = Hugr {
        graph: g,
        hierarchy: h,
        root,
        op_types,
        metadata: Default::default(),
    };

    check_hugr_schema_roundtrip(&hugr);
}

#[test]
fn weighted_hugr_ser() {
    let hugr = {
        let mut module_builder = ModuleBuilder::new();
        module_builder.set_metadata("name", "test");

        let t_row = vec![Type::new_sum([type_row![NAT], type_row![QB]])];
        let mut f_build = module_builder
            .define_function("main", FunctionType::new(t_row.clone(), t_row).into())
            .unwrap();

        let outputs = f_build
            .input_wires()
            .map(|in_wire| {
                f_build
                    .add_dataflow_op(
                        Noop {
                            ty: f_build.get_wire_type(in_wire).unwrap(),
                        },
                        [in_wire],
                    )
                    .unwrap()
                    .out_wire(0)
            })
            .collect_vec();
        f_build.set_metadata("val", 42);
        f_build.finish_with_outputs(outputs).unwrap();

        module_builder.finish_prelude_hugr().unwrap()
    };

    check_hugr_schema_roundtrip(&hugr);
}

#[test]
fn dfg_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tp: Vec<Type> = vec![BOOL_T; 2];
    let mut dfg = DFGBuilder::new(FunctionType::new(tp.clone(), tp))?;
    let mut params: [_; 2] = dfg.input_wires_arr();
    for p in params.iter_mut() {
        *p = dfg
            .add_dataflow_op(Noop { ty: BOOL_T }, [*p])
            .unwrap()
            .out_wire(0);
    }
    let hugr = dfg.finish_hugr_with_outputs(params, &EMPTY_REG)?;

    check_hugr_schema_roundtrip(&hugr);
    Ok(())
}

#[test]
fn opaque_ops() -> Result<(), Box<dyn std::error::Error>> {
    let tp: Vec<Type> = vec![BOOL_T; 1];
    let mut dfg = DFGBuilder::new(FunctionType::new_endo(tp))?;
    let [wire] = dfg.input_wires_arr();

    // Add an extension operation
    let extension_op: ExtensionOp = NotOp.to_extension_op().unwrap();
    let wire = dfg
        .add_dataflow_op(extension_op.clone(), [wire])
        .unwrap()
        .out_wire(0);

    // Add an unresolved opaque operation
    let opaque_op: OpaqueOp = extension_op.into();
    let wire = dfg.add_dataflow_op(opaque_op, [wire]).unwrap().out_wire(0);

    let hugr = dfg.finish_hugr_with_outputs([wire], &PRELUDE_REGISTRY)?;

    check_hugr_schema_roundtrip(&hugr);
    Ok(())
}

#[test]
fn function_type() -> Result<(), Box<dyn std::error::Error>> {
    let fn_ty = Type::new_function(FunctionType::new_endo(type_row![BOOL_T]));
    let mut bldr = DFGBuilder::new(FunctionType::new_endo(vec![fn_ty.clone()]))?;
    let op = bldr.add_dataflow_op(Noop { ty: fn_ty }, bldr.input_wires())?;
    let h = bldr.finish_prelude_hugr_with_outputs(op.outputs())?;

    check_hugr_schema_roundtrip(&h);
    Ok(())
}

#[test]
fn hierarchy_order() -> Result<(), Box<dyn std::error::Error>> {
    let mut hugr = closed_dfg_root_hugr(FunctionType::new(vec![QB], vec![QB]));
    let [old_in, out] = hugr.get_io(hugr.root()).unwrap();
    hugr.connect(old_in, 0, out, 0);

    // Now add a new input
    let new_in = hugr.add_node(Input::new([QB].to_vec()).into());
    hugr.disconnect(old_in, OutgoingPort::from(0));
    hugr.connect(new_in, 0, out, 0);
    hugr.move_before_sibling(new_in, old_in);
    hugr.remove_node(old_in);
    hugr.update_validate(&PRELUDE_REGISTRY)?;

    let new_hugr: Hugr = check_hugr_schema_roundtrip(&hugr);
    new_hugr.validate(&EMPTY_REG).unwrap_err();
    new_hugr.validate(&PRELUDE_REGISTRY)?;
    Ok(())
}

#[test]
fn constants_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = DFGBuilder::new(FunctionType::new(vec![], vec![FLOAT64_TYPE])).unwrap();
    let w = builder.add_load_value(ConstF64::new(0.5));
    let hugr = builder.finish_hugr_with_outputs([w], &FLOAT_OPS_REGISTRY)?;

    let ser = serde_json::to_string(&hugr)?;
    let deser = serde_json::from_str(&ser)?;

    assert_eq!(hugr, deser);

    Ok(())
}

#[test]
fn serialize_types_roundtrip() {
    let g: Type = Type::new_function(FunctionType::new_endo(vec![]));

    assert_eq!(ser_roundtrip(&g), g);

    // A Simple tuple
    let t = Type::new_tuple(vec![USIZE_T, g]);
    assert_eq!(ser_roundtrip(&t), t);

    // A Classic sum
    let t = Type::new_sum([type_row![USIZE_T], type_row![FLOAT64_TYPE]]);
    assert_eq!(ser_roundtrip(&t), t);

    let t = Type::new_unit_sum(4);
    assert_eq!(ser_roundtrip(&t), t);
}

#[cfg(feature = "proptest")]
mod proptest {
    use super::super::NodeSer;
    use super::check_testing_roundtrip;
    use crate::extension::ExtensionSet;
    use crate::ops::{OpType, Value};
    use crate::types::{PolyFuncType, Type};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_roundtrip_type(t:  Type) {
            check_testing_roundtrip(t)
        }

        #[test]
        fn prop_roundtrip_poly_func_type(t: PolyFuncType) {
            check_testing_roundtrip(t)
        }

        #[test]
        fn prop_roundtrip_value(t: Value) {
            check_testing_roundtrip(t)
        }

        #[test]
        fn prop_roundtrip_optype(op in ((0..(std::u32::MAX / 2) as usize). prop_map(|x| portgraph::NodeIndex::new(x).into()), any::<Option<ExtensionSet>>(), any::<OpType>()).prop_map(|(parent, input_extensions, op)| NodeSer { parent, input_extensions, op })) {
            check_testing_roundtrip(op)
        }
    }
}

use super::*;
use crate::builder::{
    test::closed_dfg_root_hugr, Container, DFGBuilder, Dataflow, DataflowHugr,
    DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::{BOOL_T, PRELUDE_ID, QB_T, USIZE_T};
use crate::extension::simple_op::MakeRegisteredOp;
use crate::extension::{test::SimpleOpDef, EMPTY_REG, PRELUDE_REGISTRY};
use crate::hugr::internal::HugrMutInternals;
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::ops::{self, dataflow::IOTrait, Input, Module, Noop, Output, Value, DFG};
use crate::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use crate::std_extensions::arithmetic::int_ops::INT_OPS_REGISTRY;
use crate::std_extensions::arithmetic::int_types::{int_custom_type, ConstInt, INT_TYPES};
use crate::std_extensions::logic::NotOp;
use crate::types::{
    type_param::TypeParam, FunctionType, FunTypeVarArgs, PolyFuncType, SumType, Type, TypeArg, TypeBound,
};
use crate::{type_row, OutgoingPort};

use itertools::Itertools;
use jsonschema::{Draft, JSONSchema};
use lazy_static::lazy_static;
use portgraph::LinkView;
use portgraph::{multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, UnmanagedDenseMap};
use rstest::rstest;

const NAT: Type = crate::extension::prelude::USIZE_T;
const QB: Type = crate::extension::prelude::QB_T;

/// Version 1 of the Testing HUGR serialisation format, see `testing_hugr.py`.
#[derive(Serialize, Deserialize, PartialEq, Debug, Default)]
struct SerTestingV1 {
    typ: Option<crate::types::Type<true>>,
    sum_type: Option<crate::types::SumType>,
    poly_func_type: Option<crate::types::PolyFuncType>,
    value: Option<crate::ops::Value>,
    optype: Option<NodeSer>,
    op_def: Option<SimpleOpDef>,
}

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

impl_sertesting_from!(crate::types::Type<true>, typ);
impl_sertesting_from!(crate::types::SumType, sum_type);
impl_sertesting_from!(crate::types::PolyFuncType, poly_func_type);
impl_sertesting_from!(crate::ops::Value, value);
impl_sertesting_from!(NodeSer, optype);
impl_sertesting_from!(SimpleOpDef, op_def);

#[cfg(test)]
impl From<PolyFuncType<false>> for TestingModel {
    fn from(v: PolyFuncType<false>) -> Self {
        let mut r: Self = Default::default();
        r.poly_func_type = Some(v.into());
        r
    }
}


#[cfg(test)]
impl From<Type<false>> for TestingModel {
    fn from(v: Type<false>) -> Self {
        let mut r: Self = Default::default();
        r.typ = Some(v.into());
        r
    }
}

#[test]
fn empty_hugr_serialize() {
    check_hugr_roundtrip(&Hugr::default(), true);
}

/// Serialize and deserialize a value, optionally validating against a schema.
pub fn ser_serialize_check_schema<T: Serialize + serde::de::DeserializeOwned>(
    g: &T,
    schema: Option<&JSONSchema>,
) -> serde_json::Value {
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
    val
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
/// Note that we do not literally compare the before and after `Hugr`s for
/// equality, because impls of `CustomConst` are not required to implement
/// equality checking.
///
/// Returns the deserialized HUGR.
pub fn check_hugr_roundtrip(hugr: &Hugr, check_schema: bool) -> Hugr {
    let hugr_ser = ser_serialize_check_schema(hugr, check_schema.then_some(&SCHEMA));
    let _ = ser_serialize_check_schema(hugr, check_schema.then_some(&SCHEMA_STRICT));
    let new_hugr: Hugr = serde_json::from_value(hugr_ser).unwrap();
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
        if !new_op.is_const() {
            assert_eq!(new_op, old_op);
        }
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

fn check_testing_roundtrip(t: impl Into<TestingModel>) {
    let before = Versioned::new(t.into());
    let after_strict = serde_json::from_value(ser_serialize_check_schema(
        &before,
        Some(&TESTING_SCHEMA_STRICT),
    ))
    .unwrap();
    let after =
        serde_json::from_value(ser_serialize_check_schema(&before, Some(&TESTING_SCHEMA))).unwrap();
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
    let mut builder =
        DFGBuilder::new(FunctionType::new(vec![], vec![INT_TYPES[4].clone()])).unwrap();
    let w = builder.add_load_value(ConstInt::new_s(4, -2).unwrap());
    let hugr = builder.finish_hugr_with_outputs([w], &INT_OPS_REGISTRY)?;

    let ser = serde_json::to_string(&hugr)?;
    let deser = serde_json::from_str(&ser)?;

    assert_eq!(hugr, deser);

    Ok(())
}

#[test]
fn serialize_types_roundtrip() {
    let g: Type<true> = Type::new_function(FunctionType::new_endo(vec![]));

    check_testing_roundtrip(g.clone());

    // A Simple tuple
    let t: Type = Type::new_tuple(vec![USIZE_T.into(), g]);
    check_testing_roundtrip(t);

    // A Classic sum
    let t: Type<true> = Type::new_sum([type_row![USIZE_T], type_row![FLOAT64_TYPE]]);
    check_testing_roundtrip(t);

    let t: Type = Type::new_unit_sum(4);
    check_testing_roundtrip(t);
}

#[rstest]
#[case(BOOL_T)]
#[case(USIZE_T)]
#[case(INT_TYPES[2].clone())]
#[case(Type::new_alias(crate::ops::AliasDecl::new("t", TypeBound::Any)))]
#[case(Type::new_var_use(2, TypeBound::Copyable))]
#[case(Type::new_tuple(type_row![BOOL_T,QB_T]))]
#[case(Type::new_sum([type_row![BOOL_T,QB_T], type_row![Type::new_unit_sum(4)]]))]
#[case(Type::new_function(FunctionType::new_endo(type_row![QB_T,BOOL_T,USIZE_T])))]
fn roundtrip_type(#[case] typ: Type) {
    check_testing_roundtrip(typ);
}

#[rstest]
#[case(SumType::new_unary(2))]
#[case(SumType::new([type_row![USIZE_T, QB_T], type_row![]]))]
fn roundtrip_sumtype(#[case] sum_type: SumType) {
    check_testing_roundtrip(sum_type);
}

#[rstest]
#[case(Value::unit())]
#[case(Value::true_val())]
#[case(Value::unit_sum(3,5).unwrap())]
#[case(Value::extension(ConstInt::new_u(2,1).unwrap()))]
#[case(Value::sum(1,[Value::extension(ConstInt::new_u(2,1).unwrap())], SumType::new([vec![], vec![INT_TYPES[2].clone()]])).unwrap())]
#[case(Value::tuple([Value::false_val(), Value::extension(ConstInt::new_s(2,1).unwrap())]))]
#[case(Value::function(crate::builder::test::simple_dfg_hugr()).unwrap())]
fn roundtrip_value(#[case] value: Value) {
    check_testing_roundtrip(value);
}

fn polyfunctype1() -> PolyFuncType<false> {
    let mut extension_set = ExtensionSet::new();
    extension_set.insert_type_var(1);
    let function_type = FunctionType::new_endo(type_row![]).with_extension_delta(extension_set);
    PolyFuncType::new([TypeParam::max_nat(), TypeParam::Extensions], function_type)
}

fn polyfunctype2() -> PolyFuncType {
    let tv0 = Type::new_row_var_use(0, TypeBound::Any);
    let tv1 = Type::new_row_var_use(1, TypeBound::Eq);
    let params = [TypeBound::Any, TypeBound::Eq].map(TypeParam::new_list);
    let inputs = vec![
        Type::new_function(FunTypeVarArgs::new(tv0.clone(), tv1.clone())),
        tv0,
    ];
    let res = PolyFuncType::new(params, FunTypeVarArgs::new(inputs, tv1));
    // Just check we've got the arguments the right way round
    // (not that it really matters for the serialization schema we have)
    res.validate(&EMPTY_REG).unwrap();
    res
}

#[rstest]
#[case(FunctionType::new_endo(type_row![]).into())]
#[case(polyfunctype1())]
#[case(PolyFuncType::new([TypeParam::Opaque { ty: int_custom_type(TypeArg::BoundedNat { n: 1 }) }], FunctionType::new_endo(type_row![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncType::new([TypeBound::Eq.into()], FunctionType::new_endo(type_row![Type::new_var_use(0, TypeBound::Eq)])))]
#[case(PolyFuncType::new([TypeParam::new_list(TypeBound::Any)], FunctionType::new_endo(type_row![])))]
#[case(PolyFuncType::new([TypeParam::Tuple { params: [TypeBound::Any.into(), TypeParam::bounded_nat(2.try_into().unwrap())].into() }], FunctionType::new_endo(type_row![])))]
#[case(PolyFuncType::new(
    [TypeParam::new_list(TypeBound::Any)],
    FunctionType::new_endo(Type::new_tuple(Type::new_row_var_use(0, TypeBound::Any)))))]
fn roundtrip_polyfunctype_fixedlen(#[case] poly_func_type: PolyFuncType<false>) {
    check_testing_roundtrip(poly_func_type)
}

#[rstest]
#[case(FunTypeVarArgs::new_endo(type_row![]).into())]
#[case(PolyFuncType::new([TypeParam::Opaque { ty: int_custom_type(TypeArg::BoundedNat { n: 1 }) }], FunctionType::new_endo(type_row![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncType::new([TypeBound::Eq.into()], FunctionType::new_endo(type_row![Type::new_var_use(0, TypeBound::Eq)])))]
#[case(PolyFuncType::new([TypeParam::new_list(TypeBound::Any)], FunctionType::new_endo(type_row![])))]
#[case(PolyFuncType::new([TypeParam::Tuple { params: [TypeBound::Any.into(), TypeParam::bounded_nat(2.try_into().unwrap())].into() }], FunctionType::new_endo(type_row![])))]
#[case(PolyFuncType::new(
    [TypeParam::new_list(TypeBound::Any)],
    FunTypeVarArgs::new_endo(Type::new_row_var_use(0, TypeBound::Any))))]
#[case(polyfunctype2())]
fn roundtrip_polyfunctype_varlen(#[case] poly_func_type: PolyFuncType<true>) {
    check_testing_roundtrip(poly_func_type)
}

#[rstest]
#[case(ops::Module)]
#[case(ops::FuncDefn { name: "polyfunc1".into(), signature: polyfunctype1()})]
#[case(ops::FuncDecl { name: "polyfunc2".into(), signature: polyfunctype1()})]
#[case(ops::AliasDefn { name: "aliasdefn".into(), definition: Type::new_unit_sum(4)})]
#[case(ops::AliasDecl { name: "aliasdecl".into(), bound: TypeBound::Any})]
#[case(ops::Const::new(Value::false_val()))]
#[case(ops::Const::new(Value::function(crate::builder::test::simple_dfg_hugr()).unwrap()))]
#[case(ops::Input::new(type_row![Type::new_var_use(3,TypeBound::Eq)]))]
#[case(ops::Output::new(vec![Type::new_function(FunctionType::new_endo(type_row![]))]))]
#[case(ops::Call::try_new(polyfunctype1(), [TypeArg::BoundedNat{n: 1}, TypeArg::Extensions{ es: ExtensionSet::singleton(&PRELUDE_ID)} ], &EMPTY_REG).unwrap())]
#[case(ops::CallIndirect { signature : FunctionType::new_endo(type_row![BOOL_T]) })]
fn roundtrip_optype(#[case] optype: impl Into<OpType> + std::fmt::Debug) {
    check_testing_roundtrip(NodeSer {
        parent: portgraph::NodeIndex::new(0).into(),
        input_extensions: None,
        op: optype.into(),
    });
}

mod proptest {
    use super::check_testing_roundtrip;
    use super::{NodeSer, SimpleOpDef};
    use crate::extension::ExtensionSet;
    use crate::ops::{OpType, Value};
    use crate::types::{PolyFuncType, Type};
    use proptest::prelude::*;

    impl Arbitrary for NodeSer {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                (0..i32::MAX as usize).prop_map(|x| portgraph::NodeIndex::new(x).into()),
                any::<Option<ExtensionSet>>(),
                any::<OpType>(),
            )
                .prop_map(|(parent, input_extensions, op)| NodeSer {
                    parent,
                    input_extensions,
                    op,
                })
                .boxed()
        }
    }

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
        fn prop_roundtrip_optype(op: NodeSer ) {
            check_testing_roundtrip(op)
        }

        #[test]
        fn prop_roundtrip_opdef(opdef: SimpleOpDef) {
            check_testing_roundtrip(opdef)
        }
    }
}

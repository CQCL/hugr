//! Tests for the HUGR serialization format.

use super::*;
use crate::builder::{
    endo_sig, inout_sig, test::closed_dfg_root_hugr, Container, DFGBuilder, Dataflow, DataflowHugr,
    DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::Noop;
use crate::extension::prelude::{bool_t, qb_t, usize_t, PRELUDE_ID};
use crate::extension::simple_op::MakeRegisteredOp;
use crate::extension::ExtensionId;
use crate::extension::{test::SimpleOpDef, ExtensionSet, EMPTY_REG, PRELUDE_REGISTRY};
use crate::hugr::internal::HugrMutInternals;
use crate::hugr::validate::ValidationError;
use crate::hugr::ExtensionResolutionError;
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::ops::{self, dataflow::IOTrait, Input, Module, Output, Value, DFG};
use crate::std_extensions::arithmetic::float_types::float64_type;
use crate::std_extensions::arithmetic::int_ops::INT_OPS_REGISTRY;
use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
use crate::std_extensions::logic::LogicOp;
use crate::types::type_param::TypeParam;
use crate::types::{
    FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature, SumType, Type, TypeArg, TypeBound,
    TypeRV,
};
use crate::utils::test_quantum_extension;
use crate::{type_row, OutgoingPort};

use itertools::Itertools;
use jsonschema::{Draft, Validator};
use lazy_static::lazy_static;
use portgraph::LinkView;
use portgraph::{multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, UnmanagedDenseMap};
use rstest::rstest;

/// Version 1 of the Testing HUGR serialization format, see `testing_hugr.py`.
#[derive(Serialize, Deserialize, PartialEq, Debug, Default)]
struct SerTestingLatest {
    typ: Option<crate::types::TypeRV>,
    sum_type: Option<crate::types::SumType>,
    poly_func_type: Option<crate::types::PolyFuncTypeRV>,
    value: Option<crate::ops::Value>,
    optype: Option<NodeSer>,
    op_def: Option<SimpleOpDef>,
}

struct NamedSchema {
    name: &'static str,
    schema: Validator,
}

impl NamedSchema {
    pub fn new(name: &'static str, schema: Validator) -> Self {
        Self { name, schema }
    }

    pub fn check(&self, val: &serde_json::Value) {
        let mut errors = self.schema.iter_errors(val).peekable();
        if errors.peek().is_some() {
            // errors don't necessarily implement Debug
            eprintln!("Schema failed to validate: {}", self.name);
            for error in errors {
                eprintln!("Validation error: {}", error);
                eprintln!("Instance path: {}", error.instance_path);
            }
            panic!("Serialization test failed.");
        }
    }

    pub fn check_schemas(
        val: &serde_json::Value,
        schemas: impl IntoIterator<Item = &'static Self>,
    ) {
        for schema in schemas {
            schema.check(val);
        }
    }
}

macro_rules! include_schema {
    ($name:ident, $path:literal) => {
        lazy_static! {
            static ref $name: NamedSchema =
                NamedSchema::new("$name", {
                    let schema_val: serde_json::Value = serde_json::from_str(include_str!(
                        concat!("../../../../specification/schema/", $path, "_live.json")
                    ))
                    .unwrap();
                    Validator::options()
                        .with_draft(Draft::Draft7)
                        .build(&schema_val)
                        .expect("Schema is invalid.")
                });
        }
    };
}

include_schema!(SCHEMA, "hugr_schema");
include_schema!(SCHEMA_STRICT, "hugr_schema_strict");
include_schema!(TESTING_SCHEMA, "testing_hugr_schema");
include_schema!(TESTING_SCHEMA_STRICT, "testing_hugr_schema_strict");

fn get_schemas(b: bool) -> impl IntoIterator<Item = &'static NamedSchema> {
    let schemas: [&'static NamedSchema; 2] = [&SCHEMA, &SCHEMA_STRICT];
    b.then_some(schemas.into_iter()).into_iter().flatten()
}

fn get_testing_schemas(b: bool) -> impl IntoIterator<Item = &'static NamedSchema> {
    let schemas: Vec<&'static NamedSchema> = vec![&TESTING_SCHEMA, &TESTING_SCHEMA_STRICT];
    b.then_some(schemas.into_iter()).into_iter().flatten()
}

macro_rules! impl_sertesting_from {
    ($typ:ty, $field:ident) => {
        #[cfg(test)]
        impl From<$typ> for SerTestingLatest {
            fn from(v: $typ) -> Self {
                let mut r: Self = Default::default();
                r.$field = Some(v);
                r
            }
        }
    };
}

impl_sertesting_from!(crate::types::TypeRV, typ);
impl_sertesting_from!(crate::types::SumType, sum_type);
impl_sertesting_from!(crate::types::PolyFuncTypeRV, poly_func_type);
impl_sertesting_from!(crate::ops::Value, value);
impl_sertesting_from!(NodeSer, optype);
impl_sertesting_from!(SimpleOpDef, op_def);

impl From<PolyFuncType> for SerTestingLatest {
    fn from(v: PolyFuncType) -> Self {
        let v: PolyFuncTypeRV = v.into();
        v.into()
    }
}

impl From<Type> for SerTestingLatest {
    fn from(v: Type) -> Self {
        let t: TypeRV = v.into();
        t.into()
    }
}

#[test]
fn empty_hugr_serialize() {
    check_hugr_roundtrip(&Hugr::default(), true);
}

fn ser_deserialize_check_schema<T: serde::de::DeserializeOwned>(
    val: serde_json::Value,
    schemas: impl IntoIterator<Item = &'static NamedSchema>,
) -> T {
    NamedSchema::check_schemas(&val, schemas);
    serde_json::from_value(val).unwrap()
}

/// Serialize and deserialize a value, validating against a schema.
fn ser_roundtrip_check_schema<T: Serialize + serde::de::DeserializeOwned>(
    g: &T,
    schemas: impl IntoIterator<Item = &'static NamedSchema>,
) -> T {
    let val = serde_json::to_value(g).unwrap();
    NamedSchema::check_schemas(&val, schemas);
    serde_json::from_value(val).unwrap()
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
    let new_hugr = ser_roundtrip_check_schema(hugr, get_schemas(check_schema));

    check_hugr(hugr, &new_hugr);
    new_hugr
}

/// Deserialize a HUGR json, ensuring that it is valid against the schema.
pub fn check_hugr_deserialize(hugr: &Hugr, value: serde_json::Value, check_schema: bool) -> Hugr {
    let new_hugr = ser_deserialize_check_schema(value, get_schemas(check_schema));

    check_hugr(hugr, &new_hugr);
    new_hugr
}

/// Check that two HUGRs are equivalent, up to node renumbering.
pub fn check_hugr(lhs: &Hugr, rhs: &Hugr) {
    // Original HUGR, with canonicalized node indices
    //
    // The internal port indices may still be different.
    let mut h_canon = lhs.clone();
    h_canon.canonicalize_nodes(|_, _| {});

    assert_eq!(rhs.root, h_canon.root);
    assert_eq!(rhs.hierarchy, h_canon.hierarchy);
    assert_eq!(rhs.metadata, h_canon.metadata);

    // Extension operations may have been downgraded to opaque operations.
    for node in rhs.nodes() {
        let new_op = rhs.get_optype(node);
        let old_op = h_canon.get_optype(node);
        if !new_op.is_const() {
            match (new_op, old_op) {
                (OpType::ExtensionOp(ext), OpType::OpaqueOp(opaque))
                | (OpType::OpaqueOp(opaque), OpType::ExtensionOp(ext)) => {
                    let ext_opaque: OpaqueOp = ext.clone().into();
                    assert_eq!(ext_opaque, opaque.clone());
                }
                _ => assert_eq!(new_op, old_op),
            }
        }
    }

    // Check that the graphs are equivalent up to port renumbering.
    let new_graph = &rhs.graph;
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
}

fn check_testing_roundtrip(t: impl Into<SerTestingLatest>) {
    let before = Versioned::new_latest(t.into());
    let after = ser_roundtrip_check_schema(&before, get_testing_schemas(true));
    assert_eq!(before, after);
}

/// Generate an optype for a node with a matching amount of inputs and outputs.
fn gen_optype(g: &MultiPortGraph, node: portgraph::NodeIndex) -> OpType {
    let inputs = g.num_inputs(node);
    let outputs = g.num_outputs(node);
    match (inputs == 0, outputs == 0) {
        (false, false) => DFG {
            signature: Signature::new(vec![usize_t(); inputs - 1], vec![usize_t(); outputs - 1]),
        }
        .into(),
        (true, false) => Input::new(vec![usize_t(); outputs - 1]).into(),
        (false, true) => Output::new(vec![usize_t(); inputs - 1]).into(),
        (true, true) => Module::new().into(),
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

    op_types[root] = gen_optype(&g, root);

    for n in [a, b, c] {
        h.push_child(n, root).unwrap();
        op_types[n] = gen_optype(&g, n);
    }

    let hugr = Hugr {
        graph: g,
        hierarchy: h,
        root,
        op_types,
        metadata: Default::default(),
    };

    check_hugr_roundtrip(&hugr, true);
}

#[test]
fn weighted_hugr_ser() {
    let hugr = {
        let mut module_builder = ModuleBuilder::new();
        module_builder.set_metadata("name", "test");

        let t_row = vec![Type::new_sum([vec![usize_t()], vec![qb_t()]])];
        let mut f_build = module_builder
            .define_function("main", Signature::new(t_row.clone(), t_row).with_prelude())
            .unwrap();

        let outputs = f_build
            .input_wires()
            .map(|in_wire| {
                f_build
                    .add_dataflow_op(Noop(f_build.get_wire_type(in_wire).unwrap()), [in_wire])
                    .unwrap()
                    .out_wire(0)
            })
            .collect_vec();
        f_build.set_metadata("val", 42);
        f_build.finish_with_outputs(outputs).unwrap();

        module_builder.finish_prelude_hugr().unwrap()
    };

    check_hugr_roundtrip(&hugr, true);
}

#[test]
fn dfg_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let tp: Vec<Type> = vec![bool_t(); 2];
    let mut dfg = DFGBuilder::new(Signature::new(tp.clone(), tp).with_prelude())?;
    let mut params: [_; 2] = dfg.input_wires_arr();
    for p in params.iter_mut() {
        *p = dfg
            .add_dataflow_op(Noop(bool_t()), [*p])
            .unwrap()
            .out_wire(0);
    }
    let hugr = dfg.finish_hugr_with_outputs(params, &test_quantum_extension::REG)?;

    check_hugr_roundtrip(&hugr, true);
    Ok(())
}

#[test]
fn extension_ops() -> Result<(), Box<dyn std::error::Error>> {
    let tp: Vec<Type> = vec![bool_t(); 1];
    let mut dfg = DFGBuilder::new(endo_sig(tp))?;
    let [wire] = dfg.input_wires_arr();

    // Add an extension operation
    let extension_op: ExtensionOp = LogicOp::Not.to_extension_op().unwrap();
    let wire = dfg
        .add_dataflow_op(extension_op.clone(), [wire])
        .unwrap()
        .out_wire(0);

    let hugr = dfg.finish_hugr_with_outputs([wire], &test_quantum_extension::REG)?;

    check_hugr_roundtrip(&hugr, true);
    Ok(())
}

#[test]
fn opaque_ops() -> Result<(), Box<dyn std::error::Error>> {
    let tp: Vec<Type> = vec![bool_t(); 1];
    let mut dfg = DFGBuilder::new(endo_sig(tp))?;
    let [wire] = dfg.input_wires_arr();

    // Add an extension operation
    let extension_op: ExtensionOp = LogicOp::Not.to_extension_op().unwrap();
    let wire = dfg
        .add_dataflow_op(extension_op.clone(), [wire])
        .unwrap()
        .out_wire(0);
    let not_node = wire.node();

    // Add an unresolved opaque operation
    let opaque_op: OpaqueOp = extension_op.into();
    let ext_name = opaque_op.extension().clone();
    let wire = dfg.add_dataflow_op(opaque_op, [wire]).unwrap().out_wire(0);

    assert_eq!(
        dfg.finish_hugr_with_outputs([wire], &PRELUDE_REGISTRY),
        Err(ValidationError::ExtensionResolutionError(
            ExtensionResolutionError::MissingOpExtension {
                node: not_node,
                op: "logic.Not".into(),
                missing_extension: ext_name,
                available_extensions: vec![ExtensionId::new("prelude").unwrap()]
            }
        )
        .into())
    );

    Ok(())
}

#[test]
fn function_type() -> Result<(), Box<dyn std::error::Error>> {
    let fn_ty = Type::new_function(Signature::new_endo(vec![bool_t()]).with_prelude());
    let mut bldr = DFGBuilder::new(Signature::new_endo(vec![fn_ty.clone()]).with_prelude())?;
    let op = bldr.add_dataflow_op(Noop(fn_ty), bldr.input_wires())?;
    let h = bldr.finish_prelude_hugr_with_outputs(op.outputs())?;

    check_hugr_roundtrip(&h, true);
    Ok(())
}

#[test]
fn hierarchy_order() -> Result<(), Box<dyn std::error::Error>> {
    let mut hugr = closed_dfg_root_hugr(Signature::new(vec![qb_t()], vec![qb_t()]));
    let [old_in, out] = hugr.get_io(hugr.root()).unwrap();
    hugr.connect(old_in, 0, out, 0);

    // Now add a new input
    let new_in = hugr.add_node(Input::new([qb_t()].to_vec()).into());
    hugr.disconnect(old_in, OutgoingPort::from(0));
    hugr.connect(new_in, 0, out, 0);
    hugr.move_before_sibling(new_in, old_in);
    hugr.remove_node(old_in);
    hugr.update_validate(&PRELUDE_REGISTRY)?;

    let rhs: Hugr = check_hugr_roundtrip(&hugr, true);
    rhs.validate(&EMPTY_REG).unwrap_err();
    rhs.validate(&PRELUDE_REGISTRY)?;
    Ok(())
}

#[test]
fn constants_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = DFGBuilder::new(inout_sig(vec![], INT_TYPES[4].clone())).unwrap();
    let w = builder.add_load_value(ConstInt::new_s(4, -2).unwrap());
    let hugr = builder.finish_hugr_with_outputs([w], &INT_OPS_REGISTRY)?;

    let ser = serde_json::to_string(&hugr)?;
    let deser = serde_json::from_str(&ser)?;

    assert_eq!(hugr, deser);

    Ok(())
}

#[test]
fn serialize_types_roundtrip() {
    let g: Type = Type::new_function(Signature::new_endo(vec![]));
    check_testing_roundtrip(g.clone());

    // A Simple tuple
    let t = Type::new_tuple(vec![usize_t(), g]);
    check_testing_roundtrip(t);

    // A Classic sum
    let t = TypeRV::new_sum([vec![usize_t()], vec![float64_type()]]);
    check_testing_roundtrip(t);

    let t = Type::new_unit_sum(4);
    check_testing_roundtrip(t);
}

#[rstest]
#[case(bool_t())]
#[case(usize_t())]
#[case(INT_TYPES[2].clone())]
#[case(Type::new_alias(crate::ops::AliasDecl::new("t", TypeBound::Any)))]
#[case(Type::new_var_use(2, TypeBound::Copyable))]
#[case(Type::new_tuple(vec![bool_t(),qb_t()]))]
#[case(Type::new_sum([vec![bool_t(),qb_t()], vec![Type::new_unit_sum(4)]]))]
#[case(Type::new_function(Signature::new_endo(vec![qb_t(),bool_t(),usize_t()])))]
fn roundtrip_type(#[case] typ: Type) {
    check_testing_roundtrip(typ);
}

#[rstest]
#[case(SumType::new_unary(2))]
#[case(SumType::new([vec![usize_t(), qb_t()].into(), type_row![]]))]
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

fn polyfunctype1() -> PolyFuncType {
    let mut extension_set = ExtensionSet::new();
    extension_set.insert_type_var(1);
    let function_type = Signature::new_endo(type_row![]).with_extension_delta(extension_set);
    PolyFuncType::new([TypeParam::max_nat(), TypeParam::Extensions], function_type)
}

fn polyfunctype2() -> PolyFuncTypeRV {
    let tv0 = TypeRV::new_row_var_use(0, TypeBound::Any);
    let tv1 = TypeRV::new_row_var_use(1, TypeBound::Copyable);
    let params = [TypeBound::Any, TypeBound::Copyable].map(TypeParam::new_list);
    let inputs = vec![
        TypeRV::new_function(FuncValueType::new(tv0.clone(), tv1.clone())),
        tv0,
    ];
    let res = PolyFuncTypeRV::new(params, FuncValueType::new(inputs, tv1));
    // Just check we've got the arguments the right way round
    // (not that it really matters for the serialization schema we have)
    res.validate(&EMPTY_REG).unwrap();
    res
}

#[rstest]
#[case(Signature::new_endo(type_row![]).into())]
#[case(polyfunctype1())]
#[case(PolyFuncType::new([TypeParam::String], Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncType::new([TypeBound::Copyable.into()], Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncType::new([TypeParam::new_list(TypeBound::Any)], Signature::new_endo(type_row![])))]
#[case(PolyFuncType::new([TypeParam::Tuple { params: [TypeBound::Any.into(), TypeParam::bounded_nat(2.try_into().unwrap())].into() }], Signature::new_endo(type_row![])))]
#[case(PolyFuncType::new(
    [TypeParam::new_list(TypeBound::Any)],
    Signature::new_endo(Type::new_tuple(TypeRV::new_row_var_use(0, TypeBound::Any)))))]
fn roundtrip_polyfunctype_fixedlen(#[case] poly_func_type: PolyFuncType) {
    check_testing_roundtrip(poly_func_type)
}

#[rstest]
#[case(FuncValueType::new_endo(type_row![]).into())]
#[case(PolyFuncTypeRV::new([TypeParam::String], FuncValueType::new_endo(vec![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncTypeRV::new([TypeBound::Copyable.into()], FuncValueType::new_endo(vec![Type::new_var_use(0, TypeBound::Copyable)])))]
#[case(PolyFuncTypeRV::new([TypeParam::new_list(TypeBound::Any)], FuncValueType::new_endo(type_row![])))]
#[case(PolyFuncTypeRV::new([TypeParam::Tuple { params: [TypeBound::Any.into(), TypeParam::bounded_nat(2.try_into().unwrap())].into() }], FuncValueType::new_endo(type_row![])))]
#[case(PolyFuncTypeRV::new(
    [TypeParam::new_list(TypeBound::Any)],
    FuncValueType::new_endo(TypeRV::new_row_var_use(0, TypeBound::Any))))]
#[case(polyfunctype2())]
fn roundtrip_polyfunctype_varlen(#[case] poly_func_type: PolyFuncTypeRV) {
    check_testing_roundtrip(poly_func_type)
}

#[rstest]
#[case(ops::Module::new())]
#[case(ops::FuncDefn { name: "polyfunc1".into(), signature: polyfunctype1()})]
#[case(ops::FuncDecl { name: "polyfunc2".into(), signature: polyfunctype1()})]
#[case(ops::AliasDefn { name: "aliasdefn".into(), definition: Type::new_unit_sum(4)})]
#[case(ops::AliasDecl { name: "aliasdecl".into(), bound: TypeBound::Any})]
#[case(ops::Const::new(Value::false_val()))]
#[case(ops::Const::new(Value::function(crate::builder::test::simple_dfg_hugr()).unwrap()))]
#[case(ops::Input::new(vec![Type::new_var_use(3,TypeBound::Copyable)]))]
#[case(ops::Output::new(vec![Type::new_function(FuncValueType::new_endo(type_row![]))]))]
#[case(ops::Call::try_new(polyfunctype1(), [TypeArg::BoundedNat{n: 1}, TypeArg::Extensions{ es: ExtensionSet::singleton(&PRELUDE_ID)} ], &EMPTY_REG).unwrap())]
#[case(ops::CallIndirect { signature : Signature::new_endo(vec![bool_t()]) })]
fn roundtrip_optype(#[case] optype: impl Into<OpType> + std::fmt::Debug) {
    check_testing_roundtrip(NodeSer {
        parent: portgraph::NodeIndex::new(0).into(),
        op: optype.into(),
    });
}

#[test]
// test all standard extension serialisations are valid against scheme
fn std_extensions_valid() {
    let std_reg = crate::std_extensions::std_reg();
    for ext in std_reg {
        let val = serde_json::to_value(ext).unwrap();
        NamedSchema::check_schemas(&val, get_schemas(true));
        // check deserialises correctly, can't check equality because of custom binaries.
        let deser: crate::extension::Extension = serde_json::from_value(val.clone()).unwrap();
        assert_eq!(serde_json::to_value(deser).unwrap(), val);
    }
}

mod proptest {
    use super::check_testing_roundtrip;
    use super::{NodeSer, SimpleOpDef};
    use crate::ops::{OpType, OpaqueOp, Value};
    use crate::types::{PolyFuncTypeRV, Type};
    use proptest::prelude::*;

    impl Arbitrary for NodeSer {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                (0..i32::MAX as usize).prop_map(|x| portgraph::NodeIndex::new(x).into()),
                any::<OpType>(),
            )
                .prop_map(|(parent, op)| {
                    if let OpType::ExtensionOp(ext_op) = op {
                        let opaque: OpaqueOp = ext_op.into();
                        NodeSer {
                            parent,
                            op: opaque.into(),
                        }
                    } else {
                        NodeSer { parent, op }
                    }
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
        fn prop_roundtrip_poly_func_type(t: PolyFuncTypeRV) {
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

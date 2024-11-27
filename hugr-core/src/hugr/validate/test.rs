use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use cool_asserts::assert_matches;

use super::*;
use crate::builder::test::closed_dfg_root_hugr;
use crate::builder::{
    inout_sig, BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    FunctionBuilder, HugrBuilder, ModuleBuilder, SubContainer,
};
use crate::extension::prelude::Noop;
use crate::extension::prelude::{BOOL_T, PRELUDE, PRELUDE_ID, QB_T, USIZE_T};
use crate::extension::{Extension, ExtensionSet, TypeDefBound, EMPTY_REG, PRELUDE_REGISTRY};
use crate::hugr::internal::HugrMutInternals;
use crate::hugr::HugrMut;
use crate::ops::dataflow::IOTrait;
use crate::ops::handle::NodeHandle;
use crate::ops::{self, OpType, Value};
use crate::std_extensions::logic::test::{and_op, or_op};
use crate::std_extensions::logic::LogicOp;
use crate::std_extensions::logic::{self};
use crate::types::type_param::{TypeArg, TypeArgError};
use crate::types::{
    CustomType, FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature, Type, TypeBound, TypeRV,
    TypeRow,
};
use crate::{
    const_extension_ids, test_file, type_row, Direction, IncomingPort, Node, OutgoingPort,
};

const NAT: Type = crate::extension::prelude::USIZE_T;

/// Creates a hugr with a single function definition that copies a bit `copies` times.
///
/// Returns the hugr and the node index of the definition.
fn make_simple_hugr(copies: usize) -> (Hugr, Node) {
    let def_op: OpType = ops::FuncDefn {
        name: "main".into(),
        signature: Signature::new(type_row![BOOL_T], vec![BOOL_T; copies])
            .with_prelude()
            .into(),
    }
    .into();

    let mut b = Hugr::default();
    let root = b.root();

    let def = b.add_node_with_parent(root, def_op);
    let _ = add_df_children(&mut b, def, copies);

    (b, def)
}

/// Adds an input{BOOL_T}, copy{BOOL_T -> BOOL_T^copies}, and output{BOOL_T^copies} operation to a dataflow container.
///
/// Returns the node indices of each of the operations.
fn add_df_children(b: &mut Hugr, parent: Node, copies: usize) -> (Node, Node, Node) {
    let input = b.add_node_with_parent(parent, ops::Input::new(type_row![BOOL_T]));
    let output = b.add_node_with_parent(parent, ops::Output::new(vec![BOOL_T; copies]));
    let copy = b.add_node_with_parent(parent, Noop(BOOL_T));

    b.connect(input, 0, copy, 0);
    for i in 0..copies {
        b.connect(copy, 0, output, i);
    }

    (input, copy, output)
}

#[test]
fn invalid_root() {
    let mut b = Hugr::new(LogicOp::Not);
    let root = b.root();
    assert_eq!(b.validate(&PRELUDE_REGISTRY), Ok(()));

    // Change the number of ports in the root
    b.set_num_ports(root, 1, 0);
    assert_matches!(
        b.validate(&PRELUDE_REGISTRY),
        Err(ValidationError::WrongNumberOfPorts { node, .. }) => assert_eq!(node, root)
    );
    b.set_num_ports(root, 2, 2);

    // Connect it to itself
    b.connect(root, 0, root, 0);
    assert_matches!(
        b.validate(&PRELUDE_REGISTRY),
        Err(ValidationError::RootWithEdges { node, .. }) => assert_eq!(node, root)
    );
    b.disconnect(root, OutgoingPort::from(0));

    // Add another hierarchy root
    let module = b.add_node(ops::Module::new().into());
    assert_matches!(
        b.validate(&PRELUDE_REGISTRY),
        Err(ValidationError::NoParent { node }) => assert_eq!(node, module)
    );

    // Make the hugr root not a hierarchy root
    b.set_parent(root, module);
    assert_matches!(
        b.validate(&PRELUDE_REGISTRY),
        Err(ValidationError::RootNotRoot { node }) => assert_eq!(node, root)
    );

    // Fix the root
    b.root = module.pg_index();
    b.remove_node(root);
    assert_eq!(b.validate(&PRELUDE_REGISTRY), Ok(()));
}

#[test]
fn leaf_root() {
    let leaf_op: OpType = Noop(USIZE_T).into();

    let b = Hugr::new(leaf_op);
    assert_eq!(b.validate(&PRELUDE_REGISTRY), Ok(()));
}

#[test]
fn dfg_root() {
    let dfg_op: OpType = ops::DFG {
        signature: Signature::new_endo(type_row![BOOL_T]).with_prelude(),
    }
    .into();

    let mut b = Hugr::new(dfg_op);
    let root = b.root();
    add_df_children(&mut b, root, 1);
    assert_eq!(b.update_validate(&EMPTY_REG), Ok(()));
}

#[test]
fn simple_hugr() {
    let mut b = make_simple_hugr(2).0;
    assert_eq!(b.update_validate(&EMPTY_REG), Ok(()));
}

#[test]
/// General children restrictions.
fn children_restrictions() {
    let (mut b, def) = make_simple_hugr(2);
    let root = b.root();
    let (_input, copy, _output) = b
        .hierarchy
        .children(def.pg_index())
        .map_into()
        .collect_tuple()
        .unwrap();

    // Add a definition without children
    let def_sig = Signature::new(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]);
    let new_def = b.add_node_with_parent(
        root,
        ops::FuncDefn {
            signature: def_sig.into(),
            name: "main".into(),
        },
    );
    assert_matches!(
        b.update_validate(&EMPTY_REG),
        Err(ValidationError::ContainerWithoutChildren { node, .. }) => assert_eq!(node, new_def)
    );

    // Add children to the definition, but move it to be a child of the copy
    add_df_children(&mut b, new_def, 2);
    b.set_parent(new_def, copy);
    assert_matches!(
        b.update_validate(&EMPTY_REG),
        Err(ValidationError::NonContainerWithChildren { node, .. }) => assert_eq!(node, copy)
    );
    b.set_parent(new_def, root);

    // After moving the previous definition to a valid place,
    // add an input node to the module subgraph
    let new_input = b.add_node_with_parent(root, ops::Input::new(type_row![]));
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidParentOp { parent, child, .. }) => {assert_eq!(parent, root); assert_eq!(child, new_input)}
    );
}

#[test]
/// Validation errors in a dataflow subgraph.
fn df_children_restrictions() {
    let (mut b, def) = make_simple_hugr(2);
    let (_input, output, copy) = b
        .hierarchy
        .children(def.pg_index())
        .map_into()
        .collect_tuple()
        .unwrap();

    // Replace the output operation of the df subgraph with a copy
    b.replace_op(output, Noop(NAT)).unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidInitialChild { parent, .. }) => assert_eq!(parent, def)
    );

    // Revert it back to an output, but with the wrong number of ports
    b.replace_op(output, ops::Output::new(type_row![BOOL_T]))
        .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, output.pg_index())}
    );
    b.replace_op(output, ops::Output::new(type_row![BOOL_T, BOOL_T]))
        .unwrap();

    // After fixing the output back, replace the copy with an output op
    b.replace_op(copy, ops::Output::new(type_row![BOOL_T, BOOL_T]))
        .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalIOChildren { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, copy.pg_index())}
    );
}

#[test]
fn test_ext_edge() {
    let mut h = closed_dfg_root_hugr(
        Signature::new(type_row![BOOL_T, BOOL_T], type_row![BOOL_T])
            .with_extension_delta(TO_BE_INFERRED),
    );
    let [input, output] = h.get_io(h.root()).unwrap();

    // Nested DFG BOOL_T -> BOOL_T
    let sub_dfg = h.add_node_with_parent(
        h.root(),
        ops::DFG {
            signature: Signature::new_endo(type_row![BOOL_T]).with_extension_delta(TO_BE_INFERRED),
        },
    );
    // this Xor has its 2nd input unconnected
    let sub_op = {
        let sub_input = h.add_node_with_parent(sub_dfg, ops::Input::new(type_row![BOOL_T]));
        let sub_output = h.add_node_with_parent(sub_dfg, ops::Output::new(type_row![BOOL_T]));
        let sub_op = h.add_node_with_parent(sub_dfg, and_op());
        h.connect(sub_input, 0, sub_op, 0);
        h.connect(sub_op, 0, sub_output, 0);
        sub_op
    };

    h.connect(input, 0, sub_dfg, 0);
    h.connect(sub_dfg, 0, output, 0);

    assert_matches!(
        h.update_validate(&EMPTY_REG),
        Err(ValidationError::UnconnectedPort { .. })
    );

    h.connect(input, 1, sub_op, 1);
    assert_matches!(
        h.update_validate(&EMPTY_REG),
        Err(ValidationError::InterGraphEdgeError(
            InterGraphEdgeError::MissingOrderEdge { .. }
        ))
    );
    //Order edge. This will need metadata indicating its purpose.
    h.add_other_edge(input, sub_dfg);
    h.update_validate(&EMPTY_REG).unwrap();
}

#[test]
fn no_ext_edge_into_func() -> Result<(), Box<dyn std::error::Error>> {
    let b2b = Signature::new_endo(BOOL_T);
    let mut h = DFGBuilder::new(Signature::new(BOOL_T, Type::new_function(b2b.clone())))?;
    let [input] = h.input_wires_arr();

    let mut dfg = h.dfg_builder(Signature::new(vec![], Type::new_function(b2b.clone())), [])?;
    let mut func = dfg.define_function("AndWithOuter", b2b.clone())?;
    let [fn_input] = func.input_wires_arr();
    let and_op = func.add_dataflow_op(and_op(), [fn_input, input])?; // 'ext' edge
    let func = func.finish_with_outputs(and_op.outputs())?;
    let loadfn = dfg.load_func(func.handle(), &[], &EMPTY_REG)?;
    let dfg = dfg.finish_with_outputs([loadfn])?;
    let res = h.finish_hugr_with_outputs(dfg.outputs(), &EMPTY_REG);
    assert_eq!(
        res,
        Err(BuildError::InvalidHUGR(
            ValidationError::InterGraphEdgeError(InterGraphEdgeError::ValueEdgeIntoFunc {
                from: input.node(),
                from_offset: input.source().into(),
                to: and_op.node(),
                to_offset: IncomingPort::from(1).into(),
                func: func.node()
            })
        ))
    );
    Ok(())
}

#[test]
fn test_local_const() {
    let mut h =
        closed_dfg_root_hugr(Signature::new_endo(BOOL_T).with_extension_delta(TO_BE_INFERRED));
    let [input, output] = h.get_io(h.root()).unwrap();
    let and = h.add_node_with_parent(h.root(), and_op());
    h.connect(input, 0, and, 0);
    h.connect(and, 0, output, 0);
    assert_eq!(
        h.update_validate(&EMPTY_REG),
        Err(ValidationError::UnconnectedPort {
            node: and,
            port: IncomingPort::from(1).into(),
            port_kind: EdgeKind::Value(BOOL_T)
        })
    );
    let const_op: ops::Const = logic::EXTENSION
        .get_value(&logic::TRUE_NAME)
        .unwrap()
        .typed_value()
        .clone()
        .into();
    // Second input of Xor from a constant
    let cst = h.add_node_with_parent(h.root(), const_op);
    let lcst = h.add_node_with_parent(h.root(), ops::LoadConstant { datatype: BOOL_T });

    h.connect(cst, 0, lcst, 0);
    h.connect(lcst, 0, and, 1);
    assert_eq!(h.static_source(lcst), Some(cst));
    // There is no edge from Input to LoadConstant, but that's OK:
    h.update_validate(&EMPTY_REG).unwrap();
}

#[test]
fn dfg_with_cycles() {
    let mut h = closed_dfg_root_hugr(Signature::new(type_row![BOOL_T, BOOL_T], type_row![BOOL_T]));
    let [input, output] = h.get_io(h.root()).unwrap();
    let or = h.add_node_with_parent(h.root(), or_op());
    let not1 = h.add_node_with_parent(h.root(), LogicOp::Not);
    let not2 = h.add_node_with_parent(h.root(), LogicOp::Not);
    h.connect(input, 0, or, 0);
    h.connect(or, 0, not1, 0);
    h.connect(not1, 0, or, 1);
    h.connect(input, 1, not2, 0);
    h.connect(not2, 0, output, 0);
    // The graph contains a cycle:
    assert_matches!(h.validate(&EMPTY_REG), Err(ValidationError::NotADag { .. }));
}

fn identity_hugr_with_type(t: Type) -> (Hugr, Node) {
    let mut b = Hugr::default();
    let row: TypeRow = vec![t].into();

    let def = b.add_node_with_parent(
        b.root(),
        ops::FuncDefn {
            name: "main".into(),
            signature: Signature::new(row.clone(), row.clone()).into(),
        },
    );

    let input = b.add_node_with_parent(def, ops::Input::new(row.clone()));
    let output = b.add_node_with_parent(def, ops::Output::new(row));
    b.connect(input, 0, output, 0);
    (b, def)
}
#[test]
fn unregistered_extension() {
    let (mut h, def) = identity_hugr_with_type(USIZE_T);
    assert_eq!(
        h.validate(&EMPTY_REG),
        Err(ValidationError::SignatureError {
            node: def,
            cause: SignatureError::ExtensionNotFound(PRELUDE.name.clone())
        })
    );
    h.update_validate(&PRELUDE_REGISTRY).unwrap();
}

const_extension_ids! {
    const EXT_ID: ExtensionId = "MyExt";
}
#[test]
fn invalid_types() {
    let ext = Extension::new_test_arc(EXT_ID, |ext, extension_ref| {
        ext.add_type(
            "MyContainer".into(),
            vec![TypeBound::Copyable.into()],
            "".into(),
            TypeDefBound::any(),
            extension_ref,
        )
        .unwrap();
    });
    let reg = ExtensionRegistry::try_new([ext, PRELUDE.clone()]).unwrap();

    let validate_to_sig_error = |t: CustomType| {
        let (h, def) = identity_hugr_with_type(Type::new_extension(t));
        match h.validate(&reg) {
            Err(ValidationError::SignatureError { node, cause }) if node == def => cause,
            e => panic!(
                "Expected SignatureError at def node, got {}",
                match e {
                    Ok(()) => "Ok".to_owned(),
                    Err(e) => format!("{}", e),
                }
            ),
        }
    };

    let valid = Type::new_extension(CustomType::new(
        "MyContainer",
        vec![TypeArg::Type { ty: USIZE_T }],
        EXT_ID,
        TypeBound::Any,
    ));
    assert_eq!(
        identity_hugr_with_type(valid.clone())
            .0
            .update_validate(&reg),
        Ok(())
    );

    // valid is Any, so is not allowed as an element of an outer MyContainer.
    let element_outside_bound = CustomType::new(
        "MyContainer",
        vec![TypeArg::Type { ty: valid.clone() }],
        EXT_ID,
        TypeBound::Any,
    );
    assert_eq!(
        validate_to_sig_error(element_outside_bound),
        SignatureError::TypeArgMismatch(TypeArgError::TypeMismatch {
            param: TypeBound::Copyable.into(),
            arg: TypeArg::Type { ty: valid }
        })
    );

    let bad_bound = CustomType::new(
        "MyContainer",
        vec![TypeArg::Type { ty: USIZE_T }],
        EXT_ID,
        TypeBound::Copyable,
    );
    assert_eq!(
        validate_to_sig_error(bad_bound.clone()),
        SignatureError::WrongBound {
            actual: TypeBound::Copyable,
            expected: TypeBound::Any
        }
    );

    // bad_bound claims to be Copyable, which is valid as an element for the outer MyContainer.
    let nested = CustomType::new(
        "MyContainer",
        vec![TypeArg::Type {
            ty: Type::new_extension(bad_bound),
        }],
        EXT_ID,
        TypeBound::Any,
    );
    assert_eq!(
        validate_to_sig_error(nested),
        SignatureError::WrongBound {
            actual: TypeBound::Copyable,
            expected: TypeBound::Any
        }
    );

    let too_many_type_args = CustomType::new(
        "MyContainer",
        vec![TypeArg::Type { ty: USIZE_T }, TypeArg::BoundedNat { n: 3 }],
        EXT_ID,
        TypeBound::Any,
    );
    assert_eq!(
        validate_to_sig_error(too_many_type_args),
        SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(2, 1))
    );
}

#[test]
fn typevars_declared() -> Result<(), Box<dyn std::error::Error>> {
    // Base case
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Any.into()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    f.finish_prelude_hugr_with_outputs([w])?;
    // Type refers to undeclared variable
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Any.into()],
            Signature::new_endo(vec![Type::new_var_use(1, TypeBound::Any)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    assert!(f.finish_prelude_hugr_with_outputs([w]).is_err());
    // Variable declaration incorrectly copied to use site
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Any.into()],
            Signature::new_endo(vec![Type::new_var_use(1, TypeBound::Copyable)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    assert!(f.finish_prelude_hugr_with_outputs([w]).is_err());
    Ok(())
}

/// Test that nested FuncDefns cannot use Type Variables declared by enclosing FuncDefns
#[test]
fn nested_typevars() -> Result<(), Box<dyn std::error::Error>> {
    const OUTER_BOUND: TypeBound = TypeBound::Any;
    const INNER_BOUND: TypeBound = TypeBound::Copyable;
    fn build(t: Type) -> Result<Hugr, BuildError> {
        let mut outer = FunctionBuilder::new(
            "outer",
            PolyFuncType::new(
                [OUTER_BOUND.into()],
                Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
            ),
        )?;
        let inner = outer.define_function(
            "inner",
            PolyFuncType::new([INNER_BOUND.into()], Signature::new_endo(vec![t])),
        )?;
        let [w] = inner.input_wires_arr();
        inner.finish_with_outputs([w])?;
        let [w] = outer.input_wires_arr();
        outer.finish_prelude_hugr_with_outputs([w])
    }
    assert!(build(Type::new_var_use(0, INNER_BOUND)).is_ok());
    assert_matches!(
        build(Type::new_var_use(1, OUTER_BOUND)).unwrap_err(),
        BuildError::InvalidHUGR(ValidationError::SignatureError {
            cause: SignatureError::FreeTypeVar {
                idx: 1,
                num_decls: 1
            },
            ..
        })
    );
    assert_matches!(build(Type::new_var_use(0, OUTER_BOUND)).unwrap_err(),
        BuildError::InvalidHUGR(ValidationError::SignatureError { cause: SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached }, .. }) =>
        {assert_eq!(actual, INNER_BOUND.into()); assert_eq!(cached, OUTER_BOUND.into())});
    Ok(())
}

#[test]
fn no_polymorphic_consts() -> Result<(), Box<dyn std::error::Error>> {
    use crate::std_extensions::collections;
    const BOUND: TypeParam = TypeParam::Type {
        b: TypeBound::Copyable,
    };
    let list_of_var = Type::new_extension(
        collections::EXTENSION
            .get_type(&collections::LIST_TYPENAME)
            .unwrap()
            .instantiate(vec![TypeArg::new_var_use(0, BOUND)])?,
    );
    let reg = ExtensionRegistry::try_new([collections::EXTENSION.to_owned()]).unwrap();
    let mut def = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [BOUND],
            Signature::new(vec![], vec![list_of_var.clone()])
                .with_extension_delta(collections::EXTENSION_ID),
        ),
    )?;
    let empty_list = Value::extension(collections::ListValue::new_empty(Type::new_var_use(
        0,
        TypeBound::Copyable,
    )));
    let cst = def.add_load_const(empty_list);
    let res = def.finish_hugr_with_outputs([cst], &reg);
    assert_matches!(
        res.unwrap_err(),
        BuildError::InvalidHUGR(ValidationError::SignatureError {
            cause: SignatureError::FreeTypeVar {
                idx: 0,
                num_decls: 0
            },
            ..
        })
    );
    Ok(())
}

pub(crate) fn extension_with_eval_parallel() -> Arc<Extension> {
    let rowp = TypeParam::new_list(TypeBound::Any);
    Extension::new_test_arc(EXT_ID, |ext, extension_ref| {
        let inputs = TypeRV::new_row_var_use(0, TypeBound::Any);
        let outputs = TypeRV::new_row_var_use(1, TypeBound::Any);
        let evaled_fn = TypeRV::new_function(FuncValueType::new(inputs.clone(), outputs.clone()));
        let pf = PolyFuncTypeRV::new(
            [rowp.clone(), rowp.clone()],
            FuncValueType::new(vec![evaled_fn, inputs], outputs),
        );
        ext.add_op("eval".into(), "".into(), pf, extension_ref)
            .unwrap();

        let rv = |idx| TypeRV::new_row_var_use(idx, TypeBound::Any);
        let pf = PolyFuncTypeRV::new(
            [rowp.clone(), rowp.clone(), rowp.clone(), rowp.clone()],
            Signature::new(
                vec![
                    Type::new_function(FuncValueType::new(rv(0), rv(2))),
                    Type::new_function(FuncValueType::new(rv(1), rv(3))),
                ],
                Type::new_function(FuncValueType::new(vec![rv(0), rv(1)], vec![rv(2), rv(3)])),
            ),
        );
        ext.add_op("parallel".into(), "".into(), pf, extension_ref)
            .unwrap();
    })
}

#[test]
fn instantiate_row_variables() -> Result<(), Box<dyn std::error::Error>> {
    fn uint_seq(i: usize) -> TypeArg {
        vec![TypeArg::Type { ty: USIZE_T }; i].into()
    }
    let e = extension_with_eval_parallel();
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![
            Type::new_function(Signature::new(USIZE_T, vec![USIZE_T, USIZE_T])),
            USIZE_T,
        ], // inputs: function + its argument
        vec![USIZE_T; 4], // outputs (*2^2, three calls)
    ))?;
    let [func, int] = dfb.input_wires_arr();
    let eval = e.instantiate_extension_op("eval", [uint_seq(1), uint_seq(2)], &PRELUDE_REGISTRY)?;
    let [a, b] = dfb.add_dataflow_op(eval, [func, int])?.outputs_arr();
    let par = e.instantiate_extension_op(
        "parallel",
        [uint_seq(1), uint_seq(1), uint_seq(2), uint_seq(2)],
        &PRELUDE_REGISTRY,
    )?;
    let [par_func] = dfb.add_dataflow_op(par, [func, func])?.outputs_arr();
    let eval2 =
        e.instantiate_extension_op("eval", [uint_seq(2), uint_seq(4)], &PRELUDE_REGISTRY)?;
    let eval2 = dfb.add_dataflow_op(eval2, [par_func, a, b])?;
    dfb.finish_hugr_with_outputs(
        eval2.outputs(),
        &ExtensionRegistry::try_new([PRELUDE.clone(), e]).unwrap(),
    )?;
    Ok(())
}

fn seq1ty(t: TypeRV) -> TypeArg {
    TypeArg::Sequence {
        elems: vec![t.into()],
    }
}

#[test]
fn row_variables() -> Result<(), Box<dyn std::error::Error>> {
    let e = extension_with_eval_parallel();
    let tv = TypeRV::new_row_var_use(0, TypeBound::Any);
    let inner_ft = Type::new_function(FuncValueType::new_endo(tv.clone()));
    let ft_usz = Type::new_function(FuncValueType::new_endo(vec![tv.clone(), USIZE_T.into()]));
    let mut fb = FunctionBuilder::new(
        "id",
        PolyFuncType::new(
            [TypeParam::new_list(TypeBound::Any)],
            Signature::new(inner_ft.clone(), ft_usz).with_extension_delta(e.name.clone()),
        ),
    )?;
    // All the wires here are carrying higher-order Function values
    let [func_arg] = fb.input_wires_arr();
    let id_usz = {
        let bldr = fb.define_function("id_usz", Signature::new_endo(USIZE_T))?;
        let vals = bldr.input_wires();
        let inner_def = bldr.finish_with_outputs(vals)?;
        fb.load_func(inner_def.handle(), &[], &PRELUDE_REGISTRY)?
    };
    let par = e.instantiate_extension_op(
        "parallel",
        [tv.clone(), USIZE_T.into(), tv.clone(), USIZE_T.into()].map(seq1ty),
        &PRELUDE_REGISTRY,
    )?;
    let par_func = fb.add_dataflow_op(par, [func_arg, id_usz])?;
    fb.finish_hugr_with_outputs(
        par_func.outputs(),
        &ExtensionRegistry::try_new([PRELUDE.clone(), e]).unwrap(),
    )?;
    Ok(())
}

#[test]
fn test_polymorphic_call() -> Result<(), Box<dyn std::error::Error>> {
    let e = Extension::try_new_test_arc(EXT_ID, |ext, extension_ref| {
        let params: Vec<TypeParam> = vec![
            TypeBound::Any.into(),
            TypeParam::Extensions,
            TypeBound::Any.into(),
        ];
        let evaled_fn = Type::new_function(
            Signature::new(
                Type::new_var_use(0, TypeBound::Any),
                Type::new_var_use(2, TypeBound::Any),
            )
            .with_extension_delta(ExtensionSet::type_var(1)),
        );
        // Single-input/output version of the higher-order "eval" operation, with extension param.
        // Note the extension-delta of the eval node includes that of the input function.
        ext.add_op(
            "eval".into(),
            "".into(),
            PolyFuncTypeRV::new(
                params.clone(),
                Signature::new(
                    vec![evaled_fn, Type::new_var_use(0, TypeBound::Any)],
                    Type::new_var_use(2, TypeBound::Any),
                )
                .with_extension_delta(ExtensionSet::type_var(1)),
            ),
            extension_ref,
        )?;

        Ok(())
    })?;

    fn utou(e: impl Into<ExtensionSet>) -> Type {
        Type::new_function(Signature::new_endo(USIZE_T).with_extension_delta(e.into()))
    }

    let int_pair = Type::new_tuple(type_row![USIZE_T; 2]);
    // Root DFG: applies a function int--PRELUDE-->int to each element of a pair of two ints
    let mut d = DFGBuilder::new(inout_sig(
        vec![utou(PRELUDE_ID), int_pair.clone()],
        vec![int_pair.clone()],
    ))?;
    // ....by calling a function parametrized<extensions E> (int--e-->int, int_pair) -> int_pair
    let f = {
        let es = ExtensionSet::type_var(0);
        let mut f = d.define_function(
            "two_ints",
            PolyFuncType::new(
                vec![TypeParam::Extensions],
                Signature::new(vec![utou(es.clone()), int_pair.clone()], int_pair.clone())
                    .with_extension_delta(EXT_ID)
                    .with_prelude()
                    .with_extension_delta(es.clone()),
            ),
        )?;
        let [func, tup] = f.input_wires_arr();
        let mut c = f.conditional_builder(
            (vec![type_row![USIZE_T; 2]], tup),
            vec![],
            type_row![USIZE_T;2],
        )?;
        let mut cc = c.case_builder(0)?;
        let [i1, i2] = cc.input_wires_arr();
        let op = e.instantiate_extension_op(
            "eval",
            vec![USIZE_T.into(), TypeArg::Extensions { es }, USIZE_T.into()],
            &PRELUDE_REGISTRY,
        )?;
        let [f1] = cc.add_dataflow_op(op.clone(), [func, i1])?.outputs_arr();
        let [f2] = cc.add_dataflow_op(op, [func, i2])?.outputs_arr();
        cc.finish_with_outputs([f1, f2])?;
        let res = c.finish_sub_container()?.outputs();
        let tup = f.make_tuple(res)?;
        f.finish_with_outputs([tup])?
    };

    let reg = ExtensionRegistry::try_new([e, PRELUDE.clone()])?;
    let [func, tup] = d.input_wires_arr();
    let call = d.call(
        f.handle(),
        &[TypeArg::Extensions {
            es: ExtensionSet::singleton(&PRELUDE_ID),
        }],
        [func, tup],
        &reg,
    )?;
    let h = d.finish_hugr_with_outputs(call.outputs(), &reg)?;
    let call_ty = h.get_optype(call.node()).dataflow_signature().unwrap();
    let exp_fun_ty = Signature::new(vec![utou(PRELUDE_ID), int_pair.clone()], int_pair)
        .with_extension_delta(EXT_ID)
        .with_prelude();
    assert_eq!(call_ty, exp_fun_ty);
    Ok(())
}

#[test]
fn test_polymorphic_load() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = ModuleBuilder::new();
    let id = m.declare(
        "id",
        PolyFuncType::new(
            vec![TypeBound::Any.into()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
        ),
    )?;
    let sig = Signature::new(
        vec![],
        vec![Type::new_function(Signature::new_endo(vec![USIZE_T]))],
    );
    let mut f = m.define_function("main", sig)?;
    let l = f.load_func(&id, &[USIZE_T.into()], &PRELUDE_REGISTRY)?;
    f.finish_with_outputs([l])?;
    let _ = m.finish_prelude_hugr()?;
    Ok(())
}

#[test]
/// Validation errors in a controlflow subgraph.
fn cfg_children_restrictions() {
    let (mut b, def) = make_simple_hugr(1);
    let (_input, _output, copy) = b
        .hierarchy
        .children(def.pg_index())
        .map_into()
        .collect_tuple()
        .unwrap();
    // Write Extension annotations into the Hugr while it's still well-formed
    // enough for us to compute them
    b.validate(&EMPTY_REG).unwrap();
    b.replace_op(
        copy,
        ops::CFG {
            signature: Signature::new(type_row![BOOL_T], type_row![BOOL_T]),
        },
    )
    .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::ContainerWithoutChildren { .. })
    );
    let cfg = copy;

    // Construct a valid CFG, with one BasicBlock node and one exit node
    let block = b.add_node_with_parent(
        cfg,
        ops::DataflowBlock {
            inputs: type_row![BOOL_T],
            sum_rows: vec![type_row![]],
            other_outputs: type_row![BOOL_T],
            extension_delta: ExtensionSet::new(),
        },
    );
    let const_op: ops::Const = ops::Value::unit_sum(0, 1).unwrap().into();
    let tag_type = Type::new_unit_sum(1);
    {
        let input = b.add_node_with_parent(block, ops::Input::new(type_row![BOOL_T]));
        let output =
            b.add_node_with_parent(block, ops::Output::new(vec![tag_type.clone(), BOOL_T]));
        let tag_def = b.add_node_with_parent(b.root(), const_op);
        let tag = b.add_node_with_parent(block, ops::LoadConstant { datatype: tag_type });

        b.connect(tag_def, 0, tag, 0);
        b.add_other_edge(input, tag);
        b.connect(tag, 0, output, 0);
        b.connect(input, 0, output, 1);
    }
    let exit = b.add_node_with_parent(
        cfg,
        ops::ExitBlock {
            cfg_outputs: type_row![BOOL_T],
        },
    );
    b.add_other_edge(block, exit);
    assert_eq!(b.update_validate(&EMPTY_REG), Ok(()));

    // Test malformed errors

    // Add an internal exit node
    let exit2 = b.add_node_after(
        exit,
        ops::ExitBlock {
            cfg_outputs: type_row![BOOL_T],
        },
    );
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalExitChildren { child, .. }, .. })
            => {assert_eq!(parent, cfg); assert_eq!(child, exit2.pg_index())}
    );
    b.remove_node(exit2);

    // Change the types in the BasicBlock node to work on qubits instead of bits
    b.replace_op(
        cfg,
        ops::CFG {
            signature: Signature::new(type_row![QB_T], type_row![BOOL_T]),
        },
    )
    .unwrap();
    b.replace_op(
        block,
        ops::DataflowBlock {
            inputs: type_row![QB_T],
            sum_rows: vec![type_row![]],
            other_outputs: type_row![QB_T],
            extension_delta: ExtensionSet::new(),
        },
    )
    .unwrap();
    let mut block_children = b.hierarchy.children(block.pg_index());
    let block_input = block_children.next().unwrap().into();
    let block_output = block_children.next_back().unwrap().into();
    b.replace_op(block_input, ops::Input::new(type_row![QB_T]))
        .unwrap();
    b.replace_op(
        block_output,
        ops::Output::new(type_row![Type::new_unit_sum(1), QB_T]),
    )
    .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidEdges { parent, source: EdgeValidationError::CFGEdgeSignatureMismatch { .. }, .. })
            => assert_eq!(parent, cfg)
    );
}

#[test]
//          /->->\
//          |    |
// Entry -> Middle -> Exit
fn cfg_connections() -> Result<(), Box<dyn std::error::Error>> {
    use crate::builder::CFGBuilder;

    let mut hugr = CFGBuilder::new(Signature::new_endo(USIZE_T))?;
    let unary_pred = hugr.add_constant(Value::unary_unit_sum());
    let mut entry = hugr.simple_entry_builder_exts(type_row![USIZE_T], 1, ExtensionSet::new())?;
    let p = entry.load_const(&unary_pred);
    let ins = entry.input_wires();
    let entry = entry.finish_with_outputs(p, ins)?;

    let mut middle = hugr.simple_block_builder(Signature::new_endo(USIZE_T), 1)?;
    let p = middle.load_const(&unary_pred);
    let ins = middle.input_wires();
    let middle = middle.finish_with_outputs(p, ins)?;

    let exit = hugr.exit_block();
    hugr.branch(&entry, 0, &middle)?;
    hugr.branch(&middle, 0, &exit)?;
    let mut h = hugr.finish_hugr(&PRELUDE_REGISTRY)?;

    h.connect(middle.node(), 0, middle.node(), 0);
    assert_eq!(
        h.validate(&PRELUDE_REGISTRY),
        Err(ValidationError::TooManyConnections {
            node: middle.node(),
            port: Port::new(Direction::Outgoing, 0),
            port_kind: EdgeKind::ControlFlow
        })
    );
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn cfg_entry_io_bug() -> Result<(), Box<dyn std::error::Error>> {
    // load test file where input node of entry block has types in reversed
    // order compared to parent CFG node.
    let mut hugr: Hugr = serde_json::from_reader(BufReader::new(
        File::open(test_file!("issue-1189.json")).unwrap(),
    ))
    .unwrap();
    assert_matches!(
        hugr.update_validate(&PRELUDE_REGISTRY),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch{..}, .. })
            => assert_eq!(parent, hugr.root())
    );

    Ok(())
}

#[cfg(feature = "extension_inference")]
mod extension_tests {
    use self::ops::handle::{BasicBlockID, TailLoopID};
    use rstest::rstest;

    use super::*;
    use crate::builder::handle::Outputs;
    use crate::builder::{BlockBuilder, BuildHandle, CFGBuilder, DFGWrapper, TailLoopBuilder};
    use crate::extension::prelude::Lift;
    use crate::extension::prelude::PRELUDE_ID;
    use crate::extension::ExtensionSet;
    use crate::macros::const_extension_ids;
    use crate::Wire;
    const_extension_ids! {
        const XA: ExtensionId = "A";
        const XB: ExtensionId = "BOOL_EXT";
    }

    #[rstest]
    #[case::d1(|signature| ops::DFG {signature}.into())]
    #[case::f1(|sig: Signature| ops::FuncDefn {name: "foo".to_string(), signature: sig.into()}.into())]
    #[case::c1(|signature| ops::Case {signature}.into())]
    fn parent_extension_mismatch(
        #[case] parent_f: impl Fn(Signature) -> OpType,
        #[values(ExtensionSet::new(), XA.into())] parent_extensions: ExtensionSet,
    ) {
        // Child graph adds extension "XB", but the parent (in all cases)
        // declares a different delta, causing a mismatch.
        let parent =
            parent_f(Signature::new_endo(USIZE_T).with_extension_delta(parent_extensions.clone()));
        let mut hugr = Hugr::new(parent);

        let input = hugr.add_node_with_parent(
            hugr.root(),
            ops::Input {
                types: type_row![USIZE_T],
            },
        );
        let output = hugr.add_node_with_parent(
            hugr.root(),
            ops::Output {
                types: type_row![USIZE_T],
            },
        );

        let lift = hugr.add_node_with_parent(hugr.root(), Lift::new(type_row![USIZE_T], XB));

        hugr.connect(input, 0, lift, 0);
        hugr.connect(lift, 0, output, 0);

        let result = hugr.validate(&PRELUDE_REGISTRY);
        assert_eq!(
            result,
            Err(ValidationError::ExtensionError(ExtensionError {
                parent: hugr.root(),
                parent_extensions,
                child: lift,
                child_extensions: ExtensionSet::from_iter([PRELUDE_ID, XB]),
            }))
        );
    }

    #[rstest]
    #[case(XA.into(), false)]
    #[case(ExtensionSet::new(), false)]
    #[case(ExtensionSet::from_iter([XA, XB]), true)]
    fn cfg_extension_mismatch(
        #[case] parent_extensions: ExtensionSet,
        #[case] success: bool,
    ) -> Result<(), BuildError> {
        let mut cfg = CFGBuilder::new(
            Signature::new_endo(USIZE_T).with_extension_delta(parent_extensions.clone()),
        )?;
        let mut bb = cfg.simple_entry_builder_exts(USIZE_T.into(), 1, XB)?;
        let pred = bb.add_load_value(Value::unary_unit_sum());
        let inputs = bb.input_wires();
        let blk = bb.finish_with_outputs(pred, inputs)?;
        let exit = cfg.exit_block();
        cfg.branch(&blk, 0, &exit)?;
        let root = cfg.hugr().root();
        let res = cfg.finish_prelude_hugr();
        if success {
            assert!(res.is_ok())
        } else {
            assert_eq!(
                res,
                Err(ValidationError::ExtensionError(ExtensionError {
                    parent: root,
                    parent_extensions,
                    child: blk.node(),
                    child_extensions: XB.into()
                }))
            );
        }
        Ok(())
    }

    #[rstest]
    #[case(XA.into(), false)]
    #[case(ExtensionSet::new(), false)]
    #[case(ExtensionSet::from_iter([XA, XB, PRELUDE_ID]), true)]
    fn conditional_extension_mismatch(
        #[case] parent_extensions: ExtensionSet,
        #[case] success: bool,
    ) {
        // Child graph adds extension "XB", but the parent
        // declares a different delta, in same cases causing a mismatch.
        let parent = ops::Conditional {
            sum_rows: vec![type_row![], type_row![]],
            other_inputs: type_row![USIZE_T],
            outputs: type_row![USIZE_T],
            extension_delta: parent_extensions.clone(),
        };
        let mut hugr = Hugr::new(parent);

        // First case with no delta should be ok in all cases. Second one may not be.
        let [_, child] = [None, Some(XB)].map(|case_ext| {
            let case_exts = if let Some(ex) = &case_ext {
                ExtensionSet::from_iter([ex.clone(), PRELUDE_ID])
            } else {
                ExtensionSet::new()
            };
            let case = hugr.add_node_with_parent(
                hugr.root(),
                ops::Case {
                    signature: Signature::new_endo(USIZE_T).with_extension_delta(case_exts),
                },
            );

            let input = hugr.add_node_with_parent(
                case,
                ops::Input {
                    types: type_row![USIZE_T],
                },
            );
            let output = hugr.add_node_with_parent(
                case,
                ops::Output {
                    types: type_row![USIZE_T],
                },
            );
            let res = match case_ext {
                None => input,
                Some(new_ext) => {
                    let lift =
                        hugr.add_node_with_parent(case, Lift::new(type_row![USIZE_T], new_ext));
                    hugr.connect(input, 0, lift, 0);
                    lift
                }
            };
            hugr.connect(res, 0, output, 0);
            case
        });
        // case is the last-assigned child, i.e. the one that requires 'XB'
        let result = hugr.validate(&PRELUDE_REGISTRY);
        let expected = if success {
            Ok(())
        } else {
            Err(ValidationError::ExtensionError(ExtensionError {
                parent: hugr.root(),
                parent_extensions,
                child,
                child_extensions: ExtensionSet::from_iter([XB, PRELUDE_ID]),
            }))
        };
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(make_bb, |bb: &mut DFGWrapper<_,_>, outs| bb.make_tuple(outs))]
    #[case(make_tailloop, |tl: &mut DFGWrapper<_,_>, outs| tl.make_break(tl.loop_signature().unwrap().clone(), outs))]
    fn bb_extension_mismatch<T>(
        #[case] dfg_fn: impl Fn(Type, ExtensionSet) -> DFGWrapper<Hugr, T>,
        #[case] make_pred: impl Fn(&mut DFGWrapper<Hugr, T>, Outputs) -> Result<Wire, BuildError>,
        #[values((ExtensionSet::from_iter([XA,PRELUDE_ID]), false), (PRELUDE_ID.into(), false), (ExtensionSet::from_iter([XA,XB,PRELUDE_ID]), true))]
        parent_exts_success: (ExtensionSet, bool),
    ) -> Result<(), BuildError> {
        let (parent_extensions, success) = parent_exts_success;
        let mut dfg = dfg_fn(USIZE_T, parent_extensions.clone());
        let lift = dfg.add_dataflow_op(Lift::new(USIZE_T.into(), XB), dfg.input_wires())?;
        let pred = make_pred(&mut dfg, lift.outputs())?;
        let root = dfg.hugr().root();
        let res = dfg.finish_prelude_hugr_with_outputs([pred]);
        if success {
            assert!(res.is_ok())
        } else {
            assert_eq!(
                res,
                Err(BuildError::InvalidHUGR(ValidationError::ExtensionError(
                    ExtensionError {
                        parent: root,
                        parent_extensions,
                        child: lift.node(),
                        child_extensions: ExtensionSet::from_iter([XB, PRELUDE_ID])
                    }
                )))
            );
        }
        Ok(())
    }

    fn make_bb(t: Type, es: ExtensionSet) -> DFGWrapper<Hugr, BasicBlockID> {
        BlockBuilder::new_exts(t.clone(), vec![t.into()], type_row![], es).unwrap()
    }

    fn make_tailloop(t: Type, es: ExtensionSet) -> DFGWrapper<Hugr, BuildHandle<TailLoopID>> {
        let row = TypeRow::from(t);
        TailLoopBuilder::new_exts(row.clone(), type_row![], row, es).unwrap()
    }
}

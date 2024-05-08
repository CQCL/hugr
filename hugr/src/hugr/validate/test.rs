use cool_asserts::assert_matches;

use super::*;
use crate::builder::test::closed_dfg_root_hugr;
use crate::builder::{
    BuildError, Container, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder,
    HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::{BOOL_T, PRELUDE, USIZE_T};
use crate::extension::{Extension, ExtensionId, TypeDefBound, EMPTY_REG, PRELUDE_REGISTRY};
use crate::hugr::hugrmut::sealed::HugrMutInternals;
use crate::hugr::HugrMut;
use crate::ops::dataflow::IOTrait;
use crate::ops::{self, Noop, Value};
use crate::std_extensions::logic::test::{and_op, or_op};
use crate::std_extensions::logic::{self, NotOp};
use crate::types::type_param::{TypeArg, TypeArgError};
use crate::types::{CustomType, FunctionType, PolyFuncType, Type, TypeBound, TypeRow};
use crate::{type_row, IncomingPort};

const NAT: Type = crate::extension::prelude::USIZE_T;

/// Creates a hugr with a single function definition that copies a bit `copies` times.
///
/// Returns the hugr and the node index of the definition.
fn make_simple_hugr(copies: usize) -> (Hugr, Node) {
    let def_op: OpType = ops::FuncDefn {
        name: "main".into(),
        signature: FunctionType::new(type_row![BOOL_T], vec![BOOL_T; copies]).into(),
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
    let copy = b.add_node_with_parent(parent, Noop { ty: BOOL_T });

    b.connect(input, 0, copy, 0);
    for i in 0..copies {
        b.connect(copy, 0, output, i);
    }

    (input, copy, output)
}

#[test]
fn invalid_root() {
    let declare_op: OpType = ops::FuncDecl {
        name: "main".into(),
        signature: Default::default(),
    }
    .into();

    let mut b = Hugr::default();
    let root = b.root();
    assert_eq!(b.validate(&EMPTY_REG), Ok(()));

    // Add another hierarchy root
    let other = b.add_node(ops::Module.into());
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::NoParent { node }) => assert_eq!(node, other)
    );
    b.set_parent(other, root);
    b.replace_op(other, NodeType::new_pure(declare_op)).unwrap();
    b.add_ports(other, Direction::Outgoing, 1);
    assert_eq!(b.validate(&EMPTY_REG), Ok(()));

    // Make the hugr root not a hierarchy root
    {
        let mut hugr = b.clone();
        hugr.root = other.pg_index();
        assert_matches!(
            hugr.validate(&EMPTY_REG),
            Err(ValidationError::RootNotRoot { node }) => assert_eq!(node, other)
        );
    }
}

#[test]
fn leaf_root() {
    let leaf_op: OpType = Noop { ty: USIZE_T }.into();

    let b = Hugr::new(NodeType::new_pure(leaf_op));
    assert_eq!(b.validate(&EMPTY_REG), Ok(()));
}

#[test]
fn dfg_root() {
    let dfg_op: OpType = ops::DFG {
        signature: FunctionType::new_endo(type_row![BOOL_T]),
    }
    .into();

    let mut b = Hugr::new(NodeType::new_pure(dfg_op));
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
    let def_sig = FunctionType::new(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]);
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
    b.infer_extensions().unwrap();
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
    b.replace_op(output, NodeType::new_pure(Noop { ty: NAT }))
        .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidInitialChild { parent, .. }) => assert_eq!(parent, def)
    );

    // Revert it back to an output, but with the wrong number of ports
    b.replace_op(
        output,
        NodeType::new_pure(ops::Output::new(type_row![BOOL_T])),
    )
    .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, output.pg_index())}
    );
    b.replace_op(
        output,
        NodeType::new_pure(ops::Output::new(type_row![BOOL_T, BOOL_T])),
    )
    .unwrap();

    // After fixing the output back, replace the copy with an output op
    b.replace_op(
        copy,
        NodeType::new_pure(ops::Output::new(type_row![BOOL_T, BOOL_T])),
    )
    .unwrap();
    assert_matches!(
        b.validate(&EMPTY_REG),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalIOChildren { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, copy.pg_index())}
    );
}

#[test]
fn test_ext_edge() {
    let mut h = closed_dfg_root_hugr(FunctionType::new(
        type_row![BOOL_T, BOOL_T],
        type_row![BOOL_T],
    ));
    let [input, output] = h.get_io(h.root()).unwrap();

    // Nested DFG BOOL_T -> BOOL_T
    let sub_dfg = h.add_node_with_parent(
        h.root(),
        ops::DFG {
            signature: FunctionType::new_endo(type_row![BOOL_T]),
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
fn test_local_const() {
    let mut h = closed_dfg_root_hugr(FunctionType::new(type_row![BOOL_T], type_row![BOOL_T]));
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
    let mut h = closed_dfg_root_hugr(FunctionType::new(
        type_row![BOOL_T, BOOL_T],
        type_row![BOOL_T],
    ));
    let [input, output] = h.get_io(h.root()).unwrap();
    let or = h.add_node_with_parent(h.root(), or_op());
    let not1 = h.add_node_with_parent(h.root(), NotOp);
    let not2 = h.add_node_with_parent(h.root(), NotOp);
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
            signature: FunctionType::new(row.clone(), row.clone()).into(),
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

#[test]
fn invalid_types() {
    let name: ExtensionId = "MyExt".try_into().unwrap();
    let mut e = Extension::new(name.clone());
    e.add_type(
        "MyContainer".into(),
        vec![TypeBound::Copyable.into()],
        "".into(),
        TypeDefBound::Explicit(TypeBound::Any),
    )
    .unwrap();
    let reg = ExtensionRegistry::try_new([e, PRELUDE.to_owned()]).unwrap();

    let validate_to_sig_error = |t: CustomType| {
        let (h, def) = identity_hugr_with_type(Type::new_extension(t));
        match h.validate(&reg) {
            Err(ValidationError::SignatureError { node, cause }) if node == def => cause,
            e => panic!("Expected SignatureError at def node, got {:?}", e),
        }
    };

    let valid = Type::new_extension(CustomType::new(
        "MyContainer",
        vec![TypeArg::Type { ty: USIZE_T }],
        name.clone(),
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
        name.clone(),
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
        name.clone(),
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
        name.clone(),
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
        name.clone(),
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
            FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    f.finish_prelude_hugr_with_outputs([w])?;
    // Type refers to undeclared variable
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Any.into()],
            FunctionType::new_endo(vec![Type::new_var_use(1, TypeBound::Any)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    assert!(f.finish_prelude_hugr_with_outputs([w]).is_err());
    // Variable declaration incorrectly copied to use site
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Any.into()],
            FunctionType::new_endo(vec![Type::new_var_use(1, TypeBound::Copyable)]),
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
                FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
            ),
        )?;
        let inner = outer.define_function(
            "inner",
            PolyFuncType::new([INNER_BOUND.into()], FunctionType::new_endo(vec![t])),
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
            FunctionType::new(vec![], vec![list_of_var.clone()])
                .with_extension_delta(collections::EXTENSION_NAME),
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

#[test]
fn test_polymorphic_call() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = ModuleBuilder::new();
    let id = m.declare(
        "id",
        PolyFuncType::new(
            vec![TypeBound::Any.into()],
            FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
        ),
    )?;
    let mut f = m.define_function("main", FunctionType::new_endo(vec![USIZE_T]).into())?;
    let c = f.call(&id, &[USIZE_T.into()], f.input_wires(), &PRELUDE_REGISTRY)?;
    f.finish_with_outputs(c.outputs())?;
    let _ = m.finish_prelude_hugr()?;
    Ok(())
}

#[test]
fn test_polymorphic_load() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = ModuleBuilder::new();
    let id = m.declare(
        "id",
        PolyFuncType::new(
            vec![TypeBound::Any.into()],
            FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
        ),
    )?;
    let sig = FunctionType::new(
        vec![],
        vec![Type::new_function(FunctionType::new_endo(vec![USIZE_T]))],
    );
    let mut f = m.define_function("main", sig.into())?;
    let l = f.load_func(&id, &[USIZE_T.into()], &PRELUDE_REGISTRY)?;
    f.finish_with_outputs([l])?;
    let _ = m.finish_prelude_hugr()?;
    Ok(())
}

#[cfg(feature = "extension_inference")]
mod extension_tests {
    use super::*;
    use crate::extension::ExtensionSet;
    use crate::macros::const_extension_ids;

    const_extension_ids! {
        const XA: ExtensionId = "A";
        const XB: ExtensionId = "BOOL_EXT";
    }

    const Q: Type = crate::extension::prelude::QB_T;

    /// Adds an input{BOOL_T}, tag_constant(0, BOOL_T^sum_size), tag(BOOL_T^sum_size), and
    /// output{Sum{unit^sum_size}, BOOL_T} operation to a dataflow container.
    /// Intended to be used to populate a BasicBlock node in a CFG.
    ///
    /// Returns the node indices of each of the operations.
    fn add_block_children(b: &mut Hugr, parent: Node, sum_size: usize) -> (Node, Node, Node, Node) {
        let const_op: ops::Const = ops::Value::unit_sum(0, sum_size as u8)
            .expect("`sum_size` must be greater than 0")
            .into();
        let tag_type = Type::new_unit_sum(sum_size as u8);

        let input = b.add_node_with_parent(parent, ops::Input::new(type_row![BOOL_T]));
        let output =
            b.add_node_with_parent(parent, ops::Output::new(vec![tag_type.clone(), BOOL_T]));
        let tag_def = b.add_node_with_parent(b.root(), const_op);
        let tag = b.add_node_with_parent(parent, ops::LoadConstant { datatype: tag_type });

        b.connect(tag_def, 0, tag, 0);
        b.add_other_edge(input, tag);
        b.connect(tag, 0, output, 0);
        b.connect(input, 0, output, 1);

        (input, tag_def, tag, output)
    }

    #[test]
    /// Validation errors in a dataflow subgraph.
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
        b.infer_extensions().unwrap();
        b.validate(&EMPTY_REG).unwrap();
        b.replace_op(
            copy,
            NodeType::new_pure(ops::CFG {
                signature: FunctionType::new(type_row![BOOL_T], type_row![BOOL_T]),
            }),
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
        add_block_children(&mut b, block, 1);
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
            block,
            NodeType::new_pure(ops::DataflowBlock {
                inputs: type_row![Q],
                sum_rows: vec![type_row![]],
                other_outputs: type_row![Q],
                extension_delta: ExtensionSet::new(),
            }),
        )
        .unwrap();
        let mut block_children = b.hierarchy.children(block.pg_index());
        let block_input = block_children.next().unwrap().into();
        let block_output = block_children.next_back().unwrap().into();
        b.replace_op(
            block_input,
            NodeType::new_pure(ops::Input::new(type_row![Q])),
        )
        .unwrap();
        b.replace_op(
            block_output,
            NodeType::new_pure(ops::Output::new(type_row![Type::new_unit_sum(1), Q])),
        )
        .unwrap();
        assert_matches!(
            b.validate(&EMPTY_REG),
            Err(ValidationError::InvalidEdges { parent, source: EdgeValidationError::CFGEdgeSignatureMismatch { .. }, .. })
                => assert_eq!(parent, cfg)
        );
    }

    #[test]
    fn parent_io_mismatch() {
        // The DFG node declares that it has an empty extension delta,
        // but it's child graph adds extension "XB", causing a mismatch.
        let mut hugr = Hugr::new(NodeType::new_pure(ops::DFG {
            signature: FunctionType::new(type_row![USIZE_T], type_row![USIZE_T]),
        }));

        let input = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::new_pure(ops::Input {
                types: type_row![USIZE_T],
            }),
        );
        let output = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::new(
                ops::Output {
                    types: type_row![USIZE_T],
                },
                Some(XB.into()),
            ),
        );

        let lift = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::new_pure(ops::Lift {
                type_row: type_row![USIZE_T],
                new_extension: XB,
            }),
        );

        hugr.connect(input, 0, lift, 0);
        hugr.connect(lift, 0, output, 0);

        let result = hugr.validate(&PRELUDE_REGISTRY);
        assert_matches!(
            result,
            Err(ValidationError::ExtensionError(
                ExtensionError::ParentIOExtensionMismatch { .. }
            ))
        );
    }

    #[test]
    /// A wire with no extension requirements is wired into a node which has
    /// [A,BOOL_T] extensions required on its inputs and outputs. This could be fixed
    /// by adding a lift node, but for validation this is an error.
    fn missing_lift_node() -> Result<(), BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let mut main = module_builder.define_function(
            "main",
            FunctionType::new(type_row![NAT], type_row![NAT]).into(),
        )?;
        let [main_input] = main.input_wires_arr();

        let f_builder = main.dfg_builder(
            FunctionType::new(type_row![NAT], type_row![NAT]),
            // Inner DFG has extension requirements that the wire wont satisfy
            Some(ExtensionSet::from_iter([XA, XB])),
            [main_input],
        )?;
        let f_inputs = f_builder.input_wires();
        let f_handle = f_builder.finish_with_outputs(f_inputs)?;
        let [f_output] = f_handle.outputs_arr();
        main.finish_with_outputs([f_output])?;
        let handle = module_builder.hugr().validate(&PRELUDE_REGISTRY);

        assert_matches!(
            handle,
            Err(ValidationError::ExtensionError(
                ExtensionError::TgtExceedsSrcExtensionsAtPort { .. }
            ))
        );
        Ok(())
    }

    #[test]
    /// A wire with extension requirement `[A]` is wired into a an output with no
    /// extension req. In the validation extension typechecking, we don't do any
    /// unification, so don't allow open extension variables on the function
    /// signature, so this fails.
    fn too_many_extension() -> Result<(), BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let main_sig = FunctionType::new(type_row![NAT], type_row![NAT]).into();

        let mut main = module_builder.define_function("main", main_sig)?;
        let [main_input] = main.input_wires_arr();

        let inner_sig = FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(XA);

        let f_builder = main.dfg_builder(inner_sig, Some(ExtensionSet::new()), [main_input])?;
        let f_inputs = f_builder.input_wires();
        let f_handle = f_builder.finish_with_outputs(f_inputs)?;
        let [f_output] = f_handle.outputs_arr();
        main.finish_with_outputs([f_output])?;
        let handle = module_builder.hugr().validate(&PRELUDE_REGISTRY);
        assert_matches!(
            handle,
            Err(ValidationError::ExtensionError(
                ExtensionError::SrcExceedsTgtExtensionsAtPort { .. }
            ))
        );
        Ok(())
    }

    #[test]
    /// A wire with extension requirements `[A]` and another with requirements
    /// `[BOOL_T]` are both wired into a node which requires its inputs to have
    /// requirements `[A,BOOL_T]`. A slightly more complex test of the error from
    /// `missing_lift_node`.
    fn extensions_mismatch() -> Result<(), BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let all_rs = ExtensionSet::from_iter([XA, XB]);

        let main_sig = FunctionType::new(type_row![], type_row![NAT])
            .with_extension_delta(all_rs.clone())
            .into();

        let mut main = module_builder.define_function("main", main_sig)?;

        let [left_wire] = main
            .dfg_builder(
                FunctionType::new(type_row![], type_row![NAT]),
                Some(XA.into()),
                [],
            )?
            .finish_with_outputs([])?
            .outputs_arr();

        let [right_wire] = main
            .dfg_builder(
                FunctionType::new(type_row![], type_row![NAT]),
                Some(XB.into()),
                [],
            )?
            .finish_with_outputs([])?
            .outputs_arr();

        let builder = main.dfg_builder(
            FunctionType::new(type_row![NAT, NAT], type_row![NAT]),
            Some(all_rs),
            [left_wire, right_wire],
        )?;
        let [_left, _right] = builder.input_wires_arr();
        let [output] = builder.finish_with_outputs([])?.outputs_arr();

        main.finish_with_outputs([output])?;
        let handle = module_builder.hugr().validate(&PRELUDE_REGISTRY);
        assert_matches!(
            handle,
            Err(ValidationError::ExtensionError(
                ExtensionError::TgtExceedsSrcExtensionsAtPort { .. }
            ))
        );
        Ok(())
    }

    #[test]
    fn parent_signature_mismatch() {
        let main_signature =
            FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(XA);

        let mut hugr = Hugr::new(NodeType::new_pure(ops::DFG {
            signature: main_signature,
        }));
        let input = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::new_pure(ops::Input {
                types: type_row![NAT],
            }),
        );
        let output = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::new(
                ops::Output {
                    types: type_row![NAT],
                },
                Some(XA.into()),
            ),
        );
        hugr.connect(input, 0, output, 0);

        assert_matches!(
            hugr.validate(&PRELUDE_REGISTRY),
            Err(ValidationError::ExtensionError(
                ExtensionError::TgtExceedsSrcExtensionsAtPort { .. }
            ))
        );
    }
}

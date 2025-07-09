use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use cool_asserts::assert_matches;
use rstest::rstest;

use super::*;
use crate::builder::test::closed_dfg_root_hugr;
use crate::builder::{
    BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    FunctionBuilder, HugrBuilder, ModuleBuilder, inout_sig,
};
use crate::extension::prelude::Noop;
use crate::extension::prelude::{bool_t, qb_t, usize_t};
use crate::extension::{Extension, ExtensionRegistry, PRELUDE, TypeDefBound};
use crate::hugr::HugrMut;
use crate::hugr::internal::HugrMutInternals;
use crate::ops::dataflow::{DataflowParent, IOTrait};
use crate::ops::handle::NodeHandle;
use crate::ops::{self, FuncDecl, FuncDefn, OpType, Value};
use crate::std_extensions::logic::LogicOp;
use crate::std_extensions::logic::test::{and_op, or_op};
use crate::types::type_param::{TermTypeError, TypeArg};
use crate::types::{
    CustomType, FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature, Term, Type, TypeBound,
    TypeRV, TypeRow,
};
use crate::{Direction, Hugr, IncomingPort, Node, const_extension_ids, test_file, type_row};

/// Creates a hugr with a single, public, function definition that copies a bit `copies` times.
///
/// Returns the hugr and the node index of the definition.
fn make_simple_hugr(copies: usize) -> (Hugr, Node) {
    let def_op: OpType = FuncDefn::new_vis(
        "main",
        Signature::new(bool_t(), vec![bool_t(); copies]),
        Visibility::Public,
    )
    .into();

    let mut b = Hugr::default();
    let root = b.entrypoint();

    let def = b.add_node_with_parent(root, def_op);
    let _ = add_df_children(&mut b, def, copies);

    (b, def)
}

/// Adds an `input{bool_t()}`, `copy{bool_t() -> bool_t()^copies}`, and `output{bool_t()^copies}` operation to a dataflow container.
///
/// Returns the node indices of each of the operations.
fn add_df_children(b: &mut Hugr, parent: Node, copies: usize) -> (Node, Node, Node) {
    let input = b.add_node_with_parent(parent, ops::Input::new(vec![bool_t()]));
    let output = b.add_node_with_parent(parent, ops::Output::new(vec![bool_t(); copies]));
    let copy = b.add_node_with_parent(parent, Noop(bool_t()));

    b.connect(input, 0, copy, 0);
    for i in 0..copies {
        b.connect(copy, 0, output, i);
    }

    (input, copy, output)
}

#[test]
fn invalid_root() {
    let build = DFGBuilder::new(Signature::default()).unwrap();
    let mut b = build.finish_hugr().unwrap();
    let root = b.module_root();
    assert_eq!(b.validate(), Ok(()));

    // Change the number of ports in the root
    b.set_num_ports(root, 1, 0);
    assert_matches!(
        b.validate(),
        Err(ValidationError::WrongNumberOfPorts { node, .. }) => assert_eq!(node, root)
    );
    b.set_num_ports(root, 0, 0);

    // Add another hierarchy root
    let module = b.add_node(ops::Module::new().into());
    assert_matches!(
        b.validate(),
        Err(ValidationError::NoParent { node }) => assert_eq!(node, module)
    );

    // Make the hugr root not a hierarchy root
    b.set_parent(root, module);
    assert_matches!(
        b.validate(),
        Err(ValidationError::NoParent { node }) => assert_eq!(node, module)
    );

    // Fix the root
    b.remove_node(module);
    assert_eq!(b.validate(), Ok(()));
}

#[test]
fn dfg_root() {
    let dfg_op: OpType = ops::DFG {
        signature: Signature::new_endo(vec![bool_t()]),
    }
    .into();

    let mut b = Hugr::new_with_entrypoint(dfg_op).unwrap();
    let root = b.entrypoint();
    add_df_children(&mut b, root, 1);
    assert_eq!(b.validate(), Ok(()));
}

#[test]
fn simple_hugr() {
    let b = make_simple_hugr(2).0;
    assert_eq!(b.validate(), Ok(()));
}

#[test]
/// General children restrictions.
fn children_restrictions() {
    let (mut b, def) = make_simple_hugr(2);
    let root = b.entrypoint();
    let (_input, copy, _output) = b
        .hierarchy
        .children(def.into_portgraph())
        .map_into()
        .collect_tuple()
        .unwrap();

    // Add a definition without children
    let def_sig = Signature::new(vec![bool_t()], vec![bool_t(), bool_t()]);
    let new_def = b.add_node_with_parent(root, FuncDefn::new("main", def_sig));
    assert_matches!(
        b.validate(),
        Err(ValidationError::ContainerWithoutChildren { node, .. }) => assert_eq!(node, new_def)
    );

    // Add children to the definition, but move it to be a child of the copy
    add_df_children(&mut b, new_def, 2);
    b.set_parent(new_def, copy);
    assert_matches!(
        b.validate(),
        Err(ValidationError::NonContainerWithChildren { node, .. }) => assert_eq!(node, copy)
    );
    b.set_parent(new_def, root);

    // After moving the previous definition to a valid place,
    // add an input node to the module subgraph
    let new_input = b.add_node_with_parent(root, ops::Input::new(type_row![]));
    assert_matches!(
        b.validate(),
        Err(ValidationError::InvalidParentOp { parent, child, .. }) => {assert_eq!(parent, root); assert_eq!(child, new_input)}
    );
}

#[test]
/// Validation errors in a dataflow subgraph.
fn df_children_restrictions() {
    let (mut b, def) = make_simple_hugr(2);
    let (_input, output, copy) = b
        .hierarchy
        .children(def.into_portgraph())
        .map_into()
        .collect_tuple()
        .unwrap();

    // Replace the output operation of the df subgraph with a copy
    b.replace_op(output, Noop(usize_t()));
    assert_matches!(
        b.validate(),
        Err(ValidationError::InvalidInitialChild { parent, .. }) => assert_eq!(parent, def)
    );

    // Revert it back to an output, but with the wrong number of ports
    b.replace_op(output, ops::Output::new(vec![bool_t()]));
    assert_matches!(
        b.validate(),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, output)}
    );
    b.replace_op(output, ops::Output::new(vec![bool_t(), bool_t()]));

    // After fixing the output back, replace the copy with an output op
    b.replace_op(copy, ops::Output::new(vec![bool_t(), bool_t()]));
    assert_matches!(
        b.validate(),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalIOChildren { child, .. }, .. })
            => {assert_eq!(parent, def); assert_eq!(child, copy)}
    );
}

#[test]
fn test_ext_edge() {
    let mut h = closed_dfg_root_hugr(Signature::new(vec![bool_t(), bool_t()], vec![bool_t()]));
    let [input, output] = h.get_io(h.entrypoint()).unwrap();

    // Nested DFG bool_t() -> bool_t()
    let sub_dfg = h.add_node_with_parent(
        h.entrypoint(),
        ops::DFG {
            signature: Signature::new_endo(vec![bool_t()]),
        },
    );
    // this Xor has its 2nd input unconnected
    let sub_op = {
        let sub_input = h.add_node_with_parent(sub_dfg, ops::Input::new(vec![bool_t()]));
        let sub_output = h.add_node_with_parent(sub_dfg, ops::Output::new(vec![bool_t()]));
        let sub_op = h.add_node_with_parent(sub_dfg, and_op());
        h.connect(sub_input, 0, sub_op, 0);
        h.connect(sub_op, 0, sub_output, 0);
        sub_op
    };

    h.connect(input, 0, sub_dfg, 0);
    h.connect(sub_dfg, 0, output, 0);

    assert_matches!(h.validate(), Err(ValidationError::UnconnectedPort { .. }));

    h.connect(input, 1, sub_op, 1);
    assert_matches!(
        h.validate(),
        Err(ValidationError::InterGraphEdgeError(
            InterGraphEdgeError::MissingOrderEdge { .. }
        ))
    );
    //Order edge. This will need metadata indicating its purpose.
    h.add_other_edge(input, sub_dfg);
    h.validate().unwrap();
}

#[test]
fn test_local_const() {
    let mut h = closed_dfg_root_hugr(Signature::new_endo(bool_t()));
    let [input, output] = h.get_io(h.entrypoint()).unwrap();
    let and = h.add_node_with_parent(h.entrypoint(), and_op());
    h.connect(input, 0, and, 0);
    h.connect(and, 0, output, 0);
    assert_eq!(
        h.validate(),
        Err(ValidationError::UnconnectedPort {
            node: and,
            port: IncomingPort::from(1).into(),
            port_kind: Box::new(EdgeKind::Value(bool_t()))
        })
    );
    let const_op: ops::Const = ops::Value::from_bool(true).into();
    // Second input of Xor from a constant
    let cst = h.add_node_with_parent(h.entrypoint(), const_op);
    let lcst = h.add_node_with_parent(h.entrypoint(), ops::LoadConstant { datatype: bool_t() });

    h.connect(cst, 0, lcst, 0);
    h.connect(lcst, 0, and, 1);
    assert_eq!(h.static_source(lcst), Some(cst));
    // There is no edge from Input to LoadConstant, but that's OK:
    h.validate().unwrap();
}

#[test]
fn dfg_with_cycles() {
    let mut h = closed_dfg_root_hugr(Signature::new(vec![bool_t(), bool_t()], vec![bool_t()]));
    let [input, output] = h.get_io(h.entrypoint()).unwrap();
    let or = h.add_node_with_parent(h.entrypoint(), or_op());
    let not1 = h.add_node_with_parent(h.entrypoint(), LogicOp::Not);
    let not2 = h.add_node_with_parent(h.entrypoint(), LogicOp::Not);
    h.connect(input, 0, or, 0);
    h.connect(or, 0, not1, 0);
    h.connect(not1, 0, or, 1);
    h.connect(input, 1, not2, 0);
    h.connect(not2, 0, output, 0);
    // The graph contains a cycle:
    assert_matches!(h.validate(), Err(ValidationError::NotADag { .. }));
}

/// An identity hugr. Note that extensions must be updated before validation,
/// as `hugr.extensions` is empty.
fn identity_hugr_with_type(t: Type) -> (Hugr, Node) {
    let mut b = Hugr::default();
    let row: TypeRow = vec![t].into();

    let def = b.add_node_with_parent(
        b.entrypoint(),
        FuncDefn::new("main", Signature::new_endo(row.clone())),
    );

    let input = b.add_node_with_parent(def, ops::Input::new(row.clone()));
    let output = b.add_node_with_parent(def, ops::Output::new(row));
    b.connect(input, 0, output, 0);
    (b, def)
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
            String::new(),
            TypeDefBound::any(),
            extension_ref,
        )
        .unwrap();
    });
    let reg = ExtensionRegistry::new([ext.clone(), PRELUDE.to_owned()]);
    reg.validate().unwrap();

    let validate_to_sig_error = |t: CustomType| -> SignatureError {
        let (mut h, def) = identity_hugr_with_type(Type::new_extension(t));
        h.resolve_extension_defs(&reg).unwrap();

        let e = h.validate().unwrap_err();
        let (node, cause) = assert_matches!(
            e,
            ValidationError::SignatureError{ node, cause, .. } => (node, cause)
        );
        assert_eq!(node, def);
        cause
    };

    let valid = Type::new_extension(CustomType::new(
        "MyContainer",
        vec![usize_t().into()],
        EXT_ID,
        TypeBound::Linear,
        &Arc::downgrade(&ext),
    ));
    let mut hugr = identity_hugr_with_type(valid.clone()).0;
    hugr.resolve_extension_defs(&reg).unwrap();
    assert_eq!(hugr.validate(), Ok(()));

    // valid is Any, so is not allowed as an element of an outer MyContainer.
    let element_outside_bound = CustomType::new(
        "MyContainer",
        vec![valid.clone().into()],
        EXT_ID,
        TypeBound::Linear,
        &Arc::downgrade(&ext),
    );
    assert_eq!(
        validate_to_sig_error(element_outside_bound),
        SignatureError::TypeArgMismatch(TermTypeError::TypeMismatch {
            type_: Box::new(TypeBound::Copyable.into()),
            term: Box::new(valid.into())
        })
    );

    let bad_bound = CustomType::new(
        "MyContainer",
        vec![usize_t().into()],
        EXT_ID,
        TypeBound::Copyable,
        &Arc::downgrade(&ext),
    );
    assert_eq!(
        validate_to_sig_error(bad_bound.clone()),
        SignatureError::WrongBound {
            actual: TypeBound::Copyable,
            expected: TypeBound::Linear
        }
    );

    // bad_bound claims to be Copyable, which is valid as an element for the outer MyContainer.
    let nested = CustomType::new(
        "MyContainer",
        vec![Type::new_extension(bad_bound).into()],
        EXT_ID,
        TypeBound::Linear,
        &Arc::downgrade(&ext),
    );
    assert_eq!(
        validate_to_sig_error(nested),
        SignatureError::WrongBound {
            actual: TypeBound::Copyable,
            expected: TypeBound::Linear
        }
    );

    let too_many_type_args = CustomType::new(
        "MyContainer",
        vec![usize_t().into(), 3u64.into()],
        EXT_ID,
        TypeBound::Linear,
        &Arc::downgrade(&ext),
    );
    assert_eq!(
        validate_to_sig_error(too_many_type_args),
        SignatureError::TypeArgMismatch(TermTypeError::WrongNumberArgs(2, 1))
    );
}

#[test]
fn typevars_declared() -> Result<(), Box<dyn std::error::Error>> {
    // Base case
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Linear.into()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Linear)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    f.finish_hugr_with_outputs([w])?;
    // Type refers to undeclared variable
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Linear.into()],
            Signature::new_endo(vec![Type::new_var_use(1, TypeBound::Linear)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    assert!(f.finish_hugr_with_outputs([w]).is_err());
    // Variable declaration incorrectly copied to use site
    let f = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new(
            [TypeBound::Linear.into()],
            Signature::new_endo(vec![Type::new_var_use(1, TypeBound::Copyable)]),
        ),
    )?;
    let [w] = f.input_wires_arr();
    assert!(f.finish_hugr_with_outputs([w]).is_err());
    Ok(())
}

/// Test that `FuncDefns` cannot be nested.
#[test]
fn no_nested_funcdefns() -> Result<(), Box<dyn std::error::Error>> {
    let mut outer = FunctionBuilder::new("outer", Signature::new_endo(usize_t()))?;
    let inner = outer
        .add_hugr({
            let inner = FunctionBuilder::new("inner", Signature::new_endo(bool_t()))?;
            let [w] = inner.input_wires_arr();
            inner.finish_hugr_with_outputs([w])?
        })
        .inserted_entrypoint;
    let [w] = outer.input_wires_arr();
    let outer_node = outer.container_node();
    let hugr = outer.finish_hugr_with_outputs([w]);
    assert_matches!(
        hugr.unwrap_err(),
        BuildError::InvalidHUGR(ValidationError::InvalidParentOp {
            child_optype,
            allowed_children: OpTag::DataflowChild,
            parent_optype,
            child, parent
        }) if matches!(*child_optype, OpType::FuncDefn(_)) && matches!(*parent_optype, OpType::FuncDefn(_)) => {
            assert_eq!(child, inner);
            assert_eq!(parent, outer_node);
        }
    );
    Ok(())
}

#[test]
fn no_polymorphic_consts() -> Result<(), Box<dyn std::error::Error>> {
    use crate::std_extensions::collections::list;
    const BOUND: TypeParam = TypeParam::RuntimeType(TypeBound::Copyable);
    let list_of_var = Type::new_extension(
        list::EXTENSION
            .get_type(&list::LIST_TYPENAME)
            .unwrap()
            .instantiate(vec![TypeArg::new_var_use(0, BOUND)])?,
    );
    let reg = ExtensionRegistry::new([list::EXTENSION.to_owned()]);
    reg.validate()?;
    let mut def = FunctionBuilder::new(
        "myfunc",
        PolyFuncType::new([BOUND], Signature::new(vec![], vec![list_of_var.clone()])),
    )?;
    let empty_list = Value::extension(list::ListValue::new_empty(Type::new_var_use(
        0,
        TypeBound::Copyable,
    )));
    let cst = def.add_load_const(empty_list);
    let res = def.finish_hugr_with_outputs([cst]);
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
    let rowp = TypeParam::new_list_type(TypeBound::Linear);
    Extension::new_test_arc(EXT_ID, |ext, extension_ref| {
        let inputs = TypeRV::new_row_var_use(0, TypeBound::Linear);
        let outputs = TypeRV::new_row_var_use(1, TypeBound::Linear);
        let evaled_fn = TypeRV::new_function(FuncValueType::new(inputs.clone(), outputs.clone()));
        let pf = PolyFuncTypeRV::new(
            [rowp.clone(), rowp.clone()],
            FuncValueType::new(vec![evaled_fn, inputs], outputs),
        );
        ext.add_op("eval".into(), String::new(), pf, extension_ref)
            .unwrap();

        let rv = |idx| TypeRV::new_row_var_use(idx, TypeBound::Linear);
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
        ext.add_op("parallel".into(), String::new(), pf, extension_ref)
            .unwrap();
    })
}

#[test]
fn instantiate_row_variables() -> Result<(), Box<dyn std::error::Error>> {
    fn uint_seq(i: usize) -> Term {
        vec![usize_t().into(); i].into()
    }
    let e = extension_with_eval_parallel();
    let mut dfb = DFGBuilder::new(inout_sig(
        vec![
            Type::new_function(Signature::new(usize_t(), vec![usize_t(), usize_t()])),
            usize_t(),
        ], // inputs: function + its argument
        vec![usize_t(); 4], // outputs (*2^2, three calls)
    ))?;
    let [func, int] = dfb.input_wires_arr();
    let eval = e.instantiate_extension_op("eval", [uint_seq(1), uint_seq(2)])?;
    let [a, b] = dfb.add_dataflow_op(eval, [func, int])?.outputs_arr();
    let par = e.instantiate_extension_op(
        "parallel",
        [uint_seq(1), uint_seq(1), uint_seq(2), uint_seq(2)],
    )?;
    let [par_func] = dfb.add_dataflow_op(par, [func, func])?.outputs_arr();
    let eval2 = e.instantiate_extension_op("eval", [uint_seq(2), uint_seq(4)])?;
    let eval2 = dfb.add_dataflow_op(eval2, [par_func, a, b])?;
    dfb.finish_hugr_with_outputs(eval2.outputs())?;
    Ok(())
}

fn list1ty(t: TypeRV) -> Term {
    Term::new_list([t.into()])
}

#[test]
fn row_variables() -> Result<(), Box<dyn std::error::Error>> {
    let e = extension_with_eval_parallel();
    let tv = TypeRV::new_row_var_use(0, TypeBound::Linear);
    let inner_ft = Type::new_function(FuncValueType::new_endo(tv.clone()));
    let ft_usz = Type::new_function(FuncValueType::new_endo(vec![tv.clone(), usize_t().into()]));
    let mut fb = FunctionBuilder::new(
        "id",
        PolyFuncType::new(
            [TypeParam::new_list_type(TypeBound::Linear)],
            Signature::new(inner_ft.clone(), ft_usz),
        ),
    )?;
    // All the wires here are carrying higher-order Function values
    let [func_arg] = fb.input_wires_arr();
    let id_usz = {
        let mut mb = fb.module_root_builder();
        let bldr = mb.define_function("id_usz", Signature::new_endo(usize_t()))?;
        let vals = bldr.input_wires();
        let helper_def = bldr.finish_with_outputs(vals)?;
        fb.load_func(helper_def.handle(), &[])?
    };
    let par = e.instantiate_extension_op(
        "parallel",
        [tv.clone(), usize_t().into(), tv.clone(), usize_t().into()].map(list1ty),
    )?;
    let par_func = fb.add_dataflow_op(par, [func_arg, id_usz])?;
    fb.finish_hugr_with_outputs(par_func.outputs())?;
    Ok(())
}

#[test]
fn test_polymorphic_load() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = ModuleBuilder::new();
    let id = m.declare(
        "id",
        PolyFuncType::new(
            vec![TypeBound::Linear.into()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Linear)]),
        ),
    )?;
    let sig = Signature::new(
        vec![],
        vec![Type::new_function(Signature::new_endo(vec![usize_t()]))],
    );
    let mut f = m.define_function("main", sig)?;
    let l = f.load_func(&id, &[usize_t().into()])?;
    f.finish_with_outputs([l])?;
    let _ = m.finish_hugr()?;
    Ok(())
}

#[test]
/// Validation errors in a controlflow subgraph.
fn cfg_children_restrictions() {
    let (mut b, def) = make_simple_hugr(1);
    let (_input, _output, copy) = b
        .hierarchy
        .children(def.into_portgraph())
        .map_into()
        .collect_tuple()
        .unwrap();
    // Write Extension annotations into the Hugr while it's still well-formed
    // enough for us to compute them
    b.validate().unwrap();
    b.replace_op(
        copy,
        ops::CFG {
            signature: Signature::new(vec![bool_t()], vec![bool_t()]),
        },
    );
    assert_matches!(
        b.validate(),
        Err(ValidationError::ContainerWithoutChildren { .. })
    );
    let cfg = copy;

    // Construct a valid CFG, with one BasicBlock node and one exit node
    let block = b.add_node_with_parent(
        cfg,
        ops::DataflowBlock {
            inputs: vec![bool_t()].into(),
            sum_rows: vec![type_row![]],
            other_outputs: vec![bool_t()].into(),
        },
    );
    let const_op: ops::Const = ops::Value::unit_sum(0, 1).unwrap().into();
    let tag_type = Type::new_unit_sum(1);
    {
        let input = b.add_node_with_parent(block, ops::Input::new(vec![bool_t()]));
        let output =
            b.add_node_with_parent(block, ops::Output::new(vec![tag_type.clone(), bool_t()]));
        let tag_def = b.add_node_with_parent(b.entrypoint(), const_op);
        let tag = b.add_node_with_parent(block, ops::LoadConstant { datatype: tag_type });

        b.connect(tag_def, 0, tag, 0);
        b.add_other_edge(input, tag);
        b.connect(tag, 0, output, 0);
        b.connect(input, 0, output, 1);
    }
    let exit = b.add_node_with_parent(
        cfg,
        ops::ExitBlock {
            cfg_outputs: vec![bool_t()].into(),
        },
    );
    b.add_other_edge(block, exit);
    assert_eq!(b.validate(), Ok(()));

    // Test malformed errors

    // Add an internal exit node
    let exit2 = b.add_node_after(
        exit,
        ops::ExitBlock {
            cfg_outputs: vec![bool_t()].into(),
        },
    );
    assert_matches!(
        b.validate(),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::InternalExitChildren { child, .. }, .. })
            => {assert_eq!(parent, cfg); assert_eq!(child, exit2)}
    );
    b.remove_node(exit2);

    // Change the types in the BasicBlock node to work on qubits instead of bits
    b.replace_op(
        cfg,
        ops::CFG {
            signature: Signature::new(vec![qb_t()], vec![bool_t()]),
        },
    );
    b.replace_op(
        block,
        ops::DataflowBlock {
            inputs: vec![qb_t()].into(),
            sum_rows: vec![type_row![]],
            other_outputs: vec![qb_t()].into(),
        },
    );
    let mut block_children = b.hierarchy.children(block.into_portgraph());
    let block_input = block_children.next().unwrap().into();
    let block_output = block_children.next_back().unwrap().into();
    b.replace_op(block_input, ops::Input::new(vec![qb_t()]));
    b.replace_op(
        block_output,
        ops::Output::new(vec![Type::new_unit_sum(1), qb_t()]),
    );
    assert_matches!(
        b.validate(),
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

    let mut hugr = CFGBuilder::new(Signature::new_endo(usize_t()))?;
    let unary_pred = hugr.add_constant(Value::unary_unit_sum());
    let mut entry = hugr.simple_entry_builder(vec![usize_t()].into(), 1)?;
    let p = entry.load_const(&unary_pred);
    let ins = entry.input_wires();
    let entry = entry.finish_with_outputs(p, ins)?;

    let mut middle = hugr.simple_block_builder(Signature::new_endo(usize_t()), 1)?;
    let p = middle.load_const(&unary_pred);
    let ins = middle.input_wires();
    let middle = middle.finish_with_outputs(p, ins)?;

    let exit = hugr.exit_block();
    hugr.branch(&entry, 0, &middle)?;
    hugr.branch(&middle, 0, &exit)?;
    let mut h = hugr.finish_hugr()?;

    h.connect(middle.node(), 0, middle.node(), 0);
    assert_eq!(
        h.validate(),
        Err(ValidationError::TooManyConnections {
            node: middle.node(),
            port: Port::new(Direction::Outgoing, 0),
            port_kind: Box::new(EdgeKind::ControlFlow)
        })
    );
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn cfg_entry_io_bug() -> Result<(), Box<dyn std::error::Error>> {
    // load test file where input node of entry block has types in reversed
    // order compared to parent CFG node.
    let hugr: Hugr = Hugr::load(
        BufReader::new(File::open(test_file!("issue-1189.hugr")).unwrap()),
        None,
    )
    .unwrap();
    assert_matches!(
        hugr.validate(),
        Err(ValidationError::InvalidChildren { parent, source: ChildrenValidationError::IOSignatureMismatch{..}, .. })
            => assert_eq!(parent, hugr.entrypoint())
    );

    Ok(())
}

fn sig1() -> Signature {
    Signature::new_endo(bool_t())
}

fn sig2() -> Signature {
    Signature::new_endo(usize_t())
}

#[rstest]
// Private FuncDefns never conflict even if different sig
#[case(
    FuncDefn::new_vis("foo", sig1(), Visibility::Public),
    FuncDefn::new("foo", sig2()),
    None
)]
#[case(FuncDefn::new("foo", sig1()), FuncDecl::new("foo", sig2()), None)]
// Public FuncDefn conflicts with anything Public even if same sig
#[case(
    FuncDefn::new_vis("foo", sig1(), Visibility::Public),
    FuncDefn::new_vis("foo", sig1(), Visibility::Public),
    Some("foo")
)]
#[case(
    FuncDefn::new_vis("foo", sig1(), Visibility::Public),
    FuncDecl::new("foo", sig1()),
    Some("foo")
)]
// Two public FuncDecls are ok with same sig
#[case(FuncDecl::new("foo", sig1()), FuncDecl::new("foo", sig1()), None)]
// But two public FuncDecls not ok if different sigs
#[case(
    FuncDecl::new("foo", sig1()),
    FuncDecl::new("foo", sig2()),
    Some("foo")
)]
fn validate_linkage(
    #[case] f1: impl Into<OpType>,
    #[case] f2: impl Into<OpType>,
    #[case] err: Option<&str>,
) {
    let mut h = Hugr::new();
    let [n1, n2] = [f1.into(), f2.into()].map(|f| {
        let def_sig = f
            .as_func_defn()
            .map(FuncDefn::inner_signature)
            .map(Cow::into_owned);
        let n = h.add_node_with_parent(h.module_root(), f);
        if let Some(Signature { input, output }) = def_sig {
            let i = h.add_node_with_parent(n, ops::Input::new(input));
            let o = h.add_node_with_parent(n, ops::Output::new(output));
            h.connect(i, 0, o, 0); // Assume all sig's used in test are 1-ary endomorphic
        }
        n
    });
    let r = h.validate();
    match err {
        None => r.unwrap(),
        Some(name) => {
            let Err(ValidationError::DuplicateExport {
                link_name,
                children,
            }) = r
            else {
                panic!("validate() should have produced DuplicateExport error not {r:?}")
            };
            assert_eq!(link_name, name);
            assert!(children == [n1, n2] || children == [n2, n1]);
        }
    }
}

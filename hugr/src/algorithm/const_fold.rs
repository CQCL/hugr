//! Constant folding routines.

use std::collections::{BTreeSet, HashMap};

use itertools::Itertools;
use thiserror::Error;

use crate::hugr::{SimpleReplacementError, ValidationError};
use crate::types::SumType;
use crate::Direction;
use crate::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionRegistry},
    hugr::{
        rewrite::consts::{RemoveConst, RemoveLoadConstant},
        views::SiblingSubgraph,
        HugrMut,
    },
    ops::{OpType, Value},
    type_row,
    types::FunctionType,
    Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
};

use super::VerifyLevel;

#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum ConstFoldError {
    #[error("Failed to verify {label} HUGR: {err}")]
    VerifyError {
        label: String,
        #[source]
        err: ValidationError,
    },
    #[error(transparent)]
    SimpleReplaceError(#[from] SimpleReplacementError),
}

impl ConstFoldError {
    fn verify_err(label: impl Into<String>, err: ValidationError) -> Self {
        Self::VerifyError {
            label: label.into(),
            err,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// TODO
pub struct ConstFoldConfig {
    verify: VerifyLevel,
}

impl ConstFoldConfig {
    /// Create a new `ConstFoldConfig` with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `ConstFoldConfig` with the given [VerifyLevel].
    pub fn with_verify(mut self, verify: VerifyLevel) -> Self {
        self.verify = verify;
        self
    }

    fn verify_impl(
        &self,
        label: &str,
        h: &impl HugrView,
        reg: &ExtensionRegistry,
    ) -> Result<(), ConstFoldError> {
        match self.verify {
            VerifyLevel::None => Ok(()),
            VerifyLevel::WithoutExtensions => h.validate_no_extensions(reg),
            VerifyLevel::WithExtensions => h.validate(reg),
        }
        .map_err(|err| ConstFoldError::verify_err(label, err))
    }

    /// Run the Constant Folding pass.
    pub fn run(&self, h: &mut impl HugrMut, reg: &ExtensionRegistry) -> Result<(), ConstFoldError> {
        self.verify_impl("input", h, reg)?;
        loop {
            // We can only safely apply a single replacement. Applying a
            // replacement removes nodes and edges which may be referenced by
            // further replacements returned by find_consts. Even worse, if we
            // attempted to apply those replacements, expecting them to fail if
            // the nodes and edges they reference had been deleted,  they may
            // succeed because new nodes and edges reused the ids.
            //
            // We could be a lot smarter here, keeping track of `LoadConstant`
            // nodes and only looking at their out neighbours.
            let Some((replace, removes)) = find_consts(h, h.nodes(), reg).next() else {
                break;
            };
            h.apply_rewrite(replace)?;
            for rem in removes {
                if let Ok(const_node) = h.apply_rewrite(rem) {
                    // if the LoadConst was removed, try removing the Const too.
                    let _ = h.apply_rewrite(RemoveConst(const_node));
                }
            }
        }
        self.verify_impl("output", h, reg)
    }
}

/// Tag some output constants with [`OutgoingPort`] inferred from the ordering.
fn out_row(consts: impl IntoIterator<Item = Value>) -> ConstFoldResult {
    let vec = consts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i.into(), c))
        .collect();
    Some(vec)
}

/// Sort folding inputs with [`IncomingPort`] as key
fn sort_by_in_port(consts: &[(IncomingPort, Value)]) -> Vec<&(IncomingPort, Value)> {
    let mut v: Vec<_> = consts.iter().collect();
    v.sort_by_key(|(i, _)| i);
    v
}

/// Sort some input constants by port and just return the constants.
pub(crate) fn sorted_consts(consts: &[(IncomingPort, Value)]) -> Vec<&Value> {
    sort_by_in_port(consts)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}

/// For a given op and consts, attempt to evaluate the op.
pub fn fold_leaf_op(op: &OpType, consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
    let fold_result = match op {
        OpType::Noop { .. } => out_row([consts.first()?.1.clone()]),
        OpType::MakeTuple { .. } => {
            out_row([Value::tuple(sorted_consts(consts).into_iter().cloned())])
        }
        OpType::UnpackTuple { .. } => {
            let c = &consts.first()?.1;
            let Value::Tuple { vs } = c else {
                panic!("This op always takes a Tuple input.");
            };
            out_row(vs.iter().cloned())
        }

        OpType::Tag(t) => out_row([Value::sum(
            t.tag,
            consts.iter().map(|(_, konst)| konst.clone()),
            SumType::new(t.variants.clone()),
        )
        .unwrap()]),
        OpType::CustomOp(op) => {
            let ext_op = op.as_extension_op()?;
            ext_op.constant_fold(consts)
        }
        _ => None,
    };
    assert!(fold_result.as_ref().map_or(true, |x| x.len()
        == op.value_port_count(Direction::Outgoing)));
    fold_result
}

/// Generate a graph that loads and outputs `consts` in order, validating
/// against `reg`.
fn const_graph(consts: Vec<Value>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Value::get_type).collect_vec();
    let mut b = DFGBuilder::new(FunctionType::new(type_row![], const_types)).unwrap();

    let outputs = consts
        .into_iter()
        .map(|c| b.add_load_const(c))
        .collect_vec();

    b.finish_hugr_with_outputs(outputs, reg).unwrap()
}

/// Given some `candidate_nodes` to search for LoadConstant operations in `hugr`,
/// return an iterator of possible constant folding rewrites. The
/// [`SimpleReplacement`] replaces an operation with constants that result from
/// evaluating it, the extension registry `reg` is used to validate the
/// replacement HUGR. The vector of [`RemoveLoadConstant`] refer to the
/// LoadConstant nodes that could be removed - they are not automatically
/// removed as they may be used by other operations.
pub fn find_consts<'a, 'r: 'a>(
    hugr: &'a impl HugrView,
    candidate_nodes: impl IntoIterator<Item = Node> + 'a,
    reg: &'r ExtensionRegistry,
) -> impl Iterator<Item = (SimpleReplacement, Vec<RemoveLoadConstant>)> + 'a {
    // track nodes for operations that have already been considered for folding
    let mut used_neighbours = BTreeSet::new();

    candidate_nodes
        .into_iter()
        .filter_map(move |n| {
            // only look at LoadConstant
            hugr.get_optype(n).is_load_constant().then_some(())?;

            let (out_p, _) = hugr.out_value_types(n).exactly_one().ok()?;
            let neighbours = hugr
                .linked_inputs(n, out_p)
                .filter(|(n, _)| used_neighbours.insert(*n))
                .collect_vec();
            if neighbours.is_empty() {
                // no uses of LoadConstant that haven't already been considered.
                return None;
            }
            let fold_iter = neighbours
                .into_iter()
                .filter_map(|(neighbour, _)| fold_op(hugr, neighbour, reg));
            Some(fold_iter)
        })
        .flatten()
}

/// Attempt to evaluate and generate rewrites for the operation at `op_node`
fn fold_op(
    hugr: &impl HugrView,
    op_node: Node,
    reg: &ExtensionRegistry,
) -> Option<(SimpleReplacement, Vec<RemoveLoadConstant>)> {
    // only support leaf folding for now.
    let neighbour_op = hugr.get_optype(op_node);
    let (in_consts, removals): (Vec<_>, Vec<_>) = hugr
        .node_inputs(op_node)
        .filter_map(|in_p| {
            let (con_op, load_n) = get_const(hugr, op_node, in_p)?;
            Some(((in_p, con_op), RemoveLoadConstant(load_n)))
        })
        .unzip();
    // attempt to evaluate op
    let (nu_out, consts): (HashMap<_, _>, Vec<_>) = fold_leaf_op(neighbour_op, &in_consts)?
        .into_iter()
        .enumerate()
        .filter_map(|(i, (op_out, konst))| {
            // for each used port of the op give the nu_out entry and the
            // corresponding Value
            hugr.single_linked_input(op_node, op_out)
                .map(|np| ((np, i.into()), konst))
        })
        .unzip();
    let replacement = const_graph(consts, reg);
    let sibling_graph = SiblingSubgraph::try_from_nodes([op_node], hugr)
        .expect("Operation should form valid subgraph.");

    let simple_replace = SimpleReplacement::new(
        sibling_graph,
        replacement,
        // no inputs to replacement
        HashMap::new(),
        nu_out,
    );
    Some((simple_replace, removals))
}

/// If `op_node` is connected to a LoadConstant at `in_p`, return the constant
/// and the LoadConstant node
fn get_const(hugr: &impl HugrView, op_node: Node, in_p: IncomingPort) -> Option<(Value, Node)> {
    let (load_n, _) = hugr.single_linked_output(op_node, in_p)?;
    let load_op = hugr.get_optype(load_n).as_load_constant()?;
    let const_node = hugr
        .single_linked_output(load_n, load_op.constant_port())?
        .0;
    let const_op = hugr.get_optype(const_node).as_const()?;

    // TODO avoid const clone here
    Some((const_op.as_ref().clone(), load_n))
}

/// Exhaustively apply constant folding to a HUGR.
pub fn constant_fold_pass<H: HugrMut>(h: &mut H, reg: &ExtensionRegistry) {
    ConstFoldConfig::default().run(h, reg).unwrap()
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::extension::prelude::{sum_with_error, BOOL_T};
    use crate::extension::{ExtensionRegistry, PRELUDE};
    use crate::ops::{OpType, UnpackTuple};
    use crate::std_extensions::arithmetic;
    use crate::std_extensions::arithmetic::conversions::ConvertOpDef;
    use crate::std_extensions::arithmetic::float_ops::FloatOps;
    use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
    use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use crate::std_extensions::logic::{self, NaryLogic, NotOp};
    use crate::utils::test::assert_fully_folded;

    use rstest::rstest;

    /// int to constant
    fn i2c(b: u64) -> Value {
        Value::extension(ConstInt::new_u(5, b).unwrap())
    }

    /// float to constant
    fn f2c(f: f64) -> Value {
        ConstF64::new(f).into()
    }

    #[rstest]
    #[case(0.0, 0.0, 0.0)]
    #[case(0.0, 1.0, 1.0)]
    #[case(23.5, 435.5, 459.0)]
    // c = a + b
    fn test_add(#[case] a: f64, #[case] b: f64, #[case] c: f64) {
        let consts = vec![(0.into(), f2c(a)), (1.into(), f2c(b))];
        let add_op: OpType = FloatOps::fadd.into();
        let out = fold_leaf_op(&add_op, &consts).unwrap();

        assert_eq!(&out[..], &[(0.into(), f2c(c))]);
    }
    #[test]
    fn test_big() {
        /*
           Test approximately calculates
           let x = (5.6, 3.2);
           int(x.0 - x.1) == 2
        */
        let sum_type = sum_with_error(INT_TYPES[5].to_owned());
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![sum_type.clone().into()],
        ))
        .unwrap();

        let tup = build.add_load_const(Value::tuple([f2c(5.6), f2c(3.2)]));

        let unpack = build
            .add_dataflow_op(
                UnpackTuple {
                    tys: type_row![FLOAT64_TYPE, FLOAT64_TYPE],
                },
                [tup],
            )
            .unwrap();

        let sub = build
            .add_dataflow_op(FloatOps::fsub, unpack.outputs())
            .unwrap();
        let to_int = build
            .add_dataflow_op(ConvertOpDef::trunc_u.with_log_width(5), sub.outputs())
            .unwrap();

        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            arithmetic::int_types::EXTENSION.to_owned(),
            arithmetic::float_types::EXTENSION.to_owned(),
            arithmetic::float_ops::EXTENSION.to_owned(),
            arithmetic::conversions::EXTENSION.to_owned(),
        ])
        .unwrap();
        let mut h = build
            .finish_hugr_with_outputs(to_int.outputs(), &reg)
            .unwrap();
        assert_eq!(h.node_count(), 8);

        constant_fold_pass(&mut h, &reg);

        let expected = Value::sum(0, [i2c(2).clone()], sum_type).unwrap();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    #[cfg_attr(
        feature = "extension_inference",
        ignore = "inference fails for test graph, it shouldn't"
    )]
    fn test_list_ops() -> Result<(), Box<dyn std::error::Error>> {
        use crate::std_extensions::collections::{self, ListOp, ListValue};

        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            logic::EXTENSION.to_owned(),
            collections::EXTENSION.to_owned(),
        ])
        .unwrap();
        let list: Value = ListValue::new(BOOL_T, [Value::unit_sum(0, 1).unwrap()]).into();
        let mut build = DFGBuilder::new(FunctionType::new(
            type_row![],
            vec![list.get_type().clone()],
        ))
        .unwrap();

        let list_wire = build.add_load_const(list.clone());

        let pop = build.add_dataflow_op(
            ListOp::Pop.with_type(BOOL_T).to_extension_op(&reg).unwrap(),
            [list_wire],
        )?;

        let push = build.add_dataflow_op(
            ListOp::Push
                .with_type(BOOL_T)
                .to_extension_op(&reg)
                .unwrap(),
            pop.outputs(),
        )?;
        let mut h = build.finish_hugr_with_outputs(push.outputs(), &reg)?;
        constant_fold_pass(&mut h, &reg);

        assert_fully_folded(&h, &list);
        Ok(())
    }

    #[test]
    fn test_fold_and() {
        // pseudocode:
        // x0, x1 := bool(true), bool(true)
        // x2 := and(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::true_val());
        let x1 = build.add_load_const(Value::true_val());
        let x2 = build
            .add_dataflow_op(NaryLogic::And.with_n_inputs(2), [x0, x1])
            .unwrap();
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_or() {
        // pseudocode:
        // x0, x1 := bool(true), bool(false)
        // x2 := or(x0, x1)
        // output x2 == true;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::true_val());
        let x1 = build.add_load_const(Value::false_val());
        let x2 = build
            .add_dataflow_op(NaryLogic::Or.with_n_inputs(2), [x0, x1])
            .unwrap();
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(x2.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::true_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn test_fold_not() {
        // pseudocode:
        // x0 := bool(true)
        // x1 := not(x0)
        // output x1 == false;
        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let x0 = build.add_load_const(Value::true_val());
        let x1 = build.add_dataflow_op(NotOp, [x0]).unwrap();
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(x1.outputs(), &reg).unwrap();
        constant_fold_pass(&mut h, &reg);
        let expected = Value::false_val();
        assert_fully_folded(&h, &expected);
    }

    #[test]
    fn orphan_output() {
        // pseudocode:
        // x0 := bool(true)
        // x1 := not(x0)
        // x2 := or(x0,x1)
        // output x2 == true;
        //
        // We arange things so that the `or` folds away first, leaving the not
        // with no outputs.
        use crate::hugr::NodeType;
        use crate::ops::handle::NodeHandle;

        let mut build = DFGBuilder::new(FunctionType::new(type_row![], vec![BOOL_T])).unwrap();
        let true_wire = build.add_load_value(Value::true_val());
        // this Not will be manually replaced
        let orig_not = build.add_dataflow_op(NotOp, [true_wire]).unwrap();
        let r = build
            .add_dataflow_op(
                NaryLogic::Or.with_n_inputs(2),
                [true_wire, orig_not.out_wire(0)],
            )
            .unwrap();
        let or_node = r.node();
        let parent = build.dfg_node;
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), logic::EXTENSION.to_owned()]).unwrap();
        let mut h = build.finish_hugr_with_outputs(r.outputs(), &reg).unwrap();

        // we delete the original Not and create a new One. This means it will be
        // traversed by `constant_fold_pass` after the Or.
        let new_not = h.add_node_with_parent(parent, NodeType::new_auto(NotOp));
        h.connect(true_wire.node(), true_wire.source(), new_not, 0);
        h.disconnect(or_node, IncomingPort::from(1));
        h.connect(new_not, 0, or_node, 1);
        h.remove_node(orig_not.node());
        constant_fold_pass(&mut h, &reg);
        assert_fully_folded(&h, &Value::true_val())
    }
}

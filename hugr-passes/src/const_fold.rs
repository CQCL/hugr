//! Constant folding routines.

use std::collections::{BTreeSet, HashMap};

use hugr_core::builder::ft2;
use itertools::Itertools;
use thiserror::Error;

use hugr_core::hugr::SimpleReplacementError;
use hugr_core::types::SumType;
use hugr_core::Direction;
use hugr_core::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::{ConstFoldResult, ExtensionRegistry},
    hugr::{
        hugrmut::HugrMut,
        rewrite::consts::{RemoveConst, RemoveLoadConstant},
        views::SiblingSubgraph,
    },
    ops::{OpType, Value},
    type_row,
    utils::sorted_consts,
    Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
};

use crate::validation::{ValidatePassError, ValidationLevel};

#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum ConstFoldError {
    #[error(transparent)]
    SimpleReplacementError(#[from] SimpleReplacementError),
    #[error(transparent)]
    ValidationError(#[from] ValidatePassError),
}

#[derive(Debug, Clone, Copy, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstantFoldPass {
    validation: ValidationLevel,
}

impl ConstantFoldPass {
    /// Create a new `ConstFoldConfig` with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `ConstFoldConfig` with the given [ValidationLevel].
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Run the Constant Folding pass.
    pub fn run<H: HugrMut>(
        &self,
        hugr: &mut H,
        reg: &ExtensionRegistry,
    ) -> Result<(), ConstFoldError> {
        self.validation
            .run_validated_pass(hugr, reg, |hugr: &mut H, _| {
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
                    let Some((replace, removes)) = find_consts(hugr, hugr.nodes(), reg).next()
                    else {
                        break Ok(());
                    };
                    hugr.apply_rewrite(replace)?;
                    for rem in removes {
                        // We are optimistically applying these [RemoveLoadConstant] and
                        // [RemoveConst] rewrites without checking whether the nodes
                        // they attempt to remove have remaining uses. If they do, then
                        // the rewrite fails and we move on.
                        if let Ok(const_node) = hugr.apply_rewrite(rem) {
                            // if the LoadConst was removed, try removing the Const too.
                            let _ = hugr.apply_rewrite(RemoveConst(const_node));
                        }
                    }
                }
            })
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
    debug_assert!(fold_result.as_ref().map_or(true, |x| x.len()
        == op.value_port_count(Direction::Outgoing)));
    fold_result
}

/// Generate a graph that loads and outputs `consts` in order, validating
/// against `reg`.
fn const_graph(consts: Vec<Value>, reg: &ExtensionRegistry) -> Hugr {
    let const_types = consts.iter().map(Value::get_type).collect_vec();
    let mut b = DFGBuilder::new(ft2(type_row![], const_types)).unwrap();

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
    ConstantFoldPass::default().run(h, reg).unwrap()
}

#[cfg(test)]
mod test;

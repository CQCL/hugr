#![warn(missing_docs)]
//! Constant-folding pass.
//! An (example) use of the [dataflow analysis framework](super::dataflow).

pub mod value_handle;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

use hugr_core::{
    HugrView, IncomingPort, Node, NodeIndex, OutgoingPort, PortIndex, Wire,
    hugr::hugrmut::HugrMut,
    ops::{
        Const, DataflowOpTrait, ExtensionOp, LoadConstant, OpType, Value, constant::OpaqueValue,
    },
    types::EdgeKind,
};
use value_handle::ValueHandle;

use crate::dead_code::{DeadCodeElimPass, PreserveNode};
use crate::{ComposablePass, composable::validate_if_test};
use crate::{
    IncludeExports,
    dataflow::{
        ConstLoader, ConstLocation, DFContext, Machine, PartialValue, TailLoopTermination,
        partial_from_const,
    },
};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
///
/// Note that by default we assume that only the entrypoint is reachable and
/// only if it is not the module root; see [Self::with_inputs]. Mutation
/// occurs anywhere beneath the entrypoint.
pub struct ConstantFoldPass {
    allow_increase_termination: bool,
    /// Each outer key Node must be either:
    ///   - a `FuncDefn` child of the module-root
    ///   - the entrypoint
    inputs: HashMap<Node, HashMap<IncomingPort, Value>>,
}

#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
/// Errors produced by [`ConstantFoldPass`].
pub enum ConstFoldError {
    /// Error raised when inputs are provided for a Node that is neither a dataflow
    /// parent, nor a [CFG](OpType::CFG), nor a [Conditional](OpType::Conditional).
    #[error("{node} has OpType {op} which cannot be an entry-point")]
    InvalidEntryPoint {
        /// The node which was specified as an entry-point
        node: Node,
        /// The `OpType` of the node
        op: OpType,
    },
    /// Inputs were provided for a node that is not in the hugr.
    #[error("Entry-point {node} is not part of the Hugr")]
    MissingEntryPoint {
        /// The missing node
        node: Node,
    },
}

impl ConstantFoldPass {
    /// Allows the pass to remove potentially-non-terminating [`TailLoop`]s and [`CFG`] if their
    /// result (if/when they do terminate) is either known or not needed.
    ///
    /// [`TailLoop`]: hugr_core::ops::TailLoop
    /// [`CFG`]: hugr_core::ops::CFG
    #[must_use]
    pub fn allow_increase_termination(mut self) -> Self {
        self.allow_increase_termination = true;
        self
    }

    /// Specifies a number of external inputs to an entry point of the Hugr.
    /// In normal use, for Module-rooted Hugrs, `node` is a `FuncDefn` (child of the root);
    /// for non-Module-rooted Hugrs, `node` is the [HugrView::entrypoint]. (This is not
    /// enforced, but it must be a container and not a module itself.)
    ///
    /// Multiple calls for the same entry-point combine their values, with later
    /// values on the same in-port replacing earlier ones.
    ///
    /// Note that providing empty `inputs` indicates that we must preserve the ability
    /// to compute the result of `node` for all possible inputs.
    /// * If the entrypoint is the module-root, this method should be called for every
    /// [FuncDefn] that is externally callable
    /// * Otherwise, i.e. if the entrypoint is not the module-root,
    ///    * The default is to assume the entrypoint is callable with any inputs;
    ///    * If `node` is the entrypoint, this method allows to restrict the possible inputs
    ///    * If `node` is beneath the entrypoint, this merely degrades the analysis. (We
    ///      will mutate only beneath the entrypoint, but using results of analysing the
    ///      whole Hugr wrt. the specified/any inputs too).
    /// 
    /// [FuncDefn]: hugr_core::ops::FuncDefn
    pub fn with_inputs(
        mut self,
        node: Node,
        inputs: impl IntoIterator<Item = (impl Into<IncomingPort>, Value)>,
    ) -> Self {
        self.inputs
            .entry(node)
            .or_default()
            .extend(inputs.into_iter().map(|(p, v)| (p.into(), v)));
        self
    }
}

impl<H: HugrMut<Node = Node> + 'static> ComposablePass<H> for ConstantFoldPass {
    type Error = ConstFoldError;
    type Result = ();

    /// Run the Constant Folding pass.
    ///
    /// # Errors
    ///
    /// [ConstFoldError] if inputs were provided via [`Self::with_inputs`] for an invalid node.
    fn run(&self, hugr: &mut H) -> Result<(), ConstFoldError> {
        let fresh_node = Node::from(portgraph::NodeIndex::new(
            hugr.nodes().max().map_or(0, |n| n.index() + 1),
        ));
        let mut m = Machine::new(&hugr);
        for (&n, in_vals) in &self.inputs {
            if !hugr.contains_node(n) {
                return Err(ConstFoldError::MissingEntryPoint { node: n });
            }
            m.prepopulate_inputs(
                n,
                in_vals.iter().map(|(p, v)| {
                    let const_with_dummy_loc = partial_from_const(
                        &ConstFoldContext,
                        ConstLocation::Field(p.index(), &fresh_node.into()),
                        v,
                    );
                    (*p, const_with_dummy_loc)
                }),
            )
            .map_err(|op| ConstFoldError::InvalidEntryPoint { node: n, op })?;
        }

        let results = m.run(ConstFoldContext, []);
        let mb_root_inp = hugr.get_io(hugr.entrypoint()).map(|[i, _]| i);

        let wires_to_break = hugr
            .entry_descendants()
            .flat_map(|n| hugr.node_inputs(n).map(move |ip| (n, ip)))
            .filter(|(n, ip)| {
                *n != hugr.entrypoint()
                    && matches!(hugr.get_optype(*n).port_kind(*ip), Some(EdgeKind::Value(_)))
            })
            .filter_map(|(n, ip)| {
                let (src, outp) = hugr.single_linked_output(n, ip).unwrap();
                // Avoid breaking edges from existing LoadConstant (we'd only add another)
                // or from root input node (any "external inputs" provided will show up here
                //   - potentially also in other places which this won't catch)
                (!hugr.get_optype(src).is_load_constant() && Some(src) != mb_root_inp).then_some((
                    n,
                    ip,
                    results
                        .try_read_wire_concrete::<Value>(Wire::new(src, outp))
                        .ok()?,
                ))
            })
            .collect::<Vec<_>>();
        // Sadly the results immutably borrow the hugr, so we must extract everything we need before mutation
        let terminating_tail_loops = hugr
            .entry_descendants()
            .filter(|n| {
                results.tail_loop_terminates(*n) == Some(TailLoopTermination::NeverContinues)
            })
            .collect::<Vec<_>>();

        for (n, inport, v) in wires_to_break {
            let parent = hugr.get_parent(n).unwrap();
            let datatype = v.get_type();
            // We could try hash-consing identical Consts, but not ATM
            let cst = hugr.add_node_with_parent(parent, Const::new(v));
            let lcst = hugr.add_node_with_parent(parent, LoadConstant { datatype });
            hugr.connect(cst, OutgoingPort::from(0), lcst, IncomingPort::from(0));
            hugr.disconnect(n, inport);
            hugr.connect(lcst, OutgoingPort::from(0), n, inport);
        }
        // Eliminate dead code not required for the same entry points.
        DeadCodeElimPass::<H>::default()
            .with_entry_points(self.inputs.keys().copied())
            .set_preserve_callback(if self.allow_increase_termination {
                Arc::new(|_, _| PreserveNode::CanRemoveIgnoringChildren)
            } else {
                Arc::new(move |h, n| {
                    if terminating_tail_loops.contains(&n) {
                        PreserveNode::DeferToChildren
                    } else {
                        PreserveNode::default_for(h, n)
                    }
                })
            })
            .run(hugr)
            .map_err(|inf| match inf {})?; // TODO use into_ok when available
        Ok(())
    }
}

const NO_INPUTS: [(IncomingPort, Value); 0] = [];

/// Exhaustively apply constant folding to a HUGR.
/// If the Hugr's entrypoint is its [`Module`], assumes all [`FuncDefn`] children are reachable.
/// Otherwise, assume that the [HugrView::entrypoint] is itself reachable.
///
/// [`FuncDefn`]: hugr_core::ops::OpType::FuncDefn
/// [`Module`]: hugr_core::ops::OpType::Module
#[deprecated(note = "Use fold_constants, or manually configure ConstantFoldPass")]
pub fn constant_fold_pass<H: HugrMut<Node = Node> + 'static>(mut h: impl AsMut<H>) {
    let h = h.as_mut();
    let c = ConstantFoldPass::default();
    let c = if h.get_optype(h.entrypoint()).is_module() {
        h.children(h.entrypoint())
            .filter(|n| h.get_optype(*n).is_func_defn())
            .fold(c, |c, n| c.with_inputs(n, NO_INPUTS.clone()))
    } else {
        c
    };
    validate_if_test(c, h).unwrap();
}

/// Exhaustively apply constant folding to a HUGR.
/// Assumes that the Hugr's entrypoint is reachable (if it is not a [`Module`]).
/// Also uses `policy` to determine which public [`FuncDefn`] children of the [`HugrView::module_root`] are reachable.
///
/// [`Module`]: hugr_core::ops::OpType::Module
/// [`FuncDefn`]: hugr_core::ops::OpType::FuncDefn
pub fn fold_constants(h: &mut (impl HugrMut<Node = Node> + 'static), policy: IncludeExports) {
    let mut funcs = Vec::new();
    if !h.entrypoint_optype().is_module() {
        funcs.push(h.entrypoint());
    }
    if policy.for_hugr(&h) {
        funcs.extend(
            h.children(h.module_root())
                .filter(|n| h.get_optype(*n).is_func_defn()),
        )
    }
    let c = funcs.into_iter().fold(ConstantFoldPass::default(), |c, n| {
        c.with_inputs(n, NO_INPUTS.clone())
    });
    validate_if_test(c, h).unwrap();
}

struct ConstFoldContext;

impl ConstLoader<ValueHandle<Node>> for ConstFoldContext {
    type Node = Node;

    fn value_from_opaque(
        &self,
        loc: ConstLocation<Node>,
        val: &OpaqueValue,
    ) -> Option<ValueHandle<Node>> {
        Some(ValueHandle::new_opaque(loc, val.clone()))
    }

    fn value_from_const_hugr(
        &self,
        loc: ConstLocation<Node>,
        h: &hugr_core::Hugr,
    ) -> Option<ValueHandle<Node>> {
        Some(ValueHandle::new_const_hugr(loc, Box::new(h.clone())))
    }
}

impl DFContext<ValueHandle<Node>> for ConstFoldContext {
    fn interpret_leaf_op(
        &mut self,
        node: Node,
        op: &ExtensionOp,
        ins: &[PartialValue<ValueHandle<Node>>],
        outs: &mut [PartialValue<ValueHandle<Node>>],
    ) {
        let sig = op.signature();
        let known_ins = sig
            .input_types()
            .iter()
            .enumerate()
            .zip(ins.iter())
            .filter_map(|((i, ty), pv)| {
                pv.clone()
                    .try_into_concrete(ty)
                    .ok()
                    .map(|v| (IncomingPort::from(i), v))
            })
            .collect::<Vec<_>>();
        for (p, v) in op.constant_fold(&known_ins).unwrap_or_default() {
            outs[p.index()] =
                partial_from_const(self, ConstLocation::Field(p.index(), &node.into()), &v);
        }
    }
}

#[cfg(test)]
mod test;

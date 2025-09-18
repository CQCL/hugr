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

use crate::dataflow::{
    ConstLoader, ConstLocation, DFContext, Machine, PartialValue, TailLoopTermination,
    partial_from_const,
};
use crate::dead_code::{DeadCodeElimError, DeadCodeElimPass, PreserveNode};
use crate::{ComposablePass, composable::validate_if_test};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstantFoldPass {
    allow_increase_termination: bool,
    /// Each outer key Node must be either:
    ///   - a `FuncDefn` child of the root, if the root is a module; or
    ///   - the root, if the root is not a Module
    inputs: HashMap<Node, HashMap<IncomingPort, Value>>,
}

#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
/// Errors produced by [`ConstantFoldPass`].
pub enum ConstFoldError {
    /// Error raised when a Node is specified as an entry-point but
    /// is neither a dataflow parent, nor a [CFG](OpType::CFG), nor
    /// a [Conditional](OpType::Conditional).
    #[error("{node} has OpType {op} which cannot be an entry-point")]
    InvalidEntryPoint {
        /// The node which was specified as an entry-point
        node: Node,
        /// The `OpType` of the node
        op: OpType,
    },
    /// The chosen entrypoint is not in the hugr.
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
    /// In normal use, for Module-rooted Hugrs, `node` is a `FuncDefn` child of the root;
    /// or for non-Module-rooted Hugrs, `node` is the root of the Hugr. (This is not
    /// enforced, but it must be a container and not a module itself.)
    ///
    /// Multiple calls for the same entry-point combine their values, with later
    /// values on the same in-port replacing earlier ones.
    ///
    /// Note that if `inputs` is empty, this still marks the node as an entry-point, i.e.
    /// we must preserve nodes required to compute its result.
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
    /// [`ConstFoldError::InvalidEntryPoint`] if an entry-point added by [`Self::with_inputs`]
    /// was of an invalid [`OpType`]
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
            .map_err(|e| match e {
                DeadCodeElimError::NodeNotFound(_) => {
                    panic!("ConstFoldError::MissingEntrypoint not raised above")
                }
            })?;
        Ok(())
    }
}

/// Exhaustively apply constant folding to a HUGR.
/// If the Hugr's entrypoint is its [`Module`], assumes all [`FuncDefn`] children are reachable.
///
/// [`FuncDefn`]: hugr_core::ops::OpType::FuncDefn
/// [`Module`]: hugr_core::ops::OpType::Module
pub fn constant_fold_pass<H: HugrMut<Node = Node> + 'static>(mut h: impl AsMut<H>) {
    let h = h.as_mut();
    let c = ConstantFoldPass::default();
    let c = if h.get_optype(h.entrypoint()).is_module() {
        let no_inputs: [(IncomingPort, _); 0] = [];
        h.children(h.entrypoint())
            .filter(|n| h.get_optype(*n).is_func_defn())
            .fold(c, |c, n| c.with_inputs(n, no_inputs.iter().cloned()))
    } else {
        c
    };
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

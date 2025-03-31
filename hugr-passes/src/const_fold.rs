#![warn(missing_docs)]
//! Constant-folding pass.
//! An (example) use of the [dataflow analysis framework](super::dataflow).

pub mod value_handle;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

use hugr_core::{
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, ExtractHugr, HierarchyView},
    },
    ops::{
        constant::OpaqueValue, handle::FuncID, Const, DataflowOpTrait, ExtensionOp, LoadConstant,
        OpType, Value,
    },
    types::{EdgeKind, TypeArg},
    HugrView, IncomingPort, Node, NodeIndex, OutgoingPort, PortIndex, Wire,
};
use value_handle::ValueHandle;

use crate::dataflow::{
    partial_from_const, ConstLoader, ConstLocation, DFContext, Machine, PartialValue,
    TailLoopTermination,
};
use crate::dead_code::{DeadCodeElimPass, PreserveNode};
use crate::validation::{ValidatePassError, ValidationLevel};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstantFoldPass {
    validation: ValidationLevel,
    allow_increase_termination: bool,
    /// Each outer key Node must be either:
    ///   - a FuncDefn child of the root, if the root is a module; or
    ///   - the root, if the root is not a Module
    inputs: HashMap<Node, HashMap<IncomingPort, Value>>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors produced by [ConstantFoldPass].
pub enum ConstFoldError {
    #[error(transparent)]
    #[allow(missing_docs)]
    ValidationError(#[from] ValidatePassError),
    /// Error raised when a Node is specified as an entry-point but
    /// is neither a dataflow parent, nor a [CFG](OpType::CFG), nor
    /// a [Conditional](OpType::Conditional).
    #[error("Node {_0} has OpType {_1} which cannot be an entry-point")]
    InvalidEntryPoint(Node, OpType),
}

impl ConstantFoldPass {
    /// Sets the validation level used before and after the pass is run
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Allows the pass to remove potentially-non-terminating [TailLoop]s and [CFG] if their
    /// result (if/when they do terminate) is either known or not needed.
    ///
    /// [TailLoop]: hugr_core::ops::TailLoop
    /// [CFG]: hugr_core::ops::CFG
    pub fn allow_increase_termination(mut self) -> Self {
        self.allow_increase_termination = true;
        self
    }

    /// Specifies a number of external inputs to an entry point of the Hugr.
    /// In normal use, for Module-rooted Hugrs, `node` is a FuncDefn child of the root;
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

    /// Run the Constant Folding pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ConstFoldError> {
        let fresh_node = Node::from(portgraph::NodeIndex::new(
            hugr.nodes().max().map_or(0, |n| n.index() + 1),
        ));
        let mut m = Machine::new(&hugr);
        for (&n, in_vals) in self.inputs.iter() {
            m.prepopulate_inputs(
                n,
                in_vals.iter().map(|(p, v)| {
                    let const_with_dummy_loc = partial_from_const(
                        &ConstFoldContext(hugr),
                        ConstLocation::Field(p.index(), &fresh_node.into()),
                        v,
                    );
                    (*p, const_with_dummy_loc)
                }),
            )
            .map_err(|opty| ConstFoldError::InvalidEntryPoint(n, opty))?;
        }

        let results = m.run(ConstFoldContext(hugr), []);
        let mb_root_inp = hugr.get_io(hugr.root()).map(|[i, _]| i);

        let wires_to_break = hugr
            .nodes()
            .flat_map(|n| hugr.node_inputs(n).map(move |ip| (n, ip)))
            .filter(|(n, ip)| {
                *n != hugr.root()
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
                        .try_read_wire_concrete::<Value, _, _>(Wire::new(src, outp))
                        .ok()?,
                ))
            })
            .collect::<Vec<_>>();
        // Sadly the results immutably borrow the hugr, so we must extract everything we need before mutation
        let terminating_tail_loops = hugr
            .nodes()
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
        DeadCodeElimPass::default()
            .with_entry_points(self.inputs.keys().cloned())
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
            .run(hugr)?;
        Ok(())
    }

    /// Run the pass using this configuration.
    ///
    /// # Errors
    ///
    /// [ConstFoldError::ValidationError] if the Hugr does not validate before/afnerwards
    /// (if [Self::validation_level] is set, or in tests)
    ///
    /// [ConstFoldError::InvalidEntryPoint] if an entry-point added by [Self::with_inputs]
    /// was of an invalid OpType
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), ConstFoldError> {
        self.validation
            .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
    }
}

/// Exhaustively apply constant folding to a HUGR.
/// If the Hugr is [Module]-rooted, assumes all [FuncDefn] children are reachable.
///
/// [FuncDefn]: hugr_core::ops::OpType::FuncDefn
/// [Module]: hugr_core::ops::OpType::Module
pub fn constant_fold_pass<H: HugrMut>(h: &mut H) {
    let c = ConstantFoldPass::default();
    let c = if h.get_optype(h.root()).is_module() {
        let no_inputs: [(IncomingPort, _); 0] = [];
        h.children(h.root())
            .filter(|n| h.get_optype(*n).is_func_defn())
            .fold(c, |c, n| c.with_inputs(n, no_inputs.iter().cloned()))
    } else {
        c
    };
    c.run(h).unwrap()
}

struct ConstFoldContext<'a, H>(&'a H);

impl<H: HugrView<Node = Node>> ConstLoader<ValueHandle<H::Node>> for ConstFoldContext<'_, H> {
    type Node = H::Node;

    fn value_from_opaque(
        &self,
        loc: ConstLocation<H::Node>,
        val: &OpaqueValue,
    ) -> Option<ValueHandle<H::Node>> {
        Some(ValueHandle::new_opaque(loc, val.clone()))
    }

    fn value_from_const_hugr(
        &self,
        loc: ConstLocation<H::Node>,
        h: &hugr_core::Hugr,
    ) -> Option<ValueHandle<H::Node>> {
        Some(ValueHandle::new_const_hugr(loc, Box::new(h.clone())))
    }

    fn value_from_function(
        &self,
        node: H::Node,
        type_args: &[TypeArg],
    ) -> Option<ValueHandle<H::Node>> {
        if !type_args.is_empty() {
            // TODO: substitution across Hugr (https://github.com/CQCL/hugr/issues/709)
            return None;
        };
        // Returning the function body as a value, here, would be sufficient for inlining IndirectCall
        // but not for transforming to a direct Call.
        let func = DescendantsGraph::<FuncID<true>>::try_new(self.0, node).ok()?;
        Some(ValueHandle::new_const_hugr(
            ConstLocation::Node(node),
            Box::new(func.extract_hugr()),
        ))
    }
}

impl<H: HugrView<Node = Node>> DFContext<ValueHandle<H::Node>> for ConstFoldContext<'_, H> {
    fn interpret_leaf_op(
        &mut self,
        node: H::Node,
        op: &ExtensionOp,
        ins: &[PartialValue<ValueHandle<H::Node>>],
        outs: &mut [PartialValue<ValueHandle<H::Node>>],
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

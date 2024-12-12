#![warn(missing_docs)]
//! Constant-folding pass.
//! An (example) use of the [dataflow analysis framework](super::dataflow).

pub mod value_handle;
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

use hugr_core::{
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, ExtractHugr, HierarchyView},
    },
    ops::{
        constant::OpaqueValue, handle::FuncID, Const, DataflowOpTrait, ExtensionOp, LoadConstant,
        Value,
    },
    types::{EdgeKind, TypeArg},
    HugrView, IncomingPort, Node, NodeIndex, OutgoingPort, PortIndex, Wire,
};
use value_handle::ValueHandle;

use crate::dataflow::{
    partial_from_const, AbstractValue, AnalysisResults, ConstLoader, ConstLocation, DFContext,
    Machine, PartialValue, TailLoopTermination,
};
use crate::validation::{ValidatePassError, ValidationLevel};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstantFoldPass {
    validation: ValidationLevel,
    allow_increase_termination: bool,
    inputs: HashMap<IncomingPort, Value>,
}

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors produced by [ConstantFoldPass].
pub enum ConstFoldError {
    #[error(transparent)]
    #[allow(missing_docs)]
    ValidationError(#[from] ValidatePassError),
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

    /// Specifies any number of external inputs to provide to the Hugr (on root-node
    /// in-ports). Each supersedes any previous value on the same in-port.
    pub fn with_inputs(
        mut self,
        inputs: impl IntoIterator<Item = (impl Into<IncomingPort>, Value)>,
    ) -> Self {
        self.inputs
            .extend(inputs.into_iter().map(|(p, v)| (p.into(), v)));
        self
    }

    /// Run the Constant Folding pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ConstFoldError> {
        let fresh_node = Node::from(portgraph::NodeIndex::new(
            hugr.nodes().max().map_or(0, |n| n.index() + 1),
        ));
        let inputs = self.inputs.iter().map(|(p, v)| {
            (
                *p,
                partial_from_const(
                    &ConstFoldContext(hugr),
                    ConstLocation::Field(p.index(), &fresh_node.into()),
                    v,
                ),
            )
        });

        let results = Machine::new(&hugr).run(ConstFoldContext(hugr), inputs);
        let mut keep_nodes = HashSet::new();
        self.find_needed_nodes(&results, &mut keep_nodes);
        let [root_inp, _] = hugr.get_io(hugr.root()).unwrap();

        let remove_nodes = hugr
            .nodes()
            .filter(|n| !keep_nodes.contains(n))
            .collect::<HashSet<_>>();
        let wires_to_break = keep_nodes
            .into_iter()
            .flat_map(|n| hugr.node_inputs(n).map(move |ip| (n, ip)))
            .filter(|(n, ip)| {
                *n != hugr.root()
                    && matches!(hugr.get_optype(*n).port_kind(*ip), Some(EdgeKind::Value(_)))
            })
            // Note we COULD filter out (avoid breaking) wires from other nodes that we are keeping.
            // This would insert fewer constants, but potentially expose less parallelism.
            .filter_map(|(n, ip)| {
                let (src, outp) = hugr.single_linked_output(n, ip).unwrap();
                // Avoid breaking edges from existing LoadConstant (we'd only add another)
                // or from root input node (any "external inputs" provided will show up here
                //   - potentially also in other places which this won't catch)
                (!hugr.get_optype(src).is_load_constant() && src != root_inp).then_some((
                    n,
                    ip,
                    results
                        .try_read_wire_concrete::<Value, _, _>(Wire::new(src, outp))
                        .ok()?,
                ))
            })
            .collect::<Vec<_>>();

        for (n, import, v) in wires_to_break {
            let parent = hugr.get_parent(n).unwrap();
            let datatype = v.get_type();
            // We could try hash-consing identical Consts, but not ATM
            let cst = hugr.add_node_with_parent(parent, Const::new(v));
            let lcst = hugr.add_node_with_parent(parent, LoadConstant { datatype });
            hugr.connect(cst, OutgoingPort::from(0), lcst, IncomingPort::from(0));
            hugr.disconnect(n, import);
            hugr.connect(lcst, OutgoingPort::from(0), n, import);
        }
        for n in remove_nodes {
            hugr.remove_node(n);
        }
        Ok(())
    }

    /// Run the pass using this configuration
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), ConstFoldError> {
        self.validation
            .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
    }

    fn find_needed_nodes<H: HugrView>(
        &self,
        results: &AnalysisResults<ValueHandle, H>,
        needed: &mut HashSet<Node>,
    ) {
        let mut q = VecDeque::new();
        let h = results.hugr();
        q.push_back(h.root());
        while let Some(n) = q.pop_front() {
            if !needed.insert(n) {
                continue;
            };

            if h.get_optype(n).is_cfg() {
                for bb in h.children(n) {
                    //if results.bb_reachable(bb).unwrap() { // no, we'd need to patch up predicates
                    q.push_back(bb);
                }
            } else if let Some(inout) = h.get_io(n) {
                // Dataflow. Find minimal nodes necessary to compute output, including StateOrder edges.
                q.extend(inout); // Input also necessary for legality even if unreachable

                if !self.allow_increase_termination {
                    // Also add on anything that might not terminate (even if results not required -
                    // if its results are required we'll add it by following dataflow, below.)
                    for ch in h.children(n) {
                        if might_diverge(results, ch) {
                            q.push_back(ch);
                        }
                    }
                }
            }
            // Also follow dataflow demand
            for (src, op) in h.all_linked_outputs(n) {
                let needs_predecessor = match h.get_optype(src).port_kind(op).unwrap() {
                    EdgeKind::Value(_) => {
                        h.get_optype(src).is_load_constant()
                            || results
                                .try_read_wire_concrete::<Value, _, _>(Wire::new(src, op))
                                .is_err()
                    }
                    EdgeKind::StateOrder | EdgeKind::Const(_) | EdgeKind::Function(_) => true,
                    EdgeKind::ControlFlow => false, // we always include all children of a CFG above
                    _ => true, // needed as EdgeKind non-exhaustive; not knowing what it is, assume the worst
                };
                if needs_predecessor {
                    q.push_back(src);
                }
            }
        }
    }
}

// "Diverge" aka "never-terminate"
// TODO would be more efficient to compute this bottom-up and cache (dynamic programming)
fn might_diverge<V: AbstractValue>(results: &AnalysisResults<V, impl HugrView>, n: Node) -> bool {
    let op = results.hugr().get_optype(n);
    if op.is_cfg() {
        // TODO if the CFG has no cycles (that are possible given predicates)
        // then we could say it definitely terminates (i.e. return false)
        true
    } else if op.is_tail_loop()
        && results.tail_loop_terminates(n).unwrap() != TailLoopTermination::NeverContinues
    {
        // If we can even figure out the number of iterations is bounded that would allow returning false.
        true
    } else {
        // Node does not introduce non-termination, but still non-terminates if any of its children does
        results
            .hugr()
            .children(n)
            .any(|ch| might_diverge(results, ch))
    }
}

/// Exhaustively apply constant folding to a HUGR.
pub fn constant_fold_pass<H: HugrMut>(h: &mut H) {
    ConstantFoldPass::default().run(h).unwrap()
}

struct ConstFoldContext<'a, H>(&'a H);

impl<H: HugrView> std::ops::Deref for ConstFoldContext<'_, H> {
    type Target = H;
    fn deref(&self) -> &H {
        self.0
    }
}

impl<H: HugrView> ConstLoader<ValueHandle> for ConstFoldContext<'_, H> {
    fn value_from_opaque(&self, loc: ConstLocation, val: &OpaqueValue) -> Option<ValueHandle> {
        Some(ValueHandle::new_opaque(loc, val.clone()))
    }

    fn value_from_const_hugr(
        &self,
        loc: ConstLocation,
        h: &hugr_core::Hugr,
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_const_hugr(loc, Box::new(h.clone())))
    }

    fn value_from_function(&self, node: Node, type_args: &[TypeArg]) -> Option<ValueHandle> {
        if !type_args.is_empty() {
            // TODO: substitution across Hugr (https://github.com/CQCL/hugr/issues/709)
            return None;
        };
        // Returning the function body as a value, here, would be sufficient for inlining IndirectCall
        // but not for transforming to a direct Call.
        let func = DescendantsGraph::<FuncID<true>>::try_new(&**self, node).ok()?;
        Some(ValueHandle::new_const_hugr(
            ConstLocation::Node(node),
            Box::new(func.extract_hugr()),
        ))
    }
}

impl<H: HugrView> DFContext<ValueHandle> for ConstFoldContext<'_, H> {
    fn interpret_leaf_op(
        &mut self,
        node: Node,
        op: &ExtensionOp,
        ins: &[PartialValue<ValueHandle>],
        outs: &mut [PartialValue<ValueHandle>],
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

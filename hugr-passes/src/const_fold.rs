#![warn(missing_docs)]
//! An (example) use of the [super::dataflow](dataflow-analysis framework)
//! to perform constant-folding.

pub mod value_handle;
use std::collections::{HashSet, VecDeque};

use hugr_core::{
    extension::ExtensionRegistry,
    hugr::hugrmut::HugrMut,
    hugr::views::{DescendantsGraph, ExtractHugr, HierarchyView},
    ops::constant::OpaqueValue,
    ops::{handle::FuncID, Const, DataflowOpTrait, ExtensionOp, LoadConstant, Value},
    types::{EdgeKind, TypeArg},
    HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire,
};
use value_handle::ValueHandle;

use crate::{
    dataflow::{
        AnalysisResults, ConstLoader, DFContext, Machine, PartialValue, TailLoopTermination,
    },
    validation::{ValidatePassError, ValidationLevel},
};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstFoldPass {
    validation: ValidationLevel,
    allow_increase_termination: bool,
}

impl ConstFoldPass {
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

    /// Run the Constant Folding pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ValidatePassError> {
        let results = Machine::default().run(ConstFoldContext(hugr), []);
        let mut keep_nodes = HashSet::new();
        self.find_needed_nodes(&results, hugr.root(), &mut keep_nodes);

        let remove_nodes = results
            .hugr()
            .nodes()
            .filter(|n| !keep_nodes.contains(n))
            .collect::<HashSet<_>>();
        let wires_to_break = keep_nodes
            .into_iter()
            .flat_map(|n| hugr.node_inputs(n).map(move |ip| (n, ip)))
            .filter(|(n, ip)| {
                matches!(
                    hugr.get_optype(*n).port_kind(*ip).unwrap(),
                    EdgeKind::Value(_)
                )
            })
            // Note we COULD filter out (avoid breaking) wires from other nodes that we are keeping.
            // This would insert fewer constants, but potentially expose less parallelism.
            .filter_map(|(n, ip)| {
                let (src, outp) = hugr.single_linked_output(n, ip).unwrap();
                (!hugr.get_optype(src).is_load_constant()).then_some((
                    n,
                    ip,
                    results
                        .try_read_wire_value::<Value, _, _>(Wire::new(src, outp))
                        .ok()?,
                ))
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
        for n in remove_nodes {
            hugr.remove_node(n);
        }
        Ok(())
    }

    /// Run the pass using this configuration
    pub fn run<H: HugrMut>(
        &self,
        hugr: &mut H,
        reg: &ExtensionRegistry,
    ) -> Result<(), ValidatePassError> {
        self.validation
            .run_validated_pass(hugr, reg, |hugr: &mut H, _| self.run_no_validate(hugr))
    }

    fn find_needed_nodes<H: HugrView>(
        &self,
        results: &AnalysisResults<ValueHandle, ConstFoldContext<H>>,
        root: Node,
        needed: &mut HashSet<Node>,
    ) {
        let mut q = VecDeque::new();
        q.push_back(root);
        let h = results.hugr();
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
                        if self.might_diverge(results, ch) {
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
                                .try_read_wire_value::<Value, _, _>(Wire::new(src, op))
                                .is_err()
                    }
                    EdgeKind::StateOrder | EdgeKind::Const(_) | EdgeKind::Function(_) => true,
                    EdgeKind::ControlFlow => panic!(),
                    _ => true, // needed as EdgeKind non-exhaustive; not knowing what it is, assume the worst
                };
                if needs_predecessor {
                    q.push_back(src);
                }
            }
        }
    }

    // "Diverge" aka "never-terminate"
    // TODO would be more efficient to compute this bottom-up and cache (dynamic programming)
    fn might_diverge(
        &self,
        results: &AnalysisResults<ValueHandle, impl DFContext<ValueHandle>>,
        n: Node,
    ) -> bool {
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
                .any(|ch| self.might_diverge(results, ch))
        }
    }
}

/// Exhaustively apply constant folding to a HUGR.
pub fn constant_fold_pass<H: HugrMut>(h: &mut H, reg: &ExtensionRegistry) {
    ConstFoldPass::default().run(h, reg).unwrap()
}

struct ConstFoldContext<'a, H>(&'a H);

impl<'a, H: HugrView> std::ops::Deref for ConstFoldContext<'a, H> {
    type Target = H;
    fn deref(&self) -> &H {
        self.0
    }
}

impl<'a, H: HugrView> ConstLoader<ValueHandle> for ConstFoldContext<'a, H> {
    fn value_from_opaque(
        &self,
        node: Node,
        fields: &[usize],
        val: &OpaqueValue,
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_opaque(node, fields, val.clone()))
    }

    fn value_from_const_hugr(
        &self,
        node: Node,
        fields: &[usize],
        h: &hugr_core::Hugr,
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_const_hugr(
            node,
            fields,
            Box::new(h.clone()),
        ))
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
            node,
            &[],
            Box::new(func.extract_hugr()),
        ))
    }
}

impl<'a, H: HugrView> DFContext<ValueHandle> for ConstFoldContext<'a, H> {
    type View = H;

    fn interpret_leaf_op(
        &self,
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
                Some((IncomingPort::from(i), pv.clone().try_into_value(ty).ok()?))
            })
            .collect::<Vec<_>>();
        for (p, v) in op.constant_fold(&known_ins).unwrap_or_default() {
            // Hmmm, we should (at least) key the value also by p
            outs[p.index()] = self.value_from_const(node, &v);
        }
    }
}

#[cfg(test)]
mod test;

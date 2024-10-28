//! An (example) use of the [super::dataflow](dataflow-analysis framework)
//! to perform constant-folding.

// These are pub because this "example" is used for testing the framework.
pub mod value_handle;
use std::collections::{HashSet, VecDeque};

use hugr_core::{
    extension::ExtensionRegistry,
    hugr::{
        hugrmut::HugrMut,
        views::{DescendantsGraph, ExtractHugr, HierarchyView},
    },
    ops::{constant::OpaqueValue, handle::FuncID, Const, ExtensionOp, LoadConstant, Value},
    types::{EdgeKind, TypeArg},
    HugrView, IncomingPort, Node, OutgoingPort, Wire,
};
use value_handle::ValueHandle;

use crate::{
    dataflow::{
        AnalysisResults, ConstLoader, Machine, PartialValue, TailLoopTermination, TotalContext,
    },
    validation::{ValidatePassError, ValidationLevel},
};

#[derive(Debug, Clone, Default)]
/// A configuration for the Constant Folding pass.
pub struct ConstFoldPass {
    validation: ValidationLevel,
    /// If true, allow to skip evaluating loops (whose results are not needed) even if
    /// we are not sure they will terminate. (If they definitely terminate then fair game.)
    pub allow_skip_loops: bool,
}

impl ConstFoldPass {
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Run the Constant Folding pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ValidatePassError> {
        let results = Machine::default().run(ConstFoldContext(hugr), []);
        let mut keep_nodes = HashSet::new();
        self.find_needed_nodes(&results, results.hugr.root(), &mut keep_nodes);

        let remove_nodes = results
            .hugr
            .nodes()
            .filter(|n| !keep_nodes.contains(n))
            .collect::<HashSet<_>>();
        for n in keep_nodes {
            // Every input either (a) is in keep_nodes, or (b) has a known value. Break all wires (b).
            for inport in results.hugr.node_inputs(n) {
                if matches!(
                    results.hugr.get_optype(n).port_kind(inport).unwrap(),
                    EdgeKind::Value(_)
                ) {
                    let (src, outp) = results.hugr.single_linked_output(n, inport).unwrap();
                    if let Ok(v) = results.try_read_wire_value(Wire::new(src, outp)) {
                        let parent = results.hugr.get_parent(n).unwrap();
                        let datatype = v.get_type();
                        // We could try hash-consing identical Consts, but not ATM
                        let hugr_mut = &mut *results.hugr.0;
                        let cst = hugr_mut.add_node_with_parent(parent, Const::new(v));
                        let lcst = hugr_mut.add_node_with_parent(parent, LoadConstant { datatype });
                        hugr_mut.connect(cst, OutgoingPort::from(0), lcst, IncomingPort::from(0));
                    }
                }
            }
        }
        for n in remove_nodes {
            hugr.remove_node(n);
        }
        Ok(())
    }

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
        results: &AnalysisResults<ValueHandle, H>,
        root: Node,
        needed: &mut HashSet<Node>,
    ) {
        let mut q = VecDeque::new();
        q.push_back(root);
        let h = &results.hugr;
        while let Some(n) = q.pop_front() {
            if !needed.insert(n) {
                continue;
            };
            if h.get_optype(n).is_cfg() {
                for bb in h.children(n) {
                    if results.bb_reachable(bb).unwrap()
                        && needed.insert(bb)
                        && h.get_optype(bb).is_dataflow_block()
                    {
                        q.push_back(bb);
                    }
                }
            } else if let Some(inout) = h.get_io(n) {
                // Dataflow. Find minimal nodes necessary to compute output, including StateOrder edges.
                q.extend(inout); // Input also necessary for legality even if unreachable

                // Also add on anything that might not terminate. We might also allow a custom predicate for extension ops?
                for ch in h.children(n) {
                    if h.get_optype(ch).is_cfg()
                        || (!self.allow_skip_loops
                            && h.get_optype(ch).is_tail_loop()
                            && results.tail_loop_terminates(ch).unwrap()
                                != TailLoopTermination::NeverContinues)
                    {
                        q.push_back(ch);
                    }
                }
            }
            // Also follow dataflow demand
            for (src, op) in h.all_linked_outputs(n) {
                let needs_predecessor = match h.get_optype(src).port_kind(op).unwrap() {
                    EdgeKind::Value(_) => results.try_read_wire_value(Wire::new(src, op)).is_err(),
                    EdgeKind::StateOrder | EdgeKind::Const(_) | EdgeKind::Function(_) => true,
                    EdgeKind::ControlFlow => panic!(),
                    _ => true, // needed for non-exhaustive; not knowing what it is, assume the worst
                };
                if needs_predecessor {
                    q.push_back(src);
                }
            }
        }
    }
}

/// Exhaustively apply constant folding to a HUGR.
pub fn constant_fold_pass<H: HugrMut>(h: &mut H, reg: &ExtensionRegistry) {
    ConstFoldPass::default().run(h, reg).unwrap()
}

struct ConstFoldContext<'a, H>(&'a mut H);

impl<'a, T: HugrView> AsRef<hugr_core::Hugr> for ConstFoldContext<'a, T> {
    fn as_ref(&self) -> &hugr_core::Hugr {
        self.0.base_hugr()
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
        if type_args.len() > 0 {
            // TODO: substitution across Hugr (https://github.com/CQCL/hugr/issues/709)
            return None;
        };
        // Returning the function body as a value, here, would be sufficient for inlining IndirectCall
        // but not for transforming to a direct Call.
        let func = DescendantsGraph::<FuncID<true>>::try_new(self, node).ok()?;
        Some(ValueHandle::new_const_hugr(
            node,
            &[],
            Box::new(func.extract_hugr()),
        ))
    }
}

impl<'a, H: HugrView> TotalContext<ValueHandle> for ConstFoldContext<'a, H> {
    type InterpretableVal = Value;

    fn interpret_leaf_op(
        &self,
        n: Node,
        op: &ExtensionOp,
        ins: &[(IncomingPort, Value)],
    ) -> Vec<(OutgoingPort, PartialValue<ValueHandle>)> {
        let ins = ins.iter().map(|(p, v)| (*p, v.clone())).collect::<Vec<_>>();
        op.constant_fold(&ins).map_or(Vec::new(), |outs| {
            outs.into_iter()
                .map(|(p, v)| {
                    (
                        p,
                        self.value_from_const(n, &v), // Hmmm, should (at least) also key by p
                    )
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod test;

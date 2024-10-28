//! An (example) use of the [super::dataflow](dataflow-analysis framework)
//! to perform constant-folding.

// These are pub because this "example" is used for testing the framework.
mod context;
pub mod value_handle;
use std::collections::{HashSet, VecDeque};

pub use context::ConstFoldContext;
use hugr_core::{
    extension::ExtensionRegistry,
    hugr::hugrmut::HugrMut,
    ops::{Const, LoadConstant},
    types::EdgeKind,
    HugrView, IncomingPort, Node, OutgoingPort, Wire,
};
use value_handle::ValueHandle;

use crate::{
    dataflow::{AnalysisResults, Machine, TailLoopTermination},
    validation::{ValidatePassError, ValidationLevel},
};

pub struct ConstFoldPass {
    validation: ValidationLevel,
    /// If true, allow to skip evaluating loops (whose results are not needed) even if
    /// we are not sure they will terminate. (If they definitely terminate then fair game.)
    pub allow_skip_loops: bool,
}

struct MutRefCell<'a, H>(&'a mut H);

impl<'a, T: HugrView> AsRef<hugr_core::Hugr> for MutRefCell<'a, T> {
    fn as_ref(&self) -> &hugr_core::Hugr {
        self.0.base_hugr()
    }
}

impl ConstFoldPass {
    /// Run the Constant Folding pass.
    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ValidatePassError> {
        let ctx = ConstFoldContext(MutRefCell(hugr));
        let results = Machine::default().run(ctx, []);
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
                        let hugr_mut = &mut *results.hugr.0 .0;
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
        container: Node,
        needed: &mut HashSet<Node>,
    ) {
        let h = &results.hugr;
        if h.get_optype(container).is_cfg() {
            for bb in h.children(container) {
                if results.bb_reachable(bb).unwrap()
                    && needed.insert(bb)
                    && h.get_optype(bb).is_dataflow_block()
                {
                    self.find_needed_nodes(results, bb, needed);
                }
            }
        } else {
            // Dataflow. Find minimal nodes necessary to compute output, including StateOrder edges.
            let [_inp, outp] = h.get_io(container).unwrap();
            let mut q = VecDeque::new();
            q.push_back(outp);
            // Add on anything that might not terminate. We might also allow a custom predicate for extension ops?
            for n in h.children(container) {
                if h.get_optype(n).is_cfg()
                    || (!self.allow_skip_loops
                        && h.get_optype(n).is_tail_loop()
                        && results.tail_loop_terminates(n).unwrap()
                            != TailLoopTermination::NeverContinues)
                {
                    q.push_back(n);
                }
            }
            while let Some(n) = q.pop_front() {
                if !needed.insert(n) {
                    continue;
                }
                for (src, op) in h.all_linked_outputs(n) {
                    let needs_predecessor = match h.get_optype(src).port_kind(op).unwrap() {
                        EdgeKind::Value(_) => {
                            results.try_read_wire_value(Wire::new(src, op)).is_err()
                        }
                        EdgeKind::StateOrder | EdgeKind::Const(_) | EdgeKind::Function(_) => true,
                        EdgeKind::ControlFlow => panic!(),
                        _ => true, // needed for non-exhaustive; not knowing what it is, assume the worst
                    };
                    if needs_predecessor {
                        q.push_back(src);
                    }
                }
                if h.get_optype(n).is_container() {
                    self.find_needed_nodes(results, container, needed);
                }
            }
        }
    }
}

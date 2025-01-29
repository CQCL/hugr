//! Pass for removing dead code, i.e. that computes values that are then discarded

use std::collections::{HashSet, VecDeque};

use hugr_core::{hugr::hugrmut::HugrMut, ops::OpType, HugrView, Node};

use crate::validation::{ValidatePassError, ValidationLevel};

/// Configuration for Dead Code Elimination pass
#[derive(Clone, Debug, Default)]
pub struct DeadCodeElimPass {
    entry_points: Vec<Node>,
    allow_increase_termination: bool,
    validation: ValidationLevel,
}

impl DeadCodeElimPass {
    /// Sets the validation level used before and after the pass is run
    #[allow(unused)]
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

    /// Mark some nodes as entry-points to the Hugr.
    /// The root node is assumed to be an entry point;
    /// for Module roots the client will want to mark some of the FuncDefn children
    /// as entry-points too.
    pub fn with_entry_points(mut self, entry_points: impl IntoIterator<Item = Node>) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    fn find_needed_nodes(&self, h: impl HugrView) -> HashSet<Node> {
        let mut needed = HashSet::new();
        let mut q = VecDeque::from_iter(self.entry_points.iter().cloned());
        q.push_front(h.root());
        while let Some(n) = q.pop_front() {
            if !needed.insert(n) {
                continue;
            };
            // TODO if we really want to support
            if matches!(
                h.get_optype(n),
                OpType::Conditional(_) | OpType::CFG(_) | OpType::Module(_)
            ) {
                // All Cases are reachable, but don't assume so for e.g. Constants/FuncDefns
                for ch in h.children(n) {
                    match h.get_optype(ch) {
                        // Include only if reachable by static edges (from Call/LoadConst/LoadFunction):
                        OpType::FuncDecl(_) | OpType::FuncDefn(_) | OpType::Const(_) => continue,
                        // Include all cases / BBs, and Aliases (we do not track their uses in types)
                        OpType::Case(_)
                        | OpType::DataflowBlock(_)
                        | OpType::ExitBlock(_)
                        | OpType::AliasDecl(_)
                        | OpType::AliasDefn(_) => q.push_back(ch),
                        op => panic!("Unexpected optype {}", op),
                    }
                }
            } else if let Some(inout) = h.get_io(n) {
                // Dataflow container, e.g. FuncDefn. Find minimal nodes necessary to compute output,
                // including StateOrder edges.
                q.extend(inout); // Input also necessary for legality even if unreachable

                if !self.allow_increase_termination {
                    // Also add on anything that might not terminate (even if results not required -
                    // if its results are required we'll add it by following dataflow, below.)
                    for ch in h.children(n) {
                        if might_diverge(&h, ch) {
                            q.push_back(ch);
                        }
                    }
                }
            }
            // Finally, follow dataflow demand (including e.g. edges from Call to FuncDefn)
            for src in h.input_neighbours(n) {
                // Following ControlFlow edges backwards is harmless, we've already assumed all
                // BBs are reachable above.
                q.push_back(src);
            }
        }
        needed
    }

    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), ValidatePassError> {
        self.validation
            .run_validated_pass(hugr, |h, _| self.run_no_validate(h))
    }

    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<(), ValidatePassError> {
        let needed = self.find_needed_nodes(&*hugr);
        let remove = hugr
            .nodes()
            .filter(|n| !needed.contains(n))
            .collect::<Vec<_>>();
        for n in remove {
            hugr.remove_node(n);
        }
        Ok(())
    }
}

// "Diverge" aka "never-terminate"
// TODO would be more efficient to compute this bottom-up and cache (dynamic programming)
fn might_diverge(h: &impl HugrView, n: Node) -> bool {
    match h.get_optype(n) {
        OpType::CFG(_) => {
            // TODO if the CFG has no cycles (that are possible given predicates)
            // then we could say it definitely terminates (i.e. return false)
            true
        }
        OpType::TailLoop(_) => {
            // If the TailLoop never continues, clearly it doesn't terminate, but we haven't got
            // dataflow results to tell us that. Instead rely on an earlier pass having rewritten
            // such a TailLoop into a non-loop.
            // Even just an upper-bound on the number of iterations would allow returning false.
            true
        }
        OpType::Call(_) => {
            // We could scan the target FuncDefn, but that might contain calls to itself, so we'd need
            // a "seen" set...instead just rely on calls being inlined if we want to remove them.
            true
        }
        _ => {
            // Node does not introduce non-termination, but still non-terminates if any of its children does
            h.children(n).any(|ch| might_diverge(h, ch))
        }
    }
}

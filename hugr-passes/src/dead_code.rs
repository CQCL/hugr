//! Pass for removing dead code, i.e. that computes values that are then discarded

use hugr_core::{hugr::hugrmut::HugrMut, ops::OpType, Hugr, HugrView, Node};
use std::fmt::{Debug, Formatter};
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use crate::validation::{ValidatePassError, ValidationLevel};

/// Configuration for Dead Code Elimination pass
#[derive(Clone, Default)]
pub struct DeadCodeElimPass {
    entry_points: Vec<Node>,
    diverge_callback: Option<Arc<DivergeCallback>>,
    validation: ValidationLevel,
}

impl Debug for DeadCodeElimPass {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Use "derive Debug" by defining an identical struct without the unprintable fields

        #[allow(unused)] // Rust ignores the derive-Debug in figuring out what's used
        #[derive(Debug)]
        struct DCEDebug<'a> {
            entry_points: &'a Vec<Node>,
            validation: ValidationLevel,
        }

        Debug::fmt(
            &DCEDebug {
                entry_points: &self.entry_points,
                validation: self.validation,
            },
            f,
        )
    }
}

pub type DivergeCallback = dyn Fn(&Hugr, Node) -> NodeDivergence;

pub enum NodeDivergence {
    #[allow(unused)]
    MustKeep,
    CanRemove,
    UseDefault,
    RemoveIfAllChildrenCanBeRemoved,
}

impl DeadCodeElimPass {
    /// Sets the validation level used before and after the pass is run
    #[allow(unused)]
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Allows setting a callback that determines whether a node is considered as diverging
    /// (non-terminating) - that is, nodes for which the callback returns
    ///   * Some(true) => cannot be removed, even if we don't need their results
    ///   * Some(false) => can be removed so long as we don't need their results
    ///     (note that this means we can remove their descendants too, *even if* said descendants diverge)
    ///   * None => use default algorithm for whether we can remove or not
    ///
    /// The default algorithm says that [Cfg], [Call] and [TailLoop] nodes can never be removed,
    /// nor can any node that (recursively) contains a diverging node.
    ///
    /// [Call]: hugr_core::ops::Call
    /// [CFG]: hugr_core::ops::CFG
    /// [TailLoop]: hugr_core::ops::TailLoop
    pub fn set_diverge_callback(mut self, cb: Arc<DivergeCallback>) -> Self {
        self.diverge_callback = Some(cb);
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
            if matches!(
                h.get_optype(n),
                OpType::Conditional(_) | OpType::CFG(_) | OpType::Module(_)
            ) {
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

                // Also add on anything that might not terminate (even if results not required -
                // if its results are required we'll add it by following dataflow, below.)
                for ch in h.children(n) {
                    if self.might_diverge(&h, ch) {
                        q.push_back(ch);
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

    // "Diverge" aka "never-terminate"
    // TODO would be more efficient to compute this bottom-up and cache (dynamic programming)
    fn might_diverge(&self, h: &impl HugrView, n: Node) -> bool {
        match self
            .diverge_callback
            .as_ref()
            .map_or(NodeDivergence::UseDefault, |f| f(h.base_hugr(), n))
        {
            NodeDivergence::MustKeep => return true,
            NodeDivergence::CanRemove => return false,
            NodeDivergence::UseDefault => {
                match h.get_optype(n) {
                    OpType::CFG(_) => {
                        // TODO if the CFG has no cycles (that are possible given predicates)
                        // then we could say it definitely terminates (i.e. return false)
                        return true;
                    }
                    OpType::TailLoop(_) => {
                        // If the TailLoop never continues, clearly it doesn't terminate, but we haven't got
                        // dataflow results to tell us that. Instead rely on an earlier pass having rewritten
                        // such a TailLoop into a non-loop.
                        // Even just an upper-bound on the number of iterations would allow returning false.
                        return true;
                    }
                    OpType::Call(_) => {
                        // We could scan the target FuncDefn, but that might contain calls to itself, so we'd need
                        // a "seen" set...instead just rely on calls being inlined if we want to remove them.
                        return true;
                    }
                    _ => (), // fall through to check children
                }
            }
            NodeDivergence::RemoveIfAllChildrenCanBeRemoved => (), // fall through to check children
        }

        // Node does not introduce non-termination, but still non-terminates if any of its children does
        h.children(n).any(|ch| self.might_diverge(h, ch))
    }
}

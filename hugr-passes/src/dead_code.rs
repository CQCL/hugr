//! Pass for removing dead code, i.e. that computes values that are then discarded

use hugr_core::{hugr::hugrmut::HugrMut, ops::OpType, Hugr, HugrView, Node};
use std::convert::Infallible;
use std::fmt::{Debug, Formatter};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use crate::ComposablePass;

/// Configuration for Dead Code Elimination pass
#[derive(Clone)]
pub struct DeadCodeElimPass {
    /// Nodes that are definitely needed - e.g. FuncDefns, but could be anything.
    /// Hugr Root is assumed to be an entry point even if not mentioned here.
    entry_points: Vec<Node>,
    /// Callback identifying nodes that must be preserved even if their
    /// results are not used. Defaults to [PreserveNode::default_for].
    preserve_callback: Arc<PreserveCallback>,
}

impl Default for DeadCodeElimPass {
    fn default() -> Self {
        Self {
            entry_points: Default::default(),
            preserve_callback: Arc::new(PreserveNode::default_for),
        }
    }
}

impl Debug for DeadCodeElimPass {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Use "derive Debug" by defining an identical struct without the unprintable fields

        #[allow(unused)] // Rust ignores the derive-Debug in figuring out what's used
        #[derive(Debug)]
        struct DCEDebug<'a> {
            entry_points: &'a Vec<Node>,
        }

        Debug::fmt(
            &DCEDebug {
                entry_points: &self.entry_points,
            },
            f,
        )
    }
}

/// Callback that identifies nodes that must be preserved even if their
/// results are not used. For example, (the default) [PreserveNode::default_for].
pub type PreserveCallback = dyn Fn(&Hugr, Node) -> PreserveNode;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// Signal that a node must be preserved even when its result is not used
pub enum PreserveNode {
    /// The node must be kept (nodes inside it may be removed)
    MustKeep,
    /// The node can be removed, even if nodes inside it must be kept
    /// - this will remove the descendants too, so use with care.
    CanRemoveIgnoringChildren,
    /// The node may be removed if-and-only-if all of its children can
    /// (must be kept iff any of its children must be kept).
    DeferToChildren,
}

impl PreserveNode {
    /// A conservative default for a given node. Just examines the node's [OpType]:
    /// * Assumes all Calls must be preserved. (One could scan the called FuncDefn, but would
    ///   also need to check for cycles in the [CallGraph](super::call_graph::CallGraph).)
    /// * Assumes all CFGs must be preserved. (One could, for example, allow acyclic
    ///   CFGs to be removed.)
    /// * Assumes all TailLoops must be preserved. (One could, for example, use dataflow
    ///   analysis to allow removal of TailLoops that never [Continue](hugr_core::ops::TailLoop::CONTINUE_TAG).)
    pub fn default_for(h: &Hugr, n: Node) -> PreserveNode {
        match h.get_optype(n) {
            OpType::CFG(_) | OpType::TailLoop(_) | OpType::Call(_) => PreserveNode::MustKeep,
            _ => Self::DeferToChildren,
        }
    }
}

impl DeadCodeElimPass {
    /// Allows setting a callback that determines whether a node must be preserved
    /// (even when its result is not used)
    pub fn set_preserve_callback(mut self, cb: Arc<PreserveCallback>) -> Self {
        self.preserve_callback = cb;
        self
    }

    /// Mark some nodes as entry points to the Hugr, i.e. so we cannot eliminate any code
    /// used to evaluate these nodes.
    /// The root node is assumed to be an entry point;
    /// for Module roots the client will want to mark some of the FuncDefn children
    /// as entry points too.
    pub fn with_entry_points(mut self, entry_points: impl IntoIterator<Item = Node>) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    fn find_needed_nodes(&self, h: impl HugrView<Node = Node>) -> HashSet<Node> {
        let mut must_preserve = HashMap::new();
        let mut needed = HashSet::new();
        let mut q = VecDeque::from_iter(self.entry_points.iter().cloned());
        q.push_front(h.root());
        while let Some(n) = q.pop_front() {
            if !needed.insert(n) {
                continue;
            };
            for ch in h.children(n) {
                if self.must_preserve(&h, &mut must_preserve, ch)
                    || matches!(
                        h.get_optype(ch),
                        OpType::Case(_) // Include all Cases in Conditionals
                    | OpType::DataflowBlock(_) // and all Basic Blocks in CFGs
                    | OpType::ExitBlock(_)
                    | OpType::AliasDecl(_) // and all Aliases (we do not track their uses in types)
                    | OpType::AliasDefn(_)
                    | OpType::Input(_) // Also Dataflow input/output, these are necessary for legality
                    | OpType::Output(_) // Do not include FuncDecl / FuncDefn / Const unless reachable by static edges
                                                                // (from Call/LoadConst/LoadFunction):
                    )
                {
                    q.push_back(ch);
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

    fn must_preserve(
        &self,
        h: &impl HugrView<Node = Node>,
        cache: &mut HashMap<Node, bool>,
        n: Node,
    ) -> bool {
        if let Some(res) = cache.get(&n) {
            return *res;
        }
        #[allow(deprecated)]
        let res = match self.preserve_callback.as_ref()(h.base_hugr(), n) {
            PreserveNode::MustKeep => true,
            PreserveNode::CanRemoveIgnoringChildren => false,
            PreserveNode::DeferToChildren => {
                h.children(n).any(|ch| self.must_preserve(h, cache, ch))
            }
        };
        cache.insert(n, res);
        res
    }
}

impl ComposablePass for DeadCodeElimPass {
    type Node = Node;
    type Error = Infallible;
    type Result = ();

    fn run(&self, hugr: &mut impl HugrMut<Node = Node>) -> Result<(), Infallible> {
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
#[cfg(test)]
mod test {
    use std::sync::Arc;

    use hugr_core::builder::{CFGBuilder, Container, Dataflow, DataflowSubContainer, HugrBuilder};
    use hugr_core::extension::prelude::{usize_t, ConstUsize};
    use hugr_core::ops::{handle::NodeHandle, OpTag, OpTrait};
    use hugr_core::types::Signature;
    use hugr_core::{ops::Value, type_row, HugrView};
    use itertools::Itertools;

    use crate::ComposablePass;

    use super::{DeadCodeElimPass, PreserveNode};

    #[test]
    fn test_cfg_callback() {
        let mut cb = CFGBuilder::new(Signature::new_endo(type_row![])).unwrap();
        let cst_unused = cb.add_constant(Value::from(ConstUsize::new(3)));
        let cst_used_in_dfg = cb.add_constant(Value::from(ConstUsize::new(5)));
        let cst_used = cb.add_constant(Value::unary_unit_sum());
        let mut block = cb.entry_builder([type_row![]], type_row![]).unwrap();
        let mut dfg_unused = block
            .dfg_builder(Signature::new(type_row![], usize_t()), [])
            .unwrap();
        let lc_unused = dfg_unused.load_const(&cst_unused);
        let lc1 = dfg_unused.load_const(&cst_used_in_dfg);
        let dfg_unused = dfg_unused.finish_with_outputs([lc1]).unwrap().node();
        let pred = block.load_const(&cst_used);
        let block = block.finish_with_outputs(pred, []).unwrap();
        let exit = cb.exit_block();
        cb.branch(&block, 0, &exit).unwrap();
        let orig = cb.finish_hugr().unwrap();

        // Callbacks that allow removing the DFG (and cst_unused)
        for dce in [
            DeadCodeElimPass::default(),
            // keep the node inside the DFG, but remove the DFG without checking its children:
            DeadCodeElimPass::default().set_preserve_callback(Arc::new(move |h, n| {
                if n == dfg_unused || h.get_optype(n).is_const() {
                    PreserveNode::CanRemoveIgnoringChildren
                } else {
                    PreserveNode::MustKeep
                }
            })),
        ] {
            let mut h = orig.clone();
            dce.run(&mut h).unwrap();
            assert_eq!(
                h.children(h.root()).collect_vec(),
                [block.node(), exit.node(), cst_used.node()]
            );
            assert_eq!(
                h.children(block.node())
                    .map(|n| h.get_optype(n).tag())
                    .collect_vec(),
                [OpTag::Input, OpTag::Output, OpTag::LoadConst]
            );
        }

        // Callbacks that prevent removing any node...
        fn keep_if(b: bool) -> PreserveNode {
            if b {
                PreserveNode::MustKeep
            } else {
                PreserveNode::DeferToChildren
            }
        }
        for dce in [
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(|_, _| PreserveNode::MustKeep)),
            // keeping the unused node in the DFG, means keeping the DFG (which uses its other children)
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == lc_unused.node()))),
        ] {
            let mut h = orig.clone();
            dce.run(&mut h).unwrap();
            assert_eq!(orig, h);
        }

        // Callbacks that keep the DFG but allow removing the unused constant
        for dce in [
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == dfg_unused))),
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == lc1.node()))),
        ] {
            let mut h = orig.clone();
            dce.run(&mut h).unwrap();
            assert_eq!(
                h.children(h.root()).collect_vec(),
                [
                    block.node(),
                    exit.node(),
                    cst_used_in_dfg.node(),
                    cst_used.node()
                ]
            );
            assert_eq!(
                h.children(block.node()).skip(2).collect_vec(),
                [dfg_unused, pred.node()]
            );
            assert_eq!(
                h.children(dfg_unused.node())
                    .map(|n| h.get_optype(n).tag())
                    .collect_vec(),
                [OpTag::Input, OpTag::Output, OpTag::LoadConst]
            );
        }

        // Callback that allows removing the DFG but require keeping cst_unused
        {
            let cst_unused = cst_unused.node();
            let mut h = orig.clone();
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == cst_unused)))
                .run(&mut h)
                .unwrap();
            assert_eq!(
                h.children(h.root()).collect_vec(),
                [block.node(), exit.node(), cst_unused, cst_used.node()]
            );
            assert_eq!(
                h.children(block.node())
                    .map(|n| h.get_optype(n).tag())
                    .collect_vec(),
                [OpTag::Input, OpTag::Output, OpTag::LoadConst]
            );
        }
    }
}

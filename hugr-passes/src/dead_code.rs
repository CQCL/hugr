//! Pass for removing dead code, i.e. that computes values that are then discarded

use hugr_core::hugr::internal::HugrInternals;
use hugr_core::{HugrView, Visibility, hugr::hugrmut::HugrMut, ops::OpType};
use std::convert::Infallible;
use std::fmt::{Debug, Formatter};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use crate::{ComposablePass, VisPolicy};

/// Configuration for Dead Code Elimination pass, i.e. which removes nodes
/// beneath the [HugrView::entrypoint] that compute only unneeded values.
#[derive(Clone)]
pub struct DeadCodeElimPass<H: HugrView> {
    /// Nodes that are definitely needed - e.g. `FuncDefns`, but could be anything.
    /// [HugrView::entrypoint] is assumed to be needed even if not mentioned here.
    entry_points: Vec<H::Node>,
    /// Callback identifying nodes that must be preserved even if their
    /// results are not used. Defaults to [`PreserveNode::default_for`].
    preserve_callback: Arc<PreserveCallback<H>>,
    include_exports: VisPolicy,
}

impl<H: HugrView + 'static> Default for DeadCodeElimPass<H> {
    fn default() -> Self {
        Self {
            entry_points: Default::default(),
            preserve_callback: Arc::new(PreserveNode::default_for),
            include_exports: VisPolicy::default(),
        }
    }
}

impl<H: HugrView> Debug for DeadCodeElimPass<H> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Use "derive Debug" by defining an identical struct without the unprintable fields

        #[allow(unused)] // Rust ignores the derive-Debug in figuring out what's used
        #[derive(Debug)]
        struct DCEDebug<'a, N> {
            entry_points: &'a Vec<N>,
            include_exports: VisPolicy,
        }

        Debug::fmt(
            &DCEDebug {
                entry_points: &self.entry_points,
                include_exports: self.include_exports,
            },
            f,
        )
    }
}

/// Callback that identifies nodes that must be preserved even if their
/// results are not used. For example, (the default) [`PreserveNode::default_for`].
pub type PreserveCallback<H> = dyn Fn(&H, <H as HugrInternals>::Node) -> PreserveNode;

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
    /// A conservative default for a given node. Just examines the node's [`OpType`]:
    /// * Assumes all Calls must be preserved. (One could scan the called `FuncDefn` for
    ///   termination, but would also need to check for cycles in the `CallGraph`.)
    /// * Assumes all CFGs must be preserved. (One could, for example, allow acyclic
    ///   CFGs to be removed.)
    /// * Assumes all `TailLoops` must be preserved. (One could use some analysis, e.g.
    ///   dataflow, to allow removal of `TailLoops` with a bounded number of iterations.)
    pub fn default_for<H: HugrView>(h: &H, n: H::Node) -> PreserveNode {
        match h.get_optype(n) {
            OpType::CFG(_) | OpType::TailLoop(_) | OpType::Call(_) => PreserveNode::MustKeep,
            _ => Self::DeferToChildren,
        }
    }
}

impl<H: HugrView> DeadCodeElimPass<H> {
    /// Allows setting a callback that determines whether a node must be preserved
    /// (even when its result is not used)
    pub fn set_preserve_callback(mut self, cb: Arc<PreserveCallback<H>>) -> Self {
        self.preserve_callback = cb;
        self
    }

    /// Mark some nodes as reachable, i.e. so we cannot eliminate any code used to
    /// evaluate their results. The [`HugrView::entrypoint`] is assumed to be reachable;
    /// if that is the [`HugrView::module_root`], then any public [FuncDefn] and
    /// [FuncDecl]s are also considered reachable by default,
    /// but this can be change by [`Self::include_module_exports`].
    ///
    /// [FuncDecl]: OpType::FuncDecl
    /// [FuncDefn]: OpType::FuncDefn
    pub fn with_entry_points(mut self, entry_points: impl IntoIterator<Item = H::Node>) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    /// Sets whether the exported [FuncDefn](OpType::FuncDefn)s and
    /// [FuncDecl](OpType::FuncDecl)s are considered reachable.
    ///
    /// Note that for non-module-entry Hugrs this has no effect, since we only remove
    /// code beneath the entrypoint: this cannot be affected by other module children.
    ///
    /// So, for module-rooted-Hugrs: [VisPolicy::PublicIfModuleEntrypoint] is
    /// equivalent to [VisPolicy::AllPublic]; and [VisPolicy::None] will remove
    /// all children, unless some are explicity added by [Self::with_entry_points].
    pub fn include_module_exports(mut self, include: VisPolicy) -> Self {
        self.include_exports = include;
        self
    }

    fn find_needed_nodes(&self, h: &H) -> HashSet<H::Node> {
        let mut must_preserve = HashMap::new();
        let mut needed = HashSet::new();
        let mut q = VecDeque::from_iter(self.entry_points.iter().copied());
        q.push_front(h.entrypoint());
        while let Some(n) = q.pop_front() {
            if !needed.insert(n) {
                continue;
            }
            for ch in h.children(n) {
                let must_keep = match h.get_optype(ch) {
                        OpType::Case(_) // Include all Cases in Conditionals
                    | OpType::DataflowBlock(_) // and all Basic Blocks in CFGs
                    | OpType::ExitBlock(_)
                    | OpType::AliasDecl(_) // and all Aliases (we do not track their uses in types)
                    | OpType::AliasDefn(_)
                    | OpType::Input(_) // Also Dataflow input/output, these are necessary for legality
                    | OpType::Output(_) => true,
                    // FuncDefns (as children of Module) only if public and including exports
                    // (will be included if static predecessors of Call/LoadFunction below,
                    // regardless of Visibility or self.include_exports)
                    OpType::FuncDefn(fd) => fd.visibility() == &Visibility::Public && self.include_exports.for_hugr(h),
                    OpType::FuncDecl(fd) => fd.visibility() == &Visibility::Public && self.include_exports.for_hugr(h),
                    // No Const, unless reached along static edges
                    _ => false
                };
                if must_keep || self.must_preserve(h, &mut must_preserve, ch) {
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

    fn must_preserve(&self, h: &H, cache: &mut HashMap<H::Node, bool>, n: H::Node) -> bool {
        if let Some(res) = cache.get(&n) {
            return *res;
        }
        let res = match self.preserve_callback.as_ref()(h, n) {
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

impl<H: HugrMut> ComposablePass<H> for DeadCodeElimPass<H> {
    type Error = Infallible;
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<(), Infallible> {
        let needed = self.find_needed_nodes(&*hugr);
        let remove = hugr
            .entry_descendants()
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

    use hugr_core::builder::{
        CFGBuilder, Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use hugr_core::extension::prelude::{ConstUsize, usize_t};
    use hugr_core::ops::{OpTag, OpTrait, Value, handle::NodeHandle};
    use hugr_core::{Hugr, HugrView, type_row, types::Signature};
    use itertools::Itertools;
    use rstest::rstest;

    use crate::{ComposablePass, VisPolicy};

    use super::{DeadCodeElimPass, PreserveNode};

    #[rstest]
    #[case(false, VisPolicy::None, true)]
    #[case(false, VisPolicy::PublicIfModuleEntrypoint, false)]
    #[case(false, VisPolicy::AllPublic, false)]
    #[case(true, VisPolicy::None, true)]
    #[case(true, VisPolicy::PublicIfModuleEntrypoint, false)]
    #[case(true, VisPolicy::AllPublic, false)]
    fn test_module_exports(
        #[case] include_dfn: bool,
        #[case] module_exports: VisPolicy,
        #[case] decl_removed: bool,
    ) {
        let mut mb = ModuleBuilder::new();
        let dfn = mb
            .define_function("foo", Signature::new_endo(usize_t()))
            .unwrap();
        let ins = dfn.input_wires();
        let dfn = dfn.finish_with_outputs(ins).unwrap();
        let dcl = mb
            .declare("bar", Signature::new_endo(usize_t()).into())
            .unwrap();
        let mut h = mb.finish_hugr().unwrap();
        let mut dce = DeadCodeElimPass::<Hugr>::default().include_module_exports(module_exports);
        if include_dfn {
            dce = dce.with_entry_points([dfn.node()]);
        }
        dce.run(&mut h).unwrap();
        let defn_retained = include_dfn;
        let decl_retained = !decl_removed;
        let children = h.children(h.module_root()).collect_vec();
        assert_eq!(defn_retained, children.iter().contains(&dfn.node()));
        assert_eq!(decl_retained, children.iter().contains(&dcl.node()));
        assert_eq!(
            children.len(),
            (defn_retained as usize) + (decl_retained as usize)
        );
    }

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
            DeadCodeElimPass::<Hugr>::default(),
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
                h.children(h.entrypoint()).collect_vec(),
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
            DeadCodeElimPass::<Hugr>::default()
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
            DeadCodeElimPass::<Hugr>::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == dfg_unused))),
            DeadCodeElimPass::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == lc1.node()))),
        ] {
            let mut h = orig.clone();
            dce.run(&mut h).unwrap();
            assert_eq!(
                h.children(h.entrypoint()).collect_vec(),
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
            DeadCodeElimPass::<Hugr>::default()
                .set_preserve_callback(Arc::new(move |_, n| keep_if(n == cst_unused)))
                .run(&mut h)
                .unwrap();
            assert_eq!(
                h.children(h.entrypoint()).collect_vec(),
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

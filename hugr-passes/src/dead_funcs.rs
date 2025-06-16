#![warn(missing_docs)]
//! Pass for removing statically-unreachable functions from a Hugr

use std::collections::HashSet;

use hugr_core::{
    HugrView, Node,
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait},
};
use petgraph::visit::{Dfs, Walker};

use crate::{
    ComposablePass,
    composable::{ValidatePassError, validate_if_test},
};

use super::call_graph::{CallGraph, CallGraphNode};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
/// Errors produced by [`RemoveDeadFuncsPass`].
pub enum RemoveDeadFuncsError<N = Node> {
    /// The specified entry point is not a `FuncDefn` node
    #[error(
        "Entrypoint for RemoveDeadFuncsPass {node} was not a function definition in the root module"
    )]
    InvalidEntryPoint {
        /// The invalid node.
        node: N,
    },
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum IncludeExports {
    Always,
    Never,
    #[default]
    OnlyIfEntrypointIsModuleRoot,
}

fn reachable_funcs<'a, H: HugrView>(
    cg: &'a CallGraph<H::Node>,
    h: &'a H,
    entry_points: impl IntoIterator<Item = H::Node>,
) -> impl Iterator<Item = H::Node> + 'a {
    let g = cg.graph();
    let mut d = Dfs::new(g, 0.into());
    d.stack.clear(); // Remove the fake 0
    for n in entry_points {
        d.stack.push(cg.node_index(n).unwrap());
    }
    d.iter(g).map(|i| match g.node_weight(i).unwrap() {
        CallGraphNode::FuncDefn(n) | CallGraphNode::FuncDecl(n) => *n,
        CallGraphNode::NonFuncRoot => h.entrypoint(),
    })
}

#[derive(Debug, Default, Clone)]
/// A configuration for the Dead Function Removal pass.
pub struct RemoveDeadFuncsPass {
    entry_points: Vec<Node>,
    include_exports: IncludeExports,
}

impl RemoveDeadFuncsPass {
    /// Adds new entry points - these must be [`FuncDefn`] nodes
    /// that are children of the [`Module`] at the root of the Hugr.
    ///
    /// [`FuncDefn`]: hugr_core::ops::OpType::FuncDefn
    /// [`Module`]: hugr_core::ops::OpType::Module
    pub fn with_module_entry_points(
        mut self,
        entry_points: impl IntoIterator<Item = Node>,
    ) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    /// Sets whether the exported [FuncDefn](hugr_core::ops::FuncDefn) children are
    /// included as entry points for reachability analysis - see [IncludeExports].
    pub fn include_module_exports(mut self, include: IncludeExports) -> Self {
        self.include_exports = include;
        self
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for RemoveDeadFuncsPass {
    type Error = RemoveDeadFuncsError;
    type Result = ();
    fn run(&self, hugr: &mut H) -> Result<(), RemoveDeadFuncsError> {
        let mut entry_points = Vec::new();
        let include_exports = match self.include_exports {
            IncludeExports::Always => true,
            IncludeExports::Never => false,
            IncludeExports::OnlyIfEntrypointIsModuleRoot => hugr.entrypoint() == hugr.module_root(),
        };
        let include_exports2 = matches!(
            (
                self.include_exports,
                hugr.entrypoint() == hugr.module_root()
            ),
            (IncludeExports::Always, _) | (IncludeExports::OnlyIfEntrypointIsModuleRoot, true)
        );
        assert_eq!(include_exports, include_exports2);
        if include_exports {
            entry_points.extend(hugr.children(hugr.module_root()).filter(|ch| {
                hugr.get_optype(*ch)
                    .as_func_defn()
                    .is_some_and(|fd| fd.link_name().is_some())
            }));
        }

        for &n in self.entry_points.iter() {
            if !hugr.get_optype(n).is_func_defn() {
                return Err(RemoveDeadFuncsError::InvalidEntryPoint { node: n });
            }
            debug_assert_eq!(hugr.get_parent(n), Some(hugr.module_root()));
            entry_points.push(n);
        }
        if hugr.entrypoint() != hugr.module_root() {
            entry_points.push(hugr.entrypoint())
        }

        let mut reachable =
            reachable_funcs(&CallGraph::new(hugr), hugr, entry_points).collect::<HashSet<_>>();
        // Also prevent removing the entrypoint itself
        let mut n = Some(hugr.entrypoint());
        while let Some(n2) = n {
            n = hugr.get_parent(n2);
            if n == Some(hugr.module_root()) {
                reachable.insert(n2);
            }
        }

        let unreachable = hugr
            .children(hugr.module_root())
            .filter(|n| {
                OpTag::Function.is_superset(hugr.get_optype(*n).tag()) && !reachable.contains(n)
            })
            .collect::<Vec<_>>();
        for n in unreachable {
            hugr.remove_subtree(n);
        }
        Ok(())
    }
}

/// Deletes from the Hugr any functions that are not used by either [`Call`] or
/// [`LoadFunction`] nodes in reachable parts.
///
/// `entry_points` may provide a list of entry points, which must be [`FuncDefn`]s (children of the root).
/// * If the [HugrView::entrypoint] is the module root, then any [`FuncDefn`] children with a [link_name]
/// will also be considered an entry point
/// * otherwise, the [HugrView::entrypoint] itself will.
///
/// # Errors
/// * If any node in `entry_points` is not a [`FuncDefn`]
///
/// [`Call`]: hugr_core::ops::OpType::Call
/// [link_name]: hugr_core::ops::FuncDefn::link_name
/// [`LoadFunction`]: hugr_core::ops::OpType::LoadFunction
/// [`Module`]: hugr_core::ops::OpType::Module
pub fn remove_dead_funcs(
    h: &mut impl HugrMut<Node = Node>,
) -> Result<(), ValidatePassError<Node, RemoveDeadFuncsError>> {
    validate_if_test(RemoveDeadFuncsPass::default(), h)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use hugr_core::ops::handle::NodeHandle;
    use itertools::Itertools;
    use rstest::rstest;

    use hugr_core::builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use hugr_core::hugr::hugrmut::HugrMut;
    use hugr_core::{HugrView, extension::prelude::usize_t, types::Signature};

    use super::RemoveDeadFuncsPass;
    use crate::ComposablePass;
    use crate::dead_funcs::IncludeExports;

    #[rstest]
    #[case(false, IncludeExports::default(), [], vec!["from_pub", "pubfunc"])]
    #[case(false, IncludeExports::Never, ["main"], vec!["from_main", "main"])]
    #[case(false, IncludeExports::Never, ["from_main", "from_pub"], vec!["from_main", "from_pub"])]
    #[case(false, IncludeExports::default(), ["from_main"], vec!["from_main", "from_pub", "pubfunc"])]
    #[case(false, IncludeExports::Always, ["main"], vec!["from_main", "from_pub", "main", "pubfunc"])]
    #[case(true, IncludeExports::default(), [], vec!["from_main", "main"])]
    #[case(true, IncludeExports::Always, [], vec!["from_main", "from_pub", "main", "pubfunc"])]
    #[case(true, IncludeExports::Never, ["from_pub"], vec!["from_main", "from_pub", "main"])]
    fn remove_dead_funcs_entry_points(
        #[case] use_hugr_entrypoint: bool,
        #[case] inc: IncludeExports,
        #[case] entry_points: impl IntoIterator<Item = &'static str>,
        #[case] retained_funcs: Vec<&'static str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut hb = ModuleBuilder::new();
        let o2 = hb.define_function("from_pub", Signature::new_endo(usize_t()))?;
        let o2inp = o2.input_wires();
        let o2 = o2.finish_with_outputs(o2inp)?;
        let mut o1 = hb.define_function_pub("pubfunc", Signature::new_endo(usize_t()))?;

        let o1c = o1.call(o2.handle(), &[], o1.input_wires())?;
        o1.finish_with_outputs(o1c.outputs())?;

        let fm = hb.define_function("from_main", Signature::new_endo(usize_t()))?;
        let f_inp = fm.input_wires();
        let fm = fm.finish_with_outputs(f_inp)?;
        // Note main here is private, but sometimes we use it as entrypoint.
        // This could be confusing if ppl expect it to be public - rename main->entryfunc, pubfunc->main?
        let mut m = hb.define_function("main", Signature::new_endo(usize_t()))?;
        let mc = m.call(fm.handle(), &[], m.input_wires())?;
        let m = m.finish_with_outputs(mc.outputs())?;

        let mut hugr = hb.finish_hugr()?;
        if use_hugr_entrypoint {
            hugr.set_entrypoint(m.node());
        }

        let avail_funcs = hugr
            .children(hugr.module_root())
            .filter_map(|n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .map(|fd| (fd.func_name().clone(), n))
            })
            .collect::<HashMap<_, _>>();

        RemoveDeadFuncsPass::default()
            .include_module_exports(inc)
            .with_module_entry_points(
                entry_points
                    .into_iter()
                    .map(|name| *avail_funcs.get(name).unwrap())
                    .collect::<Vec<_>>(),
            )
            .run(&mut hugr)
            .unwrap();

        let remaining_funcs = hugr
            .nodes()
            .filter_map(|n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .map(|fd| fd.func_name().as_str())
            })
            .sorted()
            .collect_vec();
        assert_eq!(remaining_funcs, retained_funcs);
        Ok(())
    }
}

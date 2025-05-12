#![warn(missing_docs)]
//! Pass for removing statically-unreachable functions from a Hugr

use std::collections::HashSet;

use hugr_core::{
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait},
    HugrView, Node,
};
use petgraph::visit::{Dfs, Walker};

use crate::{
    composable::{validate_if_test, ValidatePassError},
    ComposablePass,
};

use super::call_graph::{CallGraph, CallGraphNode};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
/// Errors produced by [RemoveDeadFuncsPass].
pub enum RemoveDeadFuncsError<N = Node> {
    /// The specified entry point is not a FuncDefn node or is not a child of the root.
    #[error(
        "Entrypoint for RemoveDeadFuncsPass {node} was not a function definition in the root module"
    )]
    InvalidEntryPoint {
        /// The invalid node.
        node: N,
    },
}

fn reachable_funcs<'a, H: HugrView>(
    cg: &'a CallGraph<H::Node>,
    h: &'a H,
    entry_points: impl IntoIterator<Item = H::Node>,
) -> Result<impl Iterator<Item = H::Node> + 'a, RemoveDeadFuncsError<H::Node>> {
    let g = cg.graph();
    let mut entry_points = entry_points.into_iter();
    let searcher = if h.get_optype(h.entrypoint()).is_module() {
        let mut d = Dfs::new(g, 0.into());
        d.stack.clear();
        for n in entry_points {
            if !h.get_optype(n).is_func_defn() || h.get_parent(n) != Some(h.entrypoint()) {
                return Err(RemoveDeadFuncsError::InvalidEntryPoint { node: n });
            }
            d.stack.push(cg.node_index(n).unwrap())
        }
        d
    } else {
        if let Some(n) = entry_points.next() {
            // Can't be a child of the module root as there isn't a module root!
            return Err(RemoveDeadFuncsError::InvalidEntryPoint { node: n });
        }
        Dfs::new(g, cg.node_index(h.entrypoint()).unwrap())
    };
    Ok(searcher.iter(g).map(|i| match g.node_weight(i).unwrap() {
        CallGraphNode::FuncDefn(n) | CallGraphNode::FuncDecl(n) => *n,
        CallGraphNode::NonFuncRoot => h.entrypoint(),
    }))
}

#[derive(Debug, Clone)]
/// A configuration for the Dead Function Removal pass.
pub struct RemoveDeadFuncsPass {
    entry_points: Vec<Node>,
    include_exports: bool,
}

impl Default for RemoveDeadFuncsPass {
    fn default() -> Self {
        Self {
            entry_points: Default::default(),
            include_exports: true,
        }
    }
}

impl RemoveDeadFuncsPass {
    /// Adds new entry points - these must be [FuncDefn] nodes
    /// that are children of the [Module] at the root of the Hugr.
    ///
    /// [FuncDefn]: hugr_core::ops::OpType::FuncDefn
    /// [Module]: hugr_core::ops::OpType::Module
    pub fn with_module_entry_points(
        mut self,
        entry_points: impl IntoIterator<Item = Node>,
    ) -> Self {
        self.entry_points.extend(entry_points);
        self
    }

    /// Sets whether the exported [FuncDefn](hugr_core::ops::FuncDefn) children of a
    /// [Module](hugr_core::ops::Module) are included as entry points (yes by default)
    pub fn include_module_exports(mut self, include: bool) -> Self {
        self.include_exports = include;
        self
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for RemoveDeadFuncsPass {
    type Error = RemoveDeadFuncsError;
    type Result = ();
    fn run(&self, hugr: &mut H) -> Result<(), RemoveDeadFuncsError> {
        let exports = if hugr.entrypoint() == hugr.module_root() && self.include_exports {
            hugr.children(hugr.module_root())
                .filter(|ch| {
                    hugr.get_optype(*ch)
                        .as_func_defn()
                        .is_some_and(|fd| fd.link_name.is_some())
                })
                .collect()
        } else {
            vec![]
        };
        let reachable = reachable_funcs(
            &CallGraph::new(hugr),
            hugr,
            self.entry_points.iter().cloned().chain(exports),
        )?
        .collect::<HashSet<_>>();
        let unreachable = hugr
            .entry_descendants()
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

/// Deletes from the Hugr any functions that are not used by either [Call] or
/// [LoadFunction] nodes in reachable parts.
///
/// For [Module]-rooted Hugrs, all top-level functions with [FuncDefn::link_name] set,
/// will be used as entry points.
///
pub fn remove_dead_funcs(
    h: &mut impl HugrMut<Node = Node>,
) -> Result<(), ValidatePassError<Node, RemoveDeadFuncsError>> {
    validate_if_test(RemoveDeadFuncsPass::default(), h)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use itertools::Itertools;
    use rstest::rstest;

    use hugr_core::builder::{
        Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use hugr_core::{extension::prelude::usize_t, types::Signature, HugrView};

    use super::RemoveDeadFuncsPass;
    use crate::ComposablePass;

    #[rstest]
    #[case(false, [], vec![])] // No entry_points removes everything!
    #[case(false, ["main"], vec!["from_main", "main"])]
    #[case(false, ["from_main"], vec!["from_main"])]
    #[case(false, ["other1"], vec!["other1", "other2"])]
    #[case(false, ["other2"], vec!["other2"])]
    #[case(false, ["other1", "other2"], vec!["other1", "other2"])]
    #[case(true, [], vec!["from_main", "main", "other2"])]
    #[case(true, ["other1"], vec!["from_main", "main", "other1", "other2"])]
    fn remove_dead_funcs_entry_points(
        #[case] include_exports: bool,
        #[case] entry_points: impl IntoIterator<Item = &'static str>,
        #[case] retained_funcs: Vec<&'static str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut hb = ModuleBuilder::new();
        let o2 = hb.define_function("other2", Signature::new_endo(usize_t()))?;
        let o2inp = o2.input_wires();
        let o2 = o2.finish_with_outputs(o2inp)?;
        let mut o1 =
            hb.define_function_link_name("other1", Signature::new_endo(usize_t()), None)?;

        let o1c = o1.call(o2.handle(), &[], o1.input_wires())?;
        o1.finish_with_outputs(o1c.outputs())?;

        let fm = hb.define_function_link_name("from_main", Signature::new_endo(usize_t()), None)?;
        let f_inp = fm.input_wires();
        let fm = fm.finish_with_outputs(f_inp)?;
        let mut m = hb.define_function("main", Signature::new_endo(usize_t()))?;
        let mc = m.call(fm.handle(), &[], m.input_wires())?;
        m.finish_with_outputs(mc.outputs())?;

        let mut hugr = hb.finish_hugr()?;

        let avail_funcs = hugr
            .entry_descendants()
            .filter_map(|n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .map(|fd| (fd.name.clone(), n))
            })
            .collect::<HashMap<_, _>>();

        RemoveDeadFuncsPass::default()
            .include_module_exports(include_exports)
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
            .filter_map(|n| hugr.get_optype(n).as_func_defn().map(|fd| fd.name.as_str()))
            .sorted()
            .collect_vec();
        assert_eq!(remaining_funcs, retained_funcs);
        Ok(())
    }
}

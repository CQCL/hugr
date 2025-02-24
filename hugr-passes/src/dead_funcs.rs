#![warn(missing_docs)]
//! Pass for removing statically-unreachable functions from a Hugr

use std::collections::HashSet;

use hugr_core::{
    hugr::hugrmut::HugrMut,
    ops::{OpTag, OpTrait},
    HugrView, Node,
};
use petgraph::visit::{Dfs, Walker};

use crate::validation::{ValidatePassError, ValidationLevel};

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
    #[error(transparent)]
    #[allow(missing_docs)]
    ValidationError(#[from] ValidatePassError),
}

fn reachable_funcs<'a, H: HugrView>(
    cg: &'a CallGraph<H::Node>,
    h: &'a H,
    entry_points: impl IntoIterator<Item = H::Node>,
) -> Result<impl Iterator<Item = H::Node> + 'a, RemoveDeadFuncsError<H::Node>> {
    let g = cg.graph();
    let mut entry_points = entry_points.into_iter();
    let searcher = if h.get_optype(h.root()).is_module() {
        let mut d = Dfs::new(g, 0.into());
        d.stack.clear();
        for n in entry_points {
            if !h.get_optype(n).is_func_defn() || h.get_parent(n) != Some(h.root()) {
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
        Dfs::new(g, cg.node_index(h.root()).unwrap())
    };
    Ok(searcher.iter(g).map(|i| match g.node_weight(i).unwrap() {
        CallGraphNode::FuncDefn(n) | CallGraphNode::FuncDecl(n) => *n,
        CallGraphNode::NonFuncRoot => h.root(),
    }))
}

#[derive(Debug, Clone, Default)]
/// A configuration for the Dead Function Removal pass.
pub struct RemoveDeadFuncsPass {
    validation: ValidationLevel,
    entry_points: Vec<Node>,
}

impl RemoveDeadFuncsPass {
    /// Sets the validation level used before and after the pass is run
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

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

    /// Runs the pass (see [remove_dead_funcs]) with this configuration
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<(), RemoveDeadFuncsError> {
        self.validation.run_validated_pass(hugr, |hugr: &mut H, _| {
            remove_dead_funcs(hugr, self.entry_points.iter().cloned())
        })
    }
}

/// Delete from the Hugr any functions that are not used by either [Call] or
/// [LoadFunction] nodes in reachable parts.
///
/// For [Module]-rooted Hugrs, `entry_points` may provide a list of entry points,
/// which must be children of the root. Note that if `entry_points` is empty, this will
/// result in all functions in the module being removed.
///
/// For non-[Module]-rooted Hugrs, `entry_points` must be empty; the root node is used.
///
/// # Errors
/// * If there are any `entry_points` but the root of the hugr is not a [Module]
/// * If any node in `entry_points` is
///     * not a [FuncDefn], or
///     * not a child of the root
///
/// [Call]: hugr_core::ops::OpType::Call
/// [FuncDefn]: hugr_core::ops::OpType::FuncDefn
/// [LoadFunction]: hugr_core::ops::OpType::LoadFunction
/// [Module]: hugr_core::ops::OpType::Module
pub fn remove_dead_funcs(
    h: &mut impl HugrMut,
    entry_points: impl IntoIterator<Item = Node>,
) -> Result<(), RemoveDeadFuncsError> {
    let reachable = reachable_funcs(&CallGraph::new(h), h, entry_points)?.collect::<HashSet<_>>();
    let unreachable = h
        .nodes()
        .filter(|n| OpTag::Function.is_superset(h.get_optype(*n).tag()) && !reachable.contains(n))
        .collect::<Vec<_>>();
    for n in unreachable {
        h.remove_subtree(n);
    }
    Ok(())
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

    #[rstest]
    #[case([], vec![])] // No entry_points removes everything!
    #[case(["main"], vec!["from_main", "main"])]
    #[case(["from_main"], vec!["from_main"])]
    #[case(["other1"], vec!["other1", "other2"])]
    #[case(["other2"], vec!["other2"])]
    #[case(["other1", "other2"], vec!["other1", "other2"])]
    fn remove_dead_funcs_entry_points(
        #[case] entry_points: impl IntoIterator<Item = &'static str>,
        #[case] retained_funcs: Vec<&'static str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut hb = ModuleBuilder::new();
        let o2 = hb.define_function("other2", Signature::new_endo(usize_t()))?;
        let o2inp = o2.input_wires();
        let o2 = o2.finish_with_outputs(o2inp)?;
        let mut o1 = hb.define_function("other1", Signature::new_endo(usize_t()))?;

        let o1c = o1.call(o2.handle(), &[], o1.input_wires())?;
        o1.finish_with_outputs(o1c.outputs())?;

        let fm = hb.define_function("from_main", Signature::new_endo(usize_t()))?;
        let f_inp = fm.input_wires();
        let fm = fm.finish_with_outputs(f_inp)?;
        let mut m = hb.define_function("main", Signature::new_endo(usize_t()))?;
        let mc = m.call(fm.handle(), &[], m.input_wires())?;
        m.finish_with_outputs(mc.outputs())?;

        let mut hugr = hb.finish_hugr()?;

        let avail_funcs = hugr
            .nodes()
            .filter_map(|n| {
                hugr.get_optype(n)
                    .as_func_defn()
                    .map(|fd| (fd.name.clone(), n))
            })
            .collect::<HashMap<_, _>>();

        RemoveDeadFuncsPass::default()
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

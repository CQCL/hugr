use std::convert::Infallible;

use ascent::hashbrown::HashMap;
use hugr_core::{extension::ExtensionRegistry, hugr::{hugrmut::HugrMut, views::{DescendantsGraph, ExtractHugr as _, HierarchyView}, HugrError, Rewrite, ValidationError}, ops::{DataflowOpTrait as _, OpTrait, DFG}, Direction, HugrView, Node};
use itertools::Itertools as _;
use petgraph::visit::EdgeRef as _;
use thiserror::Error;

use crate::validation::ValidationLevel;


#[derive(Debug, Clone, Default)]
/// TODO docs
pub struct InlinePass {
    validation: ValidationLevel,
}

impl InlinePass {
    /// Sets the validation level used before and after the pass is run
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    pub fn run(
        &self,
        hugr: &mut impl HugrMut,
        registry: &ExtensionRegistry,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validation.run_validated_pass_mut(hugr, registry, |hugr,_| {

            let mut calls = {
                let cg = CallGraph::new(hugr);
                let Some(calls) = cg.iter_nonrecursive() else {
                    Err("InlinePass: recursion")?
                };
                let mut calls = calls.collect_vec();
                calls.reverse();
                calls
            };
            dbg!(&calls);

            let rewrites = calls.iter().filter_map(|(caller,_)| InlineRewrite::try_new(hugr, *caller, registry).ok()).collect_vec();

            for rewrite in rewrites {
                hugr.apply_rewrite(rewrite).unwrap();
            }

            calls.reverse();

            for func_node in calls.into_iter().map(|x| x.1).dedup() {
                let Some(func) = hugr.get_optype(func_node).as_func_defn() else {
                    panic!("impossible")
                };
                if hugr.linked_inputs(func_node, 0).count() == 0 && func.name != "main" {
                    eprintln!("Removing func: {}", func.name);
                    let func_hugr = DescendantsGraph::<Node>::try_new(hugr, func_node).unwrap();
                    let to_delete = func_hugr.nodes().dedup().collect_vec();
                    for n in to_delete {
                        hugr.remove_node(n);
                    }
                }
            }
            hugr.validate(registry)?;
            eprintln!("{}", hugr.mermaid_string());
            Ok(())
        })
    }
}

pub struct CallGraph {
    g: petgraph::graph::Graph<Node, Node>,
    // node_to_cg: HashMap<Node, petgraph::graph::NodeIndex>,
}

fn func_of_node(hugr: &impl HugrView, node: Node) -> Option<Node> {
    let mut n = node;
    while let Some(parent) = hugr.get_parent(n) {
        if hugr.get_optype(parent).is_func_defn() {
            return Some(parent);
        }
        n = parent;
    }
    None
}

impl CallGraph {
    pub fn new(hugr: &impl HugrView) -> Self {
        let mut g: petgraph::graph::Graph<Node,Node> = Default::default();

        let node_to_cg: HashMap<_,_> = hugr.nodes().filter_map(|n| (hugr.get_optype(n).is_func_decl() || hugr.get_optype(n).is_func_defn()).then(|| (n, g.add_node(n)))).collect();

        for n in hugr.nodes() {
            if let Some(call) = hugr.get_optype(n).as_call() {
                if let Some(caller_func) = func_of_node(hugr, n) {
                    if let Some((callee_func,_)) = hugr.single_linked_output(n, call.called_function_port()) {
                        g.add_edge(node_to_cg[&caller_func], node_to_cg[&callee_func], n);
                    }
                }
            }
        }

        Self { g }
    }

    pub fn iter_nonrecursive(&self) -> Option<impl Iterator<Item=(Node,Node)> + '_> {
        let funcs = petgraph::algo::toposort(&self.g, None).ok()?;

        Some(funcs.into_iter().flat_map(move |f| self.g.edges(f).map(move |e| (*e.weight(), self.g[e.target()]))))
    }
}

pub struct InlineRewrite<'a> {
    call: Node,
    func: Node,
    registry: &'a ExtensionRegistry,
}

impl<'a> InlineRewrite<'a> {
    pub fn try_new(hugr: &impl HugrView, call: Node, registry: &'a ExtensionRegistry) -> Result<Self,InlineRewriteError> {

        if !hugr.valid_node(call) {
            Err(InlineRewriteError::InvalidCall)?
        }
        let Some(call_ot) = hugr.get_optype(call).as_call() else {
            Err(InlineRewriteError::InvalidCall)?
        };

        let Some((func,_)) = hugr.single_linked_output(call, call_ot.called_function_port()) else {
            Err(InlineRewriteError::InvalidCall)?
        };

        if !hugr.get_optype(func).is_func_defn() {
            Err(InlineRewriteError::InvalidFunction)?
        }

        let r = Self { call, func, registry };
        debug_assert!(r.verify(hugr).is_ok());

        Ok(r)
    }
}

#[derive(Debug,Clone,Error)]
pub enum InlineRewriteError {
    #[error("Invalid Function")]
    InvalidFunction,
    #[error("Invalid Call")]
    InvalidCall,
    #[error("Call does not target func")]
    Invalid,
    #[error(transparent)]
    HugrError(#[from] HugrError),
    #[error(transparent)]
    Validation(#[from] ValidationError),

}

impl<'a> Rewrite for InlineRewrite<'a> {
    type Error = InlineRewriteError;

    type ApplyResult = ();

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let Some(call) = h.get_optype(self.call).as_call() else {
            Err(InlineRewriteError::InvalidCall)?
        };
        if call.type_args.len() != 0 {
            Err(InlineRewriteError::InvalidCall)?
        }
        let Some(_) = h.get_optype(self.func).as_func_defn() else {
            Err(InlineRewriteError::InvalidFunction)?
        };

        if let Some((n,_)) = h.single_linked_output(self.call, call.called_function_port()) {
            if self.func != n {
                Err(InlineRewriteError::Invalid)?
            }
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;

        dbg!(self.call, self.func);

        let func_hugr = DescendantsGraph::<Node>::try_new(h, self.func).map_err(|_| InlineRewriteError::InvalidFunction)?.extract_hugr();
        func_hugr.validate(self.registry)?;

        let call = h.get_optype(self.call).as_call().unwrap().to_owned();
        let call_parent = h.get_parent(self.call).unwrap();

        let signature = call.signature();

        let insertion = h.insert_hugr(call_parent, func_hugr);

        let dfg_node = insertion.new_root;
        let dfg = DFG { signature };
        h.set_num_ports(dfg_node, dfg.signature().input_count() + dfg.non_df_port_count(Direction::Incoming), dfg.signature().output_count() + dfg.non_df_port_count(Direction::Outgoing));
        h.replace_op(dfg_node, dfg)?;

        let connections = h.node_inputs(self.call).filter(|&x| x != call.called_function_port())
            .flat_map(|in_p| h.linked_outputs(self.call, in_p)
                      .map(move |(out_n,out_p)| (out_n, out_p, dfg_node, in_p))).chain(h.node_outputs(self.call).flat_map(|out_p| h.linked_inputs(self.call, out_p).map(move |(in_n,in_p)| (dfg_node, out_p, in_n, in_p)))).collect_vec();

        for (from_n,from_p,to_n,to_p) in connections {
            h.connect(from_n, from_p, to_n, to_p)
        }

        h.remove_node(self.call);
        Ok(())
    }

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        [self.call, self.func].into_iter()
    }
}

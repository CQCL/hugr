//! Inference for resource requirements on nodes of a hugr.
//!
//! Checks if the resources requirements are sane, and comes up with concrete
//! solutions when possible. Inference operates when toplevel nodes can be open
//! variables. When resource requirements of nodes depend on these open
//! variables, then the validation check for resources will succeed regardless
//! of what the variable is instantiated to.

use super::{ResourceId, ResourceSet};
use crate::{
    hugr::{Hugr, Node},
    hugr::HugrInternalsMut,
    hugr::views::{HugrView, HierarchyView, SiblingGraph},
    Direction,
};

use super::validate::ResourceError;

use petgraph::graph as pg;

use std::collections::{HashMap, HashSet};

use thiserror::Error;
use std::collections::BTreeSet;
use std::cmp::{Ord, Ordering};

pub type ResourceSolution = HashMap<(Node, Direction), ResourceSet>;

/// Infer resources for a hugr. This is the main API exposed by this module
pub fn infer_resources(hugr: &impl HugrView) -> Result<ResourceSolution, InferResourceError> {
    let mut ctx = UnificationContext::new(hugr);
    ctx.main_loop()
}

/* TODO:
- Add more complicated solving that can handle the case of "reverse"
*/

/// Metavariables don't need much
type Meta = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
/// Things we know about metavariables
enum Constraint {
    /// Constrain a variable to a specific value
    Exactly(ResourceSet),
    /// A variable has the same value as another variable
    Equal(Meta),
    /// Variable extends the value of another by one resource
    Plus(ResourceId, Meta),
}

// Implement ordering for constraints so that we can get to the most useful
// constraints first in `solve_constraints`
impl PartialOrd for Constraint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Exactly(a), Self::Exactly(b)) => {
                let a_len = a.iter().count();
                let b_len = b.iter().count();
                if a_len == b_len {
                    for (a, b) in a.iter().zip(b.iter()) {
                        if a != b {
                            return Some((*a).cmp(b));
                        }
                    }
                    Some(Ordering::Equal)
                } else {
                    Some(a_len.cmp(&b_len))
                }

            }
            (Self::Exactly(_), _) => Some(Ordering::Greater),
            (_, Self::Exactly(_)) => Some(Ordering::Less),
            (Self::Plus(r1, m1), Self::Plus(r2, m2)) => Some(m1.cmp(m2)),
            (Self::Plus(_, _), _) => Some(Ordering::Greater),
            (_, Self::Plus(_, _)) => Some(Ordering::Less),
            (Self::Equal(m1), Self::Equal(m2)) => m1.partial_cmp(m2),
        }
    }
}

impl Ord for Constraint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
/// Errors which arise during unification
pub enum InferResourceError {
    #[error("Mismatched resource sets {expected} and {actual}")]
    /// We've solved a metavariable, then encountered a constraint
    /// that says it should be something other than our solution
    MismatchedConcrete {
        //loc: (Node, Direction),
        expected: ResourceSet,
        actual: ResourceSet,
    },
    /// A variable went unsolved that wasn't related to a parameter
    #[error("Unsolved variable at location {:?}", location)]
    Unsolved {
        location: (Node, Direction),
        //constraints: Vec<Constraint>,
    },
    /// An resource mismatch between two nodes which are connected by an edge.
    /// This should mirror (or reuse) `ValidationError`'s SrcExceedsTgtResources
    /// and TgtExceedsSrcResources
    #[error("Edge mismatch: {0}")]
    EdgeMismatch(#[from] ResourceError),
}

/// A graph of metavariables which we've found equality constraints for. Edges
/// between nodes represent equality constraints.
struct EqGraph {
    equalities: pg::Graph<Meta, (), petgraph::Undirected>,
    node_map: HashMap<Meta, pg::NodeIndex>
}

impl EqGraph {
    /// Create a new `EqGraph`
    pub fn new() -> Self {
        EqGraph {
            equalities: pg::Graph::new_undirected(),
            node_map: HashMap::new(),
        }
    }

    /// Add a metavariable to the graph as a node and return the `NodeIndex`.
    /// If it's already there, just return the existing `NodeIndex`
    pub fn add_or_retrieve(&mut self, m: Meta) -> pg::NodeIndex {
        self.node_map.get(&m).cloned().unwrap_or_else(|| {
            let ix = self.equalities.add_node(m);
            self.node_map.insert(m, ix);
            ix
        })
    }

    /// Create an edge between two nodes on the graph, declaring that they stand
    /// for metavariables which should be equal.
    pub fn register_eq(&mut self, src: Meta, tgt: Meta) {
        let src_ix = self.add_or_retrieve(src);
        let tgt_ix = self.add_or_retrieve(tgt);
        self.equalities.add_edge(src_ix, tgt_ix, ());
    }

    /// Return the connected components of the graph in terms of metavariables
    pub fn ccs(&self) -> Vec<Vec<Meta>> {
        petgraph::algo::tarjan_scc(&self.equalities)
            .into_iter()
            .map(|cc| cc
                 .into_iter()
                 .map(|n| *self
                      .equalities
                      .node_weight(n)
                      .unwrap())
                 .collect())
            .collect()
    }
}

/// Our current knowledge about the resources of the graph
struct UnificationContext {
    /// A list of constraints for each metavariable
    constraints: HashMap<Meta, BTreeSet<Constraint>>,
    /// A map which says which nodes correspond to which metavariables
    resources: HashMap<(Node, Direction), Meta>,
    /// Solutions to metavariables
    solved: HashMap<Meta, ResourceSet>,
    /// A graph which says which metavariables should be equal
    eq_graph: EqGraph,
    /// A mapping from metavariables which have been merged, to the meta they've
    // been merged to
    shunted: HashMap<Meta, Meta>,
    /// Variables we're allowed to include in solutionss
    variables: HashSet<Meta>,
    /// A name for the next metavariable we create
    fresh_name: usize,
}

/// Invariant: Constraint::Plus always points to a fresh metavariable
impl UnificationContext {
    /// Create a new unification context, and populate it with constraints from
    /// traversing the hugr which is passed in.
    pub fn new(hugr: &impl HugrView) -> Self {
        let mut ctx = Self {
            constraints: HashMap::new(),
            resources: HashMap::new(),
            solved: HashMap::new(),
            eq_graph: EqGraph::new(),
            shunted: HashMap::new(),
            variables: HashSet::new(),
            fresh_name: 0,
        };
        ctx.gen_constraints(hugr);
        ctx
    }

    /// Create a fresh metavariable, and increment `fresh_name` for next time
    fn fresh_meta(&mut self) -> Meta {
        let fresh = self.fresh_name;
        self.fresh_name += 1;
        self.constraints.insert(fresh, BTreeSet::new());
        fresh
    }

    /// Declare a constraint on the metavariable
    fn add_constraint(&mut self, m: Meta, c: Constraint) {
        self.constraints
            .entry(m)
            .and_modify(|cs| { cs.insert(c.clone()); })
            .or_insert(BTreeSet::from_iter([c]));
    }

    /// Declare that a meta has been solved
    fn add_solution(&mut self, m: Meta, rs: ResourceSet) {
        assert!(self.solved.insert(m, rs).is_none());
    }

    /// If a metavariable has been merged, return the new meta
    fn resolve<'a>(&self, m: Meta) -> Meta {
        self.shunted.get(&m).cloned().map_or(m, |m| self.resolve(m))
    }

    fn get_constraints(&self, m: &Meta) -> Option<&BTreeSet<Constraint>> {
        self.constraints.get(&self.resolve(*m))
    }

    fn get_solution(&self, m: &Meta) -> Option<&ResourceSet> {
        self.solved.get(&self.resolve(*m))
    }

    fn gen_union_constraint(&mut self, input: Meta, output: Meta, delta: ResourceSet) {
        let mut last_meta = input;
        // Create fresh metavariables with `Plus` constraints for
        // each resource that should be added by the node
        // Hence a resource delta [A, B] would lead to
        // > ma = fresh_meta()
        // > add_constraint(ma, Plus(a, input)
        // > mb = fresh_meta()
        // > add_constraint(mb, Plus(b, ma)
        // > add_constraint(output, Equal(mb))
        for r in delta.iter() {
            let curr_meta = self.fresh_meta();
            self.add_constraint(curr_meta, Constraint::Plus(r.clone(), last_meta));
            last_meta = curr_meta;
        }
        self.add_constraint(output, Constraint::Equal(last_meta));
    }

    /// Return the metavariable corresponding to the given location on the
    /// graph, either by making a new meta, or looking it up
    fn make_or_get_meta(&mut self, node: Node, dir: Direction) -> Meta {
        if let Some(m) = self.resources.get(&(node, dir)) {
            *m
        } else {
            let m = self.fresh_meta();
            self.resources.insert((node, dir), m);
            m
        }
    }

    /// Iterate over the nodes in a hugr and generate unification constraints
    fn gen_constraints<T>(&mut self, hugr: &T) where T: HugrView {
        // The toplevel sibling graph can be open, and we should note what those variables are
        let toplevel: SiblingGraph<Node, T> = SiblingGraph::new(hugr, hugr.root());
        for toplevel_node in toplevel.nodes().into_iter() {
            let m_input = self.make_or_get_meta(toplevel_node, Direction::Incoming);
            self.variables.insert(m_input);
        }

        for node in hugr.nodes() {
            let m_input = self.make_or_get_meta(node, Direction::Incoming);
            let m_output = self.make_or_get_meta(node, Direction::Outgoing);

            let node_type = hugr.get_nodetype(node);

            // Add constraints for the inputs and outputs of dataflow nodes according
            // to the signature of the parent node
            if let Some([input, output]) = hugr.get_io(node) {
                for dir in Direction::BOTH {
                    let m_input_node = self.make_or_get_meta(input, dir);
                    self.add_constraint(m_input_node, Constraint::Equal(m_input));
                    let m_output_node = self.make_or_get_meta(output, dir);
                    self.add_constraint(m_output_node, Constraint::Equal(m_output));
                }
            }

            match node_type.signature() {
                // Input resources are open
                None => {
                    self.gen_union_constraint(
                        m_input,
                        m_output,
                        node_type.op_signature().resource_reqs,
                    );
                }
                // We're in the money!
                Some(sig) => {
                    println!("node {:?}", node);
                    println!("in   {:?}", sig.input_resources);
                    println!("out  {:?}", sig.output_resources());
                    self.add_solution(m_input, sig.input_resources.clone());
                    self.add_solution(m_output, sig.output_resources());
                }
            }

        }
        // Seperate loop so that we can assume that a metavariable has been
        // added for every (Node, Direction) in the graph already.
        for node in hugr.nodes() {
            let m_input = self.resources.get(&(node, Direction::Incoming)).unwrap().clone();
            let m_output = self.resources.get(&(node, Direction::Outgoing)).unwrap().clone();
            for in_neighbour in hugr.input_neighbours(node) {
                let m_src = self.resources.get(&(in_neighbour, Direction::Outgoing)).unwrap();
                self.add_constraint(m_input, Constraint::Equal(*m_src));
            }
            for out_neighbour in hugr.output_neighbours(node) {
                let m_tgt = self.resources.get(&(out_neighbour, Direction::Incoming)).unwrap();
                self.add_constraint(m_output, Constraint::Equal(*m_tgt));
            }
        }
    }

    /// Take a group of equal metas and merge them into a new, single meta.
    /// Returns a set of metas that were merged.
    fn coalesce(&mut self) -> Result<HashSet<Meta>, InferResourceError> {
        let mut merged: HashSet<Meta> = HashSet::new();
        for cc in self.eq_graph.ccs().into_iter() {
            // Within a connected component everything is equal
            let combined_meta = self.fresh_meta();
            println!("Made: {:?}", combined_meta);
            for m in cc.iter() {
                if self.shunted.contains_key(&m) {
                    continue;
                }

                if let Some(cs) = self.constraints.get(m).cloned() {
                    for c in cs.into_iter().filter(|c| !matches!(c, Constraint::Equal(_))) {
                        self.add_constraint(combined_meta, c.clone());
                    }
                    self.constraints.remove(m).unwrap();
                    merged.insert(*m);
                    println!("Coalesce Merged {:?}", m);
                }
                if let Some(solution) = self.solved.get(m) {
                    match self.solved.get(&combined_meta) {
                        Some(existing_solution) => {
                            if solution != existing_solution {
                                return Err(InferResourceError::MismatchedConcrete { expected: solution.clone(), actual: existing_solution.clone() });
                            }
                        },
                        None => { self.solved.insert(combined_meta, solution.clone()); },
                    }
                }
                if self.variables.contains(m) {
                    self.variables.insert(combined_meta);
                }
                self.shunted.insert(*m, combined_meta);
                println!("Shunting {:?} -> {:?}", m, combined_meta);
            }
            // This doesn't do anything, but if it did it would just be
            // implementing the same mechanism as provided by `resolve`
            let mut updates = HashMap::<(Node, Direction), Meta>::new();
            for (loc, meta) in self.resources.iter() {
                // If there's any node in this CC which equals `meta`, they are
                // all equal to meta
                if cc.iter().filter(|cc_meta| meta == *cc_meta).count() > 0 {
                    updates.insert(*loc, combined_meta);
                }
            }
        }
        Ok(merged)
    }

    /// Inspect the constraints of a given metavariable and try to find a
    /// solution based on those.
    /// Returns whether a solution was found
    fn solve_meta(&mut self, meta: Meta) -> Result<bool, InferResourceError> {
        let mut solved = false;
        for c in self.get_constraints(&meta).unwrap().clone().iter() {
            match c {
                Constraint::Exactly(rs2) => {
                    match self.get_solution(&meta) {
                        None => {
                            self.add_solution(meta, rs2.clone());
                        }
                        Some(rs) => {
                            // If they're the same then we're happy
                            if *rs != *rs2 {
                                return Err(InferResourceError::MismatchedConcrete {
                                    expected: rs2.clone(),
                                    actual: rs.clone(),
                                });
                            }
                        }
                    };
                    solved = true;
                }
                // Just register the equality in the EqGraph, we'll process it later
                Constraint::Equal(other_meta) => { self.eq_graph.register_eq(meta, *other_meta); },
                Constraint::Plus(r, other_meta) => {
                    match self.get_solution(other_meta) {
                        Some(rs) => {
                            let mut rrs = rs.clone();
                            rrs.insert(r);
                            match self.get_solution(&meta) {
                                // Let's check that this is right?
                                Some(rs) => {
                                    if *rs != rrs {
                                        return Err(InferResourceError::MismatchedConcrete {
                                            expected: rs.clone(),
                                            actual: rrs,
                                        });
                                    }
                                }
                                None => self.add_solution(meta, rrs),
                            };
                            solved = true;
                        }
                        // TODO: Try and go backwards with a `Minus` constraint
                        // I.e. If we have a concrete solution for this
                        // metavariable, we can then work out what `other_meta`
                        // should be by subtracting the resource that's `Plus`d
                        //
                        // N.B. This could be the case of Plus(r, a) where a
                        // parameterises the whole graph, in which case we're done
                        None => {}
                    }
                }
            }
        }
        Ok(solved)
    }

    /// Tries to return concrete resources for each node in the graph. This only
    /// works when there are no variables in the graph!
    ///
    /// What we really want is to give the concrete resources where they're
    /// available. When there are variables, we should leave the graph as it is,
    /// but make sure that no matter what they're instantiated to, the graph
    /// still makes sense (should pass the resource validation check)
    pub fn results(
        &mut self,
    ) -> Result<ResourceSolution, InferResourceError> {
        // Check that all of the metavariables associated with nodes of the
        // graph are solved
        let mut results: ResourceSolution = HashMap::new();
        for (loc, meta) in self.resources.iter() {
            let rs = match self.get_solution(meta) {
                Some(rs) => Ok(rs.clone()),
                None => {
                    // Cut through the riff raff
                    if let Some(live_var) = self.live_var(meta) {
                        Err(InferResourceError::Unsolved { location: *loc })
                    } else {
                        continue;
                    }
                }
            }?;
            results.insert(*loc, rs);
        }
        assert!(self.live_metas().is_empty());
        Ok(results)
    }

    // Get the live var associated with a meta.
    // TODO: This should really be a list
    fn live_var(&self, m: &Meta) -> Option<Meta> {
        if self.variables.contains(m) || self.variables.contains(&self.resolve(*m)) {
            return None
        }

        // TODO: We should be doing something to ensure that these are the same check...
        if self.get_solution(m).is_none() {
            if !self.get_constraints(m).is_none() {
                for c in self.get_constraints(m).unwrap().iter() {
                    match c {
                        Constraint::Plus(_, m) => return self.live_var(m),
                        _ => panic!("we shouldn't be here!"),
                    }
                }
            }
            Some(m.clone())
        } else {
            None
        }
    }

    /// Return the set of "live" metavariables in the context.
    /// "Live" here means a metavariable:
    ///   - Is associated to a location in the graph in `UnifyContext.resources`
    ///   - Is still unsolved
    ///   - Isn't a variable
    fn live_metas(&self) -> HashSet<Meta> {
        self.resources
            .values()
            .filter_map(|m| self.live_var(m))
            .filter(|m| !self.variables.contains(m))
            .collect()
    }

    /// Returns the metas that we solved
    fn solve_constraints(
        &mut self,
        vars: &HashSet<Meta>,
    ) -> Result<HashSet<Meta>, InferResourceError> {
        let mut solved = HashSet::new();
        for m in vars.into_iter() {
            if self.solve_meta(*m)? {
                println!("Solved {:?}", m);
                solved.insert(*m);
            }
        }
        Ok(solved)
    }

    /// Once the unification context is set up, attempt to infer ResourceSets
    /// for all of the metavariables in the `UnificationContext`.
    ///
    /// Return a mapping from locations in the graph to concrete `ResourceSets`
    /// where it was possible to infer them. If it wasn't possible to infer a
    /// *concrete* `ResourceSet`, e.g. if the ResourceSet relies on an open
    /// variable in the toplevel graph, don't include that location in the map
    pub fn main_loop(
        &mut self,
    ) -> Result<ResourceSolution, InferResourceError> {
        let mut remaining = HashSet::<Meta>::from_iter(self.constraints.keys().cloned());

        // Keep going as long as we're making progress (= merging nodes)
        loop {
            let mut to_delete = self.solve_constraints(&remaining)?;
            let merged = self.coalesce()?;
            let delta: HashSet<Meta> = HashSet::from_iter(to_delete.union(&merged).into_iter().cloned());

            for m in delta.iter() {
                if !merged.contains(&m) {
                    self.constraints.remove(m);
                }
                remaining.remove(m);
            }

            if delta.is_empty() {
                break;
            }
        }
        self.results()
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::*;
    use crate::builder::{
        BuildError, Container, DFGBuilder, Dataflow, DataflowHugr,
    };
    use crate::hugr::HugrInternalsMut;
    use crate::hugr::{validate::ValidationError, Hugr, HugrView, NodeType};
    use crate::ops::{self, dataflow::IOTrait};
    use crate::resource::ResourceSet;
    use crate::type_row;
    use crate::types::{AbstractSignature, Type};

    use cool_asserts::assert_matches;
    use portgraph::NodeIndex;

    use crate::utils::test::viz_dotstr;

    const BIT: Type = crate::resource::prelude::USIZE_T;

    #[test]
    // Build up a graph with some holes in its resources, and infer them
    // See if it works!
    fn from_graph() -> Result<(), Box<dyn Error>> {
        use crate::ops::OpTrait;

        let rs = ResourceSet::from_iter(["A".into(), "B".into(), "C".into()]);
        let main_sig =
            AbstractSignature::new_df(type_row![BIT, BIT], type_row![BIT]).with_resource_delta(&rs);

        let op = ops::DFG {
            signature: main_sig,
        };

        println!("{:?}", op.clone().tag());

        let root_node = NodeType::pure(op);

        // TODO: This harder case:
        //let root_node = NodeType::open_resources(ops::DFG { signature: main_sig });
        let mut hugr = Hugr::new(root_node);

        /*
                let f_node = hugr.add_child_op(root_node
                    name: name.into(),
                    signature: signature.clone().into(),
                })?;
        */

        let input = NodeType::open_resources(ops::Input::new(type_row![BIT, BIT]));
        let output = NodeType::open_resources(ops::Output::new(type_row![BIT]));

        let input = hugr.add_node_with_parent(hugr.root(), input)?;
        let output = hugr.add_node_with_parent(hugr.root(), output)?;

        assert_matches!(hugr.get_io(hugr.root()), Some(_));

        let add_a_sig = AbstractSignature::new_df(type_row![BIT], type_row![BIT])
            .with_resource_delta(&ResourceSet::singleton(&"A".into()));

        let add_b_sig = AbstractSignature::new_df(type_row![BIT], type_row![BIT])
            .with_resource_delta(&ResourceSet::singleton(&"B".into()));

        let add_ab_sig = AbstractSignature::new_df(type_row![BIT], type_row![BIT])
            .with_resource_delta(&ResourceSet::from_iter(["A".into(), "B".into()]));

        let mult_c_sig = AbstractSignature::new_df(type_row![BIT, BIT], type_row![BIT])
            .with_resource_delta(&ResourceSet::singleton(&"C".into()));

        let add_a = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_resources(ops::DFG {
                signature: add_a_sig,
            }),
        )?;
        let add_b = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_resources(ops::DFG {
                signature: add_b_sig,
            }),
        )?;
        let add_ab = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_resources(ops::DFG {
                signature: add_ab_sig,
            }),
        )?;
        let mult_c = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_resources(ops::DFG {
                signature: mult_c_sig,
            }),
        )?;

        hugr.connect(input, 0, add_a, 0)?;
        hugr.connect(add_a, 0, add_b, 0)?;
        hugr.connect(add_b, 0, mult_c, 0)?;

        hugr.connect(input, 1, add_ab, 0)?;
        hugr.connect(add_ab, 0, mult_c, 1)?;

        hugr.connect(mult_c, 0, output, 0)?;

        hugr.infer_resources()?;
        // TODO: Add a more sensible validation check
        //hugr.validate()?;
        Ok(())
    }

    #[test]
    fn plus() -> Result<(), InferResourceError> {
        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);

        let metas: Vec<Meta> = (2..8).into_iter().map(|i| {
            let meta = ctx.fresh_meta();
            ctx.resources.insert((NodeIndex::new(i).into(), Direction::Incoming), meta);
            meta
        }).collect();

        ctx.add_constraint(metas[2], Constraint::Exactly(ResourceSet::singleton(&"A".into())));
        ctx.add_constraint(metas[1], Constraint::Equal(metas[2]));
        ctx.add_constraint(metas[0], Constraint::Plus("B".into(), metas[2]));
        ctx.add_constraint(metas[4], Constraint::Plus("C".into(), metas[0]));
        ctx.add_constraint(metas[3], Constraint::Equal(metas[4]));
        ctx.add_constraint(metas[5], Constraint::Equal(metas[0]));
        ctx.main_loop()?;

        let a = ResourceSet::singleton(&"A".into());
        let mut ab = a.clone();
        ab.insert(&"B".into());
        let mut abc = ab.clone();
        abc.insert(&"C".into());

        assert_eq!(ctx.get_solution(&metas[0]).unwrap(), &ab);
        assert_eq!(ctx.get_solution(&metas[1]).unwrap(), &a);
        assert_eq!(ctx.get_solution(&metas[2]).unwrap(), &a);
        assert_eq!(ctx.get_solution(&metas[3]).unwrap(), &abc);
        assert_eq!(ctx.get_solution(&metas[4]).unwrap(), &abc);
        assert_eq!(ctx.get_solution(&metas[5]).unwrap(), &ab);

        Ok(())
    }

    #[test]
    // idk wht to call this
    // We've got to be weird
    // Is this the same as from_graph?
    // this might be the minus test that we have elsewhere?
    fn idk() -> Result<(), Box<dyn Error>> {
        let rs = ResourceSet::singleton(&"R".into());
        let root_signature =
            AbstractSignature::new_df(type_row![BIT], type_row![BIT]).with_resource_delta(&rs);
        let mut builder = DFGBuilder::new(root_signature)?;
        let [input_wire] = builder.input_wires_arr();

        let add_r_sig =
            AbstractSignature::new_df(type_row![BIT], type_row![BIT]).with_resource_delta(&rs);

        let add_r = builder.add_dataflow_node(
            NodeType::open_resources(ops::DFG {
                signature: add_r_sig,
            }),
            [input_wire],
        )?;
        let [wl] = add_r.outputs_arr();

        // Dangling thingy
        let src_sig = AbstractSignature::new_df(type_row![], type_row![BIT])
            .with_resource_delta(&ResourceSet::new());
        let src = builder.add_dataflow_node(
            NodeType::open_resources(ops::DFG { signature: src_sig }),
            [],
        )?;
        let [wr] = src.outputs_arr();

        let mult_sig = AbstractSignature::new_df(type_row![BIT, BIT], type_row![BIT])
            .with_resource_delta(&ResourceSet::new());
        let mult = builder.add_dataflow_node(
            NodeType::open_resources(ops::DFG {
                signature: mult_sig,
            }),
            [wl, wr],
        )?;
        let [w] = mult.outputs_arr();

        let h = builder.finish_hugr_with_outputs([w])?;
        Ok(())
    }

    #[test]
    // This generates a solution that causes validation to fail
    // because of a missing lift node
    fn missing_lift_node() -> Result<(), Box<dyn Error>> {
        let mut builder = DFGBuilder::new(
            AbstractSignature::new_df(type_row![BIT], type_row![BIT])
                .with_resource_delta(&ResourceSet::singleton(&"R".into())),
        )?;
        let [w] = builder.input_wires_arr();
        let hugr = builder.finish_hugr_with_outputs([w]);

        assert_matches!(
            hugr,
            Err(BuildError::InvalidHUGR(
                ValidationError::ResourceError(ResourceError::TgtExceedsSrcResources { .. })
            ))
        );
        Ok(())
    }

    #[test]
    fn reverse() -> Result<(), Box<dyn Error>> {
        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);

        let m0 = ctx.fresh_meta();
        let m1 = ctx.fresh_meta();
        //let m2 = ctx.fresh_meta();
        let m3 = ctx.fresh_meta();
        ctx.add_constraint(
            m0,
            Constraint::Exactly(ResourceSet::from_iter(["A".into()])),
        );
        ctx.add_constraint(m1, Constraint::Plus("B".into(), m0));
        //ctx.add_constraint(m2, Constraint::Equal(m1));
        //ctx.add_constraint(m2, Constraint::Plus("A".into(), m3));
        ctx.add_constraint(m1, Constraint::Plus("A".into(), m3));
        ctx.main_loop()?;

        for s in ctx.solved.iter() {
            println!("{:?}\n", s);
        }

        assert_eq!(
            ctx.get_solution(&m3).unwrap(),
            &ResourceSet::singleton(&"B".into())
        );

        Ok(())
    }

    #[test]
    /* We should be able to find a solution for this, of the form
     forall x.
       m0 = x;
       m1 = A, x;
       m2 = A, x;
       m3 = x;
    */
    fn open() -> Result<(), InferResourceError> {
        println!("awefiawef");

        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);
        let m0 = ctx.fresh_meta();
        let m1 = ctx.fresh_meta();
        let m2 = ctx.fresh_meta();
        let m3 = ctx.fresh_meta();
        // Attach the metavariables to dummy nodes so that they're considered important
        ctx.resources
            .insert((NodeIndex::new(1).into(), Direction::Incoming), m0);
        ctx.resources
            .insert((NodeIndex::new(2).into(), Direction::Incoming), m1);
        ctx.resources
            .insert((NodeIndex::new(3).into(), Direction::Incoming), m2);
        ctx.resources
            .insert((NodeIndex::new(4).into(), Direction::Incoming), m3);

        ctx.add_constraint(m1, Constraint::Plus("A".into(), m0));
        ctx.add_constraint(m2, Constraint::Equal(m1));
        ctx.add_constraint(m3, Constraint::Equal(m0));
        ctx.variables.insert(m0);
        println!("Constraints before processing");
        for r in ctx.resources.iter() {
            println!("     {:?}", r);
        }
        for r in ctx.constraints.iter() {
            println!("     {:?}", r);
        }

        let results = ctx.main_loop()?;

        println!("results: {:?}", results);

        Ok(())
    }

    #[test]
    fn simple_dumb() -> Result<(), InferResourceError> {
        let mut ctx = UnificationContext::new(&Hugr::default());
        let m0 = ctx.fresh_meta();
        let m1 = ctx.fresh_meta();
        let m2 = ctx.fresh_meta();
        ctx.add_constraint(m1, Constraint::Plus("A".into(), m0));
        ctx.add_constraint(m2, Constraint::Plus("B".into(), m0));
        ctx.main_loop()?;
        Ok(())
    }

    #[test]
    // The dual of the thing in the previous test
    fn simple_dumb_2() -> Result<(), InferResourceError> {
        let mut ctx = UnificationContext::new(&Hugr::default());
        let a = ctx.fresh_meta();
        let b = ctx.fresh_meta();
        let ab = ctx.fresh_meta();
        // Some nonsense so that the constraints register as "live"
        ctx.resources.insert((NodeIndex::new(2).into(), Direction::Outgoing), a);
        ctx.resources.insert((NodeIndex::new(3).into(), Direction::Outgoing), b);
        ctx.resources.insert((NodeIndex::new(4).into(), Direction::Incoming), ab);
        ctx.variables.insert(a);
        ctx.variables.insert(b);
        ctx.add_constraint(ab, Constraint::Plus("A".into(), b));
        ctx.add_constraint(ab, Constraint::Plus("B".into(), a));
        ctx.main_loop()?;
        Ok(())
    }
}

//! Inference for extension requirements on nodes of a hugr.
//!
//! Checks if the extensions requirements have a solution in terms of some
//! number of starting variables, and comes up with concrete solutions when
//! possible.
//!
//! Open extension variables can come from toplevel nodes: notionally "inputs"
//! to the graph where being wired up to a larger hugr would provide the
//! information needed to solve variables. When extension requirements of nodes
//! depend on these open variables, then the validation check for extensions
//! will succeed regardless of what the variable is instantiated to.

use super::ExtensionSet;
use crate::{
    hugr::views::HugrView,
    ops::{OpTag, OpTrait},
    types::EdgeKind,
    Direction, Node,
};

use petgraph::graph as pg;
use petgraph::{Directed, EdgeType, Undirected};

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

/// A mapping from nodes on the hugr to extension requirement sets which have
/// been inferred for their inputs.
pub type ExtensionSolution = HashMap<Node, ExtensionSet>;

/// Infer extensions for a hugr. This is the main API exposed by this module.
pub fn infer_extensions(hugr: &impl HugrView) -> Result<ExtensionSolution, InferExtensionError> {
    let mut ctx = UnificationContext::new(hugr);
    ctx.main_loop()?;
    ctx.instantiate_variables();
    let all_results = ctx.main_loop()?;
    let new_results = all_results
        .into_iter()
        .filter(|(n, _sol)| hugr.get_nodetype(*n).input_extensions().is_none())
        .collect();
    Ok(new_results)
}

/// Metavariables don't need much
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Meta(u32);

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// Things we know about metavariables
enum Constraint {
    /// A variable has the same value as another variable
    Equal(Meta),
    /// Variable extends the value of another by a set of extensions
    Plus(ExtensionSet, Meta),
}

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
/// Errors which arise during unification
pub enum InferExtensionError {
    #[error("Mismatched extension sets {expected} and {actual}")]
    /// We've solved a metavariable, then encountered a constraint
    /// that says it should be something other than our solution
    MismatchedConcrete {
        /// The solution we were trying to insert for this meta
        expected: ExtensionSet,
        /// The incompatible solution that we found was already there
        actual: ExtensionSet,
    },
    #[error("Solved extensions {expected} at {expected_loc:?} and {actual} at {actual_loc:?} should be equal.")]
    /// A version of the above with info about which nodes failed to unify
    MismatchedConcreteWithLocations {
        /// Where the solution we want to insert came from
        expected_loc: (Node, Direction),
        /// The solution we were trying to insert for this meta
        expected: ExtensionSet,
        /// Which node we're trying to add a solution for
        actual_loc: (Node, Direction),
        /// The incompatible solution that we found was already there
        actual: ExtensionSet,
    },
    /// A variable went unsolved that wasn't related to a parameter
    #[error("Unsolved variable at location {:?}", location)]
    Unsolved {
        /// The location on the hugr that's associated to the unsolved meta
        location: (Node, Direction),
    },
    /// Too many extension requirements coming from src
    #[error("Extensions at source node {from:?} ({from_extensions}) exceed those at target {to:?} ({to_extensions})")]
    #[allow(missing_docs)]
    SrcExceedsTgtExtensions {
        from: Node,
        from_extensions: ExtensionSet,
        to: Node,
        to_extensions: ExtensionSet,
    },
    /// Missing lift node
    #[error("Extensions at target node {to:?} ({to_extensions}) exceed those at source {from:?} ({from_extensions})")]
    #[allow(missing_docs)]
    TgtExceedsSrcExtensions {
        from: Node,
        from_extensions: ExtensionSet,
        to: Node,
        to_extensions: ExtensionSet,
    },
}

/// A graph of metavariables connected by constraints.
/// The edges represent `Equal` constraints in the undirected graph and `Plus`
/// constraints in the directed case.
struct GraphContainer<Dir: EdgeType> {
    graph: pg::Graph<Meta, (), Dir>,
    node_map: HashMap<Meta, pg::NodeIndex>,
}

impl<T: EdgeType> GraphContainer<T> {
    /// Add a metavariable to the graph as a node and return the `NodeIndex`.
    /// If it's already there, just return the existing `NodeIndex`
    fn add_or_retrieve(&mut self, m: Meta) -> pg::NodeIndex {
        self.node_map.get(&m).cloned().unwrap_or_else(|| {
            let ix = self.graph.add_node(m);
            self.node_map.insert(m, ix);
            ix
        })
    }

    /// Create an edge between two nodes on the graph
    fn add_edge(&mut self, src: Meta, tgt: Meta) {
        let src_ix = self.add_or_retrieve(src);
        let tgt_ix = self.add_or_retrieve(tgt);
        self.graph.add_edge(src_ix, tgt_ix, ());
    }

    /// Return the strongly connected components of the graph in terms of
    /// metavariables. In the undirected case, return the connected components
    fn sccs(&self) -> Vec<Vec<Meta>> {
        petgraph::algo::tarjan_scc(&self.graph)
            .into_iter()
            .map(|cc| {
                cc.into_iter()
                    .map(|n| *self.graph.node_weight(n).unwrap())
                    .collect()
            })
            .collect()
    }
}

impl GraphContainer<Undirected> {
    fn new() -> Self {
        GraphContainer {
            graph: pg::Graph::new_undirected(),
            node_map: HashMap::new(),
        }
    }
}

impl GraphContainer<Directed> {
    fn new() -> Self {
        GraphContainer {
            graph: pg::Graph::new(),
            node_map: HashMap::new(),
        }
    }
}

type EqGraph = GraphContainer<Undirected>;

/// Our current knowledge about the extensions of the graph
struct UnificationContext {
    /// A list of constraints for each metavariable
    constraints: HashMap<Meta, HashSet<Constraint>>,
    /// A map which says which nodes correspond to which metavariables
    extensions: HashMap<(Node, Direction), Meta>,
    /// Solutions to metavariables
    solved: HashMap<Meta, ExtensionSet>,
    /// A graph which says which metavariables should be equal
    eq_graph: EqGraph,
    /// A mapping from metavariables which have been merged, to the meta they've
    // been merged to
    shunted: HashMap<Meta, Meta>,
    /// Variables we're allowed to include in solutionss
    variables: HashSet<Meta>,
    /// A name for the next metavariable we create
    fresh_name: u32,
}

/// Invariant: Constraint::Plus always points to a fresh metavariable
impl UnificationContext {
    /// Create a new unification context, and populate it with constraints from
    /// traversing the hugr which is passed in.
    fn new(hugr: &impl HugrView) -> Self {
        let mut ctx = Self {
            constraints: HashMap::new(),
            extensions: HashMap::new(),
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
        let fresh = Meta(self.fresh_name);
        self.fresh_name += 1;
        self.constraints.insert(fresh, HashSet::new());
        fresh
    }

    /// Declare a constraint on the metavariable
    fn add_constraint(&mut self, m: Meta, c: Constraint) {
        self.constraints.entry(m).or_default().insert(c);
    }

    /// Declare that a meta has been solved
    fn add_solution(&mut self, m: Meta, rs: ExtensionSet) {
        let existing_sol = self.solved.insert(m, rs);
        debug_assert!(existing_sol.is_none());
    }

    /// If a metavariable has been merged, return the new meta, otherwise return
    /// the same meta.
    ///
    /// This could loop if there were a cycle in the `shunted` list, but there
    /// shouldn't be, because we only ever shunt to *new* metas.
    fn resolve(&self, m: Meta) -> Meta {
        self.shunted.get(&m).cloned().map_or(m, |m| self.resolve(m))
    }

    /// Get the relevant constraints for a metavariable. If it's been merged,
    /// get the constraints for the merged metavariable
    fn get_constraints(&self, m: &Meta) -> Option<&HashSet<Constraint>> {
        self.constraints.get(&self.resolve(*m))
    }

    /// Get the relevant solution for a metavariable. If it's been merged, get
    /// the solution for the merged metavariable
    fn get_solution(&self, m: &Meta) -> Option<&ExtensionSet> {
        self.solved.get(&self.resolve(*m))
    }

    /// Return the metavariable corresponding to the given location on the
    /// graph, either by making a new meta, or looking it up
    fn make_or_get_meta(&mut self, node: Node, dir: Direction) -> Meta {
        if let Some(m) = self.extensions.get(&(node, dir)) {
            *m
        } else {
            let m = self.fresh_meta();
            self.extensions.insert((node, dir), m);
            m
        }
    }

    /// Iterate over the nodes in a hugr and generate unification constraints
    fn gen_constraints<T>(&mut self, hugr: &T)
    where
        T: HugrView,
    {
        if hugr.root_type().input_extensions().is_none() {
            let m_input = self.make_or_get_meta(hugr.root(), Direction::Incoming);
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
                    // If the parent node is a FuncDefn, it will have no
                    // op_signature, so the Incoming and Outgoing ports will
                    // have equal extension requirements.
                    // The function that it contains, however, may have an
                    // extension delta, so its output shouldn't be equal to the
                    // FuncDefn's output.
                    //
                    // TODO: Add a constraint that the extensions of the output
                    // node of a FuncDefn should be those of the input node plus
                    // the extension delta specified in the function signature.
                    if node_type.tag() != OpTag::FuncDefn {
                        self.add_constraint(m_output_node, Constraint::Equal(m_output));
                    }
                }
            }

            if hugr.get_optype(node).tag() == OpTag::Conditional {
                for case in hugr.children(node) {
                    let m_case_in = self.make_or_get_meta(case, Direction::Incoming);
                    let m_case_out = self.make_or_get_meta(case, Direction::Outgoing);
                    self.add_constraint(m_case_in, Constraint::Equal(m_input));
                    self.add_constraint(m_case_out, Constraint::Equal(m_output));
                }
            }

            if node_type.tag() == OpTag::Cfg {
                let mut children = hugr.children(node);
                let entry = children.next().unwrap();
                let exit = children.next().unwrap();
                let m_entry = self.make_or_get_meta(entry, Direction::Incoming);
                let m_exit = self.make_or_get_meta(exit, Direction::Outgoing);
                self.add_constraint(m_input, Constraint::Equal(m_entry));
                self.add_constraint(m_output, Constraint::Equal(m_exit));
            }

            match node_type.io_extensions() {
                // Input extensions are open
                None => {
                    let delta = node_type.op().extension_delta();
                    let c = if delta.is_empty() {
                        Constraint::Equal(m_input)
                    } else {
                        Constraint::Plus(delta, m_input)
                    };
                    self.add_constraint(m_output, c);
                }
                // We have a solution for everything!
                Some((input_exts, output_exts)) => {
                    self.add_solution(m_input, input_exts.clone());
                    self.add_solution(m_output, output_exts);
                }
            }
        }
        // Separate loop so that we can assume that a metavariable has been
        // added for every (Node, Direction) in the graph already.
        for tgt_node in hugr.nodes() {
            let sig = hugr.get_nodetype(tgt_node).op();
            // Incoming ports with an edge that should mean equal extension reqs
            for port in hugr.node_inputs(tgt_node).filter(|src_port| {
                let kind = sig.port_kind(*src_port);
                kind.as_ref().is_some_and(EdgeKind::is_static)
                    || matches!(kind, Some(EdgeKind::Value(_)) | Some(EdgeKind::ControlFlow))
            }) {
                let m_tgt = *self
                    .extensions
                    .get(&(tgt_node, Direction::Incoming))
                    .unwrap();
                for (src_node, _) in hugr.linked_ports(tgt_node, port) {
                    let m_src = self
                        .extensions
                        .get(&(src_node, Direction::Outgoing))
                        .unwrap();
                    self.add_constraint(*m_src, Constraint::Equal(m_tgt));
                }
            }
        }
    }

    /// When trying to unify two metas, check if they both correspond to
    /// different ends of the same wire. If so, return an `ExtensionError`.
    /// Otherwise check whether they both correspond to *some* location on the
    /// graph and include that info the otherwise generic `MismatchedConcrete`.
    fn report_mismatch(
        &self,
        m1: Meta,
        m2: Meta,
        rs1: ExtensionSet,
        rs2: ExtensionSet,
    ) -> InferExtensionError {
        let loc1 = self
            .extensions
            .iter()
            .find(|(_, m)| **m == m1 || self.resolve(**m) == m1)
            .map(|a| a.0);
        let loc2 = self
            .extensions
            .iter()
            .find(|(_, m)| **m == m2 || self.resolve(**m) == m2)
            .map(|a| a.0);
        if let (Some((node1, dir1)), Some((node2, dir2))) = (loc1, loc2) {
            // N.B. We're looking for the case where an equality constraint
            // arose because the two locations are connected by an edge

            // If the directions are the same, they shouldn't be connected
            // to each other. If the nodes are the same, there's no edge!
            //
            // TODO: It's still possible that the equality constraint
            // arose because one node is a dataflow parent and the other
            // is one of it's I/O nodes. In that case, the directions could be
            // the same, and we should try to detect it
            if dir1 != dir2 && node1 != node2 {
                let [(src, src_rs), (tgt, tgt_rs)] = if *dir2 == Direction::Incoming {
                    [(node1, rs1.clone()), (node2, rs2.clone())]
                } else {
                    [(node2, rs2.clone()), (node1, rs1.clone())]
                };

                return if src_rs.is_subset(&tgt_rs) {
                    InferExtensionError::TgtExceedsSrcExtensions {
                        from: *src,
                        from_extensions: src_rs,
                        to: *tgt,
                        to_extensions: tgt_rs,
                    }
                } else {
                    InferExtensionError::SrcExceedsTgtExtensions {
                        from: *src,
                        from_extensions: src_rs,
                        to: *tgt,
                        to_extensions: tgt_rs,
                    }
                };
            }
        }
        if let (Some(loc1), Some(loc2)) = (loc1, loc2) {
            InferExtensionError::MismatchedConcreteWithLocations {
                expected_loc: *loc1,
                expected: rs1,
                actual_loc: *loc2,
                actual: rs2,
            }
        } else {
            InferExtensionError::MismatchedConcrete {
                expected: rs1,
                actual: rs2,
            }
        }
    }

    /// Take a group of equal metas and merge them into a new, single meta.
    ///
    /// Returns the set of new metas created and the set of metas that were
    /// merged.
    fn merge_equal_metas(&mut self) -> Result<(HashSet<Meta>, HashSet<Meta>), InferExtensionError> {
        let mut merged: HashSet<Meta> = HashSet::new();
        let mut new_metas: HashSet<Meta> = HashSet::new();
        for cc in self.eq_graph.sccs().into_iter() {
            // Within a connected component everything is equal
            let combined_meta = self.fresh_meta();
            for m in cc.iter() {
                // The same meta shouldn't be shunted twice directly. Only
                // transitively, as we still process the meta it was shunted to
                if self.shunted.contains_key(m) {
                    continue;
                }

                if let Some(cs) = self.constraints.remove(m) {
                    for c in cs
                        .iter()
                        .filter(|c| !matches!(c, Constraint::Equal(_)))
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                    {
                        self.add_constraint(combined_meta, c.clone());
                    }
                    merged.insert(*m);
                    // Record a new meta the first time that we use it; don't
                    // bother recording a new meta if we don't add any
                    // constraints. It should be safe to call this multiple times
                    new_metas.insert(combined_meta);
                }
                // Here, solved.get is equivalent to get_solution, because if
                // `m` had already been shunted, we wouldn't skipped it
                if let Some(solution) = self.solved.get(m) {
                    match self.solved.get(&combined_meta) {
                        Some(existing_solution) => {
                            if solution != existing_solution {
                                return Err(self.report_mismatch(
                                    *m,
                                    combined_meta,
                                    solution.clone(),
                                    existing_solution.clone(),
                                ));
                            }
                        }
                        None => {
                            self.solved.insert(combined_meta, solution.clone());
                        }
                    }
                }
                if self.variables.contains(m) {
                    self.variables.insert(combined_meta);
                    self.variables.remove(m);
                }
                self.shunted.insert(*m, combined_meta);
            }
        }
        Ok((new_metas, merged))
    }

    /// Inspect the constraints of a given metavariable and try to find a
    /// solution based on those.
    /// Returns whether a solution was found
    fn solve_meta(&mut self, meta: Meta) -> Result<bool, InferExtensionError> {
        let mut solved = false;
        for c in self.get_constraints(&meta).unwrap().clone().iter() {
            match c {
                // Just register the equality in the EqGraph, we'll process it later
                Constraint::Equal(other_meta) => {
                    self.eq_graph.add_edge(meta, *other_meta);
                }
                // N.B. If `meta` is already solved, we can't use that
                // information to solve `other_meta`. This is because the Plus
                // constraint only signifies a preorder.
                // I.e. if meta = other_meta + 'R', it's still possible that the
                // solution is meta = other_meta because we could be adding 'R'
                // to a set which already contained it.
                Constraint::Plus(r, other_meta) => {
                    if let Some(rs) = self.get_solution(other_meta) {
                        let rrs = rs.clone().union(r.clone());
                        match self.get_solution(&meta) {
                            // Let's check that this is right?
                            Some(rs) => {
                                if rs != &rrs {
                                    return Err(self.report_mismatch(
                                        meta,
                                        *other_meta,
                                        rs.clone(),
                                        rrs,
                                    ));
                                }
                            }
                            None => {
                                self.add_solution(meta, rrs);
                                solved = true;
                            }
                        };
                    };
                }
            }
        }
        Ok(solved)
    }

    /// Tries to return concrete extensions for each node in the graph. This only
    /// works when there are no variables in the graph!
    ///
    /// What we really want is to give the concrete extensions where they're
    /// available. When there are variables, we should leave the graph as it is,
    /// but make sure that no matter what they're instantiated to, the graph
    /// still makes sense (should pass the extension validation check)
    fn results(&self) -> Result<ExtensionSolution, InferExtensionError> {
        // Check that all of the metavariables associated with nodes of the
        // graph are solved
        let depended_upon = {
            let mut h: HashMap<Meta, Vec<Meta>> = HashMap::new();
            for (m, m2) in self.constraints.iter().flat_map(|(m, cs)| {
                cs.iter().flat_map(|c| match c {
                    Constraint::Plus(_, m2) => Some((*m, self.resolve(*m2))),
                    _ => None,
                })
            }) {
                h.entry(m2).or_default().push(m);
            }
            h
        };
        // Calculate everything dependent upon a variable.
        // Note it would be better to find metas ALL of whose dependencies were (transitively)
        // on variables, but this is more complex, and hard to define if there are cycles
        // of PLUS constraints, so leaving that as a TODO until we've handled such cycles.
        let mut depends_on_var = HashSet::new();
        let mut queue = VecDeque::from_iter(self.variables.iter());
        while let Some(m) = queue.pop_front() {
            if depends_on_var.insert(m) {
                if let Some(d) = depended_upon.get(m) {
                    queue.extend(d.iter())
                }
            }
        }

        let mut results: ExtensionSolution = HashMap::new();
        for (loc, meta) in self.extensions.iter() {
            if let Some(rs) = self.get_solution(meta) {
                if loc.1 == Direction::Incoming {
                    results.insert(loc.0, rs.clone());
                }
            } else {
                // Unsolved nodes must be unsolved because they depend on graph variables.
                if !depends_on_var.contains(&self.resolve(*meta)) {
                    return Err(InferExtensionError::Unsolved { location: *loc });
                }
            }
        }
        Ok(results)
    }

    /// Iterates over a set of metas (the argument) and tries to solve
    /// them.
    /// Returns the metas that we solved
    fn solve_constraints(
        &mut self,
        vars: &HashSet<Meta>,
    ) -> Result<HashSet<Meta>, InferExtensionError> {
        let mut solved = HashSet::new();
        for m in vars.iter() {
            if self.solve_meta(*m)? {
                solved.insert(*m);
            }
        }
        Ok(solved)
    }

    /// Once the unification context is set up, attempt to infer ExtensionSets
    /// for all of the metavariables in the `UnificationContext`.
    ///
    /// Return a mapping from locations in the graph to concrete `ExtensionSets`
    /// where it was possible to infer them. If it wasn't possible to infer a
    /// *concrete* `ExtensionSet`, e.g. if the ExtensionSet relies on an open
    /// variable in the toplevel graph, don't include that location in the map
    fn main_loop(&mut self) -> Result<ExtensionSolution, InferExtensionError> {
        let mut remaining = HashSet::<Meta>::from_iter(self.constraints.keys().cloned());

        // Keep going as long as we're making progress (= merging and solving nodes)
        loop {
            // Try to solve metas with the information we have now. This may
            // register new equalities on the EqGraph
            let to_delete = self.solve_constraints(&remaining)?;
            // Merge metas based on the equalities we just registered
            let (new, merged) = self.merge_equal_metas()?;
            // All of the metas for which we've made progress
            let delta: HashSet<Meta> = HashSet::from_iter(to_delete.union(&merged).cloned());

            // Clean up dangling constraints on solved metavariables
            to_delete.iter().for_each(|m| {
                self.constraints.remove(m);
            });
            // Remove solved and merged metas from remaining "to solve" list
            delta.iter().for_each(|m| {
                remaining.remove(m);
            });

            // If we made no progress, we're done!
            if delta.is_empty() && new.is_empty() {
                break;
            }
            remaining.extend(new)
        }
        self.results()
    }

    /// Gather all the transitive dependencies (induced by constraints) of the
    /// variables in the context.
    fn search_variable_deps(&self) -> HashSet<Meta> {
        let mut seen = HashSet::new();
        let mut new_variables: HashSet<Meta> = self.variables.clone();
        while !new_variables.is_empty() {
            new_variables = new_variables
                .into_iter()
                .filter(|m| seen.insert(*m))
                .flat_map(|m| self.get_constraints(&m))
                .flatten()
                .map(|c| match c {
                    Constraint::Plus(_, other) => self.resolve(*other),
                    Constraint::Equal(other) => self.resolve(*other),
                })
                .collect();
        }
        seen
    }

    /// Instantiate all variables in the graph with the empty extension set, or
    /// the smallest solution possible given their constraints.
    /// This is done to solve metas which depend on variables, which allows
    /// us to come up with a fully concrete solution to pass into validation.
    ///
    /// Nodes which loop into themselves must be considered as a "minimum" set
    /// of requirements. If we have
    ///   1 = 2 + X, ...
    ///   2 = 1 + x, ...
    /// then 1 and 2 both definitely contain X, even if we don't know what else.
    /// So instead of instantiating to the empty set, we'll instantiate to `{X}`
    fn instantiate_variables(&mut self) {
        // A directed graph to keep track of `Plus` constraint relationships
        let mut relations = GraphContainer::<Directed>::new();
        let mut solutions: HashMap<Meta, ExtensionSet> = HashMap::new();

        let variable_scope = self.search_variable_deps();
        for m in variable_scope.into_iter() {
            // If `m` has been merged, [`self.variables`] entry
            // will have already been updated to the merged
            // value by [`self.merge_equal_metas`] so we don't
            // need to worry about resolving it.
            if !self.solved.contains_key(&m) {
                // Handle the case where the constraints for `m` contain a self
                // reference, i.e. "m = Plus(E, m)", in which case the variable
                // should be instantiated to E rather than the empty set.
                let plus_constraints =
                    self.get_constraints(&m)
                        .unwrap()
                        .iter()
                        .cloned()
                        .flat_map(|c| match c {
                            Constraint::Plus(r, other_m) => Some((r, self.resolve(other_m))),
                            _ => None,
                        });

                let (rs, other_ms): (Vec<_>, Vec<_>) = plus_constraints.unzip();
                let solution = ExtensionSet::union_over(rs);
                let unresolved_metas = other_ms
                    .into_iter()
                    .filter(|other_m| m != *other_m)
                    .collect::<Vec<_>>();

                // If `m` doesn't depend on any other metas then we have all the
                // information we need to come up with a solution for it.
                relations.add_or_retrieve(m);
                unresolved_metas
                    .iter()
                    .for_each(|other_m| relations.add_edge(m, *other_m));
                solutions.insert(m, solution);
            }
        }

        // Process the strongly-connected components. petgraph/sccs() returns these
        // depended-upon before dependant, as we need.
        for cc in relations.sccs() {
            // Strongly connected components are looping constraint dependencies.
            // This means that each metavariable in the CC has the same solution.
            let combined_solution = cc
                .iter()
                .flat_map(|m| self.get_constraints(m).unwrap())
                .filter_map(|c| match c {
                    Constraint::Plus(_, other_m) => solutions.get(&self.resolve(*other_m)).cloned(),
                    Constraint::Equal(_) => None,
                })
                .fold(ExtensionSet::new(), ExtensionSet::union);

            for m in cc.iter() {
                self.add_solution(*m, combined_solution.clone());
                solutions.insert(*m, combined_solution.clone());
            }
        }
        self.variables = HashSet::new();
    }
}

#[cfg(test)]
mod test;

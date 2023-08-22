//! Inference for extension requirements on nodes of a hugr.
//!
//! Checks if the extensions requirements are sane, and comes up with concrete
//! solutions when possible. Inference operates when toplevel nodes can be open
//! variables. When extension requirements of nodes depend on these open
//! variables, then the validation check for extensions will succeed regardless
//! of what the variable is instantiated to.

use super::{ExtensionId, ExtensionSet};
use crate::{
    hugr::views::{HierarchyView, HugrView, SiblingGraph},
    hugr::{Node, Port},
    ops::OpType,
    types::EdgeKind,
    Direction,
};

use super::validate::ExtensionError;

use petgraph::graph as pg;

use std::collections::{HashMap, HashSet};

use thiserror::Error;

/// A mapping from locations on the hugr to extension requirement sets which
/// have been inferred for them
pub type ExtensionSolution = HashMap<(Node, Direction), ExtensionSet>;

/// Infer extensions for a hugr. This is the main API exposed by this module
pub fn infer_extensions(hugr: &impl HugrView) -> Result<ExtensionSolution, InferExtensionError> {
    let mut ctx = UnificationContext::new(hugr);
    ctx.main_loop()
}

/// Metavariables don't need much
type Meta = usize;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// Things we know about metavariables
enum Constraint {
    /// A variable has the same value as another variable
    Equal(Meta),
    /// Variable extends the value of another by one extension
    Plus(ExtensionId, Meta),
}

#[derive(Debug, Clone, PartialEq, Error)]
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
    /// A variable went unsolved that wasn't related to a parameter
    #[error("Unsolved variable at location {:?}", location)]
    Unsolved {
        /// The location on the hugr that's associated to the unsolved meta
        location: (Node, Direction),
        //constraints: Vec<Constraint>,
    },
    /// An extension mismatch between two nodes which are connected by an edge.
    /// This should mirror (or reuse) `ValidationError`'s SrcExceedsTgtExtensions
    /// and TgtExceedsSrcExtensions
    #[error("Edge mismatch: {0}")]
    EdgeMismatch(#[from] ExtensionError),
}

/// A graph of metavariables which we've found equality constraints for. Edges
/// between nodes represent equality constraints.
struct EqGraph {
    equalities: pg::Graph<Meta, (), petgraph::Undirected>,
    node_map: HashMap<Meta, pg::NodeIndex>,
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
            .map(|cc| {
                cc.into_iter()
                    .map(|n| *self.equalities.node_weight(n).unwrap())
                    .collect()
            })
            .collect()
    }
}

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
    fresh_name: usize,
}

/// Invariant: Constraint::Plus always points to a fresh metavariable
impl UnificationContext {
    /// Create a new unification context, and populate it with constraints from
    /// traversing the hugr which is passed in.
    pub fn new(hugr: &impl HugrView) -> Self {
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
        let fresh = self.fresh_name;
        self.fresh_name += 1;
        self.constraints.insert(fresh, HashSet::new());
        fresh
    }

    /// Declare a constraint on the metavariable
    fn add_constraint(&mut self, m: Meta, c: Constraint) {
        self.constraints
            .entry(m)
            .and_modify(|cs| {
                cs.insert(c.clone());
            })
            .or_insert(HashSet::from_iter([c]));
    }

    /// Declare that a meta has been solved
    fn add_solution(&mut self, m: Meta, rs: ExtensionSet) {
        assert!(self.solved.insert(m, rs).is_none());
    }

    /// If a metavariable has been merged, return the new meta
    fn resolve(&self, m: Meta) -> Meta {
        self.shunted.get(&m).cloned().map_or(m, |m| self.resolve(m))
    }

    fn get_constraints(&self, m: &Meta) -> Option<&HashSet<Constraint>> {
        self.constraints.get(&self.resolve(*m))
    }

    fn get_solution(&self, m: &Meta) -> Option<&ExtensionSet> {
        self.solved.get(&self.resolve(*m))
    }

    fn gen_union_constraint(&mut self, input: Meta, output: Meta, delta: ExtensionSet) {
        let mut last_meta = input;
        // Create fresh metavariables with `Plus` constraints for
        // each extension that should be added by the node
        // Hence a extension delta [A, B] would lead to
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
        // The toplevel sibling graph can be open, and we should note what those variables are
        let toplevel: SiblingGraph<Node, T> = SiblingGraph::new(hugr, hugr.root());
        for toplevel_node in toplevel.nodes() {
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
                // Input extensions are open
                None => {
                    self.gen_union_constraint(
                        m_input,
                        m_output,
                        node_type.op_signature().extension_reqs,
                    );
                }
                // We have a solution for everything!
                Some(sig) => {
                    self.add_solution(m_input, sig.input_extensions.clone());
                    self.add_solution(m_output, sig.output_extensions());
                }
            }
        }
        // Seperate loop so that we can assume that a metavariable has been
        // added for every (Node, Direction) in the graph already.
        for tgt_node in hugr.nodes() {
            let sig: &OpType = hugr.get_nodetype(tgt_node).into();
            // Incoming ports with a dataflow edge
            for port in hugr
                .node_inputs(tgt_node)
                .filter(|src_port| matches!(sig.port_kind(*src_port), Some(EdgeKind::Value(_))))
            {
                for (src_node, _) in hugr.linked_ports(tgt_node, port) {
                    let m_src = self
                        .extensions
                        .get(&(src_node, Direction::Outgoing))
                        .unwrap();
                    let m_tgt = self
                        .extensions
                        .get(&(tgt_node, Direction::Incoming))
                        .unwrap();
                    self.add_constraint(*m_src, Constraint::Equal(*m_tgt));
                }
            }
        }
    }

    /// Try to turn mismatches into `ExtensionError` when possible
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
        let err = if let (Some((node1, dir1)), Some((node2, dir2))) = (loc1, loc2) {
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

                if src_rs.is_subset(&tgt_rs) {
                    Some(InferExtensionError::EdgeMismatch(
                        ExtensionError::TgtExceedsSrcExtensions {
                            from: *src,
                            from_offset: Port::new(Direction::Outgoing, 0),
                            from_extensions: src_rs,
                            to: *tgt,
                            to_offset: Port::new(Direction::Incoming, 0),
                            to_extensions: tgt_rs,
                        },
                    ))
                } else {
                    Some(InferExtensionError::EdgeMismatch(
                        ExtensionError::SrcExceedsTgtExtensions {
                            from: *src,
                            from_offset: Port::new(Direction::Outgoing, 0),
                            from_extensions: src_rs,
                            to: *tgt,
                            to_offset: Port::new(Direction::Incoming, 0),
                            to_extensions: tgt_rs,
                        },
                    ))
                }
            } else {
                None
            }
        } else {
            None
        };
        err.unwrap_or(InferExtensionError::MismatchedConcrete {
            expected: rs1,
            actual: rs2,
        })
    }

    /// Take a group of equal metas and merge them into a new, single meta.
    ///
    /// Returns the set of new metas created and the set of metas that were
    /// merged.
    fn coalesce(&mut self) -> Result<(HashSet<Meta>, HashSet<Meta>), InferExtensionError> {
        let mut merged: HashSet<Meta> = HashSet::new();
        let mut new_metas: HashSet<Meta> = HashSet::new();
        for cc in self.eq_graph.ccs().into_iter() {
            // Within a connected component everything is equal
            let combined_meta = self.fresh_meta();
            println!("Made: {:?}", combined_meta);
            for m in cc.iter() {
                if self.shunted.contains_key(m) {
                    continue;
                }

                if let Some(cs) = self.constraints.get(m).cloned() {
                    for c in cs
                        .into_iter()
                        .filter(|c| !matches!(c, Constraint::Equal(_)))
                    {
                        self.add_constraint(combined_meta, c.clone());
                    }
                    self.constraints.remove(m).unwrap();
                    merged.insert(*m);
                    // Record a new meta the first time that we use it; don't
                    // bother recording a new meta if we don't add any
                    // constraints. It should be safe to call this multiple times
                    new_metas.insert(combined_meta);
                    println!("Coalesce Merged {:?}", m);
                }
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
                }
                self.shunted.insert(*m, combined_meta);
                println!("Shunting {:?} -> {:?}", m, combined_meta);
            }
            // This doesn't do anything, but if it did it would just be
            // implementing the same mechanism as provided by `resolve`
            let mut updates = HashMap::<(Node, Direction), Meta>::new();
            for (loc, meta) in self.extensions.iter() {
                // If there's any node in this CC which equals `meta`, they are
                // all equal to meta
                if cc.iter().filter(|cc_meta| meta == *cc_meta).count() > 0 {
                    updates.insert(*loc, combined_meta);
                }
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
                    self.eq_graph.register_eq(meta, *other_meta);
                }
                Constraint::Plus(r, other_meta) => {
                    if let Some(rs) = self.get_solution(other_meta) {
                        let mut rrs = rs.clone();
                        rrs.insert(r);
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
                            None => self.add_solution(meta, rrs),
                        };
                        solved = true;
                    } else {
                        // TODO: Try and go backwards with a `Minus` constraint
                        // I.e. If we have a concrete solution for this
                        // metavariable, we can then work out what `other_meta`
                        // should be by subtracting the extension that's `Plus`d
                    }
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
    pub fn results(&mut self) -> Result<ExtensionSolution, InferExtensionError> {
        // Check that all of the metavariables associated with nodes of the
        // graph are solved
        let mut results: ExtensionSolution = HashMap::new();
        for (loc, meta) in self.extensions.iter() {
            let rs = match self.get_solution(meta) {
                Some(rs) => Ok(rs.clone()),
                None => {
                    // Cut through the riff raff
                    if self.live_var(meta).is_some() {
                        Err(InferExtensionError::Unsolved { location: *loc })
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
            return None;
        }

        // TODO: We should be doing something to ensure that these are the same check...
        if self.get_solution(m).is_none() {
            if self.get_constraints(m).is_some() {
                for c in self.get_constraints(m).unwrap().iter() {
                    match c {
                        Constraint::Plus(_, m) => return self.live_var(m),
                        _ => panic!("we shouldn't be here!"),
                    }
                }
            }
            Some(*m)
        } else {
            None
        }
    }

    /// Return the set of "live" metavariables in the context.
    /// "Live" here means a metavariable:
    ///   - Is associated to a location in the graph in `UnifyContext.extensions`
    ///   - Is still unsolved
    ///   - Isn't a variable
    fn live_metas(&self) -> HashSet<Meta> {
        self.extensions
            .values()
            .filter_map(|m| self.live_var(m))
            .filter(|m| !self.variables.contains(m))
            .collect()
    }

    /// Returns the metas that we solved
    fn solve_constraints(
        &mut self,
        vars: &HashSet<Meta>,
    ) -> Result<HashSet<Meta>, InferExtensionError> {
        let mut solved = HashSet::new();
        for m in vars.iter() {
            if self.solve_meta(*m)? {
                println!("Solved {:?}", m);
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
    pub fn main_loop(&mut self) -> Result<ExtensionSolution, InferExtensionError> {
        let mut remaining = HashSet::<Meta>::from_iter(self.constraints.keys().cloned());

        // Keep going as long as we're making progress (= merging nodes)
        loop {
            let to_delete = self.solve_constraints(&remaining)?;
            let (new, merged) = self.coalesce()?;
            let delta: HashSet<Meta> = HashSet::from_iter(to_delete.union(&merged).cloned());

            for m in delta.iter() {
                if !merged.contains(m) {
                    self.constraints.remove(m);
                }
                remaining.remove(m);
            }

            if delta.is_empty() && new.is_empty() {
                break;
            }

            new.into_iter().for_each(|m| {
                remaining.insert(m);
            });
        }
        self.results()
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::*;
    use crate::builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::ExtensionSet;
    use crate::hugr::HugrInternalsMut;
    use crate::hugr::{validate::ValidationError, Hugr, HugrView, NodeType};
    use crate::ops::{self, dataflow::IOTrait};
    use crate::type_row;
    use crate::types::{FunctionType, Type};

    use cool_asserts::assert_matches;
    use portgraph::NodeIndex;

    const BIT: Type = crate::extension::prelude::USIZE_T;

    #[test]
    // Build up a graph with some holes in its extensions, and infer them
    // See if it works!
    fn from_graph() -> Result<(), Box<dyn Error>> {
        let rs = ExtensionSet::from_iter(["A".into(), "B".into(), "C".into()]);
        let main_sig =
            FunctionType::new(type_row![BIT, BIT], type_row![BIT]).with_extension_delta(&rs);

        let op = ops::DFG {
            signature: main_sig,
        };

        let root_node = NodeType::open_extensions(op);
        let mut hugr = Hugr::new(root_node);

        let input = NodeType::open_extensions(ops::Input::new(type_row![BIT, BIT]));
        let output = NodeType::open_extensions(ops::Output::new(type_row![BIT]));

        let input = hugr.add_node_with_parent(hugr.root(), input)?;
        let output = hugr.add_node_with_parent(hugr.root(), output)?;

        assert_matches!(hugr.get_io(hugr.root()), Some(_));

        let add_a_sig = FunctionType::new(type_row![BIT], type_row![BIT])
            .with_extension_delta(&ExtensionSet::singleton(&"A".into()));

        let add_b_sig = FunctionType::new(type_row![BIT], type_row![BIT])
            .with_extension_delta(&ExtensionSet::singleton(&"B".into()));

        let add_ab_sig = FunctionType::new(type_row![BIT], type_row![BIT])
            .with_extension_delta(&ExtensionSet::from_iter(["A".into(), "B".into()]));

        let mult_c_sig = FunctionType::new(type_row![BIT, BIT], type_row![BIT])
            .with_extension_delta(&ExtensionSet::singleton(&"C".into()));

        let add_a = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::DFG {
                signature: add_a_sig,
            }),
        )?;
        let add_b = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::DFG {
                signature: add_b_sig,
            }),
        )?;
        let add_ab = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::DFG {
                signature: add_ab_sig,
            }),
        )?;
        let mult_c = hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::DFG {
                signature: mult_c_sig,
            }),
        )?;

        hugr.connect(input, 0, add_a, 0)?;
        hugr.connect(add_a, 0, add_b, 0)?;
        hugr.connect(add_b, 0, mult_c, 0)?;

        hugr.connect(input, 1, add_ab, 0)?;
        hugr.connect(add_ab, 0, mult_c, 1)?;

        hugr.connect(mult_c, 0, output, 0)?;

        hugr.infer_extensions()?;
        Ok(())
    }

    #[test]
    // Basic test that the `Plus` constraint works
    fn plus() -> Result<(), InferExtensionError> {
        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);

        let metas: Vec<Meta> = (2..8)
            .map(|i| {
                let meta = ctx.fresh_meta();
                ctx.extensions
                    .insert((NodeIndex::new(i).into(), Direction::Incoming), meta);
                meta
            })
            .collect();

        ctx.solved
            .insert(metas[2], ExtensionSet::singleton(&"A".into()));
        ctx.add_constraint(metas[1], Constraint::Equal(metas[2]));
        ctx.add_constraint(metas[0], Constraint::Plus("B".into(), metas[2]));
        ctx.add_constraint(metas[4], Constraint::Plus("C".into(), metas[0]));
        ctx.add_constraint(metas[3], Constraint::Equal(metas[4]));
        ctx.add_constraint(metas[5], Constraint::Equal(metas[0]));
        ctx.main_loop()?;

        let a = ExtensionSet::singleton(&"A".into());
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
    // This generates a solution that causes validation to fail
    // because of a missing lift node
    fn missing_lift_node() -> Result<(), Box<dyn Error>> {
        let builder = DFGBuilder::new(
            FunctionType::new(type_row![BIT], type_row![BIT])
                .with_extension_delta(&ExtensionSet::singleton(&"R".into())),
        )?;
        let [w] = builder.input_wires_arr();
        let hugr = builder.finish_hugr_with_outputs([w]);

        // Fail to catch the actual error because it's a difference between I/O
        // nodes and their parents and `report_mismatch` isn't yet smart enough
        // to handle that.
        assert_matches!(
            hugr,
            Err(BuildError::InvalidHUGR(ValidationError::CantInfer(_)))
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
    fn open() -> Result<(), InferExtensionError> {
        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);
        let m0 = ctx.fresh_meta();
        let m1 = ctx.fresh_meta();
        let m2 = ctx.fresh_meta();
        let m3 = ctx.fresh_meta();
        // Attach the metavariables to dummy nodes so that they're considered important
        ctx.extensions
            .insert((NodeIndex::new(1).into(), Direction::Incoming), m0);
        ctx.extensions
            .insert((NodeIndex::new(2).into(), Direction::Incoming), m1);
        ctx.extensions
            .insert((NodeIndex::new(3).into(), Direction::Incoming), m2);
        ctx.extensions
            .insert((NodeIndex::new(4).into(), Direction::Incoming), m3);

        ctx.add_constraint(m1, Constraint::Plus("A".into(), m0));
        ctx.add_constraint(m2, Constraint::Equal(m1));
        ctx.add_constraint(m3, Constraint::Equal(m0));
        ctx.variables.insert(m0);
        println!("Constraints before processing");
        for r in ctx.extensions.iter() {
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
    // Tests that we can succeed even when all variables don't have concrete
    // extension sets, and we have an open variable at the start of the graph.
    fn open_variables() -> Result<(), InferExtensionError> {
        let mut ctx = UnificationContext::new(&Hugr::default());
        let a = ctx.fresh_meta();
        let b = ctx.fresh_meta();
        let ab = ctx.fresh_meta();
        // Some nonsense so that the constraints register as "live"
        ctx.extensions
            .insert((NodeIndex::new(2).into(), Direction::Outgoing), a);
        ctx.extensions
            .insert((NodeIndex::new(3).into(), Direction::Outgoing), b);
        ctx.extensions
            .insert((NodeIndex::new(4).into(), Direction::Incoming), ab);
        ctx.variables.insert(a);
        ctx.variables.insert(b);
        ctx.add_constraint(ab, Constraint::Plus("A".into(), b));
        ctx.add_constraint(ab, Constraint::Plus("B".into(), a));
        ctx.main_loop()?;
        Ok(())
    }

    #[test]
    // Infer an the extensions on a child node with no inputs
    fn dangling_src() -> Result<(), Box<dyn Error>> {
        let rs = ExtensionSet::singleton(&"R".into());
        let root_signature =
            FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(&rs);
        let mut builder = DFGBuilder::new(root_signature)?;
        let [input_wire] = builder.input_wires_arr();

        let add_r_sig = FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(&rs);

        let add_r = builder.add_dataflow_node(
            NodeType::open_extensions(ops::DFG {
                signature: add_r_sig,
            }),
            [input_wire],
        )?;
        let [wl] = add_r.outputs_arr();

        // Dangling thingy
        let src_sig = FunctionType::new(type_row![], type_row![BIT])
            .with_extension_delta(&ExtensionSet::new());
        let src = builder.add_dataflow_node(
            NodeType::open_extensions(ops::DFG { signature: src_sig }),
            [],
        )?;
        let [wr] = src.outputs_arr();

        let mult_sig = FunctionType::new(type_row![BIT, BIT], type_row![BIT])
            .with_extension_delta(&ExtensionSet::new());
        // Mult has open extension requirements, which we should solve to be "R"
        let mult = builder.add_dataflow_node(
            NodeType::open_extensions(ops::DFG {
                signature: mult_sig,
            }),
            [wl, wr],
        )?;
        let [w] = mult.outputs_arr();

        builder.finish_hugr_with_outputs([w])?;
        Ok(())
    }
}

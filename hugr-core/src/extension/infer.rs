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
use crate::{hugr::views::HugrView, types::EdgeKind, Direction, Node};

use super::validate::ExtensionError;

use itertools::Itertools;
use petgraph::{graph as pg, visit::EdgeRef, Directed, EdgeDirection};

use std::collections::{hash_map::Entry, HashMap, HashSet};

use thiserror::Error;

/// A mapping from nodes on the hugr to extension requirement sets which have
/// been inferred for their inputs.
pub type ExtensionSolution = HashMap<Node, ExtensionSet>;

/// Infer extensions for a hugr. This is the main API exposed by this module.
///
/// Return all the solutions found for locations on the graph, these can be
/// passed to [`validate_with_extension_closure`]
///
/// [`validate_with_extension_closure`]: crate::Hugr::validate_with_extension_closure
pub fn infer_extensions(hugr: &impl HugrView) -> Result<ExtensionSolution, InferExtensionError> {
    let mut ctx = UnificationContext::default();
    ctx.gen_constraints(hugr);
    ctx.merge_equal_metas()?;
    let all_results = ctx.solve_all()?;
    let new_results = all_results
        .into_iter()
        .filter(|(n, _sol)| hugr.get_nodetype(*n).input_extensions().is_none())
        .collect();
    Ok(new_results)
}

/// Metavariables don't need much
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Meta(u32);

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
    /// An extension mismatch between two nodes which are connected by an edge.
    /// This should mirror (or reuse) `ValidationError`'s SrcExceedsTgtExtensions
    /// and TgtExceedsSrcExtensions
    #[error("Edge mismatch: {0}")]
    EdgeMismatch(#[from] ExtensionError),
}

/// A graph of metavariables connected by constraints.
/// The edges represent `Equal` constraints in the undirected graph and `Plus`
/// constraints in the directed case.
#[derive(Default)]
struct DirectedGraph {
    graph: pg::Graph<Meta, (), Directed>,
    node_map: HashMap<Meta, pg::NodeIndex>,
}

impl DirectedGraph {
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
    /// metavariables.
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

    fn take_successors(&mut self, m: Meta) -> impl Iterator<Item = Meta> + '_ {
        // If not in nodemap, return empty? Note neighbours_directed returns empty for unknown index.
        let index = *self.node_map.get(&m).unwrap();
        let mut out_edges = Vec::new();
        for e in self.graph.edges_directed(index, EdgeDirection::Outgoing) {
            out_edges.push((*self.graph.node_weight(e.target()).unwrap(), e.id()));
        }
        for (_, e_id) in out_edges.iter() {
            self.graph.remove_edge(*e_id);
        }
        out_edges.into_iter().map(|(tgt, _)| tgt)
    }
}

/// Our current knowledge about the extensions of the graph
#[derive(Default)]
struct UnificationContext {
    /// A graph where each Meta is a node, with edges to other metas it must include
    constraints: DirectedGraph,
    /// A map which says which nodes correspond to which metavariables.
    /// TODO ideally this would be bijective, or something...
    /// (But, some Meta's may be Deltas or Cycles instead of node inputs/outputs.)
    extensions: HashMap<(Node, Direction), Meta>,
    /// Solutions to metavariables, including those fixed by the Hugr
    /// (Node deltas, and user-provided annotations)
    solved: HashMap<Meta, ExtensionSet>,
    /// A mapping from metavariables which (were in cycles and hence) have been merged,
    /// to the meta (representing the whole cycle) they've been merged into
    merged_cycles: HashMap<Meta, Meta>,
    /// A name for the next metavariable we create.
    /// TODO: use constraints.num_nodes()
    fresh_name: u32,
}

/// Invariant: Constraint::Plus always points to a fresh metavariable
impl UnificationContext {
    /// Create a fresh metavariable, and increment `fresh_name` for next time
    /// TODO can we just let the graph assign indices?
    fn fresh_meta(&mut self) -> Meta {
        let fresh = Meta(self.fresh_name);
        self.fresh_name += 1;
        self.constraints.add_or_retrieve(fresh);
        fresh
    }

    /// Declare that `m1` must include `m2`
    fn must_include(&mut self, m1: Meta, m2: Meta) {
        self.constraints.add_edge(m1, m2)
    }

    /// Declare that a meta has been solved
    fn add_solution(&mut self, m: Meta, rs: ExtensionSet) {
        let existing_sol = self.solved.insert(m, rs);
        debug_assert!(existing_sol.is_none());
    }

    /// If a metavariable has been merged, return the new meta, otherwise return
    /// the same meta.
    fn resolve(&self, m: Meta) -> Meta {
        match self.merged_cycles.get(&m) {
            None => m,
            Some(tgt) => {
                // Cycles are all merged in one pass as SCCs, there are no cycles of cycles
                debug_assert!(!self.merged_cycles.contains_key(tgt));
                *tgt
            }
        }
    }

    /// Return the metavariable corresponding to the given location on the
    /// graph, either by making a new meta, or looking it up
    fn make_or_get_meta(&mut self, node: Node, dir: Direction) -> Meta {
        // TODO can't call fresh_meta while holding an Entry, but maybe can when fresh_meta defers to graph?
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
        for node in hugr.nodes() {
            let m_input = self.make_or_get_meta(node, Direction::Incoming);
            let m_output = self.make_or_get_meta(node, Direction::Outgoing);

            let node_type = hugr.get_nodetype(node);
            let m_delta = self.fresh_meta();
            // For e.g. FuncDefn, which has no op_signature, this will compute empty delta,
            // regardless of the function body. This is correct: FuncDefn merely *defines* the
            // function, the extensions are what's required to *execute* it.
            // TODO memoize this as a map from ExtensionSet to Meta.
            self.add_solution(
                m_delta,
                node_type
                    .op_signature()
                    .map_or_else(ExtensionSet::new, |ft| ft.extension_reqs.clone()),
            );

            self.must_include(m_output, m_input);
            self.must_include(m_output, m_delta);

            if let Some([_, output]) = hugr.get_io(node) {
                // Parent node. Constrain the *Output* child to be less than the Delta.
                // (This leaves the input free, but assuming it's connected to the output,
                //  will also be less than the delta; and we prefer minimal solutions).
                let m_output = self.make_or_get_meta(output, Direction::Outgoing);
                self.must_include(m_delta, m_output);
            }

            // For Conditional/Case, and CFG/BB, validation checks that the delta of the parent
            // contains that of every child, so nothing to do here.

            if let Some(annotation) = node_type.input_extensions() {
                self.add_solution(m_input, annotation.clone());
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
                    self.must_include(m_tgt, *m_src);
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

                return InferExtensionError::EdgeMismatch(if src_rs.is_subset(&tgt_rs) {
                    ExtensionError::TgtExceedsSrcExtensions {
                        from: *src,
                        from_extensions: src_rs,
                        to: *tgt,
                        to_extensions: tgt_rs,
                    }
                } else {
                    ExtensionError::SrcExceedsTgtExtensions {
                        from: *src,
                        from_extensions: src_rs,
                        to: *tgt,
                        to_extensions: tgt_rs,
                    }
                });
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
    fn merge_equal_metas(&mut self) -> Result<(), InferExtensionError> {
        for scc in self.constraints.sccs() {
            let combined_meta = self.fresh_meta();
            let solutions = scc.iter().flat_map(|m| self.solved.get(m).map(|s| (m, s)));
            match solutions.at_most_one() {
                Err(e) => {
                    // TODO The "actual" and "expected" labels here are totally bogus
                    let [(m1, rs1), (m2, rs2)] = e.take(2).collect::<Vec<_>>().try_into().unwrap();
                    return Err(self.report_mismatch(*m1, *m2, rs1.clone(), rs2.clone()));
                }
                Ok(Some((_, s))) => self.solved.insert(combined_meta, s.clone()),
                Ok(None) => None,
            };
            let metas = scc.into_iter().collect::<HashSet<_>>();
            for m in metas.iter() {
                self.merged_cycles.insert(*m, combined_meta);
                for s in self.constraints.take_successors(*m).collect::<Vec<_>>() {
                    if !metas.contains(&s) {
                        self.must_include(combined_meta, s);
                    }
                }
                //self.constraints.remove(m); // no need
            }
        }
        Ok(())
    }

    fn solve(&mut self, m_in: Meta) -> Result<&ExtensionSet, InferExtensionError> {
        let m = self.resolve(m_in);
        let mut min_sol = ExtensionSet::new();
        for s in self.constraints.take_successors(m).collect::<Vec<_>>() {
            min_sol = min_sol.union(self.solve(s)?.clone());
        }
        // The first time we come here, the below will check the computed solution above is
        // less than any prior solution (and then return it). To avoid the above computation
        // on every call...
        Ok(match self.solved.entry(m) {
            Entry::Vacant(ve) => ve.insert(min_sol),
            Entry::Occupied(oc) => {
                if !oc.get().is_superset(&min_sol) {
                    // TODO include location of `m`
                    return Err(InferExtensionError::MismatchedConcrete {
                        expected: min_sol,
                        actual: oc.get().clone(),
                    });
                }
                oc.into_mut()
            }
        })
    }

    fn solve_all(&mut self) -> Result<ExtensionSolution, InferExtensionError> {
        let required_metas = self
            .extensions
            .iter()
            .filter_map(|((n, d), m)| (d == &Direction::Incoming).then_some((*m, *n)))
            .collect::<HashMap<_, _>>();
        let mut all_solns = HashMap::new();
        for (m, n) in required_metas.iter() {
            all_solns.insert(*n, self.solve(*m)?.clone());
        }
        Ok(all_solns)
    }
}

#[cfg(test)]
mod test;

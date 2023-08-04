use super::{ResourceId, ResourceSet};
use crate::{
    hugr::{HugrView, Node},
    Direction,
};

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Metavariables don't need much
type Meta = usize;

#[derive(Clone, Debug)]
/// Things we know about metavariables
pub enum Constraint {
    /// Constrain a variable to a specific value
    Exactly(ResourceSet),
    /// A variable has the same value as another variable
    Equal(Meta),
    /// Variable extends the value of another by one resource
    Plus(ResourceId, Meta),
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
    #[error("Unsolved variable at location {:?}", location)]
    Unsolved {
        location: (Node, Direction),
        //constraints: Vec<Constraint>,
    },
}

#[derive(Clone)]
struct Deletion {
    src: Meta,
    tgt: Meta,
}

impl Deletion {
    #[inline]
    pub fn shunt(src: Meta, tgt: Meta) -> Self {
        Self { src, tgt }
    }
}

/// Our current knowledge about the resources of the graph
pub struct UnificationContext {
    /// A list of constraints for each metavariable
    pub constraints: HashMap<Meta, Vec<Constraint>>,
    /// A map which says which nodes correspond to which metavariables
    pub resources: HashMap<(Node, Direction), Meta>,
    /// Solutions to metavariables
    solved: HashMap<Meta, ResourceSet>,
}

/// Invariant: Constraint::Plus always points to a fresh metavariable
impl UnificationContext {
    pub fn new(hugr: &impl HugrView) -> Self {
        let mut ctx = Self {
            constraints: HashMap::new(),
            resources: HashMap::new(),
            solved: HashMap::new(),
        };
        ctx.gen_constraints(hugr);
        ctx
    }

    fn fresh_meta(&mut self) -> Meta {
        let fresh = self.constraints.len();
        self.constraints.insert(fresh, Vec::new());
        fresh
    }

    fn add_constraint(&mut self, m: Meta, c: Constraint) {
        self.constraints
            .entry(m)
            .and_modify(|cs| cs.push(c.clone()))
            .or_insert(vec![c]);
    }

    fn add_solution(&mut self, m: Meta, rs: ResourceSet) {
        assert!(self.solved.insert(m, rs).is_none());
    }

    fn gen_constraints(&mut self, hugr: &impl HugrView) {
        for node in hugr.nodes() {
            let input = self.fresh_meta();
            assert!(self
                .resources
                .insert((node, Direction::Incoming), input)
                .is_none());
            let output = self.fresh_meta();
            assert!(self
                .resources
                .insert((node, Direction::Outgoing), output)
                .is_none());

            let node_type = hugr.get_nodetype(node);
            match node_type.signature() {
                // Input resources are open
                None => {
                    let mut last_meta = input;
                    for r in node_type.op_signature().resource_reqs.iter() {
                        let curr_meta = self.fresh_meta();
                        self.add_constraint(curr_meta, Constraint::Plus(r.clone(), last_meta));
                        last_meta = curr_meta;
                    }
                    self.add_constraint(output, Constraint::Equal(last_meta));
                }
                // We're in the money!
                Some(sig) => {
                    self.add_solution(input, sig.input_resources.clone());
                    self.add_solution(output, sig.output_resources());
                }
            }
        }
    }

    // Coalesce
    fn process_deletions(&mut self, deletions: Vec<Deletion>) {
        fn sanity_check(cs: &[Constraint]) -> bool {
            cs.iter()
                .filter(|c| std::matches!(c, Constraint::Equal(_)))
                .count()
                == 1
        }

        let mut srcs = Vec::new();
        let mut tgts = Vec::new();
        deletions.iter().for_each(|Deletion { src, tgt }| {
            srcs.push(src);
            tgts.push(tgt);
        });
        assert!(srcs.len() == HashSet::<&usize>::from_iter(srcs.into_iter()).len());
        assert!(tgts.len() == HashSet::<&usize>::from_iter(tgts.into_iter()).len());

        for Deletion { src, tgt } in deletions.iter() {
            match self.constraints.get(src) {
                // She's already deleted!
                None => (),
                Some(cs) => {
                    assert!(sanity_check(cs));
                    let mut src_constraints: Vec<Constraint> = cs
                        .iter()
                        .cloned()
                        .filter(|c| !matches!(c, Constraint::Equal(_)))
                        .collect();
                    self.constraints
                        .entry(*tgt)
                        .and_modify(|cs| cs.append(&mut src_constraints))
                        .or_insert(src_constraints);
                    self.constraints.remove(src);
                }
            }
        }
    }

    /// Process the constraints of a given metavariable
    fn solve_meta(&mut self, meta: Meta) -> Result<bool, InferResourceError> {
        let mut deleted: Vec<Deletion> = Vec::new();
        let mut unfinished_business = false;
        for c in self.constraints.get(&meta).unwrap().clone().iter() {
            match c {
                Constraint::Exactly(rs2) => {
                    match self.solved.get(&meta) {
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
                }
                Constraint::Equal(other_meta) => {
                    match (self.solved.get(&meta), self.solved.get(other_meta)) {
                        (None, None) => {
                            unfinished_business = true;
                        }
                        (None, Some(rs)) => {
                            self.add_solution(meta, rs.clone());
                            deleted.push(Deletion::shunt(meta, *other_meta));
                        }
                        (Some(rs), None) => {
                            self.add_solution(*other_meta, rs.clone());
                            deleted.push(Deletion::shunt(*other_meta, meta));
                        }
                        (Some(rs1), Some(rs2)) => {
                            if rs1 != rs2 {
                                return Err(InferResourceError::MismatchedConcrete {
                                    expected: rs1.clone(),
                                    actual: rs2.clone(),
                                });
                            }
                            deleted.push(Deletion::shunt(meta, *other_meta));
                        }
                    };
                }
                Constraint::Plus(r, other_meta) => {
                    match self.solved.get(other_meta) {
                        Some(rs) => {
                            let mut rrs = rs.clone();
                            rrs.insert(r);
                            match self.solved.get(&meta) {
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
                            }
                        }
                        // Should we do something here?
                        None => {
                            unfinished_business = true;
                        }
                    }
                }
            }
        }
        self.process_deletions(deleted);
        Ok(unfinished_business)
    }

    /// Once the unification context is set up, attempt to infer resources for all of the nodes
    // TODO: This should not be the main API
    pub fn solve_constraints(&mut self,
    ) -> Result<HashMap<(Node, Direction), ResourceSet>, InferResourceError> {
        let mut remaining: Vec<Meta> = self.constraints.keys().clone().cloned().collect();
        let mut prev_len = remaining.len() + 1;
        while prev_len > remaining.len() {
            let mut new_remaining: Vec<Meta> = Vec::new();
            for m in remaining.iter() {
                let unfinished = self.solve_meta(*m)?;
                if unfinished {
                    new_remaining.push(*m);
                }
            }
            // Set up next step
            prev_len = remaining.len();
            remaining = new_remaining;
        }

        let mut results: HashMap<(Node, Direction), ResourceSet> = HashMap::new();
        for (loc, meta) in self.resources.iter() {
            let rs = match self.solved.get(meta) {
                Some(rs) => Ok(rs.clone()),
                None => Err(InferResourceError::Unsolved { location: *loc }),
            }?;
            results.insert(*loc, rs);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::*;
    use crate::builder::{BuildError, Container, DataflowHugr, DataflowSubContainer, DFGBuilder, ModuleBuilder, Dataflow};
    use crate::hugr::{Hugr, HugrMut, HugrView, NodeType, validate::ValidationError};
    use crate::ops::{self, dataflow::IOTrait};
    use crate::resource::ResourceSet;
    use crate::types::{AbstractSignature, ClassicType, SimpleType};
    use crate::type_row;

    pub(super) const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());

    #[test]
    fn plus() -> Result<(), InferResourceError> {
        let hugr = Hugr::default();
        let mut ctx = UnificationContext::new(&hugr);

        let m0 = ctx.fresh_meta();
        let m1 = ctx.fresh_meta();
        let m2 = ctx.fresh_meta();
        let m3 = ctx.fresh_meta();
        let m4 = ctx.fresh_meta();
        let m5 = ctx.fresh_meta();
        ctx.add_constraint(m2, Constraint::Exactly(ResourceSet::singleton(&"A".into())));
        ctx.add_constraint(m1, Constraint::Equal(m2));
        ctx.add_constraint(m0, Constraint::Plus("B".into(), m2));
        ctx.add_constraint(m4, Constraint::Plus("C".into(), m0));
        ctx.add_constraint(m3, Constraint::Equal(m4));
        ctx.add_constraint(m5, Constraint::Equal(m0));
        ctx.solve_constraints()?;

        let a = ResourceSet::singleton(&"A".into());
        let mut ab = a.clone();
        ab.insert(&"B".into());
        let mut abc = ab.clone();
        abc.insert(&"C".into());

        assert_eq!(ctx.solved.get(&m0).unwrap(), &ab);
        assert_eq!(ctx.solved.get(&m1).unwrap(), &a);
        assert_eq!(ctx.solved.get(&m2).unwrap(), &a);
        assert_eq!(ctx.solved.get(&m3).unwrap(), &abc);
        assert_eq!(ctx.solved.get(&m4).unwrap(), &abc);
        assert_eq!(ctx.solved.get(&m5).unwrap(), &ab);

        Ok(())
    }
}

use crate::{Direction, hugr::{Node, HugrView}, ops::OpTrait, Hugr};
use super::{ResourceId, ResourceSet};

use std::collections::HashMap;
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
    Plus(ResourceId, Meta)
}

#[derive(Debug, Error)]
/// Errors which arise during unification
pub enum InferResourceError {
    #[error("Mismatched resource sets {expected} and {actual}")]
    MismatchedConcrete {
        //loc: (Node, Direction),
        expected: ResourceSet,
        actual: ResourceSet
    },
    #[error("It's bad.")]
    Bad,
    #[error("Unsolved variable at location {:?}", location)]
    Unsolved {
        location: (Node, Direction),
        //constraints: Vec<Constraint>,
    }
}

struct UnificationContext<'a> {
    hugr: &'a Hugr,
    pub constraints: HashMap<Meta, Vec<Constraint>>,
    pub resources: HashMap<(Node, Direction), Meta>,
    solved: HashMap<Meta, ResourceSet>,
}

impl<'a> UnificationContext<'a> {
    fn new(hugr: &'a Hugr) -> Self {
        Self {
            hugr,
            constraints: HashMap::new(),
            resources: HashMap::new(),
            solved: HashMap::new(),
        }
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

    fn gen_constraints(&mut self, hugr: impl HugrView) {
        let meta = self.fresh_meta();
        for node in hugr.nodes() {
            let input = self.fresh_meta();
            assert!(self.resources.insert((node, Direction::Incoming), input).is_none());
            let output = self.fresh_meta();
            assert!(self.resources.insert((node, Direction::Outgoing), output).is_none());

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
                },
                // We're in the money!
                Some(sig) => {
                    self.add_solution(input, sig.input_resources.clone());
                    self.add_solution(output, sig.output_resources());
                }
            }
        }
    }

    fn solve_meta(&mut self, meta: &Meta) -> Result<(), InferResourceError> {
        match self.solved.get(meta).cloned() {
            // We know nothing
            None => unimplemented!(),
            Some(rs) => {
                for (curr, c) in self.constraints.get(meta).unwrap().iter().enumerate() {
                    match c {
                        Constraint::Exactly(rs2) => {
                            if rs == *rs2 {
                                self.constraints.get(meta).unwrap().remove(curr);
                            } else {
                                return Err(InferResourceError::MismatchedConcrete { expected: *rs2, actual: rs });
                            }
                        },
                        Constraint::Equal(other_meta) => {
                            match self.solved.get(other_meta) {
                                Some(rs) => {
                                    self.add_solution(*meta, rs.clone());
                                    self.constraints.get(meta).map(|remaining_constraints| {
                                        // Add remaining constraints to the other meta
                                        self.constraints
                                            .entry(*other_meta)
                                            .and_modify(|cs| cs.append(remaining_constraints.as_mut()));
                                        // and delete this one
                                        self.add_constraint(*meta, Constraint::Equal(*other_meta));
                                    });
                                },
                                // The trickier case
                                None => {
                                    todo!()
                                }
                            }
                        },
                        Constraint::Plus(r, other_meta) => {
                            match self.solved.get(other_meta) {
                                Some(rs) => {
                                    let rrs = rs.clone();
                                    rrs.insert(r);
                                    self.add_solution(*meta, rrs);
                                    self.constraints.get(meta).unwrap().remove(curr);
                                },
                                None => todo!(),
                            }
                        },
                    }
                }
                Ok(())
            }
        }
    }

    fn solve(&mut self) -> Result<HashMap<(Node, Direction), ResourceSet>, InferResourceError> {
        for m in self.constraints.keys().cloned() {
            self.solve_meta(&(m.clone()))?;
        }
        let mut results: HashMap<(Node, Direction), ResourceSet> = HashMap::new();
        for (loc, meta) in self.resources.iter() {
            let rs = match self.solved.get(&meta) {
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
    use super::*;
    use crate::types::{AbstractSignature, ClassicType, SimpleType};

    pub(super) const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());

/*
    #[test]
    fn from_graph() {
        let sig = Signature::new_df([BIT], [BIT]);
        let mut parent = DFGBuilder::new(sig);
    }

    #[test]
    fn coalesce() {
        let hugr: Hugr = Default::default();
        let mut ctx = UnificationContext::new(&hugr);
        ctx.constraints;
    }
*/

    #[test]
    fn plus() -> Result<(), InferResourceError> {
        let hugr: Hugr = Default::default();
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

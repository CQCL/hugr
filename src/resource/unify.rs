use crate::{Direction, hugr::Node, Hugr};
use super::{ResourceId, ResourceSet};

use std::collections::HashMap;
use thiserror::Error;

type Meta = usize;

enum Constraint {
    Exactly(ResourceSet),
    Equal(Meta),
    Plus(ResourceId, Meta)
}

#[derive(Debug, Error)]
enum InferResourceError {
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
        fresh
    }

    fn add_constraint(&mut self, m: Meta, c: Constraint) {
        self.constraints
            .entry(m)
            .and_modify(|&mut cs| cs.push(c))
            .or_insert(vec![c]);
    }

    fn gen_constraints(&mut self, hugr: &Hugr) {
        let meta = self.fresh_meta();
        for node in self.hugr.graph.nodes_iter().map_into() {
            let input = self.fresh_meta();
            assert!(self.resources.insert((node, Direction::Incoming), input).is_none());
            let output = self.fresh_meta();
            assert!(self.resources.insert((node, Direction::Outgoing), output).is_none());

            let added_resources = hugr.op_types.get(node.index).unwrap().resource_reqs;
            self.add_constraint(output, Constraint::Plus(add_resource, input));
        }

        for (node, rs) in input_resources.iter() {
            let meta = self.resources.get(node, Direction::Incoming).unwrap();
            self.add_constraint(meta, Exactly(rs));
        }
    }

    fn solve_meta(&mut self, meta: &Meta) -> Result<(), InferResourceError> {
        match self.solved.get(meta) {
            // We know nothing
            None => unimplemented!(),
            Some(rs) => {
                let mut curr = 0;
                for c in self.constraints.get(meta).iter() {
                    match c {
                        Constraint::Exactly(rs2) => {
                            if rs == rs2 {
                                self.constraints.get(meta).remove(curr);
                            } else {
                                return Err(MismatchedConcrete { expected: rs2, actual: rs });
                            }
                        },
                        Constraint::Equal(other_meta) => {
                            match self.solved.get(other_meta) {
                                Some(rs) => {
                                    assert!(!self.solved.insert(meta, rs).is_none());
                                    let remaining_constraints = self.constraints.get(meta);
                                    // Add remaining constraints to the other meta
                                    self.constraints
                                        .entry(other_meta)
                                        .map(|cs| cs.append(remaining_constraints));
                                    // and delete this one
                                    self.constraints.insert(meta, Equal(other_meta));
                                },
                                None => {

                                }
                            }
                        },
                        Constraint::Plus(r, other_meta) => {
                            match self.solved.get(other_meta) {
                                Some(rs) => {
                                    assert!(!self.solved.insert(meta, rs.insert(r)).is_none());
                                    self.constraints.get(meta).remove(curr)
                                },
                                None => todo!(),
                            }
                        },
                    }
                    curr += 1;
                }
            }
        }
    }

    fn solve(&mut self) -> Result<HashMap<(Node, Direction), ResourceSet>, InferResourceError> {
        for m in self.constraints.iter() {
            self.solve_meta(m)?;
        }
        let mut results = HashMap::new();
        for (loc, meta) in self.resources.iter() {
            let rs = match self.solutions.get(meta) {
                Some(rs) => Ok(rs),
                None => Err(InferResourceError::Unsolved(loc)),
            }?;
            results.insert(loc, rs);
        }
        results
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{ClassicType, Signature, SimpleType};

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
        let m1 = ctx.fresh_meta();
        let m2 = ctx.fresh_meta();
        let m3 = ctx.fresh_meta();
        let m4 = ctx.fresh_meta();
        let m5 = ctx.fresh_meta();
        let m6 = ctx.fresh_meta();
        ctx.add_constraint(m3, Constraint::Exactly(ResourceSet::singleton(&"A".into())));
        ctx.add_constraint(m2, Constraint::Equal(m3));
        ctx.add_constraint(m1, Constraint::Plus("B".into(), m3));
        ctx.add_constraint(m5, Constraint::Plus("C".into(), m1));
        ctx.add_constraint(m4, Constraint::Equal(m5));
        ctx.add_constraint(m6, Constraint::Equal(m1));
        ctx.solve()?;

        let a = ResourceSet::singleton(&"A".into());
        let mut ab = a.clone();
        ab.insert(&"B".into());
        let mut abc = ab.clone();
        abc.insert(&"C".into());

        assert_eq!(ctx.solved.get(&m1).unwrap(), &ab);
        assert_eq!(ctx.solved.get(&m2).unwrap(), &a);
        assert_eq!(ctx.solved.get(&m3).unwrap(), &a);
        assert_eq!(ctx.solved.get(&m4).unwrap(), &abc);
        assert_eq!(ctx.solved.get(&m5).unwrap(), &abc);
        assert_eq!(ctx.solved.get(&m6).unwrap(), &ab);

        Ok(())
    }
}

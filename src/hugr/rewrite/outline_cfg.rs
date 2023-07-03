//! Rewrite for inserting a CFG-node into the hierarchy containing a subsection of an existing CFG
use std::collections::HashSet;

use itertools::Itertools;
use thiserror::Error;

use crate::builder::{CFGBuilder, Container, Dataflow, SubContainer};
use crate::hugr::rewrite::Rewrite;
use crate::hugr::{HugrMut, HugrView};
use crate::ops::handle::NodeHandle;
use crate::ops::{BasicBlock, ConstValue, OpType};
use crate::{type_row, Hugr, Node};

/// Moves part of a Control-flow Sibling Graph into a new CFG-node
/// that is the only child of a new Basic Block in the original CSG.
pub struct OutlineCfg(HashSet<Node>);

impl OutlineCfg {
    /// Create a new OutlineCfg rewrite that will move the provided blocks.
    pub fn new(blocks: impl IntoIterator<Item = Node>) -> Self {
        Self(HashSet::from_iter(blocks))
    }

    fn compute_entry_exit_outside(&self, h: &Hugr) -> Result<(Node, Node, Node), OutlineCfgError> {
        let cfg_n = match self
            .0
            .iter()
            .map(|n| h.get_parent(*n))
            .unique()
            .exactly_one()
        {
            Ok(Some(n)) => n,
            _ => return Err(OutlineCfgError::NotSiblings),
        };
        let o = h.get_optype(cfg_n);
        if !matches!(o, OpType::CFG(_)) {
            return Err(OutlineCfgError::ParentNotCfg(cfg_n, o.clone()));
        };
        let mut entry = None;
        let mut exit_succ = None;
        for &n in self.0.iter() {
            if h.input_neighbours(n).any(|pred| !self.0.contains(&pred)) {
                match entry {
                    None => {
                        entry = Some(n);
                    }
                    Some(prev) => {
                        return Err(OutlineCfgError::MultipleEntryNodes(prev, n));
                    }
                }
            }
            let external = h.output_neighbours(n).filter(|s| !self.0.contains(s));
            match external.at_most_one() {
                Ok(None) => (), // No external successors
                Ok(Some(o)) => match exit_succ {
                    None => {
                        exit_succ = Some((n, o));
                    }
                    Some((prev, _)) => {
                        return Err(OutlineCfgError::MultipleExitNodes(prev, n));
                    }
                },
                Err(ext) => return Err(OutlineCfgError::MultipleExitEdges(n, ext.collect())),
            };
        }
        match (entry, exit_succ) {
            (Some(e), Some((x, o))) => Ok((e, x, o)),
            (None, _) => Err(OutlineCfgError::NoEntryNode),
            (_, None) => Err(OutlineCfgError::NoExitNode),
        }
    }
}

impl Rewrite for OutlineCfg {
    type Error = OutlineCfgError;
    const UNCHANGED_ON_FAILURE: bool = true;
    fn verify(&self, h: &Hugr) -> Result<(), OutlineCfgError> {
        self.compute_entry_exit_outside(h)?;
        Ok(())
    }
    fn apply(self, h: &mut Hugr) -> Result<(), OutlineCfgError> {
        let (entry, exit, outside) = self.compute_entry_exit_outside(h)?;
        // 1. Compute signature
        // These panic()s only happen if the Hugr would not have passed validate()
        let OpType::BasicBlock(BasicBlock::DFB {inputs, ..}) = h.get_optype(entry) else {panic!("Entry node is not a basic block")};
        let inputs = inputs.clone();
        let outputs = {
            let OpType::BasicBlock(s_type) = h.get_optype(exit) else {panic!()};
            match s_type {
                BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                BasicBlock::DFB { inputs, .. } => inputs,
            }
        }
        .clone();
        let mut existing_cfg = {
            let parent = h.get_parent(entry).unwrap();
            CFGBuilder::from_existing(h, parent).unwrap()
        };

        // 2. New CFG node will be contained in new single-successor BB
        let mut new_block = existing_cfg
            .block_builder(inputs.clone(), vec![type_row![]], outputs.clone())
            .unwrap();

        // 3. new_block contains input node, sub-cfg, exit node all connected
        let wires_in = inputs.iter().cloned().zip(new_block.input_wires());
        let cfg = new_block.cfg_builder(wires_in, outputs).unwrap();
        let cfg_node = cfg.container_node();
        let inner_exit = cfg.exit_block().node();
        let cfg_outputs = cfg.finish_sub_container().unwrap().outputs();
        let predicate = new_block
            .add_constant(ConstValue::simple_predicate(0, 1))
            .unwrap();
        let pred_wire = new_block.load_const(&predicate).unwrap();
        let new_block = new_block
            .finish_with_outputs(pred_wire, cfg_outputs)
            .unwrap();

        // 4. Entry edges. Change any edges into entry_block from outside, to target new_block
        let h = existing_cfg.hugr_mut();

        let preds: Vec<_> = h
            .linked_ports(entry, h.node_inputs(entry).exactly_one().unwrap())
            .collect();
        for (pred, br) in preds {
            if !self.0.contains(&pred) {
                h.disconnect(pred, br).unwrap();
                h.connect(pred, br.index(), new_block.node(), 0).unwrap();
            }
        }

        // 5. Children of new CFG.
        // Entry node must be first
        h.hierarchy.detach(entry.index);
        h.hierarchy
            .insert_before(entry.index, inner_exit.index)
            .unwrap();
        // And remaining nodes
        for n in self.0 {
            // Do not move the entry node, as we have already
            if n != entry {
                h.hierarchy.detach(n.index);
                h.hierarchy.push_child(n.index, cfg_node.index).unwrap();
            }
        }

        // 6. Exit edges.
        // Retarget edge from exit_node (that used to target outside) to inner_exit
        let exit_port = h
            .node_outputs(exit)
            .filter(|p| {
                let (t, p2) = h.linked_ports(exit, *p).exactly_one().ok().unwrap();
                assert!(p2.index() == 0);
                t == outside
            })
            .exactly_one()
            .unwrap();
        h.disconnect(exit, exit_port).unwrap();
        h.connect(exit, exit_port.index(), inner_exit, 0).unwrap();
        // And connect new_block to outside instead
        h.connect(new_block.node(), 0, outside, 0).unwrap();

        Ok(())
    }
}

/// Errors that can occur in expressing an OutlineCfg rewrite.
#[derive(Debug, Error)]
pub enum OutlineCfgError {
    /// The set of blocks were not siblings
    #[error("The nodes did not all have the same parent")]
    NotSiblings,
    /// The parent node was not a CFG node
    #[error("The parent node {0:?} was not a CFG but an {1:?}")]
    ParentNotCfg(Node, OpType),
    /// Multiple blocks had incoming edges
    #[error("Multiple blocks had predecessors outside the set - at least {0:?} and {1:?}")]
    MultipleEntryNodes(Node, Node),
    /// Multiple blocks had outgoing edegs
    #[error("Multiple blocks had edges leaving the set - at least {0:?} and {1:?}")]
    MultipleExitNodes(Node, Node),
    /// One block had multiple outgoing edges
    #[error("Exit block {0:?} had edges to multiple external blocks {1:?}")]
    MultipleExitEdges(Node, Vec<Node>),
    /// No block was identified as an entry block
    #[error("No block had predecessors outside the set")]
    NoEntryNode,
    /// No block was identified as an exit block
    #[error("No block had a successor outside the set")]
    NoExitNode,
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::algorithm::nest_cfgs::test::build_conditional_in_loop_cfg;
    use crate::ops::handle::NodeHandle;
    use crate::{HugrView, Node};
    use cool_asserts::assert_matches;
    use itertools::Itertools;

    use super::{OutlineCfg, OutlineCfgError};

    fn depth(h: &impl HugrView, n: Node) -> u32 {
        match h.get_parent(n) {
            Some(p) => 1 + depth(h, p),
            None => 0,
        }
    }

    #[test]
    fn test_outline_cfg_errors() {
        let (mut h, head, tail) = build_conditional_in_loop_cfg(false).unwrap();
        let head = head.node();
        let tail = tail.node();
        //               /-> left --\
        //  entry -> head            > merge -> tail -> exit
        //            |  \-> right -/             |
        //             \---<---<---<---<---<--<---/
        // merge is unique predecessor of tail
        let merge = h.input_neighbours(tail).exactly_one().unwrap();
        h.validate().unwrap();
        let backup = h.clone();
        let r = h.apply_rewrite(OutlineCfg::new([merge, tail]));
        assert_matches!(r, Err(OutlineCfgError::MultipleExitEdges(_, _)));
        assert_eq!(h, backup);

        let [left, right]: [Node; 2] = h.output_neighbours(head).collect_vec().try_into().unwrap();
        let r = h.apply_rewrite(OutlineCfg::new([left, right, head]));
        assert_matches!(r, Err(OutlineCfgError::MultipleExitNodes(a,b)) => HashSet::from([a,b]) == HashSet::from_iter([left, right, head]));
        assert_eq!(h, backup);

        let r = h.apply_rewrite(OutlineCfg::new([left, right, merge]));
        assert_matches!(r, Err(OutlineCfgError::MultipleEntryNodes(a,b)) => HashSet::from([a,b]) == HashSet::from([left, right]));
        assert_eq!(h, backup);
    }

    #[test]
    fn test_outline_cfg() {
        let (mut h, head, tail) = build_conditional_in_loop_cfg(false).unwrap();
        let head = head.node();
        let tail = tail.node();
        let parent = h.get_parent(head).unwrap();
        let [entry, exit]: [Node; 2] = h.children(parent).take(2).collect_vec().try_into().unwrap();
        //               /-> left --\
        //  entry -> head            > merge -> tail -> exit
        //            |  \-> right -/             |
        //             \---<---<---<---<---<--<---/
        // merge is unique predecessor of tail
        let merge = h.input_neighbours(tail).exactly_one().unwrap();
        let [left, right]: [Node; 2] = h.output_neighbours(head).collect_vec().try_into().unwrap();
        for n in [head, tail, merge] {
            assert_eq!(depth(&h, n), 1);
        }
        h.validate().unwrap();
        let blocks = [head, left, right, merge];
        h.apply_rewrite(OutlineCfg::new(blocks)).unwrap();
        h.validate().unwrap();
        for n in blocks {
            assert_eq!(depth(&h, n), 3);
        }
        let new_block = h.output_neighbours(entry).exactly_one().unwrap();
        for n in [entry, exit, tail, new_block] {
            assert_eq!(depth(&h, n), 1);
        }
        assert_eq!(h.input_neighbours(tail).exactly_one().unwrap(), new_block);
        assert_eq!(
            h.output_neighbours(tail).take(2).collect::<HashSet<Node>>(),
            HashSet::from([exit, new_block])
        );
    }
}

use std::collections::{HashSet, VecDeque};

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
pub struct OutlineCfg {
    /// Will become the entry block in the new sub-cfg.
    /// All edges "in" to this block will be redirected to the new block
    /// (excluding backedges from internal BBs)
    entry_node: Node,
    /// Unique node in the parent CFG that has a successor (indeed, exactly one)
    /// that is not reachable from the entry_node (without going through this exit_node).
    exit_node: Node,
}

impl OutlineCfg {
    fn compute_all_blocks(&self, h: &Hugr) -> Result<(HashSet<Node>, Node), OutlineCfgError> {
        let cfg_n = match (h.get_parent(self.entry_node), h.get_parent(self.exit_node)) {
            (Some(p1), Some(p2)) if p1 == p2 => p1,
            (p1, p2) => return Err(OutlineCfgError::EntryExitNotSiblings(p1, p2)),
        };
        let o = h.get_optype(cfg_n);
        if !matches!(o, OpType::CFG(_)) {
            return Err(OutlineCfgError::ParentNotCfg(cfg_n, o.clone()));
        };
        let mut all_blocks = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.entry_node);
        while let Some(n) = queue.pop_front() {
            // This puts the exit_node into 'all_blocks' but not its successors
            if all_blocks.insert(n) && n != self.exit_node {
                queue.extend(h.output_neighbours(n));
            }
        }
        if !all_blocks.contains(&self.exit_node) {
            return Err(OutlineCfgError::ExitNodeUnreachable);
        }
        let exit_succs = h
            .output_neighbours(self.exit_node)
            .filter(|n| !all_blocks.contains(n))
            .collect::<Vec<_>>();
        match exit_succs.into_iter().at_most_one() {
            Err(e) => Err(OutlineCfgError::MultipleExitSuccessors(Box::new(e))),
            Ok(None) => Err(OutlineCfgError::NoExitSuccessors),
            Ok(Some(exit_succ)) => Ok((all_blocks, exit_succ)),
        }
    }
}

impl Rewrite for OutlineCfg {
    type Error = OutlineCfgError;
    const UNCHANGED_ON_FAILURE: bool = true;
    fn verify(&self, h: &Hugr) -> Result<(), OutlineCfgError> {
        self.compute_all_blocks(h)?;
        Ok(())
    }
    fn apply(self, h: &mut Hugr) -> Result<(), OutlineCfgError> {
        let (all_blocks, exit_succ) = self.compute_all_blocks(h)?;
        // 1. Compute signature
        // These panic()s only happen if the Hugr would not have passed validate()
        let OpType::BasicBlock(BasicBlock::DFB {inputs, ..}) = h.get_optype(self.entry_node) else {panic!("Entry node is not a basic block")};
        let inputs = inputs.clone();
        let outputs = {
            let OpType::BasicBlock(s_type) = h.get_optype(self.exit_node) else {panic!()};
            match s_type {
                BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                BasicBlock::DFB { inputs, .. } => inputs,
            }
        }
        .clone();
        let mut existing_cfg = {
            let parent = h.get_parent(self.entry_node).unwrap();
            CFGBuilder::from_existing(h, parent).unwrap()
        };

        // 2. New CFG node will be contained in new single-successor BB
        let mut new_block = existing_cfg
            .block_builder(inputs.clone(), vec![type_row![]], outputs.clone())
            .unwrap();

        // 3. new_block contains input node, sub-cfg, exit node all connected
        let wires_in = inputs.iter().cloned().zip(new_block.input_wires());
        let cfg = new_block.cfg_builder(wires_in, outputs.clone()).unwrap();
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
            .linked_ports(
                self.entry_node,
                h.node_inputs(self.entry_node).exactly_one().unwrap(),
            )
            .collect();
        for (pred, br) in preds {
            if !all_blocks.contains(&pred) {
                h.disconnect(pred, br).unwrap();
                h.connect(pred, br.index(), new_block.node(), 0).unwrap();
            }
        }

        // 5. Children of new CFG.
        // Entry node must be first
        h.hierarchy.detach(self.entry_node.index);
        h.hierarchy
            .insert_before(self.entry_node.index, inner_exit.index)
            .unwrap();
        // And remaining nodes
        for n in all_blocks {
            // Do not move the entry node, as we have already
            if n != self.entry_node {
                h.hierarchy.detach(n.index);
                h.hierarchy.push_child(n.index, cfg_node.index).unwrap();
            }
        }

        // 6. Exit edges.
        // Retarget edge from exit_node (that used to target exit_succ) to inner_exit
        let exit_port = h
            .node_outputs(self.exit_node)
            .filter(|p| {
                let (t, p2) = h
                    .linked_ports(self.exit_node, *p)
                    .exactly_one()
                    .ok()
                    .unwrap();
                assert!(p2.index() == 0);
                t == exit_succ
            })
            .exactly_one()
            .unwrap();
        h.disconnect(self.exit_node, exit_port).unwrap();
        h.connect(self.exit_node, exit_port.index(), inner_exit, 0)
            .unwrap();
        // And connect new_block to exit_succ instead
        h.connect(new_block.node(), 0, exit_succ, 0).unwrap();

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum OutlineCfgError {
    #[error("The exit node could not be reached from the entry node")]
    ExitNodeUnreachable,
    #[error("Exit node does not exit - all its successors can be reached from the entry")]
    NoExitSuccessors,
    #[error("Entry node {0:?} and exit node {1:?} are not siblings")]
    EntryExitNotSiblings(Option<Node>, Option<Node>),
    #[error("The parent node {0:?} of entry and exit was not a CFG but an {1:?}")]
    ParentNotCfg(Node, OpType),
    #[error("Exit node had multiple successors outside CFG: {0}")]
    MultipleExitSuccessors(#[source] Box<dyn std::error::Error>),
}

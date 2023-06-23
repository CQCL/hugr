use std::collections::{VecDeque, HashSet};

use thiserror::Error;

use crate::hugr::{HugrView, HugrMut};
use crate::ops::{OpType, BasicBlock, CFG, OpTrait};
use crate::{Hugr, Node, type_row};
use crate::hugr::rewrite::Rewrite;

/// Moves part of a Control-flow Sibling Graph into a new CFG-node
/// that is the only child of a new Basic Block in the original CSG.
pub struct OutlineCfg {
    /// Will become the entry block in the new sub-cfg.
    /// All edges "in" to this block will be redirected to the new block
    /// (excluding backedges from internal BBs)
    entry_node: Node,
    /// Either the exit node of the parent CFG; or, another node, with
    /// exactly one successor that is not reachable from the entry_node
    /// (without going through the exit_node).
    exit_node: Node
}

impl OutlineCfg {
    fn compute_all_blocks(&self, h: &Hugr) -> Result<(HashSet<Node>, Option<Node>), OutlineCfgError> {
        let cfg_n = match (h.get_parent(self.entry_node), h.get_parent(self.exit_node)) {
            (Some(p1), Some(p2)) if p1==p2 => p1,
            (p1, p2) => {return Err(OutlineCfgError::EntryExitNotSiblings(p1, p2))}
        };
        match h.get_optype(cfg_n) {
            OpType::CFG(_) => (),
            o => {return Err(OutlineCfgError::ParentNotCfg(cfg_n, o.clone()));}
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
        if !all_blocks.contains(&self.exit_node) { return Err(OutlineCfgError::ExitNodeUnreachable);}
        if matches!(h.get_optype(self.exit_node), OpType::BasicBlock(BasicBlock::Exit { .. })) {
            return Ok((all_blocks, None));
        };
        let mut succ = None;
        for exit_succ in h.output_neighbours(self.exit_node) {
            if all_blocks.contains(&exit_succ) {continue;}
            if let Some(s) = succ {
                return Err(OutlineCfgError::MultipleExitSuccessors(s, exit_succ));
            }
            succ = Some(exit_succ);
        }
        assert!(succ.is_some());
        Ok((all_blocks, succ))
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
        let (all_blocks, cfg_succ) = self.compute_all_blocks(h)?;
        // 1. Add new CFG node
        // These panic()s only happen if the Hugr would not have passed validate()
        let inputs = h.get_optype(h.children(self.entry_node).next().unwrap() // input node is first child
            ).signature().output;
        let cfg_outputs = {
            let OpType::BasicBlock(exit_type) = h.get_optype(self.exit_node) else {panic!()};
            match exit_type {
                BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                BasicBlock::DFB { .. } => {
                    let Some(s) = cfg_succ else {panic!();};
                    assert!(h.output_neighbours(self.exit_node).collect::<HashSet<_>>().contains(&s));
                    let OpType::BasicBlock(s_type) = h.get_optype(self.exit_node) else {panic!()};
                    match s_type {
                        BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                        BasicBlock::DFB { inputs, .. } => inputs
                    }
                }
            }
        }.clone();
        
        let sub_cfg = h.add_op(CFG {
            inputs: inputs.clone(), outputs: cfg_outputs.clone()
        });
        // 2. New CFG node will be contained in new BB
        let new_block = h.add_op(BasicBlock::DFB {
            inputs, other_outputs: cfg_outputs.clone(), predicate_variants: vec![type_row![]]
        });
        //TODO: dfg Input and Output nodes. Use builder instead?
        h.hierarchy.push_child(sub_cfg.index, new_block.index);
        // 3. Inner CFG needs an exit node
        let inner_exit = h.add_op(OpType::BasicBlock(BasicBlock::Exit { cfg_outputs }));
        h.hierarchy.push_child(inner_exit.index, sub_cfg.index);
        // 4. Reparent nodes - entry node must be first
        h.hierarchy.detach(self.entry_node.index);
        h.hierarchy.insert_before(self.entry_node.index, inner_exit.index);
        for n in all_blocks {
            // Do not move the entry node, as we have already;
            // TODO??? don't move the exit_node if it's the exit-block of the outer CFG
            if n == self.entry_node {continue};
            h.hierarchy.detach(n.index);
            h.hierarchy.insert_before(n.index, inner_exit.index);
        }
        
        // 5.a. and redirect exitting edges from exit_node to the new (inner) exit block
        // 5.b. and also from new_block to the old successors
        // 6. redirect edges to entry_block from outside, to new_block
        
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum OutlineCfgError {
    #[error("The exit node could not be reached from the entry node")]
    ExitNodeUnreachable,
    #[error("Entry node {0:?} and exit node {1:?} are not siblings")]
    EntryExitNotSiblings(Option<Node>,Option<Node>),
    #[error("The parent node {0:?} of entry and exit was not a CFG but an {1:?}")]
    ParentNotCfg(Node, OpType),
    #[error("Exit node had multiple successors outside CFG - at least {0:?} and {1:?}")]
    MultipleExitSuccessors(Node, Node)
}
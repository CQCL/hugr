use std::collections::{HashSet, VecDeque};

use itertools::Itertools;
use thiserror::Error;

use crate::builder::{Container, DFGBuilder, Dataflow, DataflowSubContainer, SubContainer};
use crate::hugr::rewrite::Rewrite;
use crate::hugr::{HugrMut, HugrView};
use crate::ops::{BasicBlock, OpTrait, OpType};
use crate::types::Signature;
use crate::{type_row, Hugr, Node};

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
    exit_node: Node,
}

impl OutlineCfg {
    fn compute_all_blocks(
        &self,
        h: &Hugr,
    ) -> Result<(HashSet<Node>, Option<Node>), OutlineCfgError> {
        let cfg_n = match (h.get_parent(self.entry_node), h.get_parent(self.exit_node)) {
            (Some(p1), Some(p2)) if p1 == p2 => p1,
            (p1, p2) => return Err(OutlineCfgError::EntryExitNotSiblings(p1, p2)),
        };
        match h.get_optype(cfg_n) {
            OpType::CFG(_) => (),
            o => {
                return Err(OutlineCfgError::ParentNotCfg(cfg_n, o.clone()));
            }
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
        if matches!(
            h.get_optype(self.exit_node),
            OpType::BasicBlock(BasicBlock::Exit { .. })
        ) {
            return Ok((all_blocks, None));
        };
        let mut succ = None;
        for exit_succ in h.output_neighbours(self.exit_node) {
            if all_blocks.contains(&exit_succ) {
                continue;
            }
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
        // Gather data.
        // These panic()s only happen if the Hugr would not have passed validate()
        let inputs = h
            .get_optype(
                h.children(self.entry_node).next().unwrap(), // input node is first child
            )
            .signature()
            .output;
        let outputs = {
            let OpType::BasicBlock(exit_type) = h.get_optype(self.exit_node) else {panic!()};
            match exit_type {
                BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                BasicBlock::DFB { .. } => {
                    let Some(s) = cfg_succ else {panic!();};
                    assert!(h
                        .output_neighbours(self.exit_node)
                        .collect::<HashSet<_>>()
                        .contains(&s));
                    let OpType::BasicBlock(s_type) = h.get_optype(self.exit_node) else {panic!()};
                    match s_type {
                        BasicBlock::Exit { cfg_outputs } => cfg_outputs,
                        BasicBlock::DFB { inputs, .. } => inputs,
                    }
                }
            }
        }
        .clone();
        let parent = h.get_parent(self.entry_node).unwrap();

        // 2. New CFG node will be contained in new single-successor BB
        let new_block = h.add_op(BasicBlock::DFB {
            inputs: inputs.clone(),
            other_outputs: outputs.clone(),
            predicate_variants: vec![type_row![]],
        });
        h.hierarchy
            .push_child(new_block.index, parent.index)
            .unwrap();
        // Change any edges into entry_block from outside, to target new_block
        let preds: Vec<_> = h
            .linked_ports(
                self.entry_node,
                h.node_inputs(self.entry_node).exactly_one().unwrap(),
            )
            .collect();
        for (pred, br) in preds {
            if !all_blocks.contains(&pred) {
                h.disconnect(pred, br).unwrap();
                h.connect(pred, br.index(), new_block, 0).unwrap();
            }
        }
        // 3. new_block contains input node, sub-cfg, exit node all connected
        let mut b = DFGBuilder::create_with_io(
            &mut *h,
            new_block,
            Signature::new_df(inputs.clone(), outputs.clone()),
        )
        .unwrap();
        let wires_in = inputs.into_iter().cloned().zip(b.input_wires());
        let cfg = b.cfg_builder(wires_in, outputs.clone()).unwrap();
        let cfg_index = cfg.container_node().index;
        let cfg_outputs = cfg.finish_sub_container().unwrap().outputs();
        b.finish_with_outputs(cfg_outputs).unwrap();

        // 4. Children of new CFG - entry node must be first
        h.hierarchy.detach(self.entry_node.index);
        h.hierarchy
            .push_child(self.entry_node.index, cfg_index)
            .unwrap();
        // Make an exit node, next
        let inner_exit = h
            .add_op_after(
                self.entry_node,
                OpType::BasicBlock(BasicBlock::Exit {
                    cfg_outputs: outputs,
                }),
            )
            .unwrap();
        // Then reparent remainder
        for n in all_blocks {
            // Do not move the entry node, as we have already;
            // Also don't move the exit_node if it's the exit-block of the outer CFG
            if n == self.entry_node || (n == self.exit_node && cfg_succ.is_none()) {
                continue;
            };
            h.hierarchy.detach(n.index);
            h.hierarchy.insert_after(n.index, inner_exit.index).unwrap();
        }
        // Connect new_block to (old) successor of exit block
        if let Some(s) = cfg_succ {
            let exit_port = h
                .node_outputs(self.exit_node)
                .filter(|p| {
                    let (t, p2) = h
                        .linked_ports(self.exit_node, *p)
                        .exactly_one()
                        .ok()
                        .unwrap();
                    assert!(p2.index() == 0);
                    t == s
                })
                .exactly_one()
                .unwrap();
            h.disconnect(self.exit_node, exit_port).unwrap();
            h.connect(self.exit_node, exit_port.index(), inner_exit, 0)
                .unwrap();
        } else {
            // We left the exit block outside. So, it's *predecessors* need to be retargetted to the inner_exit.
            let in_p = h.node_inputs(self.exit_node).exactly_one().unwrap();
            assert!(in_p.index() == 0);
            let preds = h.linked_ports(self.exit_node, in_p).collect::<Vec<_>>();
            for (src_n, src_p) in preds {
                h.disconnect(src_n, src_p).unwrap();
                h.connect(src_n, src_p.index(), inner_exit, 0).unwrap();
            }
        }
        h.connect(new_block, 0, cfg_succ.unwrap_or(self.exit_node), 0)
            .unwrap();
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum OutlineCfgError {
    #[error("The exit node could not be reached from the entry node")]
    ExitNodeUnreachable,
    #[error("Entry node {0:?} and exit node {1:?} are not siblings")]
    EntryExitNotSiblings(Option<Node>, Option<Node>),
    #[error("The parent node {0:?} of entry and exit was not a CFG but an {1:?}")]
    ParentNotCfg(Node, OpType),
    #[error("Exit node had multiple successors outside CFG - at least {0:?} and {1:?}")]
    MultipleExitSuccessors(Node, Node),
}

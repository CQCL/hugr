//! Rewrite for inserting a CFG-node into the hierarchy containing a subsection
//! of an existing CFG
use std::collections::HashSet;

use itertools::Itertools;
use thiserror::Error;

use crate::PortIndex;
use crate::builder::{BlockBuilder, Container, Dataflow, SubContainer};
use crate::hugr::{HugrMut, HugrView};
use crate::ops;
use crate::ops::controlflow::BasicBlock;
use crate::ops::handle::NodeHandle;
use crate::ops::{DataflowBlock, OpType};
use crate::{Node, type_row};

use super::{PatchHugrMut, PatchVerification};

/// Moves some of the blocks in a Control-flow region into a new CFG-node that
/// is the only child of a new Basic Block in the original region.
pub struct OutlineCfg {
    blocks: HashSet<Node>,
}

impl OutlineCfg {
    /// Create a new `OutlineCfg` rewrite that will move the provided blocks.
    pub fn new(blocks: impl IntoIterator<Item = Node>) -> Self {
        Self {
            blocks: HashSet::from_iter(blocks),
        }
    }

    /// Compute the entry and exit nodes of the CFG which contains
    /// [`self.blocks`], along with the output neighbour its parent graph.
    fn compute_entry_exit(
        &self,
        h: &impl HugrView<Node = Node>,
    ) -> Result<(Node, Node, Node), OutlineCfgError> {
        let cfg_n = match self
            .blocks
            .iter()
            .map(|n| h.get_parent(*n))
            .unique()
            .exactly_one()
        {
            Ok(Some(n)) => n,
            _ => return Err(OutlineCfgError::NotSiblings),
        };
        let o = h.get_optype(cfg_n);
        let OpType::CFG(_) = o else {
            return Err(OutlineCfgError::ParentNotCfg(cfg_n, Box::new(o.clone())));
        };
        let cfg_entry = h.children(cfg_n).next().unwrap();
        let mut entry = None;
        let mut exit_succ = None;
        for &n in &self.blocks {
            if n == cfg_entry
                || h.input_neighbours(n)
                    .any(|pred| !self.blocks.contains(&pred))
            {
                match entry {
                    None => {
                        entry = Some(n);
                    }
                    Some(prev) => {
                        return Err(OutlineCfgError::MultipleEntryNodes(prev, n));
                    }
                }
            }
            let external_succs = h.output_neighbours(n).filter(|s| !self.blocks.contains(s));
            match external_succs.at_most_one() {
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
            }
        }
        match (entry, exit_succ) {
            (Some(e), Some((x, o))) => Ok((e, x, o)),
            (None, _) => Err(OutlineCfgError::NoEntryNode),
            (_, None) => Err(OutlineCfgError::NoExitNode),
        }
    }
}

impl PatchVerification for OutlineCfg {
    type Error = OutlineCfgError;
    type Node = Node;
    fn verify(&self, h: &impl HugrView<Node = Node>) -> Result<(), OutlineCfgError> {
        self.compute_entry_exit(h)?;
        Ok(())
    }

    fn invalidated_nodes(
        &self,
        _: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        self.blocks.iter().copied()
    }
}

impl PatchHugrMut for OutlineCfg {
    /// The newly-created basic block, and the [CFG] node inside it
    ///
    /// [CFG]: OpType::CFG
    type Outcome = [Node; 2];

    const UNCHANGED_ON_FAILURE: bool = true;
    fn apply_hugr_mut(
        self,
        h: &mut impl HugrMut<Node = Node>,
    ) -> Result<[Node; 2], OutlineCfgError> {
        let (entry, exit, outside) = self.compute_entry_exit(h)?;
        // 1. Compute signature
        // These panic()s only happen if the Hugr would not have passed validate()
        let OpType::DataflowBlock(DataflowBlock { inputs, .. }) = h.get_optype(entry) else {
            panic!("Entry node is not a basic block")
        };
        let inputs = inputs.clone();
        let outputs = match h.get_optype(outside) {
            OpType::DataflowBlock(dfb) => dfb.dataflow_input().clone(),
            OpType::ExitBlock(exit) => exit.dataflow_input().clone(),
            _ => panic!("External successor not a basic block"),
        };
        let outer_cfg = h.get_parent(entry).unwrap();
        let outer_entry = h.children(outer_cfg).next().unwrap();

        // 2. new_block contains input node, sub-cfg, exit node all connected
        let (new_block, cfg_node) = {
            let mut new_block_bldr =
                BlockBuilder::new(inputs.clone(), vec![type_row![]], outputs.clone()).unwrap();
            let wires_in = inputs.iter().cloned().zip(new_block_bldr.input_wires());
            let cfg = new_block_bldr.cfg_builder(wires_in, outputs).unwrap();
            let cfg = cfg.finish_sub_container().unwrap();
            let unit_sum = new_block_bldr.add_constant(ops::Value::unary_unit_sum());
            let pred_wire = new_block_bldr.load_const(&unit_sum);
            new_block_bldr
                .set_outputs(pred_wire, cfg.outputs())
                .unwrap();
            let new_block_hugr = std::mem::take(new_block_bldr.hugr_mut());
            let ins_res = h.insert_hugr(outer_cfg, new_block_hugr);
            (
                ins_res.inserted_entrypoint,
                *ins_res.node_map.get(&cfg.node()).unwrap(),
            )
        };

        // 3. Entry edges. Change any edges into entry_block from outside, to target new_block
        let preds: Vec<_> = h
            .linked_outputs(entry, h.node_inputs(entry).exactly_one().ok().unwrap())
            .collect();
        for (pred, br) in preds {
            if !self.blocks.contains(&pred) {
                h.disconnect(pred, br);
                h.connect(pred, br, new_block, 0);
            }
        }
        if entry == outer_entry {
            // new_block must be the entry node, i.e. first child, of the enclosing CFG
            // (the current entry node will be reparented inside new_block below)
            h.move_before_sibling(new_block, outer_entry);
        }

        // 4(a). Exit edges.
        // Remove edge from exit_node (that used to target outside)
        let exit_port = h
            .node_outputs(exit)
            .filter(|p| {
                let (t, p2) = h.single_linked_input(exit, *p).unwrap();
                assert!(p2.index() == 0);
                t == outside
            })
            .exactly_one()
            .ok() // NodePorts does not implement Debug
            .unwrap();
        h.disconnect(exit, exit_port);
        // And connect new_block to outside instead
        h.connect(new_block, 0, outside, 0);

        // 5. Children of new CFG.
        let inner_exit = {
            let inner_exit = h.children(cfg_node).exactly_one().ok().unwrap();

            // Entry node must be first
            h.move_before_sibling(entry, inner_exit);
            // And remaining nodes
            for n in self.blocks {
                // Do not move the entry node, as we have already
                if n != entry {
                    h.set_parent(n, cfg_node);
                }
            }
            inner_exit
        };

        // 4(b). Reconnect exit edge to the new exit node within the inner CFG
        h.connect(exit, exit_port, inner_exit, 0);

        Ok([new_block, cfg_node])
    }
}

/// Errors that can occur in expressing an `OutlineCfg` rewrite.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OutlineCfgError {
    /// The set of blocks were not siblings
    #[error("The nodes did not all have the same parent")]
    NotSiblings,
    /// The parent node was not a CFG node
    #[error("The parent node {0} was not a CFG but a {1}")]
    ParentNotCfg(Node, Box<OpType>),
    /// Multiple blocks had incoming edges
    #[error("Multiple blocks had predecessors outside the set - at least {0} and {1}")]
    MultipleEntryNodes(Node, Node),
    /// Multiple blocks had outgoing edges
    // Note possible TODO: straightforward if all outgoing edges target the same BB
    #[error("Multiple blocks had edges leaving the set - at least {0} and {1}")]
    MultipleExitNodes(Node, Node),
    /// One block had multiple outgoing edges
    #[error("Exit block {0} had edges to multiple external blocks {1:?}")]
    MultipleExitEdges(Node, Vec<Node>),
    /// No block was identified as an entry block
    #[error("No block had predecessors outside the set")]
    NoEntryNode,
    /// No block was found with an edge leaving the set (so, must be an infinite loop)
    #[error("No block had a successor outside the set")]
    NoExitNode,
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::builder::{
        BlockBuilder, BuildError, CFGBuilder, Container, Dataflow, DataflowSubContainer,
        HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::usize_t;
    use crate::hugr::HugrMut;
    use crate::ops::constant::Value;
    use crate::ops::handle::{BasicBlockID, ConstID, NodeHandle};
    use crate::types::Signature;
    use crate::{Hugr, HugrView, Node};
    use cool_asserts::assert_matches;
    use itertools::Itertools;
    use rstest::rstest;

    use super::{OutlineCfg, OutlineCfgError};

    ///      /-> left --\
    /// entry            > merge -> head -> tail -> exit
    ///      \-> right -/             \-<--<-/
    struct CondThenLoopCfg {
        h: Hugr,
        left: Node,
        right: Node,
        merge: Node,
        head: Node,
        tail: Node,
    }
    impl CondThenLoopCfg {
        fn new() -> Result<CondThenLoopCfg, BuildError> {
            let block_ty = Signature::new_endo(usize_t());
            let mut cfg_builder = CFGBuilder::new(block_ty.clone())?;
            let pred_const = cfg_builder.add_constant(Value::unit_sum(0, 2).expect("0 < 2"));
            let const_unit = cfg_builder.add_constant(Value::unary_unit_sum());
            fn n_identity(
                mut bbldr: BlockBuilder<&mut Hugr>,
                cst: &ConstID,
            ) -> Result<BasicBlockID, BuildError> {
                let pred = bbldr.load_const(cst);
                let vals = bbldr.input_wires();
                bbldr.finish_with_outputs(pred, vals)
            }
            let id_block = |c: &mut CFGBuilder<_>| {
                n_identity(c.simple_block_builder(block_ty.clone(), 1)?, &const_unit)
            };

            let entry = n_identity(
                cfg_builder.simple_entry_builder(usize_t().into(), 2)?,
                &pred_const,
            )?;

            let left = id_block(&mut cfg_builder)?;
            let right = id_block(&mut cfg_builder)?;
            cfg_builder.branch(&entry, 0, &left)?;
            cfg_builder.branch(&entry, 1, &right)?;

            let merge = id_block(&mut cfg_builder)?;
            cfg_builder.branch(&left, 0, &merge)?;
            cfg_builder.branch(&right, 0, &merge)?;

            let head = id_block(&mut cfg_builder)?;
            cfg_builder.branch(&merge, 0, &head)?;
            let tail = n_identity(
                cfg_builder.simple_block_builder(Signature::new_endo(usize_t()), 2)?,
                &pred_const,
            )?;
            cfg_builder.branch(&tail, 1, &head)?;
            cfg_builder.branch(&head, 0, &tail)?; // trivial "loop body"
            let exit = cfg_builder.exit_block();
            cfg_builder.branch(&tail, 0, &exit)?;

            let h = cfg_builder.finish_hugr()?;
            let (left, right) = (left.node(), right.node());
            let (merge, head, tail) = (merge.node(), head.node(), tail.node());
            Ok(Self {
                h,
                left,
                right,
                merge,
                head,
                tail,
            })
        }
        fn entry_exit(&self) -> (Node, Node) {
            self.h
                .children(self.h.entrypoint())
                .take(2)
                .collect_tuple()
                .unwrap()
        }
    }

    #[rstest::fixture]
    fn cond_then_loop_cfg() -> CondThenLoopCfg {
        CondThenLoopCfg::new().unwrap()
    }

    #[rstest]
    fn test_outline_cfg_errors(cond_then_loop_cfg: CondThenLoopCfg) {
        let (entry, _) = cond_then_loop_cfg.entry_exit();
        let CondThenLoopCfg {
            mut h,
            left,
            right,
            merge,
            head,
            tail,
        } = cond_then_loop_cfg;
        let backup = h.clone();

        let r = h.apply_patch(OutlineCfg::new([tail]));
        assert_matches!(r, Err(OutlineCfgError::MultipleExitEdges(_, _)));
        assert_eq!(h, backup);

        let r = h.apply_patch(OutlineCfg::new([entry, left, right]));
        assert_matches!(r, Err(OutlineCfgError::MultipleExitNodes(a,b))
            => assert_eq!(HashSet::from([a,b]), HashSet::from_iter([left, right])));
        assert_eq!(h, backup);

        let r = h.apply_patch(OutlineCfg::new([left, right, merge]));
        assert_matches!(r, Err(OutlineCfgError::MultipleEntryNodes(a,b))
            => assert_eq!(HashSet::from([a,b]), HashSet::from([left, right])));
        assert_eq!(h, backup);

        // The entry node implicitly has an extra incoming edge
        let r = h.apply_patch(OutlineCfg::new([entry, left, right, merge, head]));
        assert_matches!(r, Err(OutlineCfgError::MultipleEntryNodes(a,b))
            => assert_eq!(HashSet::from([a,b]), HashSet::from([entry, head])));
        assert_eq!(h, backup);
    }

    #[rstest::rstest]
    fn test_outline_cfg(cond_then_loop_cfg: CondThenLoopCfg) {
        // Outline the loop, producing:
        //     /-> left -->\
        // entry            merge -> newblock -> exit
        //     \-> right ->/
        let (_, exit) = cond_then_loop_cfg.entry_exit();
        let CondThenLoopCfg {
            mut h,
            merge,
            head,
            tail,
            ..
        } = cond_then_loop_cfg;
        let root = h.entrypoint();
        let (new_block, _, exit_block) = outline_cfg_check_parents(&mut h, root, vec![head, tail]);
        assert_eq!(h.output_neighbours(merge).collect_vec(), vec![new_block]);
        assert_eq!(h.input_neighbours(exit).collect_vec(), vec![new_block]);
        assert_eq!(
            h.output_neighbours(tail).collect::<HashSet<Node>>(),
            HashSet::from([head, exit_block])
        );
    }

    #[rstest]
    fn test_outline_cfg_multiple_in_edges(cond_then_loop_cfg: CondThenLoopCfg) {
        // Outline merge, head and tail, producing
        //     /-> left -->\
        // entry            newblock -> exit
        //     \-> right ->/
        let (_, exit) = cond_then_loop_cfg.entry_exit();
        let CondThenLoopCfg {
            mut h,
            left,
            right,
            merge,
            head,
            tail,
        } = cond_then_loop_cfg;

        let root = h.entrypoint();
        let (new_block, _, inner_exit) =
            outline_cfg_check_parents(&mut h, root, vec![merge, head, tail]);
        assert_eq!(h.input_neighbours(exit).collect_vec(), vec![new_block]);
        assert_eq!(
            h.input_neighbours(new_block).collect::<HashSet<_>>(),
            HashSet::from([left, right])
        );
        assert_eq!(
            h.output_neighbours(tail).collect::<HashSet<Node>>(),
            HashSet::from([head, inner_exit])
        );
    }

    #[rstest]
    fn test_outline_cfg_subregion(cond_then_loop_cfg: CondThenLoopCfg) {
        // Outline the loop, as above, but with the CFG inside a Function + Module,
        // operating via a SiblingMut
        let mut module_builder = ModuleBuilder::new();
        let mut fbuild = module_builder
            .define_function("main", Signature::new(vec![usize_t()], vec![usize_t()]))
            .unwrap();
        let [i1] = fbuild.input_wires_arr();
        let cfg = fbuild
            .add_hugr_with_wires(cond_then_loop_cfg.h, [i1])
            .unwrap();
        fbuild.finish_with_outputs(cfg.outputs()).unwrap();
        let mut h = module_builder.finish_hugr().unwrap();
        // `add_hugr_with_wires` does not return an InsertionResult, so recover the nodes manually:
        let cfg = cfg.node();
        let exit_node = h.children(cfg).nth(1).unwrap();
        let tail = h.input_neighbours(exit_node).exactly_one().ok().unwrap();
        let head = h.input_neighbours(tail).exactly_one().ok().unwrap();
        // Just sanity-check we have the correct nodes
        assert!(h.get_optype(exit_node).is_exit_block());
        assert_eq!(
            h.output_neighbours(tail).collect::<HashSet<_>>(),
            HashSet::from([head, exit_node])
        );
        outline_cfg_check_parents(&mut h, cfg, vec![head, tail]);
        h.validate().unwrap();
    }

    #[rstest]
    fn test_outline_cfg_move_entry(cond_then_loop_cfg: CondThenLoopCfg) {
        // Outline the conditional, producing
        //
        //  newblock -> head -> tail -> exit
        //                 \<--</
        // (where the new block becomes the entry block)
        let (entry, _) = cond_then_loop_cfg.entry_exit();
        let CondThenLoopCfg {
            mut h,
            left,
            right,
            merge,
            head,
            ..
        } = cond_then_loop_cfg;

        let root = h.entrypoint();
        let (new_block, _, _) =
            outline_cfg_check_parents(&mut h, root, vec![entry, left, right, merge]);
        h.validate().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(new_block, h.children(h.entrypoint()).next().unwrap());
        assert_eq!(h.output_neighbours(new_block).collect_vec(), [head]);
    }

    fn outline_cfg_check_parents(
        h: &mut impl HugrMut<Node = Node>,
        cfg: Node,
        blocks: Vec<Node>,
    ) -> (Node, Node, Node) {
        let mut other_blocks = h.children(cfg).collect::<HashSet<_>>();
        assert!(blocks.iter().all(|b| other_blocks.remove(b)));
        let [new_block, new_cfg] = h.apply_patch(OutlineCfg::new(blocks.clone())).unwrap();

        for n in other_blocks {
            assert_eq!(h.get_parent(n), Some(cfg));
        }
        assert_eq!(h.get_parent(new_block), Some(cfg));
        assert!(h.get_optype(new_block).is_dataflow_block());
        assert_eq!(h.get_parent(new_cfg), Some(new_block));
        for n in blocks {
            assert_eq!(h.get_parent(n), Some(new_cfg));
        }
        assert!(h.get_optype(new_cfg).is_cfg());
        let exit_block = h.children(new_cfg).nth(1).unwrap();
        assert!(h.get_optype(exit_block).is_exit_block());
        (new_block, new_cfg, exit_block)
    }
}

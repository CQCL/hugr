use std::hash::Hash;

use crate::hugr::nest_cfgs::CfgView;
use crate::ops::{controlflow::ControlFlowOp, DataflowOp, OpType};
use crate::Hugr;
use portgraph::{NodeIndex, PortIndex};

/// We provide a view of a cfg where every node has at most one of
/// (multiple predecessors, multiple successors).
/// So, for BBs with multiple preds + succs, we generate TWO HalfNode's.
/// TODO: this unfortunately doesn't capture all cases: when a node has multiple preds and succs,
/// we could "merge" *any subset* of the in-edges into a single in-edge via an extra empty BB;
/// the in-edge from that extra/empty BB, might be the endpoint of a useful SESE region,
/// but we don't have a way to identify *which subset* to select. (Here we say *all preds* if >1 succ)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum HalfNode {
    /// All predecessors of original BB; successors if this does not break rule, else the X
    N(NodeIndex),
    // Exists only for BBs with multiple preds _and_ succs; has a single pred (the N), plus original succs
    X(NodeIndex),
}

struct HalfNodeView<'a> {
    h: &'a Hugr,
    parent: NodeIndex,
}

impl<'a> HalfNodeView<'a> {
    pub fn new(h: &'a Hugr, parent: NodeIndex) -> Result<Self, String> {
        if let OpType::Function(DataflowOp::ControlFlow {
            op: ControlFlowOp::CFG { .. },
        }) = h.get_optype(parent)
        {
            Ok(Self {
                h: h,
                parent: parent,
            })
        } else {
            Err("Not a kappa-node".to_string())
        }
    }

    fn is_multi_node(&self, n: NodeIndex) -> bool {
        self.bb_preds(n).take(2).count() + self.bb_succs(n).take(2).count() == 4
    }
    fn resolve_out(&self, n: NodeIndex) -> HalfNode {
        if self.is_multi_node(n) {
            HalfNode::X(n)
        } else {
            HalfNode::N(n)
        }
    }

    fn bb_succs(&self, n: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        // TODO filter out duplicate successors (and duplicate predecessors)
        // - but not duplicate (successor + predecessors) i.e. where edge directions are reversed
        self.h
            .graph
            .output_links(n)
            .into_iter()
            .map(|p| self.port_owner(p))
    }
    fn bb_preds(&self, n: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.h
            .graph
            .input_links(n)
            .into_iter()
            .map(|p| self.port_owner(p))
    }
    fn port_owner(&self, p: Option<PortIndex>) -> NodeIndex {
        self.h.graph.port_node(p.unwrap()).unwrap()
    }
}

impl CfgView<HalfNode> for HalfNodeView<'_> {
    type Iterator<'c> = <Vec<HalfNode> as IntoIterator>::IntoIter where Self: 'c;
    fn entry_node(&self) -> HalfNode {
        HalfNode::N(self.h.hierarchy.first(self.parent).unwrap())
    }
    fn exit_node(&self) -> HalfNode {
        self.resolve_out(self.h.hierarchy.last(self.parent).unwrap())
    }
    fn predecessors<'a>(&'a self, h: HalfNode) -> Self::Iterator<'a> {
        let mut ps = Vec::new();
        match h {
            HalfNode::N(ni) => ps.extend(self.bb_preds(ni).map(|n| self.resolve_out(n))),
            HalfNode::X(ni) => ps.push(HalfNode::N(ni)),
        };
        if h == self.entry_node() {
            ps.push(self.exit_node());
        }
        ps.into_iter()
    }
    fn successors<'a>(&'a self, n: HalfNode) -> Self::Iterator<'a> {
        let mut succs = Vec::new();
        match n {
            HalfNode::N(ni) if self.is_multi_node(ni) => succs.push(HalfNode::X(ni)),
            HalfNode::N(ni) | HalfNode::X(ni) => succs.extend(self.bb_succs(ni).map(HalfNode::N)),
        };
        succs.into_iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hugr::nest_cfgs::get_edge_classes;
    use crate::ops::{controlflow::ControlFlowOp, BasicBlockOp, DataflowOp, OpType};
    use crate::type_row;
    use crate::types::{ClassicType, SimpleType};
    use crate::{hugr::HugrError, Hugr};
    use std::collections::{HashMap, HashSet};
    use test_case::test_case;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::Nat);

    fn kappa() -> OpType {
        OpType::Function(DataflowOp::ControlFlow {
            op: ControlFlowOp::CFG {
                inputs: type_row![NAT],
                outputs: type_row![NAT],
            },
        })
    }

    fn add_block(
        h: &mut Hugr,
        parent: NodeIndex,
        num_inputs: usize,
        num_outputs: usize,
    ) -> Result<NodeIndex, HugrError> {
        let idx = h.add_node(OpType::BasicBlock(BasicBlockOp {
            inputs: type_row![NAT],
            outputs: type_row![NAT],
        }));
        h.set_parent(idx, parent)?;
        h.graph
            .set_num_ports(idx, num_inputs, num_outputs, |_, _| {});
        Ok(idx)
    }

    #[test_case(true; "separate merge + loop-header")]
    #[test_case(false; "combined merge + loop-header")]
    fn test_branch_then_loop(separate_header: bool) -> Result<(), HugrError> {
        let mut h = Hugr::new();
        let k = h.add_node(kappa());
        let entry = add_block(&mut h, k, 0, 1)?;
        let split = add_block(&mut h, k, 1, 2)?;
        h.connect(entry, 0, split, 0)?;

        let left = add_block(&mut h, k, 1, 1)?;
        h.connect(split, 0, left, 0)?;
        let right = add_block(&mut h, k, 1, 1)?;
        h.connect(split, 1, right, 0)?;
        let merge = add_block(&mut h, k, if separate_header { 2 } else { 3 }, 1)?;
        h.connect(left, 0, merge, 0)?;
        h.connect(right, 0, merge, 1)?;
        let loop_header = if separate_header {
            let hdr = add_block(&mut h, k, 2, 1)?;
            h.connect(merge, 0, hdr, 0)?;
            hdr
        } else {
            merge
        };
        let loop_tail = add_block(&mut h, k, 1, 2)?;
        h.connect(loop_header, 0, loop_tail, 0)?;
        h.connect(
            loop_tail,
            0,
            loop_header,
            if separate_header { 1 } else { 2 },
        )?;
        let exit = add_block(&mut h, k, 1, 0)?;
        h.connect(loop_tail, 1, exit, 0)?;
        let classes = get_edge_classes(&HalfNodeView::new(&h, k).unwrap());
        let mut groups = HashMap::new();
        for (e, c) in classes {
            groups.entry(c).or_insert(HashSet::new()).insert(e);
        }
        let g: Vec<_> = groups.into_values().filter(|s| s.len() > 1).collect();
        assert_eq!(g.len(), 3);
        let outer_class = if separate_header {
            HashSet::from([
                (HalfNode::N(entry), HalfNode::N(split)),
                (HalfNode::N(merge), HalfNode::N(loop_header)),
                (HalfNode::N(loop_tail), HalfNode::N(exit)),
            ])
        } else {
            HashSet::from([
                (HalfNode::N(entry), HalfNode::N(split)),
                (HalfNode::N(loop_tail), HalfNode::N(exit)),
            ])
        };
        assert!(g.contains(&outer_class));
        assert!(g.contains(&HashSet::from([
            (HalfNode::N(split), HalfNode::N(left)),
            (HalfNode::N(left), HalfNode::N(merge))
        ])));
        assert!(g.contains(&HashSet::from([
            (HalfNode::N(split), HalfNode::N(right)),
            (HalfNode::N(right), HalfNode::N(merge))
        ])));
        Ok(())
    }

    #[test]
    fn test_branch_in_loop() -> Result<(), HugrError> {
        let mut h = Hugr::new();
        let k = h.add_node(kappa());
        let entry = add_block(&mut h, k, 0, 1)?;
        // (Cases like) this test are about the only ones where "HalfNode" splits nodes in the
        // right place (all in-edges THEN all out-edges)
        let split_header = add_block(&mut h, k, 2, 2)?;
        h.connect(entry, 0, split_header, 0)?;
        let left = add_block(&mut h, k, 1, 1)?;
        h.connect(split_header, 0, left, 0)?;
        let right = add_block(&mut h, k, 1, 1)?;
        h.connect(split_header, 1, right, 0)?;
        // And symmetrically here, the merge/loop is split correctly by the HalfNode
        let merge_tail = add_block(&mut h, k, 2, 2)?;
        h.connect(left, 0, merge_tail, 0)?;
        h.connect(right, 0, merge_tail, 1)?;
        h.connect(merge_tail, 0, split_header, 1)?;
        let exit = add_block(&mut h, k, 1, 0)?;
        h.connect(merge_tail, 1, exit, 0)?;
        let classes = get_edge_classes(&HalfNodeView::new(&h, k).unwrap());
        let mut groups = HashMap::new();
        for (e, c) in classes {
            groups.entry(c).or_insert(HashSet::new()).insert(e);
        }
        let g: Vec<_> = groups.into_values().filter(|s| s.len() > 1).collect();
        assert_eq!(g.len(), 4);
        assert!(g.contains(&HashSet::from([
            (HalfNode::N(entry), HalfNode::N(split_header)),
            (HalfNode::X(merge_tail), HalfNode::N(exit))
        ])));
        assert!(g.contains(&HashSet::from([
            (HalfNode::N(split_header), HalfNode::X(split_header)),
            (HalfNode::N(merge_tail), HalfNode::X(merge_tail))
        ])));
        assert!(g.contains(&HashSet::from([
            (HalfNode::X(split_header), HalfNode::N(left)),
            (HalfNode::N(left), HalfNode::N(merge_tail))
        ])));
        assert!(g.contains(&HashSet::from([
            (HalfNode::X(split_header), HalfNode::N(right)),
            (HalfNode::N(right), HalfNode::N(merge_tail))
        ])));
        Ok(())
    }
}

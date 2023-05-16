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
        if let OpType::Dataflow(DataflowOp::ControlFlow {
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

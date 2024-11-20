use std::hash::Hash;

use super::nest_cfgs::CfgNodeMap;

use hugr_core::hugr::RootTagged;

use hugr_core::ops::handle::CfgID;
use hugr_core::ops::{OpTag, OpTrait};

use hugr_core::{Direction, Node};

/// We provide a view of a cfg where every node has at most one of
/// (multiple predecessors, multiple successors).
/// So for BBs with multiple preds + succs, we generate TWO HalfNode's with a single edge between
/// them; that single edge can then be a region boundary that did not exist before.
/// TODO: this unfortunately doesn't capture all cases: when a node has multiple preds and succs,
/// we could "merge" *any subset* of the in-edges into a single in-edge via an extra empty BB;
/// the in-edge from that extra/empty BB, might be the endpoint of a useful SESE region,
/// but we don't have a way to identify *which subset* to select. (Here we say *all preds* if >1 succ)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HalfNode {
    /// All predecessors of original BB; successors if this does not break rule, else the X
    N(Node),
    // Exists only for BBs with multiple preds _and_ succs; has a single pred (the N), plus original succs
    X(Node),
}

struct HalfNodeView<H> {
    h: H,
    entry: Node,
    exit: Node,
}

impl<H: RootTagged<RootHandle = CfgID>> HalfNodeView<H> {
    #[allow(unused)]
    pub(crate) fn new(h: H) -> Self {
        let (entry, exit) = {
            let mut children = h.children(h.root());
            (children.next().unwrap(), children.next().unwrap())
        };
        assert_eq!(h.get_optype(exit).tag(), OpTag::BasicBlockExit);
        Self { h, entry, exit }
    }

    fn is_multi_node(&self, n: Node) -> bool {
        // TODO if <n> is the entry-node, should we pretend there's an extra predecessor? (The "outside")
        // We could also setify here before counting, but never
        self.bb_preds(n).take(2).count() + self.bb_succs(n).take(2).count() == 4
    }
    fn resolve_out(&self, n: Node) -> HalfNode {
        if self.is_multi_node(n) {
            HalfNode::X(n)
        } else {
            HalfNode::N(n)
        }
    }

    fn bb_succs(&self, n: Node) -> impl Iterator<Item = Node> + '_ {
        self.h.neighbours(n, Direction::Outgoing)
    }
    fn bb_preds(&self, n: Node) -> impl Iterator<Item = Node> + '_ {
        self.h.neighbours(n, Direction::Incoming)
    }
}

impl<H: RootTagged<RootHandle = CfgID>> CfgNodeMap<HalfNode> for HalfNodeView<H> {
    fn entry_node(&self) -> HalfNode {
        HalfNode::N(self.entry)
    }
    fn exit_node(&self) -> HalfNode {
        assert!(self.bb_succs(self.exit).count() == 0);
        HalfNode::N(self.exit)
    }
    fn predecessors(&self, h: HalfNode) -> impl Iterator<Item = HalfNode> {
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
    fn successors(&self, n: HalfNode) -> impl Iterator<Item = HalfNode> {
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
    use super::super::nest_cfgs::{test::*, EdgeClassifier};
    use super::{HalfNode, HalfNodeView};
    use hugr_core::builder::BuildError;
    use hugr_core::hugr::views::RootChecked;
    use hugr_core::ops::handle::NodeHandle;

    use itertools::Itertools;
    use std::collections::HashSet;
    #[test]
    fn test_cond_in_loop_combined_headers() -> Result<(), BuildError> {
        let (h, main, tail) = build_conditional_in_loop_cfg(false)?;
        //               /-> left --\
        //  entry -> main            > merge -> tail -> exit
        //            |  \-> right -/             |
        //             \---<---<---<---<---<--<---/
        // The "main" has two predecessors (entry and tail) and two successors (left and right) so
        // we get HalfNode::N(main) aka "head" and HalfNode::X(main) aka "split" in this form:
        //                          /-> left --\
        // N(entry) -> head -> split            > N(merge) -> N(tail) -> N(exit)
        //               |          \-> right -/                 |
        //               \---<---<---<---<---<---<---<---<---<---/
        // Allowing to identify two nested regions (and fixing the problem with an IdentityCfgMap on the same example)

        let v = HalfNodeView::new(RootChecked::try_new(&h).unwrap());

        let edge_classes = EdgeClassifier::get_edge_classes(&v);
        let HalfNodeView { h: _, entry, exit } = v;

        let head = HalfNode::N(main.node());
        let tail = HalfNode::N(tail.node());
        let split = HalfNode::X(main.node());
        let (entry, exit) = (HalfNode::N(entry), HalfNode::N(exit));
        // merge is unique predecessor of tail
        let merge = *edge_classes
            .keys()
            .filter(|(_, t)| *t == tail)
            .map(|(s, _)| s)
            .exactly_one()
            .unwrap();
        let [&left, &right] = edge_classes
            .keys()
            .filter(|(s, _)| *s == split)
            .map(|(_, t)| t)
            .collect::<Vec<_>>()[..]
        else {
            panic!("Split node should have two successors");
        };
        let classes = group_by(edge_classes);
        assert_eq!(
            classes,
            HashSet::from([
                sorted([(split, left), (left, merge)]), // Region containing single BB 'left'.
                sorted([(split, right), (right, merge)]), // Region containing single BB 'right'.
                sorted([(head, split), (merge, tail)]), // The inner "conditional" region.
                sorted([(entry, head), (tail, exit)]), // "Loop" region containing body (conditional) + back-edge.
                Vec::from([(tail, head)])              // The loop back-edge.
            ])
        );
        Ok(())
    }

    // Sadly this HalfNode logic is too simple to fix the test_cond_then_loop_combined case
    // (The "merge" node is not split, but needs to be split with the tail->merge edge incoming
    // to the *second* node after splitting).
}

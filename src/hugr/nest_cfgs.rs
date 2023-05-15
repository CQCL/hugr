use portgraph::{NodeIndex, PortIndex};
use std::collections::{HashMap, HashSet, LinkedList};
use std::hash::Hash;

use crate::ops::{controlflow::ControlFlowOp, DataflowOp, OpType};
use crate::Hugr;

/// Identify Single-Entry-Single-Exit regions in the CFG.
/// These are pairs of edges (a,b) where
/// a dominates b, b postdominates a, and there are no other edges in/out of the nodes inbetween
/// (the third condition is necessary because loop backedges do not affect (post)dominance).
/// Algorithm here: https://dl.acm.org/doi/10.1145/773473.178258, approximately:
/// (1) those three conditions are equivalent to:
/// >>>a and b are cycle-equivalent in the CFG with an extra edge from the exit node to the entry<<<
/// where cycle-equivalent means every cycle has either both a and b, or neither
/// (2) cycle equivalence is unaffected if all edges are considered *un*directed
///     (not obvious, see paper for proof)
/// (3) take undirected CFG, perform depth-first traversal
///     => all edges are *tree edges* or *backedges* where one endpoint is an ancestor of the other
/// (4) identify the "bracketlist" of each tree edge - the set of backedges going from a descendant to an ancestor
///     -- post-order traversal, merging bracketlists of children,
///            then delete backedges from below to here, add backedges from here to above
///     => tree edges with the same bracketlist are cycle-equivalent;
///        + a tree edge with a single-element bracketlist is cycle-equivalent with that single element
/// (5) this would be expensive (comparing large sets of backedges) - so to optimize,
///     - the backedge most recently added (at the top) of the bracketlist, plus the size of the bracketlist,
///       is sufficient to identify the set *when the UDFS tree is linear*;
///     - when UDFS is treelike, any ancestor with backedges from >1 subtree cannot be cycle-equivalent with any descendant,
///       so add (onto top of bracketlist) a fake "capping" backedge from here to the highest ancestor reached by >1 subtree.
///       (Thus, edges from here up to that ancestor, cannot be cycle-equivalent with any edges elsewhere.)

/// TODO: transform the CFG: each SESE region can be turned into its own Kappa-node
/// (in a BB with one predecessor and one successor, which may then be merged
///     and contents parallelized with predecessor or successor).

// A view of a CFG. Although we can think of each T being a BasicBlock i.e. a NodeIndex in the HUGR,
// this extra level of indirection allows "splitting" of one HUGR basic block into many (or vice versa).
// Since regions are identified by edges between pairs of such T, such splitting may allow to identify
// more regions than existed in the underlying CFG (without mutating the underlying CFG perhaps in vain).
pub trait CfgView<T> {
    fn entry_node(&self) -> T;
    fn exit_node(&self) -> T;
    fn successors(&self, item: T) -> Box<dyn Iterator<Item = T>>;
    fn predecessors(&self, item: T) -> Box<dyn Iterator<Item = T>>;
}

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
    fn entry_node(&self) -> HalfNode {
        HalfNode::N(self.h.hierarchy.first(self.parent).unwrap())
    }
    fn exit_node(&self) -> HalfNode {
        self.resolve_out(self.h.hierarchy.last(self.parent).unwrap())
    }
    fn predecessors(&self, h: HalfNode) -> Box<dyn Iterator<Item = HalfNode>> {
        let mut ps = Vec::new();
        match h {
            HalfNode::N(ni) => ps.extend(self.bb_preds(ni).map(|n| self.resolve_out(n))),
            HalfNode::X(ni) => ps.push(HalfNode::N(ni)),
        };
        if h == self.entry_node() {
            ps.push(self.exit_node());
        }
        Box::new(ps.into_iter())
    }
    fn successors(&self, n: HalfNode) -> Box<dyn Iterator<Item = HalfNode>> {
        let mut succs = Vec::new();
        match n {
            HalfNode::N(ni) if self.is_multi_node(ni) => succs.push(HalfNode::X(ni)),
            HalfNode::N(ni) | HalfNode::X(ni) => succs.extend(self.bb_succs(ni).map(HalfNode::N)),
        };
        Box::new(succs.into_iter())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum EdgeDest<T: Copy + Clone + PartialEq + Eq + Hash> {
    Forward(T),
    Backward(T),
}
impl<T> EdgeDest<T>
where
    T: Copy + Clone + PartialEq + Eq + Hash,
{
    pub fn target(&self) -> T {
        match self {
            EdgeDest::Forward(i) => *i,
            EdgeDest::Backward(i) => *i,
        }
    }
}

fn undirected_edges<T: Copy + Clone + PartialEq + Eq + Hash>(
    cfg: &dyn CfgView<T>,
    n: T,
) -> impl Iterator<Item = EdgeDest<T>> + '_ {
    let mut succs: Vec<_> = cfg.successors(n).collect();
    if n == cfg.exit_node() {
        succs.push(cfg.entry_node());
    }
    succs
        .into_iter()
        .map(EdgeDest::Forward)
        .chain(cfg.predecessors(n).map(EdgeDest::Backward))
}

fn flip<T: Copy + Clone + PartialEq + Eq + Hash>(src: T, d: EdgeDest<T>) -> (T, EdgeDest<T>) {
    match d {
        EdgeDest::Forward(tgt) => (tgt, EdgeDest::Backward(src)),
        EdgeDest::Backward(tgt) => (tgt, EdgeDest::Forward(src)),
    }
}

struct UndirectedDFSTree<T: Copy + Clone + PartialEq + Eq + Hash> {
    dfs_num: HashMap<T, usize>,
    dfs_parents: HashMap<T, EdgeDest<T>>, // value is direction + source of edge along which key was reached
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> UndirectedDFSTree<T> {
    pub fn new(cfg: &dyn CfgView<T>) -> Self {
        //1. Traverse backwards-only from exit building bitset of reachable nodes
        let mut reachable = HashSet::new();
        {
            let mut pending = LinkedList::new();
            pending.push_back(cfg.exit_node());
            while let Some(n) = pending.pop_front() {
                if reachable.insert(n) {
                    pending.extend(cfg.predecessors(n));
                }
            }
        }
        //2. Traverse undirected from entry node, building dfs_num and setting dfs_parents
        let mut dfs_num = HashMap::new();
        let mut dfs_parents = HashMap::new();
        {
            // Node, and directed edge along which reached
            let mut pending = vec![(cfg.entry_node(), EdgeDest::Backward(cfg.exit_node()))];
            while let Some((n, p_edge)) = pending.pop() {
                if !dfs_num.contains_key(&n) && reachable.contains(&n) {
                    dfs_num.insert(n, dfs_num.len());
                    dfs_parents.insert(n, p_edge);
                    for e in undirected_edges(cfg, n) {
                        pending.push(flip(n, e));
                    }
                }
            }
            dfs_parents.remove(&cfg.entry_node()).unwrap();
        }
        UndirectedDFSTree {
            dfs_num,
            dfs_parents,
        }
    }
}

struct TraversalState<T> {
    deleted_backedges: HashSet<UDEdge<T>>, // Allows constant-time deletion
    capping_edges: HashMap<usize, Vec<CappingEdge<T>>>, // Indexed by DFS num
    edge_classes: HashMap<CFEdge<T>, Option<CycleClass<T>>>, // Accumulates result (never overwritten)
}

/// Computes equivalence class of each edge, i.e. two edges with the same value
/// are cycle-equivalent.
pub fn get_edge_classes<T: Copy + Clone + PartialEq + Eq + Hash>(
    cfg: &dyn CfgView<T>,
) -> HashMap<CFEdge<T>, Option<CycleClass<T>>> {
    let tree = UndirectedDFSTree::new(cfg);
    let mut st = TraversalState {
        deleted_backedges: HashSet::new(),
        capping_edges: HashMap::new(),
        edge_classes: HashMap::new(),
    };
    traverse(cfg, &tree, &mut st, cfg.entry_node());
    assert!(st.capping_edges.is_empty());
    st.edge_classes
        .remove(&CFEdge(cfg.exit_node(), cfg.entry_node()));
    st.edge_classes
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CFEdge<T>(T, T); // TODO use tuple?
impl<T: Copy + Clone + PartialEq + Eq + Hash> CFEdge<T> {
    fn from(s: T, t: EdgeDest<T>) -> Self {
        match t {
            EdgeDest::Forward(t) => Self(s, t),
            EdgeDest::Backward(t) => Self(t, s),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CappingEdge<T> {
    // TODO hide
    common_parent: T,
    dfs_target: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum UDEdge<T> {
    // TODO hide
    RealEdge(CFEdge<T>),
    FakeEdge(CappingEdge<T>),
}

pub type CycleClass<T> = (UDEdge<T>, usize); // TODO hide (replace in output)

struct BracketList<T: Copy + Clone + PartialEq + Eq + Hash> {
    items: LinkedList<UDEdge<T>>,
    size: usize, // deleted items already taken off
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> BracketList<T> {
    pub fn new() -> Self {
        BracketList {
            items: LinkedList::new(),
            size: 0,
        }
    }

    pub fn tag(&mut self, deleted: &HashSet<UDEdge<T>>) -> Option<CycleClass<T>> {
        while let Some(e) = self.items.front() {
            // Pop deleted elements to save time (and memory)
            if deleted.contains(e) {
                self.items.pop_front();
                //deleted.remove(e); // Only saves memory
            } else {
                return Some((e.clone(), self.size));
            }
        }
        None
    }

    pub fn concat(&mut self, other: BracketList<T>) -> () {
        let BracketList { mut items, size } = other;
        self.items.append(&mut items);
        assert!(items.len() == 0);
        self.size += size;
    }

    pub fn delete(&mut self, b: &UDEdge<T>, deleted: &mut HashSet<UDEdge<T>>) {
        // Ideally, here we would also assert that no *other* BracketList contains b.
        debug_assert!(self.items.contains(b)); // Makes operation O(n), otherwise O(1)
        assert!(!deleted.contains(b));
        deleted.insert(b.clone());
        self.size -= 1;
    }

    pub fn push(&mut self, e: UDEdge<T>) {
        self.items.push_back(e);
        self.size += 1;
    }
}

fn traverse<T: Copy + Clone + PartialEq + Eq + Hash>(
    cfg: &dyn CfgView<T>,
    tree: &UndirectedDFSTree<T>,
    st: &mut TraversalState<T>,
    n: T,
) -> (usize, BracketList<T>) {
    let n_dfs = *tree.dfs_num.get(&n).unwrap(); // should only be called for nodes on path to exit
    let (children, non_capping_backedges): (Vec<_>, Vec<_>) = undirected_edges(cfg, n)
        .filter(|e| tree.dfs_num.contains_key(&e.target()))
        .partition(|e| {
            // The tree edges are those whose *targets* list the edge as parent-edge
            let (tgt, from) = flip(n, *e);
            tree.dfs_parents.get(&tgt) == Some(&from)
        });
    let child_results: Vec<_> = children
        .iter()
        .map(|c| traverse(cfg, tree, st, c.target()))
        .collect();
    let mut min_dfs_target: [Option<usize>; 2] = [None, None]; // We want highest-but-one
    let mut bs = BracketList::new();
    for (tgt, brs) in child_results {
        if tgt < min_dfs_target[0].unwrap_or(usize::MAX) {
            min_dfs_target = [Some(tgt), min_dfs_target[0]]
        } else if tgt < min_dfs_target[1].unwrap_or(usize::MAX) {
            min_dfs_target[1] = Some(tgt)
        }
        bs.concat(brs);
    }
    // Add capping backedge
    if let Some(min1dfs) = min_dfs_target[1] {
        if min1dfs < n_dfs {
            let capping_edge = CappingEdge {
                common_parent: n,
                dfs_target: min1dfs,
            };
            bs.push(UDEdge::FakeEdge(capping_edge.clone()));
            // mark capping edge to be removed when we return out to the other end
            st.capping_edges
                .entry(min1dfs)
                .or_insert(Vec::new())
                .push(capping_edge);
        }
    }

    let parent_edge = tree.dfs_parents.get(&n);
    let (be_up, be_down): (Vec<_>, Vec<_>) = non_capping_backedges
        .into_iter()
        .map(|e| (*tree.dfs_num.get(&e.target()).unwrap(), e))
        .partition(|(dfs, _)| *dfs < n_dfs);

    // Remove edges to here from beneath
    for (_, e) in be_down {
        let e = UDEdge::RealEdge(CFEdge::from(n, e));
        bs.delete(&e, &mut st.deleted_backedges);
    }
    // And capping backedges
    for e in st.capping_edges.remove(&n_dfs).unwrap_or(Vec::new()) {
        bs.delete(&UDEdge::FakeEdge(e), &mut st.deleted_backedges)
    }

    // Add backedges from here to ancestors (not the parent edge, but perhaps other edges to the same node)
    be_up
        .iter()
        .filter(|(_, e)| Some(e) != parent_edge)
        .for_each(|(_, e)| bs.push(UDEdge::RealEdge(CFEdge::from(n, *e))));

    // Now calculate edge classes
    let class = bs.tag(&st.deleted_backedges);
    if let Some((UDEdge::RealEdge(e), 1)) = &class {
        st.edge_classes.insert(e.clone(), class.clone());
    }
    if let Some(parent_edge) = tree.dfs_parents.get(&n) {
        st.edge_classes.insert(CFEdge::from(n, *parent_edge), class);
    }
    let highest_target = be_up
        .into_iter()
        .map(|(dfs, _)| dfs)
        .chain(min_dfs_target[0].into_iter());
    (highest_target.min().unwrap_or(usize::MAX), bs)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::{controlflow::ControlFlowOp, BasicBlockOp, DataflowOp, OpType};
    use crate::type_row;
    use crate::types::{ClassicType, SimpleType};
    use crate::{hugr::HugrError, Hugr};
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
                CFEdge(HalfNode::N(entry), HalfNode::N(split)),
                CFEdge(HalfNode::N(merge), HalfNode::N(loop_header)),
                CFEdge(HalfNode::N(loop_tail), HalfNode::N(exit)),
            ])
        } else {
            HashSet::from([
                CFEdge(HalfNode::N(entry), HalfNode::N(split)),
                CFEdge(HalfNode::N(loop_tail), HalfNode::N(exit)),
            ])
        };
        assert!(g.contains(&outer_class));
        assert!(g.contains(&HashSet::from([
            CFEdge(HalfNode::N(split), HalfNode::N(left)),
            CFEdge(HalfNode::N(left), HalfNode::N(merge))
        ])));
        assert!(g.contains(&HashSet::from([
            CFEdge(HalfNode::N(split), HalfNode::N(right)),
            CFEdge(HalfNode::N(right), HalfNode::N(merge))
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
            CFEdge(HalfNode::N(entry), HalfNode::N(split_header)),
            CFEdge(HalfNode::X(merge_tail), HalfNode::N(exit))
        ])));
        assert!(g.contains(&HashSet::from([
            CFEdge(HalfNode::N(split_header), HalfNode::X(split_header)),
            CFEdge(HalfNode::N(merge_tail), HalfNode::X(merge_tail))
        ])));
        assert!(g.contains(&HashSet::from([
            CFEdge(HalfNode::X(split_header), HalfNode::N(left)),
            CFEdge(HalfNode::N(left), HalfNode::N(merge_tail))
        ])));
        assert!(g.contains(&HashSet::from([
            CFEdge(HalfNode::X(split_header), HalfNode::N(right)),
            CFEdge(HalfNode::N(right), HalfNode::N(merge_tail))
        ])));
        Ok(())
    }
}

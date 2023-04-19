use portgraph::{NodeIndex, PortIndex};
use std::collections::{HashMap, HashSet, LinkedList};
use std::hash::Hash;

use crate::ops::{controlflow::ControlFlowOp, DataflowOp, OpType};
use crate::Hugr;

/// We provide a view of a cfg where every node has at most one of
/// (multiple predecessors, multiple successors).
// So, for BBs with multiple preds + succs, we generate TWO HalfNode's.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum HalfNode {
    /// All predecessors of original BB; successors if this does not break rule, else the X
    N(NodeIndex),
    // Exists only for BBs with multiple preds _and_ succs; has a single pred (the N), plus original succs
    X(NodeIndex),
}

struct CfgView<'a> {
    h: &'a Hugr,
    parent: NodeIndex,
}

impl<'a> CfgView<'a> {
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
    pub fn entry_node(&self) -> HalfNode {
        HalfNode::N(self.h.hierarchy.first(self.parent).unwrap())
    }
    pub fn exit_node(&self) -> HalfNode {
        self.resolve_out(self.h.hierarchy.last(self.parent).unwrap())
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
    fn predecessors(&self, h: HalfNode) -> impl Iterator<Item = HalfNode> + '_ {
        let mut ps = Vec::new();
        match h {
            HalfNode::N(ni) => ps.extend(self.bb_preds(ni).map(|n| self.resolve_out(n))),
            HalfNode::X(ni) => ps.push(HalfNode::N(ni)),
        };
        ps.into_iter()
    }
    fn bb_succs(&self, n: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
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
    pub fn undirected_edges(&self, n: HalfNode) -> impl Iterator<Item = EdgeDest> + '_ {
        let mut succs = Vec::new();
        match n {
            HalfNode::N(ni) if self.is_multi_node(ni) => succs.push(HalfNode::X(ni)),
            HalfNode::N(ni) | HalfNode::X(ni) => succs.extend(self.bb_succs(ni).map(HalfNode::N)),
        };
        succs
            .into_iter()
            .map(EdgeDest::Forward)
            .chain(self.predecessors(n).map(EdgeDest::Backward))
    }
    pub fn get_edge_classes(&self) -> HashMap<CFEdge, CycleClass> {
        let tree = UndirectedDFSTree::new(self);
        let mut st = TraversalState {
            deleted_backedges: HashSet::new(),
            capping_edges: HashMap::new(),
            edge_classes: HashMap::new(),
        };
        tree.traverse(&mut st, self.entry_node());
        assert!(st.capping_edges.is_empty());
        st.edge_classes
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum EdgeDest {
    Forward(HalfNode),
    Backward(HalfNode),
}
impl EdgeDest {
    pub fn target(&self) -> HalfNode {
        match self {
            EdgeDest::Forward(i) => *i,
            EdgeDest::Backward(i) => *i,
        }
    }
}

#[derive(Copy, Clone, Eq)]
struct CFEdge(HalfNode, EdgeDest);
impl CFEdge {
    pub fn flip(&self) -> Self {
        match self.1 {
            EdgeDest::Forward(tgt) => Self(tgt, EdgeDest::Backward(self.0)),
            EdgeDest::Backward(tgt) => Self(tgt, EdgeDest::Forward(self.0)),
        }
    }
}

impl PartialEq for CFEdge {
    fn eq(&self, other: &CFEdge) -> bool {
        let &CFEdge(n1, d1) = self;
        let &CFEdge(n2, d2) = other;
        (n1, d1) == (n2, d2) || {
            let CFEdge(n1, d1) = self.flip();
            (n1, d1) == (n2, d2)
        }
    }
}

impl Hash for CFEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.1 {
            EdgeDest::Forward(d) => (self.0, d).hash(state),
            EdgeDest::Backward(_) => self.flip().hash(state),
        };
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CappingEdge {
    common_parent: HalfNode,
    dfs_target: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum UDEdge {
    RealEdge(CFEdge),
    FakeEdge(CappingEdge),
}

type CycleClass = (UDEdge, usize);

struct BracketList {
    items: LinkedList<UDEdge>,
    size: usize, // deleted items already taken off
}

impl BracketList {
    pub fn new() -> Self {
        BracketList {
            items: LinkedList::new(),
            size: 0,
        }
    }

    pub fn tag(&mut self, deleted: &HashSet<UDEdge>) -> Option<CycleClass> {
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

    pub fn concat(&mut self, other: BracketList) -> () {
        let BracketList { mut items, size } = other;
        self.items.append(&mut items);
        assert!(items.len() == 0);
        self.size += size;
    }

    pub fn delete(&mut self, b: &UDEdge, deleted: &mut HashSet<UDEdge>) {
        // Ideally, here we would also assert that no *other* BracketList contains b.
        debug_assert!(self.items.contains(b));
        assert!(!deleted.contains(b));
        deleted.insert(b.clone());
        self.size -= 1;
    }

    pub fn push(&mut self, e: UDEdge) {
        self.items.push_back(e);
        self.size += 1;
    }
}

struct UndirectedDFSTree<'a> {
    h: &'a CfgView<'a>,
    dfs_num: HashMap<HalfNode, usize>,
    dfs_parents: HashMap<HalfNode, EdgeDest>, // value is direction + source of edge along which key was reached
}

struct TraversalState {
    deleted_backedges: HashSet<UDEdge>,
    capping_edges: HashMap<usize, Vec<CappingEdge>>, // Indexed by DFS num, elems all CappingBackedge's
    edge_classes: HashMap<CFEdge, CycleClass>,
}

impl<'a> UndirectedDFSTree<'a> {
    pub fn new(h: &'a CfgView) -> Self {
        //1. Traverse backwards-only from exit building bitset of reachable nodes
        let mut reachable = HashSet::new();
        {
            let mut pending = LinkedList::new();
            pending.push_back(h.exit_node());
            while let Some(n) = pending.pop_front() {
                if reachable.insert(n) {
                    pending.extend(h.predecessors(n));
                }
            }
        }
        //2. Traverse undirected from entry node, building dfs_num and setting dfs_parents
        let mut dfs_num = HashMap::new();
        let mut dfs_parents = HashMap::new();
        {
            // Node, and edge along which reached
            let mut pending = vec![CFEdge(h.entry_node(), EdgeDest::Backward(h.exit_node()))];
            while let Some(CFEdge(n, p_edge)) = pending.pop() {
                if !dfs_num.contains_key(&n) && reachable.contains(&n) {
                    dfs_num.insert(n, dfs_num.len());
                    dfs_parents.insert(n, p_edge);
                    for e in h.undirected_edges(n) {
                        pending.push(CFEdge(n, e));
                    }
                }
            }
        }
        UndirectedDFSTree {
            h,
            dfs_num,
            dfs_parents,
        }
    }

    pub fn children_backedges(&self, n: HalfNode) -> (Vec<EdgeDest>, Vec<EdgeDest>) {
        self.h
            .undirected_edges(n)
            .filter(|e| self.dfs_parents.contains_key(&e.target()))
            .partition(|e| {
                // The tree edges are those whose *targets* list the edge as parent-edge
                let CFEdge(tgt, from) = CFEdge(n, *e).flip();
                (*self.dfs_parents.get(&tgt).unwrap()) == from
            })
    }

    fn traverse(&self, st: &mut TraversalState, n: HalfNode) -> (usize, BracketList) {
        let n_dfs = *self.dfs_num.get(&n).unwrap(); // should only be called for nodes on path to exit
        let (children, non_capping_backedges) = self.children_backedges(n);
        let child_results: Vec<_> = children
            .iter()
            .map(|c| self.traverse(st, c.target()))
            .collect();
        let mut min_dfs_target: [Option<usize>; 2] = [None, None];
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
                match st.capping_edges.get_mut(&min1dfs) {
                    Some(v) => v.push(capping_edge),
                    None => {
                        st.capping_edges.insert(min1dfs, vec![capping_edge]);
                    }
                };
            }
        }

        let num_backedges = non_capping_backedges.len();
        let parent_edge = *self.dfs_parents.get(&n).unwrap();
        let (be_up, be_down): (Vec<_>, Vec<_>) = non_capping_backedges
            .into_iter()
            .filter(|e| *e != parent_edge)
            .map(|e| (*self.dfs_num.get(&e.target()).unwrap(), e))
            .partition(|(dfs, _)| *dfs < n_dfs);
        assert!(be_down.len() + be_up.len() + 1 == num_backedges); // Parent found exactly once

        // Remove edges to here from beneath
        for e in be_down
            .into_iter()
            .map(|(_, e)| UDEdge::RealEdge(CFEdge(n, e)))
            .chain(
                // Also capping backedges
                st.capping_edges
                    .remove(&n_dfs)
                    .into_iter()
                    .flat_map(|v| v.into_iter())
                    .map(UDEdge::FakeEdge),
            )
        {
            bs.delete(&e, &mut st.deleted_backedges);
        }
        // Add backedges from here to ancestors
        be_up
            .iter()
            .for_each(|(_, e)| bs.push(UDEdge::RealEdge(CFEdge(n, *e))));

        // Now calculate edge classes
        let class = bs.tag(&st.deleted_backedges).unwrap();
        if let (UDEdge::RealEdge(e), 1) = &class {
            st.edge_classes.insert(e.clone(), class.clone());
        }
        st.edge_classes.insert(CFEdge(n, parent_edge), class);
        let highest_target = be_up
            .into_iter()
            .map(|(dfs, _)| dfs)
            .chain(min_dfs_target[0].into_iter());
        (highest_target.min().unwrap_or(usize::MAX), bs)
    }
}

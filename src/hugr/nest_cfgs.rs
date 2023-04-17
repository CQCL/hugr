use portgraph::NodeIndex;
use std::collections::{HashMap, HashSet, LinkedList};
use std::cmp::min;
use std::iter::empty;

struct HugrView {
}
impl HugrView {
    pub fn entry_node(&self) -> NodeIndex {
        todo!()
    }
    pub fn exit_node(&self) -> NodeIndex {
        todo!()
    }
    pub fn successors(&self, n: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        todo!();
        empty()
    }
    pub fn predecessors(&self, n: NodeIndex) -> impl Iterator<Item = NodeIndex> {
        todo!();
        empty()
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum EdgeDest {
    Forward(NodeIndex),
    Backward(NodeIndex)
}
impl EdgeDest {
    pub fn node_index(&self) -> NodeIndex {
        match self {
            EdgeDest::Forward(i) => *i,
            EdgeDest::Backward(i) => *i,
        }
    }
    pub fn flip(&self, src: NodeIndex) -> (EdgeDest, NodeIndex) {
        match self {
            EdgeDest::Forward(tgt) => (EdgeDest::Forward(src), *tgt),
            EdgeDest::Backward(tgt) => (EdgeDest::Backward(src), *tgt),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum UDEdge {
    CFGEdge(NodeIndex, EdgeDest),
    CappingBackedge(NodeIndex, usize)
}

type CycleClass = (UDEdge, usize);

struct BracketList {
    items: LinkedList<UDEdge>,
    size: usize // deleted items already taken off
}

impl BracketList {
    pub fn new() -> Self {
        BracketList {
            items: LinkedList::new(),
            size: 0
        }
    }

    pub fn tag(&mut self, deleted: &HashSet<UDEdge>) -> Option<CycleClass> {
        while let Some(e) = self.items.front() {
            // Pop deleted elements to save time (and memory)
            if deleted.contains(e) {
                self.items.pop_front();
                //deleted.remove(e); // Only saves memory
            } else {
                return Some((e.clone(), self.size))
            }
        }
        None
    }

    pub fn concat(&mut self, other: BracketList) -> () {
        let BracketList {mut items, size} = other;
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
    h: &'a HugrView,
    dfs_num: HashMap<NodeIndex, usize>,
    dfs_parents: HashMap<NodeIndex, EdgeDest>, // value is direction + source of edge along which key was reached
}

struct TraversalState {
    deleted_backedges: HashSet<UDEdge>,
    capping_edges: HashMap<usize, Vec<UDEdge>>, // Indexed by DFS num, elems all CappingBackedge's
    edge_classes: HashMap<UDEdge, CycleClass>
}

impl<'a> UndirectedDFSTree<'a> {
    pub fn new(h: &'a HugrView) -> Self {
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
        let mut t = UndirectedDFSTree {
            h: h,
            dfs_num: HashMap::new(),
            dfs_parents: HashMap::new()
        };
        let mut pending = vec![(EdgeDest::Forward(h.exit_node()), h.entry_node())];
        while let Some((p_edge, n)) = pending.pop() {
            if !t.dfs_num.contains_key(&n) && reachable.contains(&n) {
                t.dfs_num.insert(n, t.dfs_num.len());
                t.dfs_parents.insert(n, p_edge);
                for e in t.undirected_edges(n) {
                    pending.push(e.flip(n));
                }
            }
        };
        t
    }

    fn undirected_edges(&self, n: NodeIndex) -> impl Iterator<Item=EdgeDest> {
        // Filter reachable? Or no?
        self.h.successors(n).map(EdgeDest::Forward).chain(self.h.predecessors(n).map(EdgeDest::Backward))
    }

    pub fn children_backedges(&self, n: NodeIndex) -> (Vec<EdgeDest>, Vec<EdgeDest>) {
        // If we didn't filter reachable above, we should do so here (not unwrap)
        // Also, exclude the edge from this node's parent!
        self.undirected_edges(n).partition(|e| {
            let (from, tgt) = e.flip(n);
            (*self.dfs_parents.get(&tgt).unwrap()) == from
        })
    }

    fn traverse(&self, st: &mut TraversalState, n: NodeIndex) -> (usize, BracketList) {
        let n_dfs = *self.dfs_num.get(&n).unwrap(); // should only be called for nodes on path to exit
        let (children, non_capping_backedges) = self.children_backedges(n);
        let child_results: Vec<_> = children.iter().map(|c| self.traverse(st, c.node_index())).collect();
        let mut min_dfs_target: [Option<usize>; 2] = [None, None];
        let mut bs = BracketList::new();
        for (tgt,brs) in child_results {
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
                let capping_edge = UDEdge::CappingBackedge(n, min1dfs);
                bs.push(capping_edge.clone());
                // mark capping edge to be removed when we return out to the other end
                match st.capping_edges.get_mut(&min1dfs) {
                    Some(v) => v.push(capping_edge),
                    None => {
                        st.capping_edges.insert(min1dfs, vec![capping_edge]);
                    }
                };
            }
        }
        
        // remove edges to here - these are <backedges> to nodes with greater dfs_num
        // add backedges from here - i.e. any <backedges> to nodes with lower dfs_num
        for e in non_capping_backedges.iter() {
            let target_dfs = *self.dfs_num.get(&e.node_index()).unwrap();
            let b = UDEdge::CFGEdge(n, e.clone());
            if target_dfs < n_dfs {
                // add backedges from here to higher up
                bs.push(b);
                min_dfs_target[0] = Some(match min_dfs_target[0] {
                    None => target_dfs,
                    Some(d) => min(target_dfs, d)
                });
            } else if target_dfs > n_dfs {
                // Remove backedges to here from lower down, including capping
                bs.delete(&b, &mut st.deleted_backedges);
            }
        }
        // also remove any capping backedges
        for e in st.capping_edges.remove(&n_dfs).into_iter().flat_map(|v|v.into_iter()) {
            bs.delete(&e, &mut st.deleted_backedges);
        }
        let parent_edge = self.dfs_parents.get(&n).unwrap();
        st.edge_classes.insert(UDEdge::CFGEdge(n, parent_edge.clone()), bs.tag(&st.deleted_backedges).unwrap());
        (min_dfs_target[0].unwrap(), bs)
    }
    
}
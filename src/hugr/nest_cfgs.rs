use std::collections::{HashMap, HashSet, LinkedList};
use std::hash::Hash;

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
    type Iterator<'c>: Iterator<Item = T>
    where
        Self: 'c;
    fn successors<'c>(&'c self, item: T) -> Self::Iterator<'c>;
    fn predecessors<'c>(&'c self, item: T) -> Self::Iterator<'c>;
}

// The next enum + few functions allow to abstract over the edge directions
// in a CfgView.

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

fn undirected_edges<'a, T: Copy + Clone + PartialEq + Eq + Hash + 'a>(
    cfg: &'a impl CfgView<T>,
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

fn cfg_edge<T: Copy + Clone + PartialEq + Eq + Hash>(s: T, d: EdgeDest<T>) -> (T, T) {
    match d {
        EdgeDest::Forward(t) => (s, t),
        EdgeDest::Backward(t) => (t, s),
    }
}

/// Records an undirected Depth First Search over a CfgView,
///   restricted to nodes forwards-reachable from the entry.
/// That is, the DFS traversal goes both ways along the edges of the CFG.
/// *Undirected* DFS classifies all edges into *only two* categories
///   * tree edges, which on their own (with the nodes) form a tree (minimum spanning tree);
///   * backedges, i.e. those for which, when DFS tried to traverse them, the other endpoint was an ancestor
/// Moreover, we record *which way* along the underlying CFG edge we went.
struct UndirectedDFSTree<T: Copy + Clone + PartialEq + Eq + Hash> {
    /// Pre-order traversal numbering
    dfs_num: HashMap<T, usize>,
    /// For each node, the edge along which it was reached from its parent
    dfs_parents: HashMap<T, EdgeDest<T>>,
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> UndirectedDFSTree<T> {
    pub fn new(cfg: &impl CfgView<T>) -> Self {
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

/// Mutable state updated during traversal of the UndirectedDFSTree by the cycle equivalence algorithm.
struct TraversalState<T> {
    deleted_backedges: HashSet<UDEdge<T>>, // Allows constant-time deletion
    capping_edges: HashMap<usize, Vec<CappingEdge<T>>>, // Indexed by DFS num
    edge_classes: HashMap<(T, T), Option<(UDEdge<T>, usize)>>, // Accumulates result (never overwritten)
}

/// Computes equivalence class of each edge, i.e. two edges with the same value
/// are cycle-equivalent.
pub fn get_edge_classes<T: Copy + Clone + PartialEq + Eq + Hash>(
    cfg: &impl CfgView<T>,
) -> HashMap<(T, T), usize> {
    let tree = UndirectedDFSTree::new(cfg);
    let mut st = TraversalState {
        deleted_backedges: HashSet::new(),
        capping_edges: HashMap::new(),
        edge_classes: HashMap::new(),
    };
    traverse(cfg, &tree, &mut st, cfg.entry_node());
    assert!(st.capping_edges.is_empty());
    st.edge_classes.remove(&(cfg.exit_node(), cfg.entry_node()));
    let mut cycle_class_idxs = HashMap::new();
    st.edge_classes
        .into_iter()
        .map(|(k, v)| {
            let l = cycle_class_idxs.len();
            (k, *cycle_class_idxs.entry(v).or_insert(l))
        })
        .collect()
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CappingEdge<T> {
    /// Node which is the root of the split in the DFS tree
    common_parent: T,
    /// Lowest number (highest ancestor)
    /// to which backedges reached from >1 subtree of the split
    dfs_target: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum UDEdge<T> {
    RealEdge((T, T)),
    FakeEdge(CappingEdge<T>),
}

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

    pub fn tag(&mut self, deleted: &HashSet<UDEdge<T>>) -> Option<(UDEdge<T>, usize)> {
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
    cfg: &impl CfgView<T>,
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
        let e = UDEdge::RealEdge(cfg_edge(n, e));
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
        .for_each(|(_, e)| bs.push(UDEdge::RealEdge(cfg_edge(n, *e))));

    // Now calculate edge classes
    let class = bs.tag(&st.deleted_backedges);
    if let Some((UDEdge::RealEdge(e), 1)) = &class {
        st.edge_classes.insert(e.clone(), class.clone());
    }
    if let Some(parent_edge) = tree.dfs_parents.get(&n) {
        st.edge_classes.insert(cfg_edge(n, *parent_edge), class);
    }
    let highest_target = be_up
        .into_iter()
        .map(|(dfs, _)| dfs)
        .chain(min_dfs_target[0].into_iter());
    (highest_target.min().unwrap_or(usize::MAX), bs)
}

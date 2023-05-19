//! # Nest CFGs
//!
//! Identify Single-Entry-Single-Exit regions in the CFG.
//! These are pairs of edges (a,b) where
//! a dominates b, b postdominates a, and there are no other edges in/out of the nodes inbetween
//! (the third condition is necessary because loop backedges do not affect (post)dominance).
//!
//! Algorithm here: <https://dl.acm.org/doi/10.1145/773473.178258>, approximately:
//! 1. those three conditions are equivalent to:
//! *a and b are cycle-equivalent in the CFG with an extra edge from the exit node to the entry*
//! where cycle-equivalent means every cycle has either both a and b, or neither
//! 2. cycle equivalence is unaffected if all edges are considered *un*directed
//!     (not obvious, see paper for proof)
//! 3. take undirected CFG, perform depth-first traversal
//!     => all edges are either *tree edges*, or *backedges* where one endpoint is an ancestor of the other
//! 4. identify the "bracketlist" of each tree edge - the set of backedges going from a descendant of that edge to an ancestor
//!     -- post-order traversal, merging bracketlists of children,
//!            then delete backedges from below to here, add backedges from here to above
//!     => tree edges with the same bracketlist are cycle-equivalent;
//!        + a tree edge with a single-element bracketlist is cycle-equivalent with that single element
//! 5. this would be expensive (comparing large sets of backedges) - so to optimize,
//!     - the backedge most recently added (at the top) of the bracketlist, plus the size of the bracketlist,
//!       is sufficient to identify the set *when the UDFS tree is linear*;
//!     - when UDFS is treelike, any ancestor with brackets from >1 subtree cannot be cycle-equivalent with any descendant
//!       (as the brackets of said descendant come from beneath it to its ancestors, not from any sibling/etc. in the other subtree).
//!       So, add (onto top of bracketlist) a fake "capping" backedge from here to the highest ancestor reached by >1 subtree.
//!       (Thus, edges from here up to that ancestor, cannot be cycle-equivalent with any edges elsewhere.)

use portgraph::portgraph::Neighbours;
use portgraph::NodeIndex;
use std::collections::{HashMap, HashSet, LinkedList};
use std::hash::Hash;

use crate::ops::handle::{CfgID, NodeHandle};
use crate::ops::{controlflow::BasicBlockOp, OpType};
use crate::Hugr;

// TODO: transform the CFG: each SESE region can be turned into its own Kappa-node
// (in a BB with one predecessor and one successor, which may then be merged
//     and contents parallelized with predecessor or successor).

/// A view of a CFG. `T` is the type of basic block; one interpretation of `T` would be a BasicBlock
/// (e.g. `NodeIndex`) in the Hugr, but this extra level of indirection allows "splitting" one HUGR BB
/// into many (or vice versa). Since SESE regions are bounded by edges between pairs of such `T`, such
/// splitting may allow the algorithm to identify more regions than existed in the underlying CFG
/// (without mutating the underlying CFG perhaps in vain).
pub trait CfgView<T> {
    /// The unique entry node of the CFG. It may any n>=0 of incoming edges; we assume an extra edge in "from outside"
    fn entry_node(&self) -> T;
    /// The unique exit node of the CFG. The only node to have no successors.
    fn exit_node(&self) -> T;
    /// Allows the trait implementor to define a type of iterator it will return from
    /// `successors` and `predecessors`.
    type Iterator<'c>: Iterator<Item = T>
    where
        Self: 'c;
    /// Returns an iterator over the successors of the specified basic block.
    fn successors<'c>(&'c self, node: T) -> Self::Iterator<'c>;
    /// Returns an iterator over the predecessors of the specified basic block.
    fn predecessors<'c>(&'c self, node: T) -> Self::Iterator<'c>;
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

fn all_edges<'a, T: Copy + Clone + PartialEq + Eq + Hash + 'a>(
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

/// A straightforward view of a Cfg as it appears in a Hugr
pub struct SimpleCfgView<'a> {
    h: &'a Hugr,
    entry: NodeIndex,
    exit: NodeIndex,
}
impl<'a> SimpleCfgView<'a> {
    /// Creates a SimpleCfgView for the specified CSG of a Hugr
    pub fn new(h: &'a Hugr, cfg: CfgID) -> Self {
        let mut children = h.children(cfg.node());
        let entry = children.next().unwrap(); // Panic if malformed
        let exit = children.last().unwrap();
        assert!(match h.get_optype(exit) {
            OpType::BasicBlock(BasicBlockOp::Exit { .. }) => true,
            _ => false,
        });
        Self { h, entry, exit }
    }
}
impl CfgView<NodeIndex> for SimpleCfgView<'_> {
    fn entry_node(&self) -> NodeIndex {
        self.entry
    }

    fn exit_node(&self) -> NodeIndex {
        self.exit
    }

    type Iterator<'c> = Neighbours<'c>
    where
        Self: 'c;

    fn successors<'c>(&'c self, node: NodeIndex) -> Self::Iterator<'c> {
        self.h.graph.output_neighbours(node)
    }

    fn predecessors<'c>(&'c self, node: NodeIndex) -> Self::Iterator<'c> {
        self.h.graph.input_neighbours(node)
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
                    for e in all_edges(cfg, n) {
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
    /// Edges we have marked as deleted, allowing constant-time deletion without searching BracketList
    deleted_backedges: HashSet<Bracket<T>>,
    /// Key is DFS num of highest ancestor
    ///   to which backedges reached from >1 sibling subtree;
    /// Value is the LCA i.e. parent of those siblings.
    capping_edges: HashMap<usize, Vec<T>>,
    /// Result of traversal - accumulated here, entries should never be overwritten
    edge_classes: HashMap<(T, T), Option<(Bracket<T>, usize)>>,
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
enum Bracket<T> {
    CfgEdge((T, T)),
    Capping(usize, T),
}

/// Manages a list of brackets. The goal here is to allow constant-time deletion
/// out of the middle of the list - which isn't really possible, so instead we
/// track deleted items (in an external set) and the remaining number (here).
struct BracketList<T: Copy + Clone + PartialEq + Eq + Hash> {
    items: LinkedList<Bracket<T>>,
    size: usize, // deleted items already taken off
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> BracketList<T> {
    pub fn new() -> Self {
        BracketList {
            items: LinkedList::new(),
            size: 0,
        }
    }

    pub fn tag(&mut self, deleted: &HashSet<Bracket<T>>) -> Option<(Bracket<T>, usize)> {
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

    pub fn delete(&mut self, b: &Bracket<T>, deleted: &mut HashSet<Bracket<T>>) {
        // Ideally, here we would also assert that no *other* BracketList contains b.
        debug_assert!(self.items.contains(b)); // Makes operation O(n), otherwise O(1)
        assert!(!deleted.contains(b));
        deleted.insert(b.clone());
        self.size -= 1;
    }

    pub fn push(&mut self, e: Bracket<T>) {
        self.items.push_back(e);
        self.size += 1;
    }
}

/// Returns the lowest DFS num (highest ancestor) reached by any bracket leaving
/// the subtree, and the list of said brackets.
fn traverse<T: Copy + Clone + PartialEq + Eq + Hash>(
    cfg: &impl CfgView<T>,
    tree: &UndirectedDFSTree<T>,
    st: &mut TraversalState<T>,
    n: T,
) -> (usize, BracketList<T>) {
    let n_dfs = *tree.dfs_num.get(&n).unwrap(); // should only be called for nodes on path to exit
    let (children, non_capping_backedges): (Vec<_>, Vec<_>) = all_edges(cfg, n)
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
            bs.push(Bracket::Capping(min1dfs, n));
            // mark capping edge to be removed when we return out to the other end
            st.capping_edges
                .entry(min1dfs)
                .or_insert(Vec::new())
                .push(n);
        }
    }

    let parent_edge = tree.dfs_parents.get(&n);
    let (be_up, be_down): (Vec<_>, Vec<_>) = non_capping_backedges
        .into_iter()
        .map(|e| (*tree.dfs_num.get(&e.target()).unwrap(), e))
        .partition(|(dfs, _)| *dfs < n_dfs);

    // Remove edges to here from beneath
    for (_, e) in be_down {
        let e = Bracket::CfgEdge(cfg_edge(n, e));
        bs.delete(&e, &mut st.deleted_backedges);
    }
    // And capping backedges
    for src in st.capping_edges.remove(&n_dfs).unwrap_or(Vec::new()) {
        bs.delete(&Bracket::Capping(n_dfs, src), &mut st.deleted_backedges)
    }

    // Add backedges from here to ancestors (not the parent edge, but perhaps other edges to the same node)
    be_up
        .iter()
        .filter(|(_, e)| Some(e) != parent_edge)
        .for_each(|(_, e)| bs.push(Bracket::CfgEdge(cfg_edge(n, *e))));

    // Now calculate edge classes
    let class = bs.tag(&st.deleted_backedges);
    if let Some((Bracket::CfgEdge(e), 1)) = &class {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::builder::{BuildError, CFGBuilder, Container, Dataflow, ModuleBuilder};
    use crate::ops::{
        handle::{BasicBlockID, CfgID, ConstID, NodeHandle},
        ConstValue,
    };
    use crate::types::{ClassicType, Signature, SimpleType};
    use crate::{type_row, Hugr};
    use itertools::Itertools;
    //use crate::hugr::nest_cfgs::get_edge_classes;
    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());

    fn group_by<E: Eq + Hash + Ord, V: Eq + Hash>(h: HashMap<E, V>) -> HashSet<Vec<E>> {
        let mut res = HashMap::new();
        for (k, v) in h.into_iter() {
            res.entry(v).or_insert_with(Vec::new).push(k);
        }
        res.into_values().map(sorted).collect()
    }

    fn sorted<E: Ord>(items: impl IntoIterator<Item = E>) -> Vec<E> {
        let mut v: Vec<_> = items.into_iter().collect();
        v.sort();
        v
    }

    #[test]
    fn test_cond_in_loop_separate_headers() -> Result<(), BuildError> {
        let (h, cfg_id, head, tail) = conditional_in_loop(true)?;
        let head = head.node();
        let tail = tail.node();
        //                       /-> left  -\
        //  entry -> head -> split           > merge -> tail -> exit
        //             |          \-> right -/             |
        //             \---<---<---<---<---<---<---<---<---/
        let v = SimpleCfgView::new(&h, cfg_id);
        let edge_classes = get_edge_classes(&v);
        let SimpleCfgView { h: _, entry, exit } = v;
        // split is unique successor of head
        let split = *edge_classes
            .keys()
            .filter(|(s, _)| *s == head)
            .map(|(_, t)| t)
            .exactly_one()
            .unwrap();
        // merge is unique predecessor of tail
        let merge = *edge_classes
            .keys()
            .filter(|(_, t)| *t == tail)
            .map(|(s, _)| s)
            .exactly_one()
            .unwrap();
        let [&left,&right] = edge_classes.keys().filter(|(s,_)| *s == split).map(|(_,t)|t).collect::<Vec<_>>()[..] else {panic!("Split should have two successors");};
        let classes = group_by(edge_classes);
        assert_eq!(
            classes,
            HashSet::from([
                sorted([(split, left), (left, merge)]), // Region containing single BB 'left'
                sorted([(split, right), (right, merge)]), // Region containing single BB 'right'
                sorted([(head, split), (merge, tail)]), // "Conditional" region containing split+merge choosing between left/right
                sorted([(entry, head), (tail, exit)]), // "Loop" region containing body (conditional) + back-edge
                Vec::from([(tail, head)])              // The loop back-edge
            ])
        );
        Ok(())
    }

    fn n_identity<T: Dataflow>(
        mut dataflow_builder: T,
        unit_const: &ConstID,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        let u = dataflow_builder.load_const(unit_const)?;
        dataflow_builder.finish_with_outputs([u].into_iter().chain(w))
    }

    fn branch_block(
        cfg: &mut CFGBuilder,
        const_pred: &ConstID,
    ) -> Result<BasicBlockID, BuildError> {
        let mut bldr = cfg.simple_block_builder(type_row![NAT], type_row![NAT], 2)?;
        let c = bldr.load_const(const_pred)?;
        let [inw] = bldr.input_wires_arr();
        bldr.finish_with_outputs(c, [inw])
    }

    fn build_if_then_else(
        cfg: &mut CFGBuilder,
        const_pred: &ConstID,
        unit_const: &ConstID,
        merge: BasicBlockID,
    ) -> Result<BasicBlockID, BuildError> {
        let entry = branch_block(cfg, const_pred)?;
        let left = n_identity(
            cfg.simple_block_builder(type_row![NAT], type_row![NAT], 1)?,
            unit_const,
        )?;
        let right = n_identity(
            cfg.simple_block_builder(type_row![NAT], type_row![NAT], 1)?,
            unit_const,
        )?;
        cfg.branch(&entry, 0, &left)?;
        cfg.branch(&entry, 1, &right)?;
        cfg.branch(&left, 0, &merge)?;
        cfg.branch(&right, 0, &merge)?;
        Ok(entry)
    }

    fn build_if_then_else_merge(
        cfg: &mut CFGBuilder,
        const_pred: &ConstID,
        unit_const: &ConstID,
    ) -> Result<(BasicBlockID, BasicBlockID), BuildError> {
        let exit = n_identity(
            cfg.simple_block_builder(type_row![NAT], type_row![NAT], 1)?,
            unit_const,
        )?;
        let head = build_if_then_else(cfg, const_pred, unit_const, exit)?;
        Ok((head, exit))
    }

    // Result is header (new or provided) and tail. Caller must provide 0th successor of header and tail,
    // and give tail at least one predecessor.
    fn build_loop(
        cfg: &mut CFGBuilder,
        const_pred: &ConstID,
        unit_const: &ConstID,
        body_in: Option<BasicBlockID>,
    ) -> Result<(BasicBlockID, BasicBlockID), BuildError> {
        let header = match body_in {
            Some(i) => i,
            None => {
                // Caller responsible for giving this node a successor
                n_identity(
                    cfg.simple_block_builder(type_row![NAT], type_row![NAT], 1)?,
                    unit_const,
                )?
            }
        };
        let tail = branch_block(cfg, const_pred)?;
        cfg.branch(&tail, 1, &header)?;
        Ok((header, tail))
    }

    // Build a CFG, returning the Hu
    fn conditional_in_loop(
        separate_headers: bool,
    ) -> Result<(Hugr, CfgID, BasicBlockID, BasicBlockID), BuildError> {
        //let sum2_type = SimpleType::new_predicate(2);

        let mut module_builder = ModuleBuilder::new();
        let main = module_builder.declare("main", Signature::new_df(vec![NAT], type_row![NAT]))?;
        let pred_const = module_builder.constant(ConstValue::simple_predicate(0, 2))?; // Nothing here cares which
        let const_unit = module_builder.constant(ConstValue::simple_unary_predicate())?;

        let mut func_builder = module_builder.define_function(&main)?;
        let [int] = func_builder.input_wires_arr();

        let mut cfg_builder = func_builder.cfg_builder(vec![(NAT, int)], type_row![NAT])?;
        let entry = n_identity(
            cfg_builder.simple_entry_builder(type_row![NAT], 1)?,
            &const_unit,
        )?;
        let (split, merge) = build_if_then_else_merge(&mut cfg_builder, &pred_const, &const_unit)?;

        let (head, tail) = if separate_headers {
            let (head, tail) = build_loop(&mut cfg_builder, &pred_const, &const_unit, None)?;
            cfg_builder.branch(&head, 0, &split)?;
            (head, tail)
        } else {
            // Combine loop header with split.
            build_loop(&mut cfg_builder, &pred_const, &const_unit, Some(split))?
        };
        cfg_builder.branch(&merge, 0, &tail)?;

        let exit = cfg_builder.exit_block();

        cfg_builder.branch(&entry, 0, &head)?;
        cfg_builder.branch(&tail, 0, &exit)?;

        let cfg_id = cfg_builder.finish();

        func_builder.finish_with_outputs(cfg_id.outputs())?;

        let h = module_builder.finish()?;

        Ok((h, *cfg_id.handle(), head, tail))
    }
}

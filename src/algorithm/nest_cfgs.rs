//! # Nest CFGs
//!
//! Identify Single-Entry-Single-Exit (SESE) regions in the CFG.
//! These are pairs of edges (a,b) where
//! * a dominates b
//! * b postdominates a
//! * there are no other edges in/out of the nodes inbetween
//!  (this last condition is necessary because loop backedges do not affect (post)dominance).
//!
//! # Algorithm
//! See paper: <https://dl.acm.org/doi/10.1145/773473.178258>, approximately:
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
//!
//! # Restrictions
//! * The paper assumes that all CFG nodes are on paths from entry to exit, i.e. no loops without exits.
//! HUGR assumes only that they are all reachable from entry, so we do a backward traversal from exit node
//! first and restrict to the CFG nodes in the reachable set. (This means we will not discover SESE regions
//! in exit-free loops, but that doesn't seem a major concern.)
//! * Multiple edges in the same direction between the same BBs will "confuse" the algorithm in the paper.
//! However it is straightforward for us to treat successors and predecessors as sets. (Two edges between
//! the same BBs but in opposite directions must be distinct!)

use std::collections::{HashMap, HashSet, LinkedList, VecDeque};
use std::hash::Hash;

use itertools::Itertools;

use crate::hugr::views::HugrView;
use crate::ops::OpTag;
use crate::ops::OpTrait;
use crate::{Direction, Node};

// TODO: transform the CFG: each SESE region can be turned into its own Kappa-node
// (in a BB with one predecessor and one successor, which may then be merged
//     and contents parallelized with predecessor or successor).

/// A view of a CFG. `T` is the type of basic block; one interpretation of `T` would be a BasicBlock
/// (e.g. `Node`) in the Hugr, but this extra level of indirection allows "splitting" one HUGR BB
/// into many (or vice versa). Since SESE regions are bounded by edges between pairs of such `T`, such
/// splitting may allow the algorithm to identify more regions than existed in the underlying CFG
/// (without mutating the underlying CFG perhaps in vain).
pub trait CfgView<T> {
    /// The unique entry node of the CFG. It may any n>=0 of incoming edges; we assume control arrives here from "outside".
    fn entry_node(&self) -> T;
    /// The unique exit node of the CFG. The only node to have no successors.
    fn exit_node(&self) -> T;
    /// Allows the trait implementor to define a type of iterator it will return from
    /// `successors` and `predecessors`.
    type Iterator<'c>: Iterator<Item = T>
    where
        Self: 'c;
    /// Returns an iterator over the successors of the specified basic block.
    fn successors(&self, node: T) -> Self::Iterator<'_>;
    /// Returns an iterator over the predecessors of the specified basic block.
    fn predecessors(&self, node: T) -> Self::Iterator<'_>;
}

/// Directed edges in a Cfg - i.e. along which control flows from first to second only.
type CfgEdge<T> = (T, T);

// The next enum + few functions allow to abstract over the edge directions
// in a CfgView.

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum EdgeDest<T> {
    Forward(T),
    Backward(T),
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> EdgeDest<T> {
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
    let extra = if n == cfg.exit_node() {
        vec![cfg.entry_node()]
    } else {
        vec![]
    };
    cfg.successors(n)
        .chain(extra)
        .map(EdgeDest::Forward)
        .chain(cfg.predecessors(n).map(EdgeDest::Backward))
        .unique()
}

fn flip<T: Copy + Clone + PartialEq + Eq + Hash>(src: T, d: EdgeDest<T>) -> (T, EdgeDest<T>) {
    match d {
        EdgeDest::Forward(tgt) => (tgt, EdgeDest::Backward(src)),
        EdgeDest::Backward(tgt) => (tgt, EdgeDest::Forward(src)),
    }
}

fn cfg_edge<T: Copy + Clone + PartialEq + Eq + Hash>(s: T, d: EdgeDest<T>) -> CfgEdge<T> {
    match d {
        EdgeDest::Forward(t) => (s, t),
        EdgeDest::Backward(t) => (t, s),
    }
}

/// A straightforward view of a Cfg as it appears in a Hugr
pub struct SimpleCfgView<'a, H> {
    h: &'a H,
    entry: Node,
    exit: Node,
}
impl<'a, H: HugrView> SimpleCfgView<'a, H> {
    /// Creates a SimpleCfgView for the specified CSG of a Hugr
    pub fn new(h: &'a H) -> Self {
        let mut children = h.children(h.root());
        let entry = children.next().unwrap(); // Panic if malformed
        let exit = children.next().unwrap();
        debug_assert_eq!(h.get_optype(exit).tag(), OpTag::BasicBlockExit);
        Self { h, entry, exit }
    }
}
impl<H: HugrView> CfgView<Node> for SimpleCfgView<'_, H> {
    fn entry_node(&self) -> Node {
        self.entry
    }

    fn exit_node(&self) -> Node {
        self.exit
    }

    type Iterator<'c> = H::Neighbours<'c>
    where
        Self: 'c;

    fn successors(&self, node: Node) -> Self::Iterator<'_> {
        self.h.neighbours(node, Direction::Outgoing)
    }

    fn predecessors(&self, node: Node) -> Self::Iterator<'_> {
        self.h.neighbours(node, Direction::Incoming)
    }
}

/// Records an undirected Depth First Search over a CfgView,
///   restricted to nodes forwards-reachable from the entry.
/// That is, the DFS traversal goes both ways along the edges of the CFG.
/// *Undirected* DFS classifies all edges into *only two* categories
///   * tree edges, which on their own (with the nodes) form a tree (minimum spanning tree);
///   * backedges, i.e. those for which, when DFS tried to traverse them, the other endpoint was an ancestor
/// Moreover, we record *which way* along the underlying CFG edge we went.
struct UndirectedDFSTree<T> {
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
            let mut pending = VecDeque::new();
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

#[derive(Clone, PartialEq, Eq, Hash)]
enum Bracket<T> {
    Real(CfgEdge<T>),
    Capping(usize, T),
}

/// Manages a list of brackets. The goal here is to allow constant-time deletion
/// out of the middle of the list - which isn't really possible, so instead we
/// track deleted items (in an external set) and the remaining number (here).
///
/// Note - we could put the items deleted from *this* BracketList here, and merge in concat().
/// That would be cleaner, but repeated set-merging would be slower than adding the
/// deleted items to a single set in the `TraversalState`
struct BracketList<T> {
    items: LinkedList<Bracket<T>>, // Allows O(1) `append` of two lists.
    size: usize,                   // Not counting deleted items.
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
                //deleted.remove(e); // Would only save memory, so keep as immutable
            } else {
                return Some((e.clone(), self.size));
            }
        }
        None
    }

    pub fn concat(&mut self, other: BracketList<T>) {
        let BracketList { mut items, size } = other;
        self.items.append(&mut items);
        assert!(items.is_empty());
        self.size += size;
    }

    pub fn delete(&mut self, b: &Bracket<T>, deleted: &mut HashSet<Bracket<T>>) {
        // Ideally, here we would also assert that no *other* BracketList contains b.
        debug_assert!(self.items.contains(b)); // Makes operation O(n), otherwise O(1)
        let was_new = deleted.insert(b.clone());
        assert!(was_new);
        self.size -= 1;
    }

    pub fn push(&mut self, e: Bracket<T>) {
        self.items.push_back(e);
        self.size += 1;
    }
}

/// Mutable state updated during traversal of the UndirectedDFSTree by the cycle equivalence algorithm.
pub struct EdgeClassifier<T> {
    /// Edges we have marked as deleted, allowing constant-time deletion without searching BracketList
    deleted_backedges: HashSet<Bracket<T>>,
    /// Key is DFS num of highest ancestor
    ///   to which backedges reached from >1 sibling subtree;
    /// Value is the LCA i.e. parent of those siblings.
    capping_edges: HashMap<usize, Vec<T>>,
    /// Result of traversal - accumulated here, entries should never be overwritten
    edge_classes: HashMap<CfgEdge<T>, Option<(Bracket<T>, usize)>>,
}

impl<T: Copy + Clone + PartialEq + Eq + Hash> EdgeClassifier<T> {
    /// Computes equivalence class of each edge, i.e. two edges with the same value
    /// are cycle-equivalent. Any two consecutive edges in the same class define a SESE region
    /// (where "consecutive" means on any path in the original directed CFG, as the edges
    /// in a class all dominate + postdominate each other as part of defn of cycle equivalence).
    pub fn get_edge_classes(cfg: &impl CfgView<T>) -> HashMap<CfgEdge<T>, usize> {
        let tree = UndirectedDFSTree::new(cfg);
        let mut s = Self {
            deleted_backedges: HashSet::new(),
            capping_edges: HashMap::new(),
            edge_classes: HashMap::new(),
        };
        s.traverse(cfg, &tree, cfg.entry_node());
        assert!(s.capping_edges.is_empty());
        s.edge_classes.remove(&(cfg.exit_node(), cfg.entry_node()));
        let mut cycle_class_idxs = HashMap::new();
        s.edge_classes
            .into_iter()
            .map(|(k, v)| {
                let l = cycle_class_idxs.len();
                (k, *cycle_class_idxs.entry(v).or_insert(l))
            })
            .collect()
    }

    /// Returns the lowest DFS num (highest ancestor) reached by any bracket leaving
    /// the subtree, and the list of said brackets.
    fn traverse(
        &mut self,
        cfg: &impl CfgView<T>,
        tree: &UndirectedDFSTree<T>,
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
            .map(|c| self.traverse(cfg, tree, c.target()))
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
                self.capping_edges
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
            let e = cfg_edge(n, e);
            let b = Bracket::Real(e);
            bs.delete(&b, &mut self.deleted_backedges);
            // Last chance to assign an edge class! This will be a singleton class,
            // but assign for consistency with other singletons.
            self.edge_classes.entry(e).or_insert_with(|| Some((b, 0)));
        }
        // And capping backedges
        for src in self.capping_edges.remove(&n_dfs).unwrap_or(Vec::new()) {
            bs.delete(&Bracket::Capping(n_dfs, src), &mut self.deleted_backedges)
        }

        // Add backedges from here to ancestors (not the parent edge, but perhaps other edges to the same node)
        be_up
            .iter()
            .filter(|(_, e)| Some(e) != parent_edge)
            .for_each(|(_, e)| bs.push(Bracket::Real(cfg_edge(n, *e))));

        // Now calculate edge classes
        let class = bs.tag(&self.deleted_backedges);
        if let Some((Bracket::Real(e), 1)) = &class {
            self.edge_classes.insert(*e, class.clone());
        }
        if let Some(parent_edge) = tree.dfs_parents.get(&n) {
            self.edge_classes.insert(cfg_edge(n, *parent_edge), class);
        }
        let highest_target = be_up
            .into_iter()
            .map(|(dfs, _)| dfs)
            .chain(min_dfs_target[0]);
        (highest_target.min().unwrap_or(usize::MAX), bs)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::builder::{BuildError, CFGBuilder, Container, DataflowSubContainer, HugrBuilder};
    use crate::extension::{prelude::USIZE_T, ExtensionSet};

    use crate::hugr::views::{HierarchyView, SiblingGraph};
    use crate::ops::handle::{BasicBlockID, ConstID, NodeHandle};
    use crate::ops::Const;
    use crate::types::{FunctionType, Type};
    use crate::{type_row, Hugr};
    const NAT: Type = USIZE_T;

    pub fn group_by<E: Eq + Hash + Ord, V: Eq + Hash>(h: HashMap<E, V>) -> HashSet<Vec<E>> {
        let mut res = HashMap::new();
        for (k, v) in h.into_iter() {
            res.entry(v).or_insert_with(Vec::new).push(k);
        }
        res.into_values().map(sorted).collect()
    }

    pub fn sorted<E: Ord>(items: impl IntoIterator<Item = E>) -> Vec<E> {
        let mut v: Vec<_> = items.into_iter().collect();
        v.sort();
        v
    }

    #[test]
    fn test_cond_then_loop_separate() -> Result<(), BuildError> {
        //               /-> left --\
        // entry -> split            > merge -> head -> tail -> exit
        //               \-> right -/             \-<--<-/
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;

        let pred_const = cfg_builder.add_constant(Const::simple_predicate(0, 2), None)?; // Nothing here cares which
        let const_unit = cfg_builder.add_constant(Const::simple_unary_predicate(), None)?;

        let entry = n_identity(
            cfg_builder.simple_entry_builder(type_row![NAT], 1, ExtensionSet::new())?,
            &const_unit,
        )?;
        let (split, merge) = build_if_then_else_merge(&mut cfg_builder, &pred_const, &const_unit)?;
        cfg_builder.branch(&entry, 0, &split)?;
        let (head, tail) = build_loop(&mut cfg_builder, &pred_const, &const_unit)?;
        cfg_builder.branch(&head, 0, &tail)?; // trivial "loop body"
        cfg_builder.branch(&merge, 0, &head)?;
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&tail, 0, &exit)?;

        let h = cfg_builder.finish_prelude_hugr()?;

        let (entry, exit) = (entry.node(), exit.node());
        let (split, merge, head, tail) = (split.node(), merge.node(), head.node(), tail.node());
        // There's no need to use a FlatRegionView here but we do so just to check
        // that we *can* (as we'll need to for "real" module Hugr's).
        let v: SiblingGraph = SiblingGraph::new(&h, h.root());
        let edge_classes = EdgeClassifier::get_edge_classes(&SimpleCfgView::new(&v));
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
                Vec::from([(head, tail)]), // Loop body and backedges are in their own classes because
                Vec::from([(tail, head)]), // the path executing the loop exactly once skips the backedge.
                sorted([(entry, split), (merge, head), (tail, exit)]), // Two regions, conditional and then loop.
            ])
        );
        Ok(())
    }

    #[test]
    fn test_cond_then_loop_combined() -> Result<(), BuildError> {
        //      /-> left --\
        // entry            > merge -> tail -> exit
        //      \-> right -/     \-<--<-/
        // Here we would like two consecutive regions, but there is no *edge* between
        // the conditional and the loop to indicate the boundary, so we cannot separate them.
        let (h, merge, tail) = build_cond_then_loop_cfg(false)?;
        let (merge, tail) = (merge.node(), tail.node());
        let [entry, exit]: [Node; 2] = h
            .children(h.root())
            .take(2)
            .collect_vec()
            .try_into()
            .unwrap();

        let edge_classes = EdgeClassifier::get_edge_classes(&SimpleCfgView::new(&h));
        let [&left, &right] = edge_classes
            .keys()
            .filter(|(s, _)| *s == entry)
            .map(|(_, t)| t)
            .collect::<Vec<_>>()[..]
        else {
            panic!("Entry node should have two successors");
        };

        let classes = group_by(edge_classes);
        assert_eq!(
            classes,
            HashSet::from([
                sorted([(entry, left), (left, merge)]), // Region containing single BB 'left'.
                sorted([(entry, right), (right, merge)]), // Region containing single BB 'right'.
                Vec::from([(tail, exit)]), // The only edge in neither conditional nor loop.
                Vec::from([(merge, tail)]), // Loop body (at least once per execution).
                Vec::from([(tail, merge)]), // Loop backedge (0 or more times per execution).
            ])
        );
        Ok(())
    }

    #[test]
    fn test_cond_in_loop_separate_headers() -> Result<(), BuildError> {
        let (h, head, tail) = build_conditional_in_loop_cfg(true)?;
        let head = head.node();
        let tail = tail.node();
        //                        /-> left --\
        //  entry -> head -> split            > merge -> tail -> exit
        //             |          \-> right -/             |
        //             \---<---<---<---<---<---<---<---<---/
        // split is unique successor of head
        let split = h.output_neighbours(head).exactly_one().unwrap();
        // merge is unique predecessor of tail
        let merge = h.input_neighbours(tail).exactly_one().unwrap();

        let v = SimpleCfgView::new(&h);
        let edge_classes = EdgeClassifier::get_edge_classes(&v);
        let SimpleCfgView { h: _, entry, exit } = v;
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
                sorted([(split, left), (left, merge)]), // Region containing single BB 'left'
                sorted([(split, right), (right, merge)]), // Region containing single BB 'right'
                sorted([(head, split), (merge, tail)]), // "Conditional" region containing split+merge choosing between left/right
                sorted([(entry, head), (tail, exit)]), // "Loop" region containing body (conditional) + back-edge
                Vec::from([(tail, head)])              // The loop back-edge
            ])
        );
        Ok(())
    }

    #[test]
    fn test_cond_in_loop_combined_headers() -> Result<(), BuildError> {
        let (h, head, tail) = build_conditional_in_loop_cfg(false)?;
        let head = head.node();
        let tail = tail.node();
        //               /-> left --\
        //  entry -> head            > merge -> tail -> exit
        //            |  \-> right -/             |
        //             \---<---<---<---<---<--<---/
        // Here we would like an indication that we can make two nested regions,
        // but there is no edge to act as entry to a region containing just the conditional :-(.
        let v = SimpleCfgView::new(&h);
        let edge_classes = EdgeClassifier::get_edge_classes(&v);
        let SimpleCfgView { h: _, entry, exit } = v;
        // merge is unique predecessor of tail
        let merge = *edge_classes
            .keys()
            .filter(|(_, t)| *t == tail)
            .map(|(s, _)| s)
            .exactly_one()
            .unwrap();
        let [&left, &right] = edge_classes
            .keys()
            .filter(|(s, _)| *s == head)
            .map(|(_, t)| t)
            .collect::<Vec<_>>()[..]
        else {
            panic!("Loop header should have two successors");
        };
        let classes = group_by(edge_classes);
        assert_eq!(
            classes,
            HashSet::from([
                sorted([(head, left), (left, merge)]), // Region containing single BB 'left'
                sorted([(head, right), (right, merge)]), // Region containing single BB 'right'
                Vec::from([(merge, tail)]), // The edge "in the loop", but no other edge in its class to define SESE region
                sorted([(entry, head), (tail, exit)]), // "Loop" region containing body (conditional) + back-edge
                Vec::from([(tail, head)])              // The loop back-edge
            ])
        );
        Ok(())
    }

    fn n_identity<T: DataflowSubContainer>(
        mut dataflow_builder: T,
        pred_const: &ConstID,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        let u = dataflow_builder.load_const(pred_const)?;
        dataflow_builder.finish_with_outputs([u].into_iter().chain(w))
    }

    fn build_if_then_else_merge<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg: &mut CFGBuilder<T>,
        const_pred: &ConstID,
        unit_const: &ConstID,
    ) -> Result<(BasicBlockID, BasicBlockID), BuildError> {
        let split = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 2)?,
            const_pred,
        )?;
        let merge = build_then_else_merge_from_if(cfg, unit_const, split)?;
        Ok((split, merge))
    }

    fn build_then_else_merge_from_if<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg: &mut CFGBuilder<T>,
        unit_const: &ConstID,
        split: BasicBlockID,
    ) -> Result<BasicBlockID, BuildError> {
        let merge = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?,
            unit_const,
        )?;
        let left = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?,
            unit_const,
        )?;
        let right = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?,
            unit_const,
        )?;
        cfg.branch(&split, 0, &left)?;
        cfg.branch(&split, 1, &right)?;
        cfg.branch(&left, 0, &merge)?;
        cfg.branch(&right, 0, &merge)?;
        Ok(merge)
    }

    // Returns loop tail - caller must link header to tail, and provide 0th successor of tail
    fn build_loop_from_header<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg: &mut CFGBuilder<T>,
        const_pred: &ConstID,
        header: BasicBlockID,
    ) -> Result<BasicBlockID, BuildError> {
        let tail = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 2)?,
            const_pred,
        )?;
        cfg.branch(&tail, 1, &header)?;
        Ok(tail)
    }

    // Result is header and tail. Caller must provide 0th successor of header (linking to tail), and 0th successor of tail.
    fn build_loop<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg: &mut CFGBuilder<T>,
        const_pred: &ConstID,
        unit_const: &ConstID,
    ) -> Result<(BasicBlockID, BasicBlockID), BuildError> {
        let header = n_identity(
            cfg.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?,
            unit_const,
        )?;
        let tail = build_loop_from_header(cfg, const_pred, header)?;
        Ok((header, tail))
    }

    // Result is merge and tail; loop header is (merge, if separate==true; unique successor of merge, if separate==false)
    pub fn build_cond_then_loop_cfg(
        separate: bool,
    ) -> Result<(Hugr, BasicBlockID, BasicBlockID), BuildError> {
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
        let pred_const = cfg_builder.add_constant(Const::simple_predicate(0, 2), None)?; // Nothing here cares which
        let const_unit = cfg_builder.add_constant(Const::simple_unary_predicate(), None)?;

        let entry = n_identity(
            cfg_builder.simple_entry_builder(type_row![NAT], 2, ExtensionSet::new())?,
            &pred_const,
        )?;
        let merge = build_then_else_merge_from_if(&mut cfg_builder, &const_unit, entry)?;
        let head = if separate {
            let h = n_identity(
                cfg_builder
                    .simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?,
                &const_unit,
            )?;
            cfg_builder.branch(&merge, 0, &h)?;
            h
        } else {
            merge
        };
        let tail = build_loop_from_header(&mut cfg_builder, &pred_const, head)?;
        cfg_builder.branch(&head, 0, &tail)?; // trivial "loop body"
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&tail, 0, &exit)?;

        let h = cfg_builder.finish_prelude_hugr()?;
        Ok((h, merge, tail))
    }

    // Build a CFG, returning the Hugr
    pub fn build_conditional_in_loop_cfg(
        separate_headers: bool,
    ) -> Result<(Hugr, BasicBlockID, BasicBlockID), BuildError> {
        //let sum2_type = Type::new_predicate(2);

        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;

        let pred_const = cfg_builder.add_constant(Const::simple_predicate(0, 2), None)?; // Nothing here cares which
        let const_unit = cfg_builder.add_constant(Const::simple_unary_predicate(), None)?;

        let entry = n_identity(
            cfg_builder.simple_entry_builder(type_row![NAT], 1, ExtensionSet::new())?,
            &const_unit,
        )?;
        let (split, merge) = build_if_then_else_merge(&mut cfg_builder, &pred_const, &const_unit)?;

        let (head, tail) = if separate_headers {
            let (head, tail) = build_loop(&mut cfg_builder, &pred_const, &const_unit)?;
            cfg_builder.branch(&head, 0, &split)?;
            (head, tail)
        } else {
            // Combine loop header with split.
            let tail = build_loop_from_header(&mut cfg_builder, &pred_const, split)?;
            (split, tail)
        };
        cfg_builder.branch(&merge, 0, &tail)?;

        let exit = cfg_builder.exit_block();

        cfg_builder.branch(&entry, 0, &head)?;
        cfg_builder.branch(&tail, 0, &exit)?;

        let h = cfg_builder.finish_prelude_hugr()?;
        Ok((h, head, tail))
    }
}

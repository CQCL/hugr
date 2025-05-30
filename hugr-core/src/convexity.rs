//! This module provides functionality for convexity checking in directed acyclic graphs.
use portgraph::algorithms::ConvexChecker;
use portgraph::{NodeIndex, PortIndex};
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet, VecDeque};

/// Maintains a dynamic topological order for a directed acyclic graph (DAG) using Pearce and Kelly's algorithm.
pub struct DynamicTopoSort {
    order: Vec<NodeIndex>,                             // Current topological order
    node_to_pos: HashMap<NodeIndex, usize>,            // Node ID to position in `order`
    graph: HashMap<NodeIndex, Vec<NodeIndex>>,         // Adjacency list: outgoing edges
    reverse_graph: HashMap<NodeIndex, Vec<NodeIndex>>, // Adjacency list: incoming edges
}

impl DynamicTopoSort {
    /// Creates an empty `DynamicTopoSort` instance.
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            node_to_pos: HashMap::new(),
            graph: HashMap::new(),
            reverse_graph: HashMap::new(),
        }
    }

    /// Adds a new node to the graph, placing it at the end of the order.
    pub fn add_node(&mut self, node: NodeIndex) {
        if self.node_to_pos.contains_key(&node) {
            return; // Node already exists
        }
        let pos = self.order.len();
        self.order.push(node);
        self.node_to_pos.insert(node, pos);
        self.graph.entry(node).or_insert_with(Vec::new);
        self.reverse_graph.entry(node).or_insert_with(Vec::new);
    }

    /// Adds an edge and updates the topological order, returning an error if a cycle is created.
    pub fn connect_nodes(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), &'static str> {
        if !self.node_to_pos.contains_key(&from) || !self.node_to_pos.contains_key(&to) {
            return Err("Node not found in graph");
        }
        if self.graph.get(&from).map_or(false, |v| v.contains(&to)) {
            return Ok(());
        }
        self.graph.entry(from).or_insert_with(Vec::new).push(to);
        self.reverse_graph
            .entry(to)
            .or_insert_with(Vec::new)
            .push(from);
        if self.would_create_cycle(to, from) {
            self.graph.get_mut(&from).unwrap().retain(|&n| n != to);
            self.reverse_graph
                .get_mut(&to)
                .unwrap()
                .retain(|&n| n != from);
            return Err("Adding this edge would create a cycle");
        }
        self.update_order(from, to);
        Ok(())
    }

    /// Removes a node and updates the topological order.
    pub fn remove_node(&mut self, node: NodeIndex) -> Result<(), &'static str> {
        if let Some(pos) = self.node_to_pos.remove(&node) {
            self.order.remove(pos);
            let successors = self.graph.remove(&node).unwrap_or_default();
            let predecessors = self.reverse_graph.get(&node).cloned().unwrap_or_default();
            for s in &successors {
                if let Some(preds) = self.reverse_graph.get_mut(s) {
                    preds.retain(|&p| p != node);
                }
            }
            for p in &predecessors {
                if let Some(neighbors) = self.graph.get_mut(p) {
                    neighbors.retain(|&n| n != node);
                }
            }
            self.reverse_graph.remove(&node);
            for (i, &n) in self.order.iter().enumerate() {
                self.node_to_pos.insert(n, i);
            }
            Ok(())
        } else {
            Err("Node not found in graph")
        }
    }

    /// Removes an edge and updates the topological order if necessary.
    pub fn disconnect_nodes(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), &'static str> {
        if !self.node_to_pos.contains_key(&from) || !self.node_to_pos.contains_key(&to) {
            return Err("Node not found in graph");
        }
        let mut edge_existed = false;
        if let Some(neighbors) = self.graph.get_mut(&from) {
            let len_before = neighbors.len();
            neighbors.retain(|&n| n != to);
            edge_existed = len_before != neighbors.len();
        }
        if let Some(predecessors) = self.reverse_graph.get_mut(&to) {
            predecessors.retain(|&n| n != from);
        }
        if edge_existed {
            Ok(())
        } else {
            Err("Edge not found")
        }
    }

    /// Checks if `start` can reach `target` via existing edges (standard BFS for cycle detection).
    fn would_create_cycle(&self, start: NodeIndex, target: NodeIndex) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if node == target {
                return true;
            }
            if visited.insert(node) {
                if let Some(neighbors) = self.graph.get(&node) {
                    queue.extend(neighbors);
                }
            }
        }
        false
    }

    /// Updates the topological order after adding an edge (from, to).
    fn update_order(&mut self, from: NodeIndex, to: NodeIndex) {
        let from_pos = *self.node_to_pos.get(&from).unwrap();
        let to_pos = *self.node_to_pos.get(&to).unwrap();
        if from_pos < to_pos {
            return;
        }
        let lb = to_pos;
        let ub = from_pos;
        let delta_f = self.future_cone(to, ub);
        let delta_b = self.past_cone(from, lb);
        let mut l = delta_b;
        l.extend(&delta_f);
        let mut new_pos = to_pos;
        for &w in l.iter() {
            *self.node_to_pos.get_mut(&w).unwrap() = new_pos;
            new_pos += 1;
        }
        self.order
            .sort_by_key(|&w| *self.node_to_pos.get(&w).unwrap());
    }

    /// Computes the future cone from a starting node, up to position `ub`.
    fn future_cone(&self, start: NodeIndex, ub: usize) -> Vec<NodeIndex> {
        let mut visited = HashSet::new();
        let mut cone = Vec::new();
        self.dfs_future(start, &mut visited, &mut cone, ub);
        cone
    }

    fn dfs_future(
        &self,
        n: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        cone: &mut Vec<NodeIndex>,
        ub: usize,
    ) {
        if visited.contains(&n) || *self.node_to_pos.get(&n).unwrap() >= ub {
            return;
        }
        visited.insert(n);
        cone.push(n);
        if let Some(neighbors) = self.graph.get(&n) {
            for &neighbor in neighbors {
                self.dfs_future(neighbor, visited, cone, ub);
            }
        }
    }

    /// Computes the past cone from a starting node, above position `lb`.
    fn past_cone(&self, start: NodeIndex, lb: usize) -> Vec<NodeIndex> {
        let mut visited = HashSet::new();
        let mut cone = Vec::new();
        self.dfs_past(start, &mut visited, &mut cone, lb);
        cone
    }

    fn dfs_past(
        &self,
        n: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        cone: &mut Vec<NodeIndex>,
        lb: usize,
    ) {
        if visited.contains(&n) || *self.node_to_pos.get(&n).unwrap() <= lb {
            return;
        }
        visited.insert(n);
        cone.push(n);
        if let Some(predecessors) = self.reverse_graph.get(&n) {
            for &predecessor in predecessors {
                self.dfs_past(predecessor, visited, cone, lb);
            }
        }
    }

    /// Returns the current topological order (for testing or inspection).
    pub fn get_order(&self) -> &Vec<NodeIndex> {
        &self.order
    }

    fn has_path_outside_subgraph(
        &self,
        start: NodeIndex,
        end: NodeIndex,
        min_pos: usize,
        max_pos: usize,
        nodes: &HashSet<NodeIndex>,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if !nodes.contains(&node) && node != start {
                // Exclude start since it's in the subgraph
                return true; // Found an external node on the path
            }
            if node == end {
                continue; // Reached end, but check all paths
            }
            if visited.insert(node) {
                if let Some(neighbors) = self.graph.get(&node) {
                    for &n in neighbors {
                        let n_pos = *self.node_to_pos.get(&n).unwrap_or(&usize::MAX);
                        if n_pos >= min_pos && n_pos <= max_pos {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }
        false // No path through external nodes found
    }
    /// Placeholder: Maps a port to its node.
    fn port_to_node(&self, _port: PortIndex) -> Option<NodeIndex> {
        None 
    }

    /// Checks if `start` can reach any input within position bounds, going outside `nodes`.
    fn can_reach_inputs(
        &self,
        start: NodeIndex,
        min_pos: usize,
        max_pos: usize,
        nodes: &HashSet<NodeIndex>,
        inputs: &[PortIndex],
    ) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if !nodes.contains(&node) {
                let pos = *self.node_to_pos.get(&node).unwrap_or(&usize::MAX);
                if pos < min_pos || pos > max_pos {
                    continue;
                }
                for &input in inputs {
                    if self.port_to_node(input) == Some(node) {
                        return true;
                    }
                }
            }
            if visited.insert(node) {
                if let Some(neighbors) = self.graph.get(&node) {
                    for &n in neighbors {
                        let n_pos = *self.node_to_pos.get(&n).unwrap_or(&usize::MAX);
                        if n_pos >= min_pos && n_pos <= max_pos {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }
        false
    }
}

/// A dynamic topological convex checker for portgraphs.
pub struct DynamicTopoConvexChecker {
    topo_sort: DynamicTopoSort,
}

impl DynamicTopoConvexChecker {
    /// Creates a new `DynamicTopoConvexChecker` instance.
    pub fn new() -> Self {
        Self {
            topo_sort: DynamicTopoSort::new(),
        }
    }

    /// Adds a new node to the graph.
    pub fn add_node(&mut self, node: NodeIndex) {
        self.topo_sort.add_node(node);
    }

    /// Removes a node from the graph.
    pub fn remove_node(&mut self, node: NodeIndex) -> Result<(), &'static str> {
        self.topo_sort.remove_node(node)
    }

    /// Connects two nodes with an edge.
    pub fn connect_nodes(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), &'static str> {
        self.topo_sort.connect_nodes(from, to)
    }

    /// Disconnects an edge between two nodes.
    pub fn disconnect_nodes(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), &'static str> {
        self.topo_sort.disconnect_nodes(from, to)
    }
}

impl ConvexChecker for DynamicTopoConvexChecker {
    fn is_convex(
        &self,
        nodes: impl IntoIterator<Item = NodeIndex>,
        inputs: impl IntoIterator<Item = PortIndex>,
        outputs: impl IntoIterator<Item = PortIndex>,
    ) -> bool {
        let nodes: HashSet<NodeIndex> = nodes.into_iter().collect();
        for &node in &nodes {
            if !self.topo_sort.node_to_pos.contains_key(&node) {
                return false; // Node not in graph
            }
        }
        if nodes.len() < 2 {
            return true; // Trivially convex
        }
        let inputs: Vec<PortIndex> = inputs.into_iter().collect();
        let outputs: Vec<PortIndex> = outputs.into_iter().collect();
        let mut min_pos = usize::MAX;
        let mut max_pos = 0;

        // Compute min and max positions, validate nodes
        for &node in &nodes {
            if let Some(&pos) = self.topo_sort.node_to_pos.get(&node) {
                min_pos = min(min_pos, pos);
                max_pos = max(max_pos, pos);
            } else {
                return false; // Node not in graph
            }
        }

        // If inputs/outputs are empty, check all pairs in the subgraph
        if inputs.is_empty() && outputs.is_empty() {
            for &start in &nodes {
                for &end in &nodes {
                    if start == end {
                        continue;
                    }
                    let start_pos = *self.topo_sort.node_to_pos.get(&start).unwrap();
                    let end_pos = *self.topo_sort.node_to_pos.get(&end).unwrap();
                    if start_pos < end_pos {
                        // Only check forward paths
                        if self
                            .topo_sort
                            .has_path_outside_subgraph(start, end, min_pos, max_pos, &nodes)
                        {
                            return false; // Path exists through an external node
                        }
                    }
                }
            }
            return true;
        }

        // If ports are provided, use original logic
        for &output in &outputs {
            if let Some(output_node) = self.topo_sort.port_to_node(output) {
                if !nodes.contains(&output_node) {
                    continue;
                }
                if self
                    .topo_sort
                    .can_reach_inputs(output_node, min_pos, max_pos, &nodes, &inputs)
                {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use portgraph::NodeIndex;
    use portgraph::algorithms::ConvexChecker;
    use std::collections::HashSet;

    fn is_valid_topological_order(topo: &DynamicTopoSort) -> bool {
        for (&node, neighbors) in &topo.graph {
            let node_pos = *topo.node_to_pos.get(&node).unwrap();
            for &neighbor in neighbors {
                let neighbor_pos = *topo.node_to_pos.get(&neighbor).unwrap();
                if node_pos >= neighbor_pos {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_basic_linear_order() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n1, n2]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_reordering() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n1, n0).unwrap();
        assert_eq!(topo.get_order(), &vec![n1, n0, n2]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_indirect_cycle() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.add_node(n3);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.connect_nodes(n2, n3).unwrap();
        match topo.connect_nodes(n3, n0) {
            Ok(()) => panic!("Should have detected cycle"),
            Err(e) => assert_eq!(e, "Adding this edge would create a cycle"),
        }
        assert_eq!(topo.get_order(), &vec![n0, n1, n2, n3]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_multiple_paths() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.add_node(n3);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n0, n2).unwrap();
        topo.connect_nodes(n1, n3).unwrap();
        topo.connect_nodes(n2, n3).unwrap();
        let order = topo.get_order();
        assert_eq!(order[0], n0);
        assert!(order[1] == n1 || order[1] == n2);
        assert!(order[2] == if order[1] == n1 { n2 } else { n1 });
        assert_eq!(order[3], n3);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_add_existing_node() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        topo.add_node(n0);
        topo.add_node(n0); // Should do nothing
        assert_eq!(topo.get_order(), &vec![n0]);
    }

    #[test]
    fn test_connect_nodes_nonexistent_node() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        match topo.connect_nodes(n0, n1) {
            Ok(()) => panic!("Should have failed"),
            Err(e) => assert_eq!(e, "Node not found in graph"),
        }
    }

    #[test]
    fn test_no_edges() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        assert_eq!(topo.get_order(), &vec![n0, n1, n2]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_disconnected_graph() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.add_node(n3);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n2, n3).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n1, n2, n3]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_connecting_components() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n2, n0).unwrap();
        assert_eq!(topo.get_order(), &vec![n2, n0, n1]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_reorder_middle() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.add_node(n3);
        topo.connect_nodes(n0, n2).unwrap();
        topo.connect_nodes(n1, n3).unwrap();
        topo.connect_nodes(n2, n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2, n1, n3]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_remove_node() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.remove_node(n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2]);
        assert!(!topo.node_to_pos.contains_key(&n1));
        assert_eq!(topo.graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![]));
        assert!(is_valid_topological_order(&topo));

        let n3 = NodeIndex::new(3);
        match topo.remove_node(n3) {
            Err(e) => assert_eq!(e, "Node not found in graph"),
            _ => panic!("Should have failed"),
        }
    }

    #[test]
    fn test_disconnect_nodes() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.disconnect_nodes(n0, n1).unwrap();
        assert_eq!(topo.graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.graph.get(&n1), Some(&vec![n2]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n1), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n2), Some(&vec![n1]));
        assert_eq!(topo.get_order(), &vec![n0, n1, n2]);
        assert!(is_valid_topological_order(&topo));

        match topo.disconnect_nodes(n0, n2) {
            Err(e) => assert_eq!(e, "Edge not found"),
            _ => panic!("Should have failed"),
        }
    }

    #[test]
    fn test_remove_isolated_node() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.remove_node(n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2]);
        assert!(!topo.node_to_pos.contains_key(&n1));
        assert_eq!(topo.graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![]));
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_remove_node_in_chain() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.remove_node(n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2]);
        assert!(!topo.node_to_pos.contains_key(&n1));
        assert_eq!(topo.graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![]));
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_remove_node_with_multiple_edges() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.add_node(n3);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.connect_nodes(n2, n3).unwrap();
        topo.remove_node(n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2, n3]);
        assert!(!topo.node_to_pos.contains_key(&n1));
        assert_eq!(topo.graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![n3]));
        assert_eq!(topo.graph.get(&n3), Some(&vec![]));
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_disconnect_edge_in_complex_graph() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n0, n2).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.disconnect_nodes(n0, n1).unwrap();
        assert_eq!(topo.graph.get(&n0), Some(&vec![n2]));
        assert_eq!(topo.graph.get(&n1), Some(&vec![n2]));
        assert_eq!(topo.graph.get(&n2), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n0), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n1), Some(&vec![]));
        assert_eq!(topo.reverse_graph.get(&n2), Some(&vec![n0, n1]));
        assert_eq!(topo.get_order(), &vec![n0, n1, n2]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_sequence_of_operations() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        topo.add_node(n0);
        topo.add_node(n1);
        topo.add_node(n2);
        topo.connect_nodes(n0, n1).unwrap();
        topo.connect_nodes(n1, n2).unwrap();
        topo.remove_node(n1).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2]);
        topo.add_node(n3);
        assert_eq!(topo.get_order(), &vec![n0, n2, n3]);
        topo.connect_nodes(n0, n3).unwrap();
        assert_eq!(topo.get_order(), &vec![n0, n2, n3]);
        assert!(is_valid_topological_order(&topo));
    }

    #[test]
    fn test_operations_on_empty_graph() {
        let mut topo = DynamicTopoSort::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        match topo.remove_node(n0) {
            Err(e) => assert_eq!(e, "Node not found in graph"),
            _ => panic!("Should have failed"),
        }
        match topo.disconnect_nodes(n0, n1) {
            Err(e) => assert_eq!(e, "Node not found in graph"),
            _ => panic!("Should have failed"),
        }
    }

    #[test]
    fn test_convex_subgraph() {
        let mut checker = DynamicTopoConvexChecker::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        let n2 = NodeIndex::new(2);
        let n3 = NodeIndex::new(3);
        checker.add_node(n0);
        checker.add_node(n1);
        checker.add_node(n2);
        checker.add_node(n3);
        checker.connect_nodes(n0, n1).unwrap();
        checker.connect_nodes(n0, n2).unwrap();
        checker.connect_nodes(n1, n3).unwrap();
        checker.connect_nodes(n2, n3).unwrap();
        let subgraph = HashSet::from([n0, n3]);
        assert!(
            !checker.is_convex(subgraph, vec![], vec![]),
            "Subgraph {{0, 3}} should not be convex"
        );
        let subgraph2 = HashSet::from([n1, n2, n3]);
        assert!(
            checker.is_convex(subgraph2, vec![], vec![]),
            "Subgraph {{1, 2, 3}} should be convex"
        );
        let subgraph3 = HashSet::from([n0, n1, n3]);
        assert!(
            !checker.is_convex(subgraph3, vec![], vec![]),
            "Subgraph {{0, 1, 3}} should not be convex"
        );
    }

    #[test]
    fn test_empty_or_invalid_subgraph() {
        let checker = DynamicTopoConvexChecker::new();
        let empty_subgraph: HashSet<NodeIndex> = HashSet::new();
        assert!(
            checker.is_convex(empty_subgraph, vec![], vec![]),
            "Empty subgraph should be convex"
        );

        let mut checker = DynamicTopoConvexChecker::new();
        let n0 = NodeIndex::new(0);
        let n1 = NodeIndex::new(1);
        checker.add_node(n0);
        let invalid_subgraph = HashSet::from([n1]);
        assert!(
            !checker.is_convex(invalid_subgraph, vec![], vec![]),
            "Subgraph with invalid node should not be convex"
        );
    }
}

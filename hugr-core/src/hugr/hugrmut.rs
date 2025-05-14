//! Low-level interface for modifying a HUGR.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use portgraph::{LinkMut, PortMut, PortView, SecondaryMap};

use crate::core::HugrNode;
use crate::extension::ExtensionRegistry;
use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrView, Node, OpType};
use crate::hugr::{NodeMetadata, Patch};
use crate::ops::OpTrait;
use crate::types::Substitution;
use crate::{Extension, Hugr, IncomingPort, OutgoingPort, Port, PortIndex};

use super::internal::HugrMutInternals;
use super::views::{
    Rerooted, panic_invalid_node, panic_invalid_non_entrypoint, panic_invalid_port,
};

/// Functions for low-level building of a HUGR.
pub trait HugrMut: HugrMutInternals {
    /// Set entrypoint to the HUGR.
    ///
    /// This node represents the execution entrypoint of the HUGR. When running
    /// local graph analysis or optimizations, the region defined under this
    /// node will be used as the starting point.
    ///
    /// For the hugr to remain valid, the entrypoint must be a region-container
    /// node, i.e. a node that can have children in the hierarchy.
    ///
    /// To get a borrowed view of the HUGR with a different entrypoint, use
    /// [`HugrView::with_entrypoint`] or [`HugrMut::with_entrypoint_mut`] instead.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_entrypoint(&mut self, root: Self::Node);

    /// Returns a mutable view of the HUGR with a different entrypoint.
    ///
    /// Changes to the returned HUGR affect the original one, and overwriting
    /// the entrypoint sets it both in the wrapper and the wrapped HUGR.
    ///
    /// For a non-mut view, use [`HugrView::with_entrypoint`] instead.
    ///
    /// # Panics
    ///
    /// Panics if the entrypoint node is not valid in the HUGR.
    fn with_entrypoint_mut(&mut self, entrypoint: Self::Node) -> Rerooted<&mut Self>
    where
        Self: Sized,
    {
        Rerooted::new(self, entrypoint)
    }

    /// Returns a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn get_metadata_mut(&mut self, node: Self::Node, key: impl AsRef<str>) -> &mut NodeMetadata;

    /// Sets a metadata value associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_metadata(
        &mut self,
        node: Self::Node,
        key: impl AsRef<str>,
        metadata: impl Into<NodeMetadata>,
    );

    /// Remove a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn remove_metadata(&mut self, node: Self::Node, key: impl AsRef<str>);

    /// Add a node to the graph with a parent in the hierarchy.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If the parent is not in the graph.
    fn add_node_with_parent(&mut self, parent: Self::Node, op: impl Into<OpType>) -> Self::Node;

    /// Add a node to the graph as the previous sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Panics
    ///
    /// If the sibling is not in the graph, or if the sibling is the root node.
    fn add_node_before(&mut self, sibling: Self::Node, nodetype: impl Into<OpType>) -> Self::Node;

    /// Add a node to the graph as the next sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Panics
    ///
    /// If the sibling is not in the graph, or if the sibling is the root node.
    fn add_node_after(&mut self, sibling: Self::Node, op: impl Into<OpType>) -> Self::Node;

    /// Remove a node from the graph and return the node weight.
    /// Note that if the node has children, they are not removed; this leaves
    /// the Hugr in an invalid state. See [`Self::remove_subtree`].
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or if the node is the root node.
    fn remove_node(&mut self, node: Self::Node) -> OpType;

    /// Remove a node from the graph, along with all its descendants in the hierarchy.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or is the root (this would leave an empty Hugr).
    fn remove_subtree(&mut self, node: Self::Node);

    /// Copies the strict descendants of `root` to under the `new_parent`, optionally applying a
    /// [Substitution] to the [`OpType`]s of the copied nodes.
    ///
    /// That is, the immediate children of root, are copied to make children of `new_parent`.
    ///
    /// Note this may invalidate the Hugr in two ways:
    /// * Adding children of `root` may make the children-list of `new_parent` invalid e.g.
    ///   leading to multiple [Input](OpType::Input), [Output](OpType::Output) or
    ///   [`ExitBlock`](OpType::ExitBlock) nodes or Input/Output in the wrong positions
    /// * Nonlocal edges incoming to the subtree of `root` will be copied to target the subtree under `new_parent`
    ///   which may be invalid if `new_parent` is not a child of `root`s parent (for `Ext` edges - or
    ///   correspondingly for `Dom` edges)
    fn copy_descendants(
        &mut self,
        root: Self::Node,
        new_parent: Self::Node,
        subst: Option<Substitution>,
    ) -> BTreeMap<Self::Node, Self::Node>;

    /// Connect two nodes at the given ports.
    ///
    /// # Panics
    ///
    /// If either node is not in the graph or if the ports are invalid.
    fn connect(
        &mut self,
        src: Self::Node,
        src_port: impl Into<OutgoingPort>,
        dst: Self::Node,
        dst_port: impl Into<IncomingPort>,
    );

    /// Disconnects all edges from the given port.
    ///
    /// The port is left in place.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or if the port is invalid.
    fn disconnect(&mut self, node: Self::Node, port: impl Into<Port>);

    /// Adds a non-dataflow edge between two nodes. The kind is given by the
    /// operation's [`OpTrait::other_input`] or [`OpTrait::other_output`].
    ///
    /// Returns the offsets of the new input and output ports.
    ///
    /// [`OpTrait::other_input`]: crate::ops::OpTrait::other_input
    /// [`OpTrait::other_output`]: crate::ops::OpTrait::other_output
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or if the port is invalid.
    fn add_other_edge(&mut self, src: Self::Node, dst: Self::Node) -> (OutgoingPort, IncomingPort);

    /// Insert another hugr into this one, under a given parent node.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    fn insert_hugr(&mut self, root: Self::Node, other: Hugr) -> InsertionResult<Node, Self::Node>;

    /// Copy another hugr into this one, under a given parent node.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node>;

    /// Copy a subgraph from another hugr into this one, under a given parent node.
    ///
    /// Sibling order is not preserved.
    ///
    /// The return value is a map from indices in `other` to the indices of the
    /// corresponding new nodes in `self`.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    //
    // TODO: Try to preserve the order when possible? We cannot always ensure
    // it, since the subgraph may have arbitrary nodes without including their
    // parent.
    fn insert_subgraph<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
        subgraph: &SiblingSubgraph<H::Node>,
    ) -> HashMap<H::Node, Self::Node>;

    /// Applies a patch to the graph.
    fn apply_patch<R, E>(&mut self, rw: impl Patch<Self, Outcome = R, Error = E>) -> Result<R, E>
    where
        Self: Sized,
    {
        rw.apply(self)
    }

    /// Registers a new extension in the set used by the hugr, keeping the one
    /// most recent one if the extension already exists.
    ///
    /// These can be queried using [`HugrView::extensions`].
    ///
    /// See [`ExtensionRegistry::register_updated`] for more information.
    fn use_extension(&mut self, extension: impl Into<Arc<Extension>>);

    /// Extend the set of extensions used by the hugr with the extensions in the
    /// registry.
    ///
    /// For each extension, keeps the most recent version if the id already
    /// exists.
    ///
    /// These can be queried using [`HugrView::extensions`].
    ///
    /// See [`ExtensionRegistry::register_updated`] for more information.
    fn use_extensions<Reg>(&mut self, registry: impl IntoIterator<Item = Reg>)
    where
        ExtensionRegistry: Extend<Reg>;
}

/// Records the result of inserting a Hugr or view
/// via [`HugrMut::insert_hugr`] or [`HugrMut::insert_from_view`].
///
/// Contains a map from the nodes in the source HUGR to the nodes in the
/// target HUGR, using their respective `Node` types.
pub struct InsertionResult<SourceN = Node, TargetN = Node> {
    /// The node, after insertion, that was the entrypoint of the inserted Hugr.
    ///
    /// That is, the value in [`InsertionResult::node_map`] under the key that was the [`HugrView::entrypoint`].
    pub inserted_entrypoint: TargetN,
    /// Map from nodes in the Hugr/view that was inserted, to their new
    /// positions in the Hugr into which said was inserted.
    pub node_map: HashMap<SourceN, TargetN>,
}

/// Translate a portgraph node index map into a map from nodes in the source
/// HUGR to nodes in the target HUGR.
///
/// This is as a helper in `insert_hugr` and `insert_subgraph`, where the source
/// HUGR may be an arbitrary `HugrView` with generic node types.
fn translate_indices<N: HugrNode>(
    mut source_node: impl FnMut(portgraph::NodeIndex) -> N,
    mut target_node: impl FnMut(portgraph::NodeIndex) -> Node,
    node_map: HashMap<portgraph::NodeIndex, portgraph::NodeIndex>,
) -> impl Iterator<Item = (N, Node)> {
    node_map
        .into_iter()
        .map(move |(k, v)| (source_node(k), target_node(v)))
}

/// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
impl HugrMut for Hugr {
    #[inline]
    fn set_entrypoint(&mut self, root: Node) {
        panic_invalid_node(self, root);
        self.entrypoint = root.into_portgraph();
    }

    fn get_metadata_mut(&mut self, node: Self::Node, key: impl AsRef<str>) -> &mut NodeMetadata {
        panic_invalid_node(self, node);
        self.node_metadata_map_mut(node)
            .entry(key.as_ref())
            .or_insert(serde_json::Value::Null)
    }

    fn set_metadata(
        &mut self,
        node: Self::Node,
        key: impl AsRef<str>,
        metadata: impl Into<NodeMetadata>,
    ) {
        let entry = self.get_metadata_mut(node, key);
        *entry = metadata.into();
    }

    fn remove_metadata(&mut self, node: Self::Node, key: impl AsRef<str>) {
        panic_invalid_node(self, node);
        let node_meta = self.node_metadata_map_mut(node);
        node_meta.remove(key.as_ref());
    }

    fn add_node_with_parent(&mut self, parent: Node, node: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(node.into());
        self.hierarchy
            .push_child(node.into_portgraph(), parent.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn add_node_before(&mut self, sibling: Node, nodetype: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(nodetype.into());
        self.hierarchy
            .insert_before(node.into_portgraph(), sibling.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn add_node_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(op.into());
        self.hierarchy
            .insert_after(node.into_portgraph(), sibling.into_portgraph())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn remove_node(&mut self, node: Node) -> OpType {
        panic_invalid_non_entrypoint(self, node);
        self.hierarchy.remove(node.into_portgraph());
        self.graph.remove_node(node.into_portgraph());
        self.op_types.take(node.into_portgraph())
    }

    fn remove_subtree(&mut self, node: Node) {
        panic_invalid_non_entrypoint(self, node);
        let mut queue = VecDeque::new();
        queue.push_back(node);
        while let Some(n) = queue.pop_front() {
            queue.extend(self.children(n));
            self.remove_node(n);
        }
    }

    fn connect(
        &mut self,
        src: Node,
        src_port: impl Into<OutgoingPort>,
        dst: Node,
        dst_port: impl Into<IncomingPort>,
    ) {
        let src_port = src_port.into();
        let dst_port = dst_port.into();
        panic_invalid_port(self, src, src_port);
        panic_invalid_port(self, dst, dst_port);
        self.graph
            .link_nodes(
                src.into_portgraph(),
                src_port.index(),
                dst.into_portgraph(),
                dst_port.index(),
            )
            .expect("The ports should exist at this point.");
    }

    fn disconnect(&mut self, node: Node, port: impl Into<Port>) {
        let port = port.into();
        let offset = port.pg_offset();
        panic_invalid_port(self, node, port);
        let port = self
            .graph
            .port_index(node.into_portgraph(), offset)
            .expect("The port should exist at this point.");
        self.graph.unlink_port(port);
    }

    fn add_other_edge(&mut self, src: Node, dst: Node) -> (OutgoingPort, IncomingPort) {
        let src_port = self
            .get_optype(src)
            .other_output_port()
            .expect("Source operation has no non-dataflow outgoing edges");
        let dst_port = self
            .get_optype(dst)
            .other_input_port()
            .expect("Destination operation has no non-dataflow incoming edges");
        self.connect(src, src_port, dst, dst_port);
        (src_port, dst_port)
    }

    fn insert_hugr(
        &mut self,
        root: Self::Node,
        mut other: Hugr,
    ) -> InsertionResult<Node, Self::Node> {
        let node_map = insert_hugr_internal(self, &other, other.entry_descendants(), |&n| {
            if n == other.entrypoint() {
                Some(root)
            } else {
                None
            }
        });
        // Merge the extension sets.
        self.extensions.extend(other.extensions());
        // Update the optypes and metadata, taking them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in &node_map {
            let node_pg = node.into_portgraph();
            let new_node_pg = new_node.into_portgraph();
            let optype = other.op_types.take(node_pg);
            self.op_types.set(new_node_pg, optype);
            let meta = other.metadata.take(node_pg);
            self.metadata.set(new_node_pg, meta);
        }
        InsertionResult {
            inserted_entrypoint: node_map[&other.entrypoint()],
            node_map,
        }
    }

    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node> {
        let node_map = insert_hugr_internal(self, other, other.entry_descendants(), |&n| {
            if n == other.entrypoint() {
                Some(root)
            } else {
                None
            }
        });
        // Merge the extension sets.
        self.extensions.extend(other.extensions());
        // Update the optypes and metadata, copying them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in &node_map {
            let nodetype = other.get_optype(node);
            self.op_types
                .set(new_node.into_portgraph(), nodetype.clone());
            let meta = other.node_metadata_map(node);
            if !meta.is_empty() {
                self.metadata
                    .set(new_node.into_portgraph(), Some(meta.clone()));
            }
        }
        InsertionResult {
            inserted_entrypoint: node_map[&other.entrypoint()],
            node_map,
        }
    }

    fn insert_subgraph<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
        subgraph: &SiblingSubgraph<H::Node>,
    ) -> HashMap<H::Node, Self::Node> {
        let node_map = insert_hugr_internal(self, other, subgraph.nodes().iter().copied(), |_| {
            Some(root)
        });
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in &node_map {
            let nodetype = other.get_optype(node);
            self.op_types
                .set(new_node.into_portgraph(), nodetype.clone());
            let meta = other.node_metadata_map(node);
            if !meta.is_empty() {
                self.metadata
                    .set(new_node.into_portgraph(), Some(meta.clone()));
            }
            // Add the required extensions to the registry.
            if let Ok(exts) = nodetype.used_extensions() {
                self.use_extensions(exts);
            }
        }
        node_map
    }

    fn copy_descendants(
        &mut self,
        root: Self::Node,
        new_parent: Self::Node,
        subst: Option<Substitution>,
    ) -> BTreeMap<Self::Node, Self::Node> {
        let mut descendants = self.hierarchy.descendants(root.into_portgraph());
        let root2 = descendants.next();
        debug_assert_eq!(root2, Some(root.into_portgraph()));
        let nodes = Vec::from_iter(descendants);
        let node_map = portgraph::view::Subgraph::with_nodes(&mut self.graph, nodes)
            .copy_in_parent()
            .expect("Is a MultiPortGraph");
        let node_map =
            translate_indices(Into::into, Into::into, node_map).collect::<BTreeMap<_, _>>();

        for node in self.children(root).collect::<Vec<_>>() {
            self.set_parent(*node_map.get(&node).unwrap(), new_parent);
        }

        // Copy the optypes, metadata, and hierarchy
        for (&node, &new_node) in &node_map {
            for ch in self.children(node).collect::<Vec<_>>() {
                self.set_parent(*node_map.get(&ch).unwrap(), new_node);
            }
            let new_optype = match (&subst, self.get_optype(node)) {
                (None, op) => op.clone(),
                (Some(subst), op) => op.substitute(subst),
            };
            self.op_types.set(new_node.into_portgraph(), new_optype);
            let meta = self.metadata.get(node.into_portgraph()).clone();
            self.metadata.set(new_node.into_portgraph(), meta);
        }
        node_map
    }

    #[inline]
    fn use_extension(&mut self, extension: impl Into<Arc<Extension>>) {
        self.extensions_mut().register_updated(extension);
    }

    #[inline]
    fn use_extensions<Reg>(&mut self, registry: impl IntoIterator<Item = Reg>)
    where
        ExtensionRegistry: Extend<Reg>,
    {
        self.extensions_mut().extend(registry);
    }
}

/// Internal implementation of `insert_hugr`, `insert_view`, and
/// `insert_subgraph`.
///
/// Inserts all the nodes in `other_nodes` into `hugr`, under the given `root` node.
///
/// Returns a mapping from the nodes in the inserted graph to their new indices
/// in `hugr`.
///
/// This function does not update the optypes of the inserted nodes, the
/// metadata, nor the hugr extensions, so the caller must do that.
///
/// # Parameters
/// - `hugr`: The hugr to insert into.
/// - `other`: The other graph to insert from.
/// - `other_nodes`: The nodes in the other graph to insert.
/// - `reroot`: A function that returns the new parent for each inserted node.
///   If `None`, the parent is set to the original parent after it has been inserted into `hugr`.
///   If that is the case, the parent must come before the child in the `other_nodes` iterator.
fn insert_hugr_internal<H: HugrView>(
    hugr: &mut Hugr,
    other: &H,
    other_nodes: impl Iterator<Item = H::Node>,
    reroot: impl Fn(&H::Node) -> Option<Node>,
) -> HashMap<H::Node, Node> {
    let new_node_count_hint = other_nodes.size_hint().1.unwrap_or_default();

    // Insert the nodes from the other graph into this one.
    let mut node_map = HashMap::with_capacity(new_node_count_hint);
    hugr.reserve(new_node_count_hint, 0);

    for old in other_nodes {
        // We use a dummy optype here. The callers take care of updating the
        // correct optype, avoiding cloning if possible.
        let op = OpType::default();
        let new = hugr.add_node(op);
        node_map.insert(old, new);

        hugr.set_num_ports(new, other.num_inputs(old), other.num_outputs(old));

        let new_parent = if let Some(new_parent) = reroot(&old) {
            new_parent
        } else {
            let old_parent = other.get_parent(old).unwrap();
            *node_map
                .get(&old_parent)
                .expect("Child node came before parent in `other_nodes` iterator")
        };
        hugr.set_parent(new, new_parent);

        // Reconnect the edges to the new node.
        for tgt in other.node_inputs(old) {
            for (neigh, src) in other.linked_outputs(old, tgt) {
                let Some(&neigh) = node_map.get(&neigh) else {
                    continue;
                };
                hugr.connect(neigh, src, new, tgt);
            }
        }
        for src in other.node_outputs(old) {
            for (neigh, tgt) in other.linked_inputs(old, src) {
                if neigh == old {
                    continue;
                }
                let Some(&neigh) = node_map.get(&neigh) else {
                    continue;
                };
                hugr.connect(new, src, neigh, tgt);
            }
        }
    }
    node_map
}

#[cfg(test)]
mod test {
    use crate::extension::PRELUDE;
    use crate::{
        extension::prelude::{Noop, usize_t},
        ops::{self, FuncDefn, Input, Output, dataflow::IOTrait},
        types::Signature,
    };

    use super::*;

    #[test]
    fn simple_function() -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr = Hugr::default();
        hugr.use_extension(PRELUDE.to_owned());

        // Create the root module definition
        let module: Node = hugr.entrypoint();

        // Start a main function with two nat inputs.
        let f: Node = hugr.add_node_with_parent(
            module,
            ops::FuncDefn::new_public(
                "main",
                Signature::new(usize_t(), vec![usize_t(), usize_t()]),
            ),
        );

        {
            let f_in = hugr.add_node_with_parent(f, ops::Input::new(vec![usize_t()]));
            let f_out = hugr.add_node_with_parent(f, ops::Output::new(vec![usize_t(), usize_t()]));
            let noop = hugr.add_node_with_parent(f, Noop(usize_t()));

            hugr.connect(f_in, 0, noop, 0);
            hugr.connect(noop, 0, f_out, 0);
            hugr.connect(noop, 0, f_out, 1);
        }

        hugr.validate()?;

        Ok(())
    }

    #[test]
    fn metadata() {
        let mut hugr = Hugr::default();

        // Create the root module definition
        let root: Node = hugr.entrypoint();

        assert_eq!(hugr.get_metadata(root, "meta"), None);

        *hugr.get_metadata_mut(root, "meta") = "test".into();
        assert_eq!(hugr.get_metadata(root, "meta"), Some(&"test".into()));

        hugr.set_metadata(root, "meta", "new");
        assert_eq!(hugr.get_metadata(root, "meta"), Some(&"new".into()));

        hugr.remove_metadata(root, "meta");
        assert_eq!(hugr.get_metadata(root, "meta"), None);
    }

    #[test]
    fn remove_subtree() {
        let mut hugr = Hugr::default();
        hugr.use_extension(PRELUDE.to_owned());
        let root = hugr.entrypoint();
        let [foo, bar] = ["foo", "bar"].map(|name| {
            let fd = hugr.add_node_with_parent(
                root,
                FuncDefn::new_private(name, Signature::new_endo(usize_t())),
            );
            let inp = hugr.add_node_with_parent(fd, Input::new(usize_t()));
            let out = hugr.add_node_with_parent(fd, Output::new(usize_t()));
            hugr.connect(inp, 0, out, 0);
            fd
        });
        hugr.validate().unwrap();
        assert_eq!(hugr.num_nodes(), 7);

        hugr.remove_subtree(foo);
        hugr.validate().unwrap();
        assert_eq!(hugr.num_nodes(), 4);

        hugr.remove_subtree(bar);
        hugr.validate().unwrap();
        assert_eq!(hugr.num_nodes(), 1);
    }
}

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

    /// Insert another hugr into this one, under a given parent node. Edges into the
    /// inserted subtree (i.e. nonlocal or static) will be disconnected in `self`.
    /// (See [Self::insert_forest] or trait [HugrLinking] for methods that can
    /// preserve such edges by also inserting their sources.)
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    ///
    /// [HugrLinking]: super::linking::HugrLinking
    fn insert_hugr(&mut self, root: Self::Node, other: Hugr) -> InsertionResult<Node, Self::Node> {
        let region = other.entrypoint();
        Self::insert_region(self, root, other, region)
    }

    /// Insert a subtree of another hugr into this one, under a given parent node.
    /// Edges into the inserted subtree (i.e. nonlocal or static) will be disconnected
    /// in `self`. (See [Self::insert_forest] or trait [HugrLinking] for methods that
    /// can preserve such edges by also inserting their sources.)
    ///
    /// # Panics
    ///
    /// - If the root node is not in the graph.
    /// - If the `region` node is not in `other`.
    ///
    /// [HugrLinking]: super::linking::HugrLinking
    fn insert_region(
        &mut self,
        root: Self::Node,
        other: Hugr,
        region: Node,
    ) -> InsertionResult<Node, Self::Node> {
        let node_map = self
            .insert_forest(other, [(region, root)])
            .expect("No errors possible for single subtree")
            .node_map;
        InsertionResult {
            inserted_entrypoint: node_map[&region],
            node_map,
        }
    }

    /// Copy the entrypoint subtree of another hugr into this one, under a given parent node.
    /// Edges into the inserted subtree (i.e. nonlocal or static) will be disconnected
    /// in `self`. (See [Self::insert_view_forest] or trait [HugrLinking] for methods that
    /// can preserve such edges by also copying their sources.)
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    ///
    /// [HugrLinking]: super::linking::HugrLinking
    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node> {
        let ep = other.entrypoint();
        let node_map = self
            .insert_view_forest(other, other.descendants(ep), [(ep, root)])
            .expect("No errors possible for single subtree")
            .node_map;
        InsertionResult {
            inserted_entrypoint: node_map[&ep],
            node_map,
        }
    }

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
    ) -> HashMap<H::Node, Self::Node> {
        self.insert_view_forest(
            other,
            subgraph.nodes().iter().cloned(),
            subgraph.nodes().iter().map(|n| (*n, root)),
        )
        .expect("SiblingSubgraph nodes are a set")
        .node_map
    }

    /// Insert a forest of nodes from another Hugr into this one.
    ///
    /// `root_parents` contains pairs of
    ///    * the root of a region in `other` to insert,
    ///    * the node in `self` that shall be parent for that region.
    ///
    /// Later entries for the same region override earlier ones.
    /// If `root_parents` is empty, nothing is inserted.
    ///
    /// # Errors
    ///
    /// [InsertForestError::SubtreeAlreadyCopied] if the regions in `root_parents` are not disjoint
    ///
    /// # Panics
    ///
    /// If any of the keys in `root_parents` are not nodes in `other`,
    /// or any of the values not in `self`.
    fn insert_forest(
        &mut self,
        other: Hugr,
        root_parents: impl IntoIterator<Item = (Node, Self::Node)>,
    ) -> InsertForestResult<Node, Self::Node>;

    /// Copy a forest of nodes from a view into this one.
    ///
    ///  `nodes` enumerates all nodes in `other` to copy.
    ///
    /// `root_parents` contains pairs of a node in `nodes` and the parent in `self` under which
    /// it should be to placed. Later entries (for the same node) override earlier ones.
    /// Note that unlike [Self::insert_forest] this allows inserting most of a subtree in one
    /// location but with subparts of that subtree placed elsewhere.
    ///
    /// Nodes in `nodes` which are not mentioned in `root_parents` and whose parent in `other`
    /// is not in `nodes`, will have no parent in `self`.
    ///
    /// # Errors
    ///
    /// [InsertForestError::DuplicateNode] if any node appears in `nodes` more than once.
    ///
    /// # Panics
    ///
    /// If any of the keys in `root_parents` are not in `nodes`, or any of the values not nodes in `self`.
    fn insert_view_forest<H: HugrView>(
        &mut self,
        other: &H,
        nodes: impl Iterator<Item = H::Node> + Clone,
        root_parents: impl IntoIterator<Item = (H::Node, Self::Node)>,
    ) -> InsertForestResult<H::Node, Self::Node>;

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

/// Result of inserting a forest from a hugr of `SN` nodes, into a hugr with
/// `TN` nodes.
///
/// On success, a map giving the new indices; or an error in the request.
/// Used by [HugrMut::insert_forest] and [HugrMut::insert_view_forest].
pub type InsertForestResult<SN, TN> = Result<InsertedForest<SN, TN>, InsertForestError<SN>>;

/// An error from [HugrMut::insert_forest] or [HugrMut::insert_view_forest].
///
/// `SN` is the type of nodes in the source Hugr
#[derive(Clone, Debug, derive_more::Display, derive_more::Error, PartialEq)]
#[non_exhaustive]
pub enum InsertForestError<SN: HugrNode = Node> {
    /// A source node was specified twice in a call to [HugrMut::insert_view_forest]
    #[display("Node {_0} would be copied twice")]
    DuplicateNode(SN),
    /// A subtree would be copied twice (i.e. it is contained in another) in a call to
    /// [HugrMut::insert_forest]
    #[display(
        "Subtree rooted at {subtree} is already being copied as part of that rooted at {parent}"
    )]
    SubtreeAlreadyCopied {
        /// Root of the inner subtree
        subtree: SN,
        /// Root of the outer subtree that also contains the inner
        parent: SN,
    },
}

/// Records the result of inserting a Hugr or view via [`HugrMut::insert_hugr`],
/// [`HugrMut::insert_from_view`], or [`HugrMut::insert_region`].
///
/// Contains a map from the nodes in the source HUGR to the nodes in the target
/// HUGR, using their respective `Node` types.
pub struct InsertionResult<SourceN = Node, TargetN = Node> {
    /// The node, after insertion, that was the root of the inserted Hugr.
    ///
    /// That is, the value in [`InsertionResult::node_map`] under the key that
    /// was the the `region` passed to [`HugrMut::insert_region`] or the
    /// [`HugrView::entrypoint`] in the other cases.
    pub inserted_entrypoint: TargetN,
    /// Map from nodes in the Hugr/view that was inserted, to their new
    /// positions in the Hugr into which said was inserted.
    pub node_map: HashMap<SourceN, TargetN>,
}

/// Records the result of inserting a Hugr or view via [`HugrMut::insert_forest`]
/// or [`HugrMut::insert_view_forest`].
///
/// Contains a map from the nodes in the source HUGR that were copied, to the
/// corresponding nodes in the target HUGR, using the respective `Node` types.
#[derive(Clone, Debug, Default)]
pub struct InsertedForest<SourceN = Node, TargetN = Node> {
    /// Map from the nodes from the source Hugr/view that were inserted,
    /// to the corresponding nodes in the Hugr into which said was inserted.
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

    fn insert_forest(
        &mut self,
        mut other: Hugr,
        root_parents: impl IntoIterator<Item = (Node, Self::Node)>,
    ) -> InsertForestResult<Node, Self::Node> {
        let roots: HashMap<_, _> = root_parents.into_iter().collect();
        for &subtree in roots.keys() {
            let mut n = subtree;
            while let Some(parent) = other.get_parent(n) {
                if roots.contains_key(&parent) {
                    return Err(InsertForestError::SubtreeAlreadyCopied { subtree, parent });
                }
                n = parent;
            }
        }
        let inserted = insert_forest_internal(
            self,
            &other,
            roots.keys().flat_map(|n| other.descendants(*n)),
            roots.iter().map(|(r, p)| (*r, *p)),
        )
        .expect("Trees disjoint so no repeated nodes");
        // Merge the extension sets.
        self.extensions.extend(other.extensions());
        // Update the optypes and metadata, taking them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in &inserted.node_map {
            let node_pg = node.into_portgraph();
            let new_node_pg = new_node.into_portgraph();
            let optype = other.op_types.take(node_pg);
            self.op_types.set(new_node_pg, optype);
            let meta = other.metadata.take(node_pg);
            self.metadata.set(new_node_pg, meta);
        }
        Ok(inserted)
    }

    fn insert_view_forest<H: HugrView>(
        &mut self,
        other: &H,
        nodes: impl Iterator<Item = H::Node> + Clone,
        root_parents: impl IntoIterator<Item = (H::Node, Self::Node)>,
    ) -> InsertForestResult<H::Node, Self::Node> {
        let inserted = insert_forest_internal(self, other, nodes, root_parents.into_iter())?;
        // Merge the extension sets.
        self.extensions.extend(other.extensions());
        // Update the optypes and metadata, copying them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in &inserted.node_map {
            let nodetype = other.get_optype(node);
            self.op_types
                .set(new_node.into_portgraph(), nodetype.clone());
            let meta = other.node_metadata_map(node);
            if !meta.is_empty() {
                self.metadata
                    .set(new_node.into_portgraph(), Some(meta.clone()));
            }
        }
        Ok(inserted)
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
/// - `root_parents`: a list of pairs of (node in `other`, parent to assign in `hugr`)
fn insert_forest_internal<H: HugrView>(
    hugr: &mut Hugr,
    other: &H,
    other_nodes: impl Iterator<Item = H::Node> + Clone,
    root_parents: impl Iterator<Item = (H::Node, Node)>,
) -> InsertForestResult<H::Node, Node> {
    let new_node_count_hint = other_nodes.size_hint().1.unwrap_or_default();

    // Insert the nodes from the other graph into this one.
    let mut node_map = HashMap::with_capacity(new_node_count_hint);
    hugr.reserve(new_node_count_hint, 0);

    for old in other_nodes.clone() {
        // We use a dummy optype here. The callers take care of updating the
        // correct optype, avoiding cloning if possible.
        let op = OpType::default();
        let new = hugr.add_node(op);
        if node_map.insert(old, new).is_some() {
            return Err(InsertForestError::DuplicateNode(old));
        }

        hugr.set_num_ports(new, other.num_inputs(old), other.num_outputs(old));

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
    for (r, p) in root_parents {
        hugr.set_parent(node_map[&r], p);
    }
    for old in other_nodes {
        let new = node_map[&old];
        if hugr.get_parent(new).is_none() {
            let old_parent = other.get_parent(old).unwrap();
            let new_parent = node_map[&old_parent];
            hugr.set_parent(new, new_parent);
        }
    }
    Ok(InsertedForest { node_map })
}

#[cfg(test)]
pub(super) mod test {
    use cool_asserts::assert_matches;
    use itertools::Itertools;
    use rstest::rstest;

    use crate::builder::test::{dfg_calling_defn_decl, simple_dfg_hugr};

    use crate::extension::PRELUDE;
    use crate::extension::prelude::{Noop, usize_t};
    use crate::hugr::ValidationError;
    use crate::ops::handle::{FuncID, NodeHandle};
    use crate::ops::{self, FuncDefn, Input, Output, dataflow::IOTrait};
    use crate::types::Signature;

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
            ops::FuncDefn::new(
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
            let fd = hugr
                .add_node_with_parent(root, FuncDefn::new(name, Signature::new_endo(usize_t())));
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

    pub(in crate::hugr) fn check_calls_defn_decl(h: &Hugr, call1_defn: bool, call2_decl: bool) {
        if call1_defn && call2_decl {
            h.validate().unwrap();
        } else {
            assert!(matches!(
                h.validate(),
                Err(ValidationError::UnconnectedPort { .. })
            ));
        }
        assert_eq!(
            h.children(h.module_root()).count(),
            1 + (call1_defn as usize) + (call2_decl as usize)
        );
        let [call1, call2] = h
            .nodes()
            .filter(|n| h.get_optype(*n).is_call())
            .collect_array()
            .unwrap();

        let tgt1 = h.nodes().find(|n| {
            h.get_optype(*n)
                .as_func_defn()
                .is_some_and(|fd| fd.func_name() == "helper_id")
        });
        assert_eq!(tgt1.is_some(), call1_defn);
        assert_eq!(h.static_source(call1), tgt1);

        let tgt2 = h.nodes().find(|n| {
            h.get_optype(*n)
                .as_func_decl()
                .is_some_and(|fd| fd.func_name() == "helper2")
        });
        assert_eq!(tgt2.is_some(), call2_decl);
        assert_eq!(h.static_source(call2), tgt2);
    }

    #[rstest]
    fn test_insert_forest(
        dfg_calling_defn_decl: (Hugr, FuncID<true>, FuncID<false>),
        #[values(false, true)] copy_defn: bool,
        #[values(false, true)] copy_decl: bool,
    ) {
        let (insert, defn, decl) = dfg_calling_defn_decl;
        let mut h = simple_dfg_hugr();
        let roots = std::iter::once((insert.entrypoint(), h.entrypoint()))
            .chain(copy_defn.then_some((defn.node(), h.module_root())))
            .chain(copy_decl.then_some((decl.node(), h.module_root())));
        h.insert_forest(insert, roots).unwrap();
        check_calls_defn_decl(&h, copy_defn, copy_decl);
    }

    #[rstest]
    fn test_insert_view_forest(dfg_calling_defn_decl: (Hugr, FuncID<true>, FuncID<false>)) {
        let (insert, defn, decl) = dfg_calling_defn_decl;
        let mut h = simple_dfg_hugr();

        let mut roots = HashMap::from([
            (insert.entrypoint(), h.entrypoint()),
            (defn.node(), h.module_root()),
            (decl.node(), h.module_root()),
        ]);

        // Straightforward case: three complete subtrees
        h.insert_view_forest(
            &insert,
            insert
                .entry_descendants()
                .chain(insert.descendants(defn.node()))
                .chain(std::iter::once(decl.node())),
            roots.clone(),
        )
        .unwrap();
        h.validate().unwrap();

        // Copy the FuncDefn node but not its children
        let mut h = simple_dfg_hugr();
        let node_map = h
            .insert_view_forest(
                &insert,
                insert.entry_descendants().chain([defn.node(), decl.node()]),
                roots.clone(),
            )
            .unwrap()
            .node_map;
        assert_matches!(h.validate(),
            Err(ValidationError::ContainerWithoutChildren { node, optype: _ }) => assert_eq!(node, node_map[&defn.node()]));

        // Copy the FuncDefn *containing* the entrypoint but transplant the entrypoint
        let func_containing_entry = insert.get_parent(insert.entrypoint()).unwrap();
        assert!(matches!(
            insert.get_optype(func_containing_entry),
            OpType::FuncDefn(_)
        ));
        roots.insert(func_containing_entry, h.module_root());
        let mut h = simple_dfg_hugr();
        let node_map = h
            .insert_view_forest(&insert, insert.nodes().skip(1), roots)
            .unwrap()
            .node_map;
        assert!(matches!(
            h.validate(),
            Err(ValidationError::InterGraphEdgeError(_))
        ));
        for c in h.nodes().filter(|n| h.get_optype(*n).is_call()) {
            assert!(h.static_source(c).is_some());
        }
        // The DFG (entrypoint) has been moved:
        let inserted_ep = node_map[&insert.entrypoint()];
        assert_eq!(h.get_parent(inserted_ep), Some(h.entrypoint()));
        let new_defn = node_map[&func_containing_entry];
        assert_eq!(h.children(new_defn).count(), 2);

        let [inp, outp] = h.get_io(new_defn).unwrap();
        assert!(matches!(h.get_optype(inp), OpType::Input(_)));
        assert!(matches!(h.get_optype(outp), OpType::Output(_)));
        // It seems the edge from Input is disconnected, but the edge to Output preserved
        assert_eq!(h.all_neighbours(inp).next(), None);
        assert_eq!(h.input_neighbours(outp).next(), Some(inserted_ep));
    }

    #[rstest]
    fn bad_insert_forest(dfg_calling_defn_decl: (Hugr, FuncID<true>, FuncID<false>)) {
        let backup = simple_dfg_hugr();
        let mut h = backup.clone();

        let (insert, _, _) = dfg_calling_defn_decl;
        let ep = insert.entrypoint();
        let epp = insert.get_parent(ep).unwrap();
        let roots = [(epp, h.module_root()), (ep, h.entrypoint())];
        let r = h.insert_view_forest(
            &insert,
            insert.descendants(epp).chain(insert.descendants(ep)),
            roots,
        );
        assert_eq!(r.err(), Some(InsertForestError::DuplicateNode(ep)));
        assert!(h.validate().is_err());

        let mut h = backup.clone();
        let r = h.insert_forest(insert, roots);
        assert_eq!(
            r.err(),
            Some(InsertForestError::SubtreeAlreadyCopied {
                subtree: ep,
                parent: epp
            })
        );
        // Here the error is detected in building `nodes` from `roots` so before any mutation
        assert_eq!(h, backup);
    }
}

//! Low-level interface for modifying a HUGR.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Display;
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

    /// Insert (the entrypoint-subtree) of another hugr into this one, under a given parent node.
    /// Unless `other.entrypoint() == other.module_root()`, then any children of
    /// `other.module_root()` except the unique ancestor of `other.entrypoint()` will be copied
    /// under the Module root of this Hugr - see [Self::insert_hugr_with_defns].
    ///
    /// Note there are two cases here which produce an invalid Hugr:
    /// 1. if `other.entrypoint() == other.module_root()` as this will insert a second
    ///    [`OpType::Module`] into `self`. The recommended way to insert a Hugr without
    ///    its root is to set the entrypoint to a child of the root.
    /// 2. If `other.entrypoint()` is a node inside (not itself) a `FuncDefn`, and
    ///    contains a (recursive) [`OpType::Call`] (or `LoadFunction`) to that ancestor
    ///    FuncDefn. In such a case, the containing FuncDefn will not be inserted, so the
    ///    `Call` will have no callee.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    fn insert_hugr(&mut self, root: Self::Node, other: Hugr) -> InsertionResult<Node, Self::Node> {
        let mut n = other.entrypoint();
        let mut children = HashSet::new();
        if n != other.module_root() {
            children.extend(other.children(other.module_root()));
            while !children.remove(&n) {
                n = other.get_parent(n).unwrap()
            }
        };

        self.insert_hugr_with_defns(root, other, children)
            .expect("Construction of `children` should ensure no possibility of DefnInsertionError")
    }

    /// Insert another Hugr into this one, copying the entrypoint-subtree under the specified
    /// 'parent', and any number of other subtrees under the Module root.
    ///
    /// # Errors
    ///
    /// If `children` are not `children` of the root of `other`; or any node in `other`
    /// is reachable in the hierarchy from both `other` and a node in `children`.
    ///
    /// # Panics
    ///
    /// If `parent` is not in this graph.
    fn insert_hugr_with_defns(
        &mut self,
        parent: Self::Node,
        other: Hugr,
        children: HashSet<Node>,
    ) -> Result<InsertionResult<Node, Self::Node>, DefnInsertionError<Node>>;

    /// Copy the entrypoint-subtree of another hugr into this one, under a given parent node.
    /// This will result in an invalid Hugr (with disconnected edges) if there are any
    /// nonlocal edges (including [Const] / [Function]) into the entrypoint-subtree.
    /// (See [Self::insert_from_view_with_defns].)
    ///
    /// # Panics
    ///
    /// If `parent` is not in the graph.
    ///
    /// [Const]: crate::types::EdgeKind::Const
    /// [Function]: crate::types::EdgeKind::Function
    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node> {
        self.insert_from_view_with_defns(root, other, HashSet::new())
            .expect("No defns being inserted so no possibility of error")
    }

    /// Copy nodes from another hugr into this one. The entrypoint-subtree of `other`
    /// is placed under the specified `parent` of this; each element of `children`,
    /// which must be a child of `other.root()`, will be copied to beneath the Module
    /// root of this HugrMut.
    ///
    /// # Errors
    ///
    /// If `children` are not `children` of the root of `other`; or if the subtrees
    /// rooted at `other.entrypoint()` and each node in `children` are not all disjoint.
    ///
    /// # Panics
    ///
    /// If `parent` is not in the graph.
    #[allow(clippy::type_complexity)]
    fn insert_from_view_with_defns<H: HugrView>(
        &mut self,
        parent: Self::Node,
        other: &H,
        children: HashSet<H::Node>,
    ) -> Result<InsertionResult<H::Node, Self::Node>, DefnInsertionError<H::Node>>;

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

    fn insert_hugr_with_defns(
        &mut self,
        parent: Self::Node,
        mut other: Hugr,
        children: HashSet<Node>,
    ) -> Result<InsertionResult<Node, Self::Node>, DefnInsertionError<Node>> {
        let node_map = insert_hugr_internal(self, parent, &other, children)?;
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
        Ok(InsertionResult {
            inserted_entrypoint: node_map[&other.entrypoint()],
            node_map,
        })
    }

    fn insert_from_view_with_defns<H: HugrView>(
        &mut self,
        parent: Self::Node,
        other: &H,
        children: HashSet<H::Node>,
    ) -> Result<InsertionResult<H::Node, Self::Node>, DefnInsertionError<H::Node>> {
        let node_map = insert_hugr_internal(self, parent, other, children)?;
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
        Ok(InsertionResult {
            inserted_entrypoint: node_map[&other.entrypoint()],
            node_map,
        })
    }

    fn insert_subgraph<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
        subgraph: &SiblingSubgraph<H::Node>,
    ) -> HashMap<H::Node, Self::Node> {
        let node_map = insert_hugr_nodes(self, other, subgraph.nodes().iter().copied(), |_| {
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

/// An error from [HugrMut::insert_hugr_with_defns] or [HugrMut::insert_from_view_with_defns]
/// caused by one of the module-children requested to insert.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DefnInsertionError<N: Display> {
    /// Module-children were requested in addition to the module entrypoint
    #[error(
        "Cannot insert children (e.g. {_0}) when already inserting whole Hugr (entrypoint == module_root)"
    )]
    ChildOfEntrypoint(N),
    /// A module-child requested contained (or was) the entrypoint
    #[error("Requested to insert module-child {_0} but this contains the entrypoint")]
    ChildContainsEntrypoint(N),
    /// A module-child requested was not a child of the module root
    #[error("{_0} was not a child of the module root")]
    NotChildOfRoot(N),
}

fn insert_hugr_internal<H: HugrView>(
    hugr: &mut Hugr,
    parent: Node,
    other: &H,
    children: HashSet<H::Node>,
) -> Result<HashMap<H::Node, Node>, DefnInsertionError<H::Node>> {
    if other.entrypoint() == other.module_root() {
        if let Some(c) = children.iter().next() {
            return Err(DefnInsertionError::ChildOfEntrypoint(*c));
        }
    } else {
        let mut on = Some(other.entrypoint());
        while let Some(n) = on {
            if children.contains(&n) {
                return Err(DefnInsertionError::ChildContainsEntrypoint(n));
            }
            on = other.get_parent(n);
        }
        for &c in children.iter() {
            if other.get_parent(c) != Some(other.module_root()) {
                return Err(DefnInsertionError::NotChildOfRoot(c));
            }
        }
    }
    let nodes = children
        .iter()
        .copied()
        .chain(std::iter::once(other.entrypoint()))
        .flat_map(|n| other.descendants(n));
    let hugr_root = hugr.module_root();
    Ok(insert_hugr_nodes(hugr, &other, nodes, |&n| {
        if n == other.entrypoint() {
            Some(parent)
        } else {
            children.contains(&n).then_some(hugr_root)
        }
    }))
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
fn insert_hugr_nodes<H: HugrView>(
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
    use itertools::Itertools;

    use crate::builder::test::simple_dfg_hugr;
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder};
    use crate::extension::PRELUDE;
    use crate::extension::prelude::{Noop, bool_t, usize_t};
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

    // Tests of insert_{hugr,from_view}(_with_defns) ====================================
    fn inserted_defn_decl() -> (Hugr, FuncID<true>, FuncID<false>) {
        let mut dfb = DFGBuilder::new(Signature::new(vec![], bool_t())).unwrap();
        let new_defn = {
            let mut mb = dfb.module_root_builder();
            let fb = mb
                .define_function("helper_id", Signature::new_endo(bool_t()))
                .unwrap();
            let [f_inp] = fb.input_wires_arr();
            fb.finish_with_outputs([f_inp]).unwrap()
        };
        let new_decl = dfb
            .module_root_builder()
            .declare("helper2", Signature::new_endo(bool_t()).into())
            .unwrap();
        let cst = dfb.add_load_value(ops::Value::true_val());
        let [c1] = dfb
            .call(new_defn.handle(), &[], [cst])
            .unwrap()
            .outputs_arr();
        let [c2] = dfb.call(&new_decl, &[], [c1]).unwrap().outputs_arr();
        (
            dfb.finish_hugr_with_outputs([c2]).unwrap(),
            *new_defn.handle(),
            new_decl,
        )
    }

    #[test]
    fn test_insert_hugr() {
        let mut h = simple_dfg_hugr();
        let (insert, _, _) = inserted_defn_decl();

        // Defaults
        h.insert_from_view(h.entrypoint(), &insert);
        check_insertion(h, false, false);

        let mut h = simple_dfg_hugr();
        h.insert_hugr(h.entrypoint(), insert);
        check_insertion(h, true, true);

        // Specify which decls to transfer
        for (call1, call2) in [(false, false), (false, true), (true, false), (true, true)] {
            let (insert, defn, decl) = inserted_defn_decl();
            let mod_children = HashSet::from_iter(
                call1
                    .then_some(defn.node())
                    .into_iter()
                    .chain(call2.then_some(decl.node())),
            );

            let mut h = simple_dfg_hugr();
            h.insert_from_view_with_defns(h.entrypoint(), &insert, mod_children.clone())
                .unwrap();
            check_insertion(h, call1, call2);

            let mut h = simple_dfg_hugr();
            h.insert_hugr_with_defns(h.entrypoint(), insert, mod_children)
                .unwrap();
            check_insertion(h, call1, call2);
        }
    }

    // ALAN TODO test errors inserting illegal children

    fn check_insertion(h: Hugr, call1_ok: bool, call2_ok: bool) {
        if call1_ok && call2_ok {
            h.validate().unwrap();
        } else {
            assert!(matches!(
                h.validate(),
                Err(ValidationError::UnconnectedPort { .. })
            ));
        }
        assert_eq!(
            h.children(h.module_root()).count(),
            1 + (call1_ok as usize) + (call2_ok as usize)
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
        assert_eq!(tgt1.is_some(), call1_ok);
        assert_eq!(h.static_source(call1), tgt1);

        let tgt2 = h.nodes().find(|n| {
            h.get_optype(*n)
                .as_func_decl()
                .is_some_and(|fd| fd.func_name() == "helper2")
        });
        assert_eq!(tgt2.is_some(), call2_ok);
        assert_eq!(h.static_source(call2), tgt2);
    }

    // (End) tests of insert_{hugr,from_view}(_with_defns) ====================================
}

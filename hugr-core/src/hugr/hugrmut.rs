//! Low-level interface for modifying a HUGR.

use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use portgraph::view::{NodeFilter, NodeFiltered};
use portgraph::{LinkMut, PortMut, PortView, SecondaryMap};

use crate::core::HugrNode;
use crate::extension::ExtensionRegistry;
use crate::hugr::internal::HugrInternals;
use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrView, Node, OpType, RootTagged};
use crate::hugr::{NodeMetadata, Rewrite};
use crate::ops::OpTrait;
use crate::types::Substitution;
use crate::{Extension, Hugr, IncomingPort, OutgoingPort, Port, PortIndex};

use super::internal::HugrMutInternals;
use super::NodeMetadataMap;

/// Functions for low-level building of a HUGR.
pub trait HugrMut: HugrMutInternals {
    /// Returns a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn get_metadata_mut(&mut self, node: Node, key: impl AsRef<str>) -> &mut NodeMetadata {
        panic_invalid_node(self, node);
        let node_meta = self
            .hugr_mut()
            .metadata
            .get_mut(node.pg_index())
            .get_or_insert_with(Default::default);
        node_meta
            .entry(key.as_ref())
            .or_insert(serde_json::Value::Null)
    }

    /// Sets a metadata value associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn set_metadata(
        &mut self,
        node: Node,
        key: impl AsRef<str>,
        metadata: impl Into<NodeMetadata>,
    ) {
        let entry = self.get_metadata_mut(node, key);
        *entry = metadata.into();
    }

    /// Remove a metadata entry associated with a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn remove_metadata(&mut self, node: Node, key: impl AsRef<str>) {
        panic_invalid_node(self, node);
        let node_meta = self.hugr_mut().metadata.get_mut(node.pg_index());
        if let Some(node_meta) = node_meta {
            node_meta.remove(key.as_ref());
        }
    }

    /// Retrieve the complete metadata map for a node.
    fn take_node_metadata(&mut self, node: Self::Node) -> Option<NodeMetadataMap> {
        if !self.valid_node(node) {
            return None;
        }
        self.hugr_mut().metadata.take(node.pg_index())
    }

    /// Overwrite the complete metadata map for a node.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph.
    fn overwrite_node_metadata(&mut self, node: Node, metadata: Option<NodeMetadataMap>) {
        panic_invalid_node(self, node);
        self.hugr_mut().metadata.set(node.pg_index(), metadata);
    }

    /// Add a node to the graph with a parent in the hierarchy.
    ///
    /// The node becomes the parent's last child.
    ///
    /// # Panics
    ///
    /// If the parent is not in the graph.
    #[inline]
    fn add_node_with_parent(&mut self, parent: Node, op: impl Into<OpType>) -> Node {
        panic_invalid_node(self, parent);
        self.hugr_mut().add_node_with_parent(parent, op)
    }

    /// Add a node to the graph as the previous sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Panics
    ///
    /// If the sibling is not in the graph, or if the sibling is the root node.
    #[inline]
    fn add_node_before(&mut self, sibling: Node, nodetype: impl Into<OpType>) -> Node {
        panic_invalid_non_root(self, sibling);
        self.hugr_mut().add_node_before(sibling, nodetype)
    }

    /// Add a node to the graph as the next sibling of another node.
    ///
    /// The sibling node's parent becomes the new node's parent.
    ///
    /// # Panics
    ///
    /// If the sibling is not in the graph, or if the sibling is the root node.
    #[inline]
    fn add_node_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Node {
        panic_invalid_non_root(self, sibling);
        self.hugr_mut().add_node_after(sibling, op)
    }

    /// Remove a node from the graph and return the node weight.
    /// Note that if the node has children, they are not removed; this leaves
    /// the Hugr in an invalid state. See [Self::remove_subtree].
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or if the node is the root node.
    #[inline]
    fn remove_node(&mut self, node: Node) -> OpType {
        panic_invalid_non_root(self, node);
        self.hugr_mut().remove_node(node)
    }

    /// Remove a node from the graph, along with all its descendants in the hierarchy.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or is the root (this would leave an empty Hugr).
    fn remove_subtree(&mut self, node: Node) {
        panic_invalid_non_root(self, node);
        while let Some(ch) = self.first_child(node) {
            self.remove_subtree(ch)
        }
        self.hugr_mut().remove_node(node);
    }

    /// Copies the strict descendants of `root` to under the `new_parent`, optionally applying a
    /// [Substitution] to the [OpType]s of the copied nodes.
    ///
    /// That is, the immediate children of root, are copied to make children of `new_parent`.
    ///
    /// Note this may invalidate the Hugr in two ways:
    /// * Adding children of `root` may make the children-list of `new_parent` invalid e.g.
    ///   leading to multiple [Input](OpType::Input), [Output](OpType::Output) or
    ///   [ExitBlock](OpType::ExitBlock) nodes or Input/Output in the wrong positions
    /// * Nonlocal edges incoming to the subtree of `root` will be copied to target the subtree under `new_parent`
    ///   which may be invalid if `new_parent` is not a child of `root`s parent (for `Ext` edges - or
    ///   correspondingly for `Dom` edges)
    fn copy_descendants(
        &mut self,
        root: Self::Node,
        new_parent: Self::Node,
        subst: Option<Substitution>,
    ) -> BTreeMap<Self::Node, Self::Node> {
        panic_invalid_node(self, root);
        panic_invalid_node(self, new_parent);
        self.hugr_mut().copy_descendants(root, new_parent, subst)
    }

    /// Connect two nodes at the given ports.
    ///
    /// # Panics
    ///
    /// If either node is not in the graph or if the ports are invalid.
    #[inline]
    fn connect(
        &mut self,
        src: Node,
        src_port: impl Into<OutgoingPort>,
        dst: Node,
        dst_port: impl Into<IncomingPort>,
    ) {
        panic_invalid_node(self, src);
        panic_invalid_node(self, dst);
        self.hugr_mut().connect(src, src_port, dst, dst_port);
    }

    /// Disconnects all edges from the given port.
    ///
    /// The port is left in place.
    ///
    /// # Panics
    ///
    /// If the node is not in the graph, or if the port is invalid.
    #[inline]
    fn disconnect(&mut self, node: Node, port: impl Into<Port>) {
        panic_invalid_node(self, node);
        self.hugr_mut().disconnect(node, port);
    }

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
    fn add_other_edge(&mut self, src: Node, dst: Node) -> (OutgoingPort, IncomingPort) {
        panic_invalid_node(self, src);
        panic_invalid_node(self, dst);
        self.hugr_mut().add_other_edge(src, dst)
    }

    /// Insert another hugr into this one, under a given root node.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    #[inline]
    fn insert_hugr(&mut self, root: Self::Node, other: Hugr) -> InsertionResult<Node, Self::Node> {
        panic_invalid_node(self, root);
        self.hugr_mut().insert_hugr(root, other)
    }

    /// Copy another hugr into this one, under a given root node.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    #[inline]
    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node> {
        panic_invalid_node(self, root);
        self.hugr_mut().insert_from_view(root, other)
    }

    /// Copy a subgraph from another hugr into this one, under a given root node.
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
        panic_invalid_node(self, root);
        self.hugr_mut().insert_subgraph(root, other, subgraph)
    }

    /// Applies a rewrite to the graph.
    fn apply_rewrite<R, E>(&mut self, rw: impl Rewrite<ApplyResult = R, Error = E>) -> Result<R, E>
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
    fn use_extension(&mut self, extension: impl Into<Arc<Extension>>) {
        self.hugr_mut().extensions.register_updated(extension);
    }

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
        ExtensionRegistry: Extend<Reg>,
    {
        self.hugr_mut().extensions.extend(registry);
    }

    /// Returns a mutable reference to the extension registry for this hugr.
    fn extensions_mut(&mut self) -> &mut ExtensionRegistry {
        &mut self.hugr_mut().extensions
    }
}

/// Records the result of inserting a Hugr or view
/// via [HugrMut::insert_hugr] or [HugrMut::insert_from_view].
///
/// Contains a map from the nodes in the source HUGR to the nodes in the
/// target HUGR, using their respective `Node` types.
pub struct InsertionResult<SourceN = Node, TargetN = Node> {
    /// The node, after insertion, that was the root of the inserted Hugr.
    ///
    /// That is, the value in [InsertionResult::node_map] under the key that was the [HugrView::root]
    pub new_root: TargetN,
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
impl<T: RootTagged<RootHandle = Node, Node = Node> + AsMut<Hugr>> HugrMut for T {
    fn add_node_with_parent(&mut self, parent: Node, node: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(node.into());
        self.as_mut()
            .hierarchy
            .push_child(node.pg_index(), parent.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn add_node_before(&mut self, sibling: Node, nodetype: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(nodetype.into());
        self.as_mut()
            .hierarchy
            .insert_before(node.pg_index(), sibling.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn add_node_after(&mut self, sibling: Node, op: impl Into<OpType>) -> Node {
        let node = self.as_mut().add_node(op.into());
        self.as_mut()
            .hierarchy
            .insert_after(node.pg_index(), sibling.pg_index())
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
        node
    }

    fn remove_node(&mut self, node: Node) -> OpType {
        panic_invalid_non_root(self, node);
        self.as_mut().hierarchy.remove(node.pg_index());
        self.as_mut().graph.remove_node(node.pg_index());
        self.as_mut().op_types.take(node.pg_index())
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
        self.as_mut()
            .graph
            .link_nodes(
                src.pg_index(),
                src_port.index(),
                dst.pg_index(),
                dst_port.index(),
            )
            .expect("The ports should exist at this point.");
    }

    fn disconnect(&mut self, node: Node, port: impl Into<Port>) {
        let port = port.into();
        let offset = port.pg_offset();
        panic_invalid_port(self, node, port);
        let port = self
            .as_mut()
            .graph
            .port_index(node.pg_index(), offset)
            .expect("The port should exist at this point.");
        self.as_mut().graph.unlink_port(port);
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
        let (new_root, node_map) = insert_hugr_internal(self.as_mut(), root, &other);
        // Update the optypes and metadata, taking them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in node_map.iter() {
            let optype = other.op_types.take(node);
            self.as_mut().op_types.set(new_node, optype);
            let meta = other.metadata.take(node);
            self.as_mut().metadata.set(new_node, meta);
        }
        debug_assert_eq!(
            Some(&new_root.pg_index()),
            node_map.get(&other.root().pg_index())
        );
        InsertionResult {
            new_root,
            node_map: translate_indices(|n| other.get_node(n), |n| self.get_node(n), node_map)
                .collect(),
        }
    }

    fn insert_from_view<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
    ) -> InsertionResult<H::Node, Self::Node> {
        let (new_root, node_map) = insert_hugr_internal(self.as_mut(), root, other);
        // Update the optypes and metadata, copying them from the other graph.
        //
        // No need to compute each node's extensions here, as we merge `other.extensions` directly.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_optype(other.get_node(node));
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.base_hugr().metadata.get(node);
            self.as_mut().metadata.set(new_node, meta.clone());
        }
        debug_assert_eq!(
            Some(&new_root.pg_index()),
            node_map.get(&other.get_pg_index(other.root()))
        );
        InsertionResult {
            new_root,
            node_map: translate_indices(|n| other.get_node(n), |n| self.get_node(n), node_map)
                .collect(),
        }
    }

    fn insert_subgraph<H: HugrView>(
        &mut self,
        root: Self::Node,
        other: &H,
        subgraph: &SiblingSubgraph<H::Node>,
    ) -> HashMap<H::Node, Self::Node> {
        // Create a portgraph view with the explicit list of nodes defined by the subgraph.
        let context: HashSet<portgraph::NodeIndex> = subgraph
            .nodes()
            .iter()
            .map(|&n| other.get_pg_index(n))
            .collect();
        let portgraph: NodeFiltered<_, NodeFilter<HashSet<portgraph::NodeIndex>>, _> =
            NodeFiltered::new_node_filtered(
                other.portgraph(),
                |node, ctx| ctx.contains(&node),
                context,
            );
        let node_map = insert_subgraph_internal(self.as_mut(), root, other, &portgraph);
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_optype(other.get_node(node));
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.base_hugr().metadata.get(node);
            self.as_mut().metadata.set(new_node, meta.clone());
            // Add the required extensions to the registry.
            if let Ok(exts) = nodetype.used_extensions() {
                self.use_extensions(exts);
            }
        }
        translate_indices(|n| other.get_node(n), |n| self.get_node(n), node_map).collect()
    }

    fn copy_descendants(
        &mut self,
        root: Self::Node,
        new_parent: Self::Node,
        subst: Option<Substitution>,
    ) -> BTreeMap<Self::Node, Self::Node> {
        let mut descendants = self.base_hugr().hierarchy.descendants(root.pg_index());
        let root2 = descendants.next();
        debug_assert_eq!(root2, Some(root.pg_index()));
        let nodes = Vec::from_iter(descendants);
        let node_map = portgraph::view::Subgraph::with_nodes(&mut self.as_mut().graph, nodes)
            .copy_in_parent()
            .expect("Is a MultiPortGraph");
        let node_map = translate_indices(|n| self.get_node(n), |n| self.get_node(n), node_map)
            .collect::<BTreeMap<_, _>>();

        for node in self.children(root).collect::<Vec<_>>() {
            self.set_parent(*node_map.get(&node).unwrap(), new_parent);
        }

        // Copy the optypes, metadata, and hierarchy
        for (&node, &new_node) in node_map.iter() {
            for ch in self.children(node).collect::<Vec<_>>() {
                self.set_parent(*node_map.get(&ch).unwrap(), new_node);
            }
            let new_optype = match (&subst, self.get_optype(node)) {
                (None, op) => op.clone(),
                (Some(subst), op) => op.substitute(subst),
            };
            self.as_mut().op_types.set(new_node.pg_index(), new_optype);
            let meta = self.base_hugr().metadata.get(node.pg_index()).clone();
            self.as_mut().metadata.set(new_node.pg_index(), meta);
        }
        node_map
    }
}

/// Internal implementation of `insert_hugr` and `insert_view` methods for
/// AsMut<Hugr>.
///
/// Returns the root node of the inserted hierarchy and a mapping from the nodes
/// in the inserted graph to their new indices in `hugr`.
///
/// This function does not update the optypes of the inserted nodes, so the
/// caller must do that.
fn insert_hugr_internal<H: HugrView>(
    hugr: &mut Hugr,
    root: Node,
    other: &H,
) -> (Node, HashMap<portgraph::NodeIndex, portgraph::NodeIndex>) {
    let node_map = hugr
        .graph
        .insert_graph(&other.portgraph())
        .unwrap_or_else(|e| panic!("Internal error while inserting a hugr into another: {e}"));
    let other_root = node_map[&other.get_pg_index(other.root())];

    // Update hierarchy and optypes
    hugr.hierarchy
        .push_child(other_root, root.pg_index())
        .expect("Inserting a newly-created node into the hierarchy should never fail.");
    for (&node, &new_node) in node_map.iter() {
        other.children(other.get_node(node)).for_each(|child| {
            hugr.hierarchy
                .push_child(node_map[&other.get_pg_index(child)], new_node)
                .expect("Inserting a newly-created node into the hierarchy should never fail.");
        });
    }

    // Merge the extension sets.
    hugr.extensions.extend(other.extensions());

    (other_root.into(), node_map)
}

/// Internal implementation of the `insert_subgraph` method for AsMut<Hugr>.
///
/// Returns a mapping from the nodes in the inserted graph to their new indices
/// in `hugr`.
///
/// This function does not update the optypes of the inserted nodes, so the
/// caller must do that.
///
/// In contrast to `insert_hugr_internal`, this function does not preserve
/// sibling order in the hierarchy. This is due to the subgraph not necessarily
/// having a single root, so the logic for reconstructing the hierarchy is not
/// able to just do a BFS.
fn insert_subgraph_internal<N: HugrNode>(
    hugr: &mut Hugr,
    root: Node,
    other: &impl HugrView<Node = N>,
    portgraph: &impl portgraph::LinkView,
) -> HashMap<portgraph::NodeIndex, portgraph::NodeIndex> {
    let node_map = hugr
        .graph
        .insert_graph(&portgraph)
        .expect("Internal error while inserting a subgraph into another");

    // A map for nodes that we inserted before their parent, so we couldn't
    // update the hierarchy with their new id.
    for (&node, &new_node) in node_map.iter() {
        let new_parent = other
            .get_parent(other.get_node(node))
            .and_then(|parent| node_map.get(&other.get_pg_index(parent)).copied())
            .unwrap_or(root.pg_index());
        hugr.hierarchy
            .push_child(new_node, new_parent)
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    node_map
}

/// Panic if [`HugrView::valid_node`] fails.
#[track_caller]
pub(super) fn panic_invalid_node<H: HugrView + ?Sized>(hugr: &H, node: H::Node) {
    if !hugr.valid_node(node) {
        panic!(
            "Received an invalid node {node} while mutating a HUGR:\n\n {}",
            hugr.mermaid_string()
        );
    }
}

/// Panic if [`HugrView::valid_non_root`] fails.
#[track_caller]
pub(super) fn panic_invalid_non_root<H: HugrView + ?Sized>(hugr: &H, node: H::Node) {
    if !hugr.valid_non_root(node) {
        panic!(
            "Received an invalid non-root node {node} while mutating a HUGR:\n\n {}",
            hugr.mermaid_string()
        );
    }
}

/// Panic if [`HugrView::valid_node`] fails.
#[track_caller]
pub(super) fn panic_invalid_port<H: HugrView + ?Sized>(
    hugr: &H,
    node: Node,
    port: impl Into<Port>,
) {
    let port = port.into();
    if hugr
        .portgraph()
        .port_index(node.pg_index(), port.pg_offset())
        .is_none()
    {
        panic!(
            "Received an invalid port {port} for node {node} while mutating a HUGR:\n\n {}",
            hugr.mermaid_string()
        );
    }
}

#[cfg(test)]
mod test {
    use crate::extension::PRELUDE;
    use crate::{
        extension::prelude::{usize_t, Noop},
        ops::{self, dataflow::IOTrait, FuncDefn, Input, Output},
        types::Signature,
    };

    use super::*;

    #[test]
    fn simple_function() -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr = Hugr::default();
        hugr.use_extension(PRELUDE.to_owned());

        // Create the root module definition
        let module: Node = hugr.root();

        // Start a main function with two nat inputs.
        let f: Node = hugr.add_node_with_parent(
            module,
            ops::FuncDefn {
                name: "main".into(),
                signature: Signature::new(vec![usize_t()], vec![usize_t(), usize_t()])
                    .with_prelude()
                    .into(),
            },
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
        let root: Node = hugr.root();

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
        let root = hugr.root();
        let [foo, bar] = ["foo", "bar"].map(|name| {
            let fd = hugr.add_node_with_parent(
                root,
                FuncDefn {
                    name: name.to_string(),
                    signature: Signature::new_endo(usize_t()).into(),
                },
            );
            let inp = hugr.add_node_with_parent(fd, Input::new(usize_t()));
            let out = hugr.add_node_with_parent(fd, Output::new(usize_t()));
            hugr.connect(inp, 0, out, 0);
            fd
        });
        hugr.validate().unwrap();
        assert_eq!(hugr.node_count(), 7);

        hugr.remove_subtree(foo);
        hugr.validate().unwrap();
        assert_eq!(hugr.node_count(), 4);

        hugr.remove_subtree(bar);
        hugr.validate().unwrap();
        assert_eq!(hugr.node_count(), 1);
    }
}

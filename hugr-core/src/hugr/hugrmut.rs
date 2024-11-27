//! Low-level interface for modifying a HUGR.

use core::panic;
use std::collections::HashMap;

use portgraph::view::{NodeFilter, NodeFiltered};
use portgraph::{LinkMut, NodeIndex, PortMut, PortView, SecondaryMap};

use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrView, Node, OpType, RootTagged};
use crate::hugr::{NodeMetadata, Rewrite};
use crate::{Hugr, IncomingPort, OutgoingPort, Port, PortIndex};

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
    fn take_node_metadata(&mut self, node: Node) -> Option<NodeMetadataMap> {
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
    /// If the node is not in the graph.
    #[inline]
    fn remove_node(&mut self, node: Node) -> OpType {
        panic_invalid_node(self, node);
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
    fn insert_hugr(&mut self, root: Node, other: Hugr) -> InsertionResult {
        panic_invalid_node(self, root);
        self.hugr_mut().insert_hugr(root, other)
    }

    /// Copy another hugr into this one, under a given root node.
    ///
    /// # Panics
    ///
    /// If the root node is not in the graph.
    #[inline]
    fn insert_from_view(&mut self, root: Node, other: &impl HugrView) -> InsertionResult {
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
    fn insert_subgraph(
        &mut self,
        root: Node,
        other: &impl HugrView,
        subgraph: &SiblingSubgraph,
    ) -> HashMap<Node, Node> {
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
}

/// Records the result of inserting a Hugr or view
/// via [HugrMut::insert_hugr] or [HugrMut::insert_from_view].
pub struct InsertionResult {
    /// The node, after insertion, that was the root of the inserted Hugr.
    ///
    /// That is, the value in [InsertionResult::node_map] under the key that was the [HugrView::root]
    pub new_root: Node,
    /// Map from nodes in the Hugr/view that was inserted, to their new
    /// positions in the Hugr into which said was inserted.
    pub node_map: HashMap<Node, Node>,
}

fn translate_indices(node_map: HashMap<NodeIndex, NodeIndex>) -> HashMap<Node, Node> {
    HashMap::from_iter(node_map.into_iter().map(|(k, v)| (k.into(), v.into())))
}

/// Impl for non-wrapped Hugrs. Overwrites the recursive default-impls to directly use the hugr.
impl<T: RootTagged<RootHandle = Node> + AsMut<Hugr>> HugrMut for T {
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

    fn insert_hugr(&mut self, root: Node, mut other: Hugr) -> InsertionResult {
        let (new_root, node_map) = insert_hugr_internal(self.as_mut(), root, &other);
        // Update the optypes and metadata, taking them from the other graph.
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
            node_map: translate_indices(node_map),
        }
    }

    fn insert_from_view(&mut self, root: Node, other: &impl HugrView) -> InsertionResult {
        let (new_root, node_map) = insert_hugr_internal(self.as_mut(), root, other);
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_optype(node.into());
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.base_hugr().metadata.get(node);
            self.as_mut().metadata.set(new_node, meta.clone());
        }
        debug_assert_eq!(
            Some(&new_root.pg_index()),
            node_map.get(&other.root().pg_index())
        );
        InsertionResult {
            new_root,
            node_map: translate_indices(node_map),
        }
    }

    fn insert_subgraph(
        &mut self,
        root: Node,
        other: &impl HugrView,
        subgraph: &SiblingSubgraph,
    ) -> HashMap<Node, Node> {
        // Create a portgraph view with the explicit list of nodes defined by the subgraph.
        let portgraph: NodeFiltered<_, NodeFilter<&[Node]>, &[Node]> =
            NodeFiltered::new_node_filtered(
                other.portgraph(),
                |node, ctx| ctx.contains(&node.into()),
                subgraph.nodes(),
            );
        let node_map = insert_subgraph_internal(self.as_mut(), root, other, &portgraph);
        // Update the optypes and metadata, copying them from the other graph.
        for (&node, &new_node) in node_map.iter() {
            let nodetype = other.get_optype(node.into());
            self.as_mut().op_types.set(new_node, nodetype.clone());
            let meta = other.base_hugr().metadata.get(node);
            self.as_mut().metadata.set(new_node, meta.clone());
        }
        translate_indices(node_map)
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
fn insert_hugr_internal(
    hugr: &mut Hugr,
    root: Node,
    other: &impl HugrView,
) -> (Node, HashMap<NodeIndex, NodeIndex>) {
    let node_map = hugr
        .graph
        .insert_graph(&other.portgraph())
        .unwrap_or_else(|e| panic!("Internal error while inserting a hugr into another: {e}"));
    let other_root = node_map[&other.root().pg_index()];

    // Update hierarchy and optypes
    hugr.hierarchy
        .push_child(other_root, root.pg_index())
        .expect("Inserting a newly-created node into the hierarchy should never fail.");
    for (&node, &new_node) in node_map.iter() {
        other.children(node.into()).for_each(|child| {
            hugr.hierarchy
                .push_child(node_map[&child.pg_index()], new_node)
                .expect("Inserting a newly-created node into the hierarchy should never fail.");
        });
    }

    // The root node didn't have any ports.
    let root_optype = other.get_optype(other.root());
    hugr.set_num_ports(
        other_root.into(),
        root_optype.input_count(),
        root_optype.output_count(),
    );

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
fn insert_subgraph_internal(
    hugr: &mut Hugr,
    root: Node,
    other: &impl HugrView,
    portgraph: &impl portgraph::LinkView,
) -> HashMap<NodeIndex, NodeIndex> {
    let node_map = hugr
        .graph
        .insert_graph(&portgraph)
        .expect("Internal error while inserting a subgraph into another");

    // A map for nodes that we inserted before their parent, so we couldn't
    // update the hierarchy with their new id.
    for (&node, &new_node) in node_map.iter() {
        let new_parent = other
            .get_parent(node.into())
            .and_then(|parent| node_map.get(&parent.pg_index()).copied())
            .unwrap_or(root.pg_index());
        hugr.hierarchy
            .push_child(new_node, new_parent)
            .expect("Inserting a newly-created node into the hierarchy should never fail.");
    }

    node_map
}

/// Panic if [`HugrView::valid_node`] fails.
#[track_caller]
pub(super) fn panic_invalid_node<H: HugrView + ?Sized>(hugr: &H, node: Node) {
    if !hugr.valid_node(node) {
        panic!(
            "Received an invalid node {node} while mutating a HUGR:\n\n {}",
            hugr.mermaid_string()
        );
    }
}

/// Panic if [`HugrView::valid_non_root`] fails.
#[track_caller]
pub(super) fn panic_invalid_non_root<H: HugrView + ?Sized>(hugr: &H, node: Node) {
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
    use crate::{
        extension::{
            prelude::{Noop, USIZE_T},
            PRELUDE_REGISTRY,
        },
        macros::type_row,
        ops::{self, dataflow::IOTrait, FuncDefn, Input, Output},
        types::{Signature, Type},
    };

    use super::*;

    const NAT: Type = USIZE_T;

    #[test]
    fn simple_function() -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr = Hugr::default();

        // Create the root module definition
        let module: Node = hugr.root();

        // Start a main function with two nat inputs.
        let f: Node = hugr.add_node_with_parent(
            module,
            ops::FuncDefn {
                name: "main".into(),
                signature: Signature::new(type_row![NAT], type_row![NAT, NAT])
                    .with_prelude()
                    .into(),
            },
        );

        {
            let f_in = hugr.add_node_with_parent(f, ops::Input::new(type_row![NAT]));
            let f_out = hugr.add_node_with_parent(f, ops::Output::new(type_row![NAT, NAT]));
            let noop = hugr.add_node_with_parent(f, Noop(NAT));

            hugr.connect(f_in, 0, noop, 0);
            hugr.connect(noop, 0, f_out, 0);
            hugr.connect(noop, 0, f_out, 1);
        }

        hugr.update_validate(&PRELUDE_REGISTRY)?;

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
        let root = hugr.root();
        let [foo, bar] = ["foo", "bar"].map(|name| {
            let fd = hugr.add_node_with_parent(
                root,
                FuncDefn {
                    name: name.to_string(),
                    signature: Signature::new_endo(NAT).into(),
                },
            );
            let inp = hugr.add_node_with_parent(fd, Input::new(NAT));
            let out = hugr.add_node_with_parent(fd, Output::new(NAT));
            hugr.connect(inp, 0, out, 0);
            fd
        });
        hugr.validate(&PRELUDE_REGISTRY).unwrap();
        assert_eq!(hugr.node_count(), 7);

        hugr.remove_subtree(foo);
        hugr.validate(&PRELUDE_REGISTRY).unwrap();
        assert_eq!(hugr.node_count(), 4);

        hugr.remove_subtree(bar);
        hugr.validate(&PRELUDE_REGISTRY).unwrap();
        assert_eq!(hugr.node_count(), 1);
    }
}

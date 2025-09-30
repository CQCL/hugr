use std::collections::{BTreeMap, HashMap};

use hugr_core::{
    Direction, Hugr, HugrView, Node, Port,
    extension::ExtensionRegistry,
    hugr::{
        self,
        internal::HugrInternals,
        views::{ExtractionResult, render},
    },
    ops::OpType,
};

use crate::Commit;

use super::{PatchNode, state_space::CommitId};

/// A HugrView on the (disjoint) union of all parent HUGRs of a commit.
///
/// Note that this is not a valid HUGR: not a single entrypoint, root etc. As
/// a consequence, not all HugrView methods are implemented.
#[derive(Debug, Clone)]
pub(crate) struct ParentsView<'a> {
    hugrs: BTreeMap<CommitId, &'a Hugr>,
}

impl<'a> ParentsView<'a> {
    pub(crate) fn from_commit(commit: &'a Commit) -> Self {
        let mut hugrs = BTreeMap::new();
        for parent in commit.parents() {
            hugrs.insert(parent.id(), parent.commit_hugr());
        }
        Self { hugrs }
    }
}

impl HugrInternals for ParentsView<'_> {
    type RegionPortgraph<'p>
        = portgraph::MultiPortGraph<u32, u32, u32>
    where
        Self: 'p;

    type Node = PatchNode;

    type RegionPortgraphNodes = HashMap<PatchNode, Node>;

    fn region_portgraph(
        &self,
        _parent: Self::Node,
    ) -> (
        portgraph::view::FlatRegion<'_, Self::RegionPortgraph<'_>>,
        Self::RegionPortgraphNodes,
    ) {
        unimplemented!()
    }

    fn node_metadata_map(&self, node: Self::Node) -> &hugr::NodeMetadataMap {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.node_metadata_map(node))
            .expect("valid node ID")
    }
}

impl HugrView for ParentsView<'_> {
    fn entrypoint(&self) -> Self::Node {
        unimplemented!()
    }

    fn module_root(&self) -> Self::Node {
        unimplemented!()
    }

    fn contains_node(&self, node: Self::Node) -> bool {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.contains_node(node))
            .expect("valid node ID")
    }

    fn get_parent(&self, node: Self::Node) -> Option<Self::Node> {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .and_then(|hugr| hugr.get_parent(node))
            .map(|parent| PatchNode(commit_id, parent))
    }

    fn get_optype(&self, node: Self::Node) -> &OpType {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.get_optype(node))
            .expect("valid node ID")
    }

    fn num_nodes(&self) -> usize {
        self.hugrs.values().map(|hugr| hugr.num_nodes()).sum()
    }

    fn num_edges(&self) -> usize {
        self.hugrs.values().map(|hugr| hugr.num_edges()).sum()
    }

    fn num_ports(&self, node: Self::Node, dir: Direction) -> usize {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.num_ports(node, dir))
            .expect("valid node ID")
    }

    fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone {
        self.hugrs
            .iter()
            .flat_map(|(commit_id, hugr)| hugr.nodes().map(move |node| (*commit_id, node)))
            .map(|(commit_id, node)| PatchNode(commit_id, node))
    }

    fn node_ports(&self, node: Self::Node, dir: Direction) -> impl Iterator<Item = Port> + Clone {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.node_ports(node, dir))
            .expect("valid node ID")
    }

    fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = Port> + Clone {
        let PatchNode(commit_id, node) = node;
        self.hugrs
            .get(&commit_id)
            .map(|hugr| hugr.all_node_ports(node))
            .expect("valid node ID")
    }

    fn linked_ports(
        &self,
        node: Self::Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Self::Node, Port)> + Clone {
        let PatchNode(commit_id, node) = node;
        let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
        hugr.linked_ports(node, port)
            .map(move |(node, port)| (PatchNode(commit_id, node), port))
    }

    /// Iterator the links between two nodes.
    fn node_connections(
        &self,
        node: Self::Node,
        other: Self::Node,
    ) -> impl Iterator<Item = [Port; 2]> + Clone {
        let PatchNode(commit_id, node) = node;
        let PatchNode(other_commit_id, other) = other;
        (commit_id == other_commit_id)
            .then(|| {
                let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
                hugr.node_connections(node, other)
            })
            .into_iter()
            .flatten()
    }

    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone {
        let PatchNode(commit_id, node) = node;
        let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
        hugr.children(node)
            .map(move |node| PatchNode(commit_id, node))
    }

    fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        let PatchNode(commit_id, node) = node;
        let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
        hugr.descendants(node)
            .map(move |node| PatchNode(commit_id, node))
    }

    fn neighbours(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Self::Node> + Clone {
        let PatchNode(commit_id, node) = node;
        let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
        hugr.neighbours(node, dir)
            .map(move |node| PatchNode(commit_id, node))
    }

    fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        let PatchNode(commit_id, node) = node;
        let hugr = self.hugrs.get(&commit_id).expect("valid node ID");
        hugr.all_neighbours(node)
            .map(move |node| PatchNode(commit_id, node))
    }

    fn mermaid_string(&self) -> String {
        unimplemented!()
    }

    #[expect(deprecated)]
    fn mermaid_string_with_config(&self, _config: render::RenderConfig<Self::Node>) -> String {
        unimplemented!()
    }

    fn dot_string(&self) -> String {
        unimplemented!()
    }

    fn extensions(&self) -> &ExtensionRegistry {
        unimplemented!()
    }

    fn extract_hugr(
        &self,
        _parent: Self::Node,
    ) -> (Hugr, impl ExtractionResult<Self::Node> + 'static) {
        #[allow(unreachable_code)]
        (unimplemented!(), HashMap::new())
    }
}

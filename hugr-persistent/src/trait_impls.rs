use std::collections::HashMap;

use itertools::{Either, Itertools};

use hugr_core::{
    Direction, Hugr, HugrView, Node, Port,
    extension::ExtensionRegistry,
    hugr::{
        self, Patch, SimpleReplacementError,
        internal::HugrInternals,
        views::{
            ExtractionResult,
            render::{self, MermaidFormatter, NodeLabel},
        },
    },
    ops::OpType,
};

use crate::CommitId;

use super::{
    InvalidCommit, PatchNode, PersistentHugr, PersistentReplacement, state_space::CommitData,
};

impl Patch<PersistentHugr> for PersistentReplacement {
    type Outcome = CommitId;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply(self, h: &mut PersistentHugr) -> Result<Self::Outcome, Self::Error> {
        match h.try_add_replacement(self) {
            Ok(commit) => Ok(commit),
            Err(
                InvalidCommit::UnknownParent(_)
                | InvalidCommit::IncompatibleHistory(_, _)
                | InvalidCommit::EmptyReplacement,
            ) => Err(SimpleReplacementError::InvalidRemovedNode()),
            _ => unreachable!(),
        }
    }
}

impl HugrInternals for PersistentHugr {
    type RegionPortgraph<'p>
        = portgraph::MultiPortGraph<u32, u32, u32>
    where
        Self: 'p;

    type Node = PatchNode;

    type RegionPortgraphNodes = HashMap<PatchNode, Node>;

    fn region_portgraph(
        &self,
        parent: Self::Node,
    ) -> (
        portgraph::view::FlatRegion<'_, Self::RegionPortgraph<'_>>,
        Self::RegionPortgraphNodes,
    ) {
        // TODO: this is currently not very efficient (see #2248)
        let (hugr, node_map) = self.apply_all();
        let parent = node_map[&parent];

        (hugr.into_region_portgraph(parent), node_map)
    }

    fn node_metadata_map(&self, PatchNode(commit_id, node): Self::Node) -> &hugr::NodeMetadataMap {
        let cm = self.get_commit(commit_id);
        cm.node_metadata_map(node)
    }
}

// TODO: A lot of these implementations (especially the ones relating to node
// hierarchies) are very inefficient as they (often unnecessarily) construct
// the whole extracted HUGR in memory. We are currently prioritizing correctness
// and clarity over performance and will optimise some of these operations in
// the future as bottlenecks are encountered. (see #2248)
impl HugrView for PersistentHugr {
    fn entrypoint(&self) -> Self::Node {
        // The entrypoint remains unchanged throughout the patch history, and is
        // found in the base hugr.
        let entry = self.base_hugr().entrypoint();
        let node = PatchNode(self.base(), entry);

        debug_assert!(self.contains_node(node), "invalid entrypoint");
        node
    }

    fn module_root(&self) -> Self::Node {
        // The module root remains unchanged throughout the patch history, and is
        // found in the base hugr.
        let root = self.base_hugr().module_root();
        let node = PatchNode(self.base(), root);

        debug_assert!(self.contains_node(node), "invalid module root");
        node
    }

    fn contains_node(&self, node: Self::Node) -> bool {
        self.contains_node(node)
    }

    fn get_parent(&self, node: Self::Node) -> Option<Self::Node> {
        assert!(self.contains_node(node), "invalid node");
        let (hugr, node_map) = self.apply_all();
        let parent = hugr.get_parent(node_map[&node])?;
        let parent_inv = node_map
            .iter()
            .find_map(|(&k, &v)| (v == parent).then_some(k))
            .expect("parent not found in node map");
        Some(parent_inv)
    }

    fn get_optype(&self, PatchNode(commit_id, node): Self::Node) -> &OpType {
        let cm = self.get_commit(commit_id);
        cm.get_optype(node)
    }

    fn num_nodes(&self) -> usize {
        let mut num_nodes = 0isize;
        for id in self.all_commit_ids() {
            let commit = self.get_commit(id);
            num_nodes += commit.inserted_nodes().count() as isize;
            num_nodes -= self.deleted_nodes(commit.id()).count() as isize;
        }
        num_nodes as usize
    }

    fn num_edges(&self) -> usize {
        self.to_hugr().num_edges()
    }

    fn num_ports(&self, PatchNode(commit_id, node): Self::Node, dir: Direction) -> usize {
        let cm = self.get_commit(commit_id);
        cm.num_ports(node, dir)
    }

    fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone {
        self.all_commit_ids()
            .flat_map(|commit_id| {
                let to_patch_node = move |node: Node| PatchNode(commit_id, node);
                match self.get_commit(commit_id).value() {
                    CommitData::Base(hugr) => Either::Left(hugr.nodes().map(to_patch_node)),
                    CommitData::Replacement(repl) => Either::Right(
                        repl.replacement()
                            .children(repl.replacement().entrypoint())
                            .filter(|&n| {
                                let ot = repl.replacement().get_optype(n);
                                !ot.is_input() && !ot.is_output()
                            })
                            .map(to_patch_node),
                    ),
                }
            })
            .filter(|&n| self.contains_node(n))
    }

    fn node_ports(
        &self,
        PatchNode(commit_id, node): Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Port> + Clone {
        let cm = self.get_commit(commit_id);
        cm.node_ports(node, dir)
    }

    fn all_node_ports(
        &self,
        PatchNode(commit_id, node): Self::Node,
    ) -> impl Iterator<Item = Port> + Clone {
        let cm = self.get_commit(commit_id);
        cm.all_node_ports(node)
    }

    fn linked_ports(
        &self,
        PatchNode(commit_id, node): Self::Node,
        port: impl Into<Port>,
    ) -> impl Iterator<Item = (Self::Node, Port)> + Clone {
        let port = port.into();
        let mut ret_ports = Vec::new();
        let cm = self.get_commit(commit_id);
        if !cm.is_value_port(node, port) {
            // currently non-value ports are not modified by patches
            let to_patch_node = |(node, port)| (cm.to_patch_node(node), port);
            ret_ports.extend(cm.commit_hugr().linked_ports(node, port).map(to_patch_node));
        } else {
            match port.as_directed() {
                Either::Left(incoming) => {
                    let (out_node, out_port) =
                        self.single_outgoing_port(cm.to_patch_node(node), incoming);
                    ret_ports.push((out_node, out_port.into()))
                }
                Either::Right(outgoing) => ret_ports.extend(
                    self.all_incoming_ports(cm.to_patch_node(node), outgoing)
                        .map(|(node, port)| (node, port.into())),
                ),
            }
        }

        ret_ports.into_iter()
    }

    fn node_connections(
        &self,
        node: Self::Node,
        other: Self::Node,
    ) -> impl Iterator<Item = [Port; 2]> + Clone {
        self.node_outputs(node)
            .flat_map(move |port| {
                self.linked_ports(node, port)
                    .map(move |(opp_node, opp_port)| (port, opp_node, opp_port))
            })
            .filter(move |&(_, opp_node, _)| opp_node == other)
            .map(|(port, _, opp_port)| [port.into(), opp_port])
    }

    fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone {
        let (hugr, node_map) = self.apply_all();
        let children = hugr.children(node_map[&node]).collect_vec();
        let inv_node_map: HashMap<_, _> = node_map.into_iter().map(|(k, v)| (v, k)).collect();
        children.into_iter().map(move |child| {
            *inv_node_map
                .get(&child)
                .expect("node not found in node map")
        })
    }

    fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        let (hugr, node_map) = self.apply_all();
        let descendants = hugr.descendants(node_map[&node]).collect_vec();
        let inv_node_map: HashMap<_, _> = node_map.into_iter().map(|(k, v)| (v, k)).collect();
        descendants.into_iter().map(move |child| {
            *inv_node_map
                .get(&child)
                .expect("node not found in node map")
        })
    }

    fn neighbours(
        &self,
        node: Self::Node,
        dir: Direction,
    ) -> impl Iterator<Item = Self::Node> + Clone {
        self.node_ports(node, dir)
            .flat_map(move |port| self.linked_ports(node, port).map(|(opp_node, _)| opp_node))
    }

    fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone {
        self.all_node_ports(node)
            .flat_map(move |port| self.linked_ports(node, port).map(|(opp_node, _)| opp_node))
    }

    #[expect(deprecated)]
    fn mermaid_string_with_config(&self, config: render::RenderConfig<Self::Node>) -> String {
        self.mermaid_string_with_formatter(MermaidFormatter::from_render_config(config, self))
    }

    fn mermaid_string_with_formatter(&self, formatter: MermaidFormatter<Self>) -> String {
        // Extract a concrete HUGR for displaying
        let (hugr, node_map) = self.apply_all();

        // Render the extracted HUGR but map the node indices back to the
        // original patch node IDs
        let entrypoint = formatter.entrypoint().map(|n| node_map[&n]);
        let node_labels = match formatter.node_labels() {
            NodeLabel::None => NodeLabel::None,
            NodeLabel::Numeric => {
                // replace node labels with patch node IDs
                let node_labels_map: HashMap<_, _> = node_map
                    .into_iter()
                    .map(|(k, v)| (v, format!("{k:?}")))
                    .collect();
                NodeLabel::Custom(node_labels_map)
            }
            NodeLabel::Custom(labels) => {
                // rekey labels to the extracted HUGR node IDs
                let labels = labels
                    .iter()
                    .map(|(k, v)| (node_map[k], v.clone()))
                    .collect();
                NodeLabel::Custom(labels)
            }
        };

        // Map config accordingly
        let config = MermaidFormatter::new(&hugr)
            .with_entrypoint(entrypoint)
            .with_node_labels(node_labels)
            .with_port_offsets(formatter.port_offsets())
            .with_type_labels(formatter.type_labels());

        config.finish()
    }

    fn dot_string(&self) -> String
    where
        Self: Sized,
    {
        unimplemented!("use mermaid_string instead")
    }

    fn extensions(&self) -> &ExtensionRegistry {
        self.base_hugr().extensions()
    }

    fn extract_hugr(
        &self,
        parent: Self::Node,
    ) -> (
        Hugr,
        impl hugr::views::ExtractionResult<Self::Node> + 'static,
    ) {
        let (hugr, apply_node_map) = self.apply_all();
        let (extracted_hugr, extracted_node_map) = hugr.extract_hugr(apply_node_map[&parent]);

        let node_map: HashMap<_, _> = apply_node_map
            .into_iter()
            .filter_map(|(patch_node, node)| {
                let extracted_node = extracted_node_map.extracted_node(node);
                if extracted_hugr.contains_node(extracted_node) {
                    Some((patch_node, extracted_node))
                } else {
                    None
                }
            })
            .collect();

        (extracted_hugr, node_map)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::tests::TestStateSpace;

    use super::super::tests::test_state_space;
    use super::*;

    use portgraph::PortView;
    use rstest::rstest;

    #[rstest]
    fn test_mermaid_string(test_state_space: TestStateSpace) {
        let [commit1, commit2, _commit3, commit4] = test_state_space.commits();

        let hugr =
            PersistentHugr::try_new([commit1.clone(), commit2.clone(), commit4.clone()]).unwrap();

        let mermaid_str = hugr
            .mermaid_format()
            .with_node_labels(NodeLabel::None)
            .finish();
        let extracted_hugr = hugr.to_hugr();
        let exp_str = extracted_hugr
            .mermaid_format()
            .with_node_labels(NodeLabel::None)
            .finish();

        assert_eq!(mermaid_str, exp_str);
    }

    #[rstest]
    fn test_hierarchy(test_state_space: TestStateSpace) {
        let [commit1, commit2, _commit3, commit4] = test_state_space.commits();

        let hugr =
            PersistentHugr::try_new([commit1.clone(), commit2.clone(), commit4.clone()]).unwrap();

        let commit2_nodes = hugr
            .nodes()
            .filter(|&n| n.owner() == commit2.id())
            .collect_vec();
        let commit4_nodes = hugr
            .nodes()
            .filter(|&n| n.owner() == commit4.id())
            .collect_vec();

        let all_children: HashSet<_> = hugr.children(hugr.entrypoint()).collect();

        assert!(commit2_nodes.iter().all(|&n| all_children.contains(&n)));
        assert!(commit4_nodes.iter().all(|&n| all_children.contains(&n)));

        let (extracted_hugr, node_map) = hugr.apply_all();

        for n in hugr.nodes() {
            assert_eq!(
                extracted_hugr.get_parent(node_map[&n]),
                hugr.get_parent(n).map(|p| node_map[&p])
            );
            assert_eq!(
                extracted_hugr.children(node_map[&n]).collect_vec(),
                hugr.children(n).map(|c| node_map[&c]).collect_vec()
            );
            assert_eq!(
                extracted_hugr.descendants(node_map[&n]).collect_vec(),
                hugr.descendants(n).map(|c| node_map[&c]).collect_vec()
            );
        }
    }

    #[rstest]
    fn test_linked_ports(test_state_space: TestStateSpace) {
        let [commit1, commit2, _commit3, commit4] = test_state_space.commits();

        let hugr =
            PersistentHugr::try_new([commit1.clone(), commit2.clone(), commit4.clone()]).unwrap();
        let (extracted_hugr, node_map) = hugr.apply_all();

        for n in hugr.nodes() {
            for port in hugr.all_node_ports(n) {
                let linked_ports = hugr
                    .linked_ports(n, port)
                    .map(|(node, port)| (node_map[&node], port))
                    .collect_vec();
                let extracted_linked_ports = extracted_hugr
                    .linked_ports(node_map[&n], port)
                    .collect_vec();

                assert_eq!(linked_ports, extracted_linked_ports);

                // Test neighbours
                for dir in [Direction::Incoming, Direction::Outgoing] {
                    let neighbours = hugr
                        .neighbours(n, dir)
                        .map(|node| node_map[&node])
                        .collect_vec();
                    let extracted_neighbours =
                        extracted_hugr.neighbours(node_map[&n], dir).collect_vec();

                    assert_eq!(neighbours, extracted_neighbours);
                }

                // Test all_neighbours
                let all_neighbours = hugr
                    .all_neighbours(n)
                    .map(|node| node_map[&node])
                    .collect_vec();
                let extracted_all_neighbours =
                    extracted_hugr.all_neighbours(node_map[&n]).collect_vec();

                assert_eq!(all_neighbours, extracted_all_neighbours);

                // Test node_connections with all other nodes
                for other in hugr.nodes() {
                    let connections = hugr.node_connections(n, other).collect_vec();
                    let extracted_connections = extracted_hugr
                        .node_connections(node_map[&n], node_map[&other])
                        .collect_vec();

                    assert_eq!(connections, extracted_connections);
                }
            }
        }
    }

    #[rstest]
    fn test_extract_hugr(test_state_space: TestStateSpace) {
        let [commit1, commit2, _commit3, commit4] = test_state_space.commits();

        let hugr =
            PersistentHugr::try_new([commit1.clone(), commit2.clone(), commit4.clone()]).unwrap();
        let extracted_hugr = hugr.to_hugr();

        assert_eq!(
            hugr.module_root(),
            hugr.base_commit()
                .to_patch_node(hugr.base_hugr().module_root())
        );

        assert_eq!(hugr.num_nodes(), extracted_hugr.num_nodes());
        assert_eq!(hugr.num_edges(), extracted_hugr.num_edges());

        let (pg, _) = hugr.region_portgraph(hugr.entrypoint());

        assert_eq!(pg.node_count(), hugr.children(hugr.entrypoint()).count());

        let (new_hugr, _) = hugr.extract_hugr(hugr.entrypoint());

        assert_eq!(new_hugr.num_nodes(), extracted_hugr.num_nodes());
    }
}

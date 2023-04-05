//! Serialization definition for [`Hugr`]
//! [`Hugr`]: crate::hugr::Hugr

use std::collections::HashMap;

use crate::{hugr::Hugr, ops::OpType};
use portgraph::{Direction, Hierarchy, NodeIndex, PortGraph, PortIndex, SecondaryMap};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Serialize, Deserialize)]
struct SerHugr {
    nodes: Vec<(NodeIndex, usize, usize)>,
    edges: Vec<[(NodeIndex, usize); 2]>,
    root: NodeIndex,
    op_types: HashMap<NodeIndex, OpType>,
}

impl From<&Hugr> for SerHugr {
    fn from(
        Hugr {
            graph,
            hierarchy,
            root,
            op_types,
        }: &Hugr,
    ) -> Self {
        let mut op_types_hsh = HashMap::new();
        let nodes: Vec<_> = graph
            .nodes_iter()
            .enumerate()
            .map(|(i, n)| {
                assert_eq!(i, n.index(), "can't serialize a non-compact graph");
                let parent = hierarchy.parent(n).unwrap_or_else(|| {
                    assert_eq!(*root, n);
                    n
                });
                let opt = &op_types[n];
                if opt != &OpType::default() {
                    op_types_hsh.insert(n, opt.clone());
                }
                (parent, graph.num_inputs(n), graph.num_outputs(n))
            })
            .collect();

        let find_offset = |p: PortIndex| {
            (
                graph.port_node(p).unwrap(),
                graph.port_offset(p).unwrap().index(),
            )
        };

        let edges: Vec<_> = graph
            .ports_iter()
            .filter_map(|p| {
                if graph.port_direction(p) == Some(Direction::Outgoing) {
                    let tgt = graph.port_link(p)?;
                    let np = [p, tgt].map(find_offset);
                    Some(np)
                } else {
                    None
                }
            })
            .collect();

        Self {
            nodes,
            edges,
            root: *root,
            op_types: op_types_hsh,
        }
    }
}

impl From<SerHugr> for Hugr {
    fn from(
        SerHugr {
            nodes,
            edges,
            root,
            mut op_types,
        }: SerHugr,
    ) -> Self {
        let mut hierarchy = Hierarchy::new();

        // if there are any unconnected ports the capacity will be an
        // underestimate
        let mut graph = PortGraph::with_capacity(nodes.len(), edges.len() * 2);
        let mut op_types_sec = SecondaryMap::new();
        for (parent, incoming, outgoing) in nodes {
            let ni = graph.add_node(incoming, outgoing);
            if parent != ni {
                hierarchy
                    .push_child(ni, parent)
                    .expect("Unexpected hierarchy error"); // TODO remove unwrap
            }
            if let Some(typ) = op_types.remove(&ni) {
                op_types_sec[ni] = typ;
            }
        }

        for [(srcn, from_offset), (tgtn, to_offset)] in edges {
            graph
                .link_nodes(srcn, from_offset, tgtn, to_offset)
                .expect("Unexpected link error");
        }

        Self {
            graph,
            hierarchy,
            root,
            op_types: op_types_sec,
        }
    }
}

impl Serialize for Hugr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shg: SerHugr = self.into();
        shg.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Hugr {
    fn deserialize<D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: Deserializer<'de>,
    {
        let shg = SerHugr::deserialize(deserializer)?;
        Ok(shg.into())
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use portgraph::proptest::gen_portgraph;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_serialization(mut graph in gen_portgraph(100, 50, 1000)) {
            let root = graph.add_node(0, 0);
            let mut hierarchy = Hierarchy::new();
            for n in graph.nodes_iter() {
                if n != root {
                    hierarchy.push_child(n, root).unwrap();
                }
            }

            let hgraph = Hugr { graph, hierarchy, root, ..Default::default()};

            prop_assert_eq!(ser_roundtrip(&hgraph), hgraph);
        }
    }

    #[test]
    fn empty_hugr_serialize() {
        let hg = Hugr::default();
        assert_eq!(ser_roundtrip(&hg), hg);
    }
    pub fn ser_roundtrip<T: Serialize + serde::de::DeserializeOwned>(g: &T) -> T {
        let v = rmp_serde::to_vec_named(g).unwrap();
        rmp_serde::from_slice(&v[..]).unwrap()
    }

    #[test]
    fn simpleser() {
        let mut g = PortGraph::new();
        let a = g.add_node(1, 1);
        let b = g.add_node(3, 2);
        let c = g.add_node(1, 1);
        let root = g.add_node(0, 0);

        g.link_nodes(a, 0, b, 0).unwrap();
        g.link_nodes(b, 0, b, 1).unwrap();
        g.link_nodes(b, 1, c, 0).unwrap();
        g.link_nodes(c, 0, a, 0).unwrap();

        let mut h = Hierarchy::new();

        for n in [a, b, c] {
            h.push_child(n, root).unwrap();
        }
        let hg = Hugr {
            graph: g,
            hierarchy: h,
            root,
            op_types: SecondaryMap::new(),
        };

        let v = rmp_serde::to_vec_named(&hg).unwrap();

        let newhg = rmp_serde::from_slice(&v[..]).unwrap();
        assert_eq!(hg, newhg);
    }
}

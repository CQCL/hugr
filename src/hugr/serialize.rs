//! Serialization definition for [`Hugr`]
//! [`Hugr`]: crate::hugr::Hugr

use std::collections::HashMap;
use thiserror::Error;

use crate::{hugr::Hugr, ops::OpType};
use portgraph::{
    hierarchy::AttachError, Direction, Hierarchy, LinkError, NodeIndex, PortGraph, PortIndex,
    SecondaryMap,
};
use serde::{Deserialize, Deserializer, Serialize};

/// A wrapper over the available HUGR serialization formats.
///
/// The implementation of `Serialize` for `Hugr` encodes the graph in the most
/// recent version of the format. We keep the `Deserialize` implementations for
/// older versions to allow for backwards compatibility.
///
/// Make sure to order the variants from newest to oldest, as the deserializer
/// will try to deserialize them in order.
#[derive(Serialize, Deserialize)]
#[serde(tag = "version", rename_all = "lowercase")]
enum Versioned {
    /// Version 0 of the HUGR serialization format.
    V0(SerHugrV0),

    #[serde(other)]
    Unsupported,
}

/// Version 0 of the HUGR serialization format.
#[derive(Serialize, Deserialize)]
struct SerHugrV0 {
    nodes: Vec<(NodeIndex, usize, usize)>,
    edges: Vec<[(NodeIndex, usize); 2]>,
    root: NodeIndex,
    op_types: HashMap<NodeIndex, OpType>,
}

/// Errors that can occur while serializing a HUGR.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HUGRSerializationError {
    /// Cannot serialize a non-compact graph.
    #[error("Cannot serialize a non-compact graph (node indices must be contiguous).")]
    NonCompactGraph,
    /// Unexpected hierarchy error.
    #[error("Failed to attach child to parent: {0:?}.")]
    AttachError(#[from] AttachError),
    /// Failed to add edge.
    #[error("Failed to build edge when deserializing: {0:?}.")]
    LinkError(#[from] LinkError),
}

impl Serialize for Hugr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shg: SerHugrV0 = self.try_into().map_err(serde::ser::Error::custom)?;
        let versioned = Versioned::V0(shg);
        versioned.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Hugr {
    fn deserialize<D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: Deserializer<'de>,
    {
        let shg = Versioned::deserialize(deserializer)?;
        match shg {
            Versioned::V0(shg) => shg.try_into().map_err(serde::de::Error::custom),
            Versioned::Unsupported => Err(serde::de::Error::custom(
                "Unsupported HUGR serialization format.",
            )),
        }
    }
}

impl TryFrom<&Hugr> for SerHugrV0 {
    type Error = HUGRSerializationError;

    fn try_from(
        Hugr {
            graph,
            hierarchy,
            root,
            op_types,
        }: &Hugr,
    ) -> Result<Self, Self::Error> {
        let mut op_types_hsh = HashMap::new();
        let nodes: Result<Vec<_>, HUGRSerializationError> = graph
            .nodes_iter()
            .enumerate()
            .map(|(i, n)| {
                if i != n.index() {
                    return Err(HUGRSerializationError::NonCompactGraph);
                }

                let parent = hierarchy.parent(n).unwrap_or_else(|| {
                    assert_eq!(*root, n);
                    n
                });
                let opt = &op_types[n];
                // secondary map holds default values for empty positions
                // whether or not the default value is present or not - the
                // serialization roundtrip will be correct
                if opt != &OpType::default() {
                    op_types_hsh.insert(n, opt.clone());
                }
                Ok((parent, graph.num_inputs(n), graph.num_outputs(n)))
            })
            .collect();
        let nodes = nodes?;

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

        Ok(Self {
            nodes,
            edges,
            root: *root,
            op_types: op_types_hsh,
        })
    }
}

impl TryFrom<SerHugrV0> for Hugr {
    type Error = HUGRSerializationError;
    fn try_from(
        SerHugrV0 {
            nodes,
            edges,
            root,
            mut op_types,
        }: SerHugrV0,
    ) -> Result<Self, Self::Error> {
        let mut hierarchy = Hierarchy::new();

        // if there are any unconnected ports the capacity will be an
        // underestimate
        let mut graph = PortGraph::with_capacity(nodes.len(), edges.len() * 2);
        let mut op_types_sec = SecondaryMap::new();
        for (parent, incoming, outgoing) in nodes {
            let ni = graph.add_node(incoming, outgoing);
            if parent != ni {
                hierarchy.push_child(ni, parent)?; // TODO remove unwrap
            }
            if let Some(typ) = op_types.remove(&ni) {
                op_types_sec[ni] = typ;
            }
        }

        for [(srcn, from_offset), (tgtn, to_offset)] in edges {
            graph.link_nodes(srcn, from_offset, tgtn, to_offset)?;
        }

        Ok(Self {
            graph,
            hierarchy,
            root,
            op_types: op_types_sec,
        })
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use portgraph::proptest::gen_portgraph;
    use proptest::prelude::*;
    proptest! {
        #[test]
        // miri fails due to proptest filesystem access
        #[cfg_attr(miri, ignore)]
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

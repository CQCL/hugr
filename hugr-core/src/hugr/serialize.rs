//! Serialization definition for [`Hugr`]
//! [`Hugr`]: crate::hugr::Hugr

use std::collections::HashMap;
use thiserror::Error;

use crate::core::NodeIndex;
use crate::extension::ExtensionSet;
use crate::hugr::{Hugr, NodeType};
use crate::ops::OpType;
use crate::{Node, PortIndex};
use portgraph::hierarchy::AttachError;
use portgraph::{Direction, LinkError, PortView};

use serde::{Deserialize, Deserializer, Serialize};

use super::{HugrMut, HugrView, NodeMetadataMap};

/// A wrapper over the available HUGR serialization formats.
///
/// The implementation of `Serialize` for `Hugr` encodes the graph in the most
/// recent version of the format. We keep the `Deserialize` implementations for
/// older versions to allow for backwards compatibility.
///
/// The Generic `SerHugr` is always instantiated to the most recent version of
/// the format outside this module.
///
/// Make sure to order the variants from newest to oldest, as the deserializer
/// will try to deserialize them in order.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "version", rename_all = "lowercase")]
enum Versioned<SerHugr> {
    /// Version 0 of the HUGR serialization format.
    V0,
    /// Version 1 of the HUGR serialization format.
    V1(SerHugr),

    #[serde(other)]
    Unsupported,
}

impl<T> Versioned<T> {
    pub fn new(t: T) -> Self {
        Self::V1(t)
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct NodeSer {
    parent: Node,
    input_extensions: Option<ExtensionSet>,
    #[serde(flatten)]
    op: OpType,
}

/// Version 1 of the HUGR serialization format.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SerHugrV1 {
    /// For each node: (parent, node_operation)
    nodes: Vec<NodeSer>,
    /// for each edge: (src, src_offset, tgt, tgt_offset)
    edges: Vec<[(Node, Option<u16>); 2]>,
    /// for each node: (metadata)
    #[serde(default)]
    metadata: Option<Vec<Option<NodeMetadataMap>>>,
    /// A metadata field with the package identifier that encoded the HUGR.
    #[serde(default)]
    encoder: Option<String>,
}

/// Errors that can occur while serializing a HUGR.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum HUGRSerializationError {
    /// Unexpected hierarchy error.
    #[error("Failed to attach child to parent: {0:?}.")]
    AttachError(#[from] AttachError),
    /// Failed to add edge.
    #[error("Failed to build edge when deserializing: {0:?}.")]
    LinkError(#[from] LinkError),
    /// Edges without port offsets cannot be present in operations without non-dataflow ports.
    #[error("Cannot connect an {dir:?} edge without port offset to node {node:?} with operation type {op_type:?}.")]
    MissingPortOffset {
        /// The node that has the port without offset.
        node: Node,
        /// The direction of the port without an offset
        dir: Direction,
        /// The operation type of the node.
        op_type: OpType,
    },
    /// Edges with wrong node indices
    #[error("The edge endpoint {node:?} is not a node in the graph.")]
    UnknownEdgeNode {
        /// The node that has the port without offset.
        node: Node,
    },
    /// First node in node list must be the HUGR root.
    #[error("The first node in the node list has parent {0:?}, should be itself (index 0)")]
    FirstNodeNotRoot(Node),
}

impl Serialize for Hugr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shg: SerHugrV1 = self.try_into().map_err(serde::ser::Error::custom)?;
        let versioned = Versioned::new(shg);
        versioned.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Hugr {
    fn deserialize<D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: Deserializer<'de>,
    {
        let shg: Versioned<SerHugrV1> = Versioned::deserialize(deserializer)?;
        match shg {
            Versioned::V0 => Err(serde::de::Error::custom(
                "Version 0 HUGR serialization format is not supported.",
            )),
            Versioned::V1(shg) => shg.try_into().map_err(serde::de::Error::custom),
            Versioned::Unsupported => Err(serde::de::Error::custom(
                "Unsupported HUGR serialization format.",
            )),
        }
    }
}

impl TryFrom<&Hugr> for SerHugrV1 {
    type Error = HUGRSerializationError;

    fn try_from(hugr: &Hugr) -> Result<Self, Self::Error> {
        // We compact the operation nodes during the serialization process,
        // and ignore the copy nodes.
        let mut node_rekey: HashMap<Node, Node> = HashMap::with_capacity(hugr.node_count());
        for (order, node) in hugr.canonical_order(hugr.root()).enumerate() {
            node_rekey.insert(node, portgraph::NodeIndex::new(order).into());
        }

        let mut nodes = vec![None; hugr.node_count()];
        let mut metadata = vec![None; hugr.node_count()];
        for n in hugr.nodes() {
            let parent = node_rekey[&hugr.get_parent(n).unwrap_or(n)];
            let opt = hugr.get_nodetype(n);
            let new_node = node_rekey[&n].index();
            nodes[new_node] = Some(NodeSer {
                parent,
                input_extensions: opt.input_extensions.clone(),
                op: opt.op.clone(),
            });
            metadata[new_node].clone_from(hugr.metadata.get(n.pg_index()));
        }
        let nodes = nodes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .expect("Could not reach one of the nodes");

        let find_offset = |node: Node, offset: usize, dir: Direction, hugr: &Hugr| {
            let op = hugr.get_optype(node);
            let is_value_port = offset < op.value_port_count(dir);
            let is_static_input = op.static_port(dir).map_or(false, |p| p.index() == offset);
            let offset = (is_value_port || is_static_input).then_some(offset as u16);
            (node_rekey[&node], offset)
        };

        let edges: Vec<_> = hugr
            .nodes()
            .flat_map(|node| {
                hugr.node_ports(node, Direction::Outgoing)
                    .enumerate()
                    .flat_map(move |(src_offset, port)| {
                        let src = find_offset(node, src_offset, Direction::Outgoing, hugr);
                        hugr.linked_ports(node, port).map(move |(tgt_node, tgt)| {
                            let tgt = find_offset(tgt_node, tgt.index(), Direction::Incoming, hugr);
                            [src, tgt]
                        })
                    })
            })
            .collect();

        let encoder = Some(format!("hugr-rs v{}", env!("CARGO_PKG_VERSION")));

        Ok(Self {
            nodes,
            edges,
            metadata: Some(metadata),
            encoder,
        })
    }
}

impl TryFrom<SerHugrV1> for Hugr {
    type Error = HUGRSerializationError;
    fn try_from(
        SerHugrV1 {
            nodes,
            edges,
            metadata,
            ..
        }: SerHugrV1,
    ) -> Result<Self, Self::Error> {
        // Root must be first node
        let mut nodes = nodes.into_iter();
        let NodeSer {
            parent: root_parent,
            input_extensions,
            op: root_type,
        } = nodes.next().unwrap();
        if root_parent.index() != 0 {
            return Err(HUGRSerializationError::FirstNodeNotRoot(root_parent));
        }
        // if there are any unconnected ports or copy nodes the capacity will be
        // an underestimate
        let mut hugr = Hugr::with_capacity(
            NodeType::new(root_type, input_extensions),
            nodes.len(),
            edges.len() * 2,
        );

        for node_ser in nodes {
            hugr.add_node_with_parent(
                node_ser.parent,
                NodeType::new(node_ser.op, node_ser.input_extensions),
            );
        }

        if let Some(metadata) = metadata {
            for (node, metadata) in metadata.into_iter().enumerate() {
                if let Some(metadata) = metadata {
                    let node = portgraph::NodeIndex::new(node);
                    hugr.metadata[node] = Some(metadata);
                }
            }
        }

        let unwrap_offset = |node: Node, offset, dir, hugr: &Hugr| -> Result<usize, Self::Error> {
            if !hugr.graph.contains_node(node.pg_index()) {
                return Err(HUGRSerializationError::UnknownEdgeNode { node });
            }
            let offset = match offset {
                Some(offset) => offset as usize,
                None => {
                    let op_type = hugr.get_optype(node);
                    op_type
                        .other_port(dir)
                        .ok_or(HUGRSerializationError::MissingPortOffset {
                            node,
                            dir,
                            op_type: op_type.clone(),
                        })?
                        .index()
                }
            };
            Ok(offset)
        };
        for [(src, from_offset), (dst, to_offset)] in edges {
            let src_port = unwrap_offset(src, from_offset, Direction::Outgoing, &hugr)?;
            let dst_port = unwrap_offset(dst, to_offset, Direction::Incoming, &hugr)?;

            hugr.connect(src, src_port, dst, dst_port);
        }

        Ok(hugr)
    }
}

#[cfg(all(test, not(miri)))]
// Miri doesn't run the extension registration required by `typetag` for
// registering `CustomConst`s.  https://github.com/rust-lang/miri/issues/450
pub mod test;

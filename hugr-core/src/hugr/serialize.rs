//! Serialization definition for [`Hugr`]
//! [`Hugr`]: `crate::hugr::Hugr`

use serde::de::DeserializeOwned;
use std::collections::HashMap;
use thiserror::Error;

use crate::core::NodeIndex;
use crate::hugr::Hugr;
use crate::ops::OpType;
use crate::types::EdgeKind;
use crate::{Node, PortIndex};
use portgraph::hierarchy::AttachError;
use portgraph::{Direction, LinkError, PortView};

use serde::{Deserialize, Deserializer, Serialize};

use self::upgrade::UpgradeError;

use super::{HugrError, HugrMut, HugrView, NodeMetadataMap};

mod upgrade;

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "version", rename_all = "lowercase")]
enum Versioned<SerHugr = SerHugrLatest> {
    #[serde(skip_serializing)]
    /// Version 0 of the HUGR serialization format.
    V0,

    V1(serde_json::Value),
    V2(serde_json::Value),
    Live(SerHugr),

    #[serde(skip_serializing)]
    #[serde(other)]
    Unsupported,
}

impl<T> Versioned<T> {
    pub fn new_latest(t: T) -> Self {
        Self::Live(t)
    }
}

impl<T: DeserializeOwned> Versioned<T> {
    fn upgrade(self) -> Result<T, UpgradeError> {
        // go is polymorphic in D. When we are upgrading to the latest version
        // D is T. When we are upgrading to a version which is not the latest D
        // is serde_json::Value.
        #[allow(unused)]
        fn go<D: serde::de::DeserializeOwned>(v: serde_json::Value) -> Result<D, UpgradeError> {
            serde_json::from_value(v).map_err(Into::into)
        }
        loop {
            match self {
                Self::V0 => Err(UpgradeError::KnownVersionUnsupported("0".into()))?,
                // the upgrade lines remain unchanged when adding a new constructor
                // Self::V1(json) => self = Self::V2(upgrade::v1_to_v2(json).and_then(go)?),
                Self::V1(_) => Err(UpgradeError::KnownVersionUnsupported("1".into()))?,
                Self::V2(_) => Err(UpgradeError::KnownVersionUnsupported("2".into()))?,
                Self::Live(ser_hugr) => return Ok(ser_hugr),
                Versioned::Unsupported => Err(UpgradeError::UnknownVersionUnsupported)?,
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct NodeSer {
    /// Node index of the parent.
    parent: Node,
    #[serde(flatten)]
    op: OpType,
}

/// Version 1 of the HUGR serialization format.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct SerHugrLatest {
    /// For each node: (parent, `node_operation`)
    nodes: Vec<NodeSer>,
    /// for each edge: (src, `src_offset`, tgt, `tgt_offset`)
    edges: Vec<[(Node, Option<u32>); 2]>,
    /// for each node: (metadata)
    #[serde(default)]
    metadata: Option<Vec<Option<NodeMetadataMap>>>,
    /// A metadata field with the package identifier that encoded the HUGR.
    #[serde(default)]
    encoder: Option<String>,
    /// The entrypoint of the HUGR.
    ///
    /// For backwards compatibility, if `None` the entrypoint is set to the root
    /// of the node hierarchy.
    #[serde(default)]
    entrypoint: Option<Node>,
}

/// Errors that can occur while serializing a HUGR.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum HUGRSerializationError {
    /// Unexpected hierarchy error.
    #[error("Failed to attach child to parent: {0}.")]
    AttachError(#[from] AttachError),
    /// Failed to add edge.
    #[error("Failed to build edge when deserializing: {0}.")]
    LinkError(#[from] LinkError<u32>),
    /// Edges without port offsets cannot be present in operations without non-dataflow ports.
    #[error(
        "Cannot connect an {dir:?} edge without port offset to node {node} with operation type {op_type}."
    )]
    MissingPortOffset {
        /// The node that has the port without offset.
        node: Node,
        /// The direction of the port without an offset
        dir: Direction,
        /// The operation type of the node.
        op_type: OpType,
    },
    /// Edges with wrong node indices
    #[error("The edge endpoint {node} is not a node in the graph.")]
    UnknownEdgeNode {
        /// The node that has the port without offset.
        node: Node,
    },
    /// First node in node list must be the HUGR root.
    #[error("The first node in the node list has parent {0}, should be itself (index 0)")]
    FirstNodeNotRoot(Node),
    /// Failed to deserialize the HUGR.
    #[error(transparent)]
    HugrError(#[from] HugrError),
}

impl Hugr {
    /// Serializes the HUGR using a serde encoder.
    ///
    /// This is an internal API, used to generate the JSON variant of the HUGR envelope format.
    pub(crate) fn serde_serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shg: SerHugrLatest = self.try_into().map_err(serde::ser::Error::custom)?;
        let versioned = Versioned::new_latest(shg);
        versioned.serialize(serializer)
    }

    /// Deserializes the HUGR using a serde decoder.
    ///
    /// This is an internal API, used to read the JSON variant of the HUGR envelope format.
    pub(crate) fn serde_deserialize<'de, D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: Deserializer<'de>,
    {
        let versioned = Versioned::deserialize(deserializer)?;
        let shl: SerHugrLatest = versioned.upgrade().map_err(serde::de::Error::custom)?;
        shl.try_into().map_err(serde::de::Error::custom)
    }
}

/// Deerialize the HUGR using a serde decoder.
///
/// This API is unstable API and will be removed in the future.
#[deprecated(
    since = "0.20.0",
    note = "This API is unstable and will be removed in the future.
            Use `Hugr::load` or the `AsStringEnvelope` adaptor instead."
)]
#[doc(hidden)]
pub fn serde_deserialize_hugr<'de, D>(deserializer: D) -> Result<Hugr, D::Error>
where
    D: Deserializer<'de>,
{
    Hugr::serde_deserialize(deserializer)
}

impl TryFrom<&Hugr> for SerHugrLatest {
    type Error = HUGRSerializationError;

    fn try_from(hugr: &Hugr) -> Result<Self, Self::Error> {
        // We compact the operation nodes during the serialization process,
        // and ignore the copy nodes.
        let mut node_rekey: HashMap<Node, Node> = HashMap::with_capacity(hugr.num_nodes());
        for (order, node) in hugr.canonical_order(hugr.module_root()).enumerate() {
            node_rekey.insert(node, portgraph::NodeIndex::new(order).into());
        }

        let mut nodes = vec![None; hugr.num_nodes()];
        let mut metadata = vec![None; hugr.num_nodes()];
        for n in hugr.nodes() {
            let parent = node_rekey[&hugr.get_parent(n).unwrap_or(n)];
            let opt = hugr.get_optype(n);
            let new_node = node_rekey[&n].index();
            nodes[new_node] = Some(NodeSer {
                parent,
                op: opt.clone(),
            });
            metadata[new_node].clone_from(hugr.metadata.get(n.into_portgraph()));
        }
        let nodes = nodes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .expect("Could not reach one of the nodes");

        let find_offset = |node: Node, offset: usize, dir: Direction, hugr: &Hugr| {
            let op = hugr.get_optype(node);
            let is_value_port = offset < op.value_port_count(dir);
            let is_static_input = op.static_port(dir).is_some_and(|p| p.index() == offset);
            let other_port_is_not_order = op.other_port_kind(dir) != Some(EdgeKind::StateOrder);
            let offset = (is_value_port || is_static_input || other_port_is_not_order)
                .then_some(offset as u32);
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
            entrypoint: Some(node_rekey[&hugr.entrypoint()]),
        })
    }
}

impl TryFrom<SerHugrLatest> for Hugr {
    type Error = HUGRSerializationError;
    fn try_from(
        SerHugrLatest {
            nodes,
            edges,
            metadata,
            encoder: _,
            entrypoint,
        }: SerHugrLatest,
    ) -> Result<Self, Self::Error> {
        // Root must be first node
        let mut nodes = nodes.into_iter();
        let NodeSer {
            parent: root_parent,
            op: root_type,
            ..
        } = nodes.next().unwrap();
        if root_parent.index() != 0 {
            return Err(HUGRSerializationError::FirstNodeNotRoot(root_parent));
        }
        // if there are any unconnected ports or copy nodes the capacity will be
        // an underestimate
        let mut hugr = Hugr::with_capacity(root_type, nodes.len(), edges.len() * 2)?;

        // Since the new Hugr may add some nodes to contain the root (if the
        // encoded file did not have a module at the root), we need a function
        // to map the node indices.
        let padding_nodes = hugr.entrypoint.index();
        let hugr_node =
            |node: Node| -> Node { portgraph::NodeIndex::new(node.index() + padding_nodes).into() };

        for node_ser in nodes {
            hugr.add_node_with_parent(hugr_node(node_ser.parent), node_ser.op);
        }

        if let Some(entrypoint) = entrypoint {
            hugr.set_entrypoint(hugr_node(entrypoint));
        }

        if let Some(metadata) = metadata {
            for (node_idx, metadata) in metadata.into_iter().enumerate() {
                if let Some(metadata) = metadata {
                    let node = hugr_node(portgraph::NodeIndex::new(node_idx).into());
                    hugr.metadata[node.into_portgraph()] = Some(metadata);
                }
            }
        }

        let unwrap_offset = |node: Node, offset, dir, hugr: &Hugr| -> Result<usize, Self::Error> {
            if !hugr.graph.contains_node(node.into_portgraph()) {
                return Err(HUGRSerializationError::UnknownEdgeNode { node });
            }
            let offset = if let Some(offset) = offset {
                offset as usize
            } else {
                let op_type = hugr.get_optype(node);
                op_type
                    .other_port(dir)
                    .ok_or(HUGRSerializationError::MissingPortOffset {
                        node,
                        dir,
                        op_type: op_type.clone(),
                    })?
                    .index()
            };
            Ok(offset)
        };
        for [(src, from_offset), (dst, to_offset)] in edges {
            let src = hugr_node(src);
            let dst = hugr_node(dst);

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

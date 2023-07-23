//! Serialization definition for [`Hugr`]
//! [`Hugr`]: crate::hugr::Hugr

use serde_json::json;
use std::collections::HashMap;
use thiserror::Error;

use crate::hugr::{Hugr, HugrMut};
use crate::ops::OpTrait;
use crate::ops::OpType;
use crate::Node;
use portgraph::hierarchy::AttachError;
use portgraph::{Direction, LinkError, NodeIndex, PortView};

use serde::{Deserialize, Deserializer, Serialize};

use super::{HugrError, HugrView};

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

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct NodeSer {
    parent: Node,
    #[serde(flatten)]
    op: OpType,
}

/// Version 0 of the HUGR serialization format.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SerHugrV0 {
    /// For each node: (parent, node_operation)
    nodes: Vec<NodeSer>,
    /// for each edge: (src, src_offset, tgt, tgt_offset)
    edges: Vec<[(Node, Option<u16>); 2]>,
    /// for each node: (metadata)
    #[serde(default)]
    metadata: Vec<serde_json::Value>,
}

/// Errors that can occur while serializing a HUGR.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HUGRSerializationError {
    /// Unexpected hierarchy error.
    #[error("Failed to attach child to parent: {0:?}.")]
    AttachError(#[from] AttachError),
    /// Failed to add edge.
    #[error("Failed to build edge when deserializing: {0:?}.")]
    LinkError(#[from] LinkError),
    /// Edges without port offsets cannot be present in operations without non-dataflow ports.
    #[error("Cannot connect an edge without port offset to node {node:?} with operation type {op_type:?}.")]
    MissingPortOffset {
        /// The node that has the port without offset.
        node: Node,
        /// The operation type of the node.
        op_type: OpType,
    },
    /// Edges with wrong node indices
    #[error("The edge endpoint {node:?} is not a node in the graph.")]
    UnknownEdgeNode {
        /// The node that has the port without offset.
        node: Node,
    },
    /// Error building HUGR.
    #[error("HugrError: {0:?}")]
    HugrError(#[from] HugrError),
    /// First node in node list must be the HUGR root.
    #[error("The first node in the node list has parent {0:?}, should be itself (index 0)")]
    FirstNodeNotRoot(Node),
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

    fn try_from(hugr: &Hugr) -> Result<Self, Self::Error> {
        // We compact the operation nodes during the serialization process,
        // and ignore the copy nodes.
        let mut node_rekey: HashMap<Node, Node> = HashMap::with_capacity(hugr.node_count());
        for (order, node) in hugr.canonical_order().enumerate() {
            node_rekey.insert(node, NodeIndex::new(order).into());
        }

        let mut nodes = vec![None; hugr.node_count()];
        let mut metadata = vec![json!(null); hugr.node_count()];
        for n in hugr.nodes() {
            let parent = node_rekey[&hugr.get_parent(n).unwrap_or(n)];
            let opt = hugr.get_optype(n);
            let new_node = node_rekey[&n].index.index();
            nodes[new_node] = Some(NodeSer {
                parent,
                op: opt.clone(),
            });
            metadata[new_node] = hugr.get_metadata(n).clone();
        }
        let nodes = nodes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .expect("Could not reach one of the nodes");

        let find_offset = |node: Node, offset: usize, dir: Direction, hugr: &Hugr| {
            let sig = hugr.get_optype(node).signature();
            let offset = match offset < sig.port_count(dir) {
                true => Some(offset as u16),
                false => None,
            };
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
                            let tgt = find_offset(
                                tgt_node,
                                tgt.offset.index(),
                                Direction::Incoming,
                                hugr,
                            );
                            [src, tgt]
                        })
                    })
            })
            .collect();

        Ok(Self {
            nodes,
            edges,
            metadata,
        })
    }
}

impl TryFrom<SerHugrV0> for Hugr {
    type Error = HUGRSerializationError;
    fn try_from(
        SerHugrV0 {
            nodes,
            edges,
            metadata,
        }: SerHugrV0,
    ) -> Result<Self, Self::Error> {
        // Root must be first node
        let mut nodes = nodes.into_iter();
        let NodeSer {
            parent: root_parent,
            op: root_type,
        } = nodes.next().unwrap();
        if root_parent.index.index() != 0 {
            return Err(HUGRSerializationError::FirstNodeNotRoot(root_parent));
        }
        // if there are any unconnected ports or copy nodes the capacity will be
        // an underestimate
        let mut hugr = Hugr::with_capacity(root_type, nodes.len(), edges.len() * 2);

        for node_ser in nodes {
            hugr.add_op_with_parent(node_ser.parent, node_ser.op)?;
        }

        for (node, metadata) in metadata.into_iter().enumerate() {
            let node = NodeIndex::new(node).into();
            hugr.set_metadata(node, metadata);
        }

        let unwrap_offset = |node: Node, offset, dir, hugr: &Hugr| -> Result<usize, Self::Error> {
            if !hugr.graph.contains_node(node.index) {
                return Err(HUGRSerializationError::UnknownEdgeNode { node });
            }
            let offset = match offset {
                Some(offset) => offset as usize,
                None => {
                    let op_type = hugr.get_optype(node);
                    op_type
                        .other_port_index(dir)
                        .ok_or(HUGRSerializationError::MissingPortOffset {
                            node,
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

            hugr.connect(src, src_port, dst, dst_port)?;
        }

        Ok(hugr)
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use crate::{
        builder::{
            Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        ops::{dataflow::IOTrait, Input, LeafOp, Module, Output, DFG},
        types::{ClassicType, Signature, SimpleType},
        Port,
    };
    use itertools::Itertools;
    use portgraph::{
        multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, PortView, UnmanagedDenseMap,
    };

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    const QB: SimpleType = SimpleType::Qubit;

    #[test]
    fn empty_hugr_serialize() {
        let hg = Hugr::default();
        assert_eq!(ser_roundtrip(&hg), hg);
    }

    pub fn ser_roundtrip<T: Serialize + serde::de::DeserializeOwned>(g: &T) -> T {
        let v = rmp_serde::to_vec_named(g).unwrap();
        rmp_serde::from_slice(&v[..]).unwrap()
    }

    /// Generate an optype for a node with a matching amount of inputs and outputs.
    fn gen_optype(g: &MultiPortGraph, node: NodeIndex) -> OpType {
        let inputs = g.num_inputs(node);
        let outputs = g.num_outputs(node);
        match (inputs == 0, outputs == 0) {
            (false, false) => DFG {
                signature: Signature::new_df(
                    vec![ClassicType::bit().into(); inputs - 1],
                    vec![ClassicType::bit().into(); outputs - 1],
                ),
            }
            .into(),
            (true, false) => Input::new(vec![ClassicType::bit().into(); outputs - 1]).into(),
            (false, true) => Output::new(vec![ClassicType::bit().into(); inputs - 1]).into(),
            (true, true) => Module.into(),
        }
    }

    #[test]
    fn simpleser() {
        let mut g = MultiPortGraph::new();

        let root = g.add_node(0, 0);
        let a = g.add_node(1, 1);
        let b = g.add_node(3, 2);
        let c = g.add_node(1, 1);

        g.link_nodes(a, 0, b, 0).unwrap();
        g.link_nodes(a, 0, b, 0).unwrap();
        g.link_nodes(b, 0, b, 1).unwrap();
        g.link_nodes(b, 1, c, 0).unwrap();
        g.link_nodes(b, 1, a, 0).unwrap();
        g.link_nodes(c, 0, a, 0).unwrap();

        let mut h = Hierarchy::new();
        let mut op_types = UnmanagedDenseMap::new();

        for n in [a, b, c] {
            h.push_child(n, root).unwrap();
            op_types[n] = gen_optype(&g, n);
        }

        let hg = Hugr {
            graph: g,
            hierarchy: h,
            root,
            op_types,
            metadata: Default::default(),
        };

        let v = rmp_serde::to_vec_named(&hg).unwrap();

        let newhg = rmp_serde::from_slice(&v[..]).unwrap();
        assert_eq!(hg, newhg);
    }

    #[test]
    fn weighted_hugr_ser() {
        let hugr = {
            let mut module_builder = ModuleBuilder::new();
            module_builder.set_metadata(json!({"name": "test"}));

            let t_row = vec![SimpleType::new_sum(vec![NAT, QB])];
            let mut f_build = module_builder
                .define_function("main", Signature::new_df(t_row.clone(), t_row))
                .unwrap();

            let outputs = f_build
                .input_wires()
                .map(|in_wire| {
                    f_build
                        .add_dataflow_op(
                            LeafOp::Noop {
                                ty: f_build.get_wire_type(in_wire).unwrap(),
                            },
                            [in_wire],
                        )
                        .unwrap()
                        .out_wire(0)
                })
                .collect_vec();
            f_build.set_metadata(json!(42));
            f_build.finish_with_outputs(outputs).unwrap();

            module_builder.finish_hugr().unwrap()
        };

        let ser_hugr: SerHugrV0 = (&hugr).try_into().unwrap();
        // HUGR internal structures are not preserved across serialization, so
        // test equality on SerHugrV0 instead.
        assert_eq!(ser_roundtrip(&ser_hugr), ser_hugr);
    }

    #[test]
    fn metadata_hugr_ser() {
        let hugr = {
            let mut module_builder = ModuleBuilder::new();
            let t_row = vec![SimpleType::new_sum(vec![NAT, QB])];
            let mut f_build = module_builder
                .define_function("main", Signature::new_df(t_row.clone(), t_row))
                .unwrap();

            let outputs = f_build
                .input_wires()
                .map(|in_wire| {
                    f_build
                        .add_dataflow_op(
                            LeafOp::Noop {
                                ty: f_build.get_wire_type(in_wire).unwrap(),
                            },
                            [in_wire],
                        )
                        .unwrap()
                        .out_wire(0)
                })
                .collect_vec();

            f_build.finish_with_outputs(outputs).unwrap();
            module_builder.finish_hugr().unwrap()
        };

        let ser_hugr: SerHugrV0 = (&hugr).try_into().unwrap();
        // HUGR internal structures are not preserved across serialization, so
        // test equality on SerHugrV0 instead.
        assert_eq!(ser_roundtrip(&ser_hugr), ser_hugr);
    }

    #[test]
    fn dfg_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let tp: Vec<SimpleType> = vec![ClassicType::bit().into(); 2];
        let mut dfg = DFGBuilder::new(tp.clone(), tp)?;
        let mut params: [_; 2] = dfg.input_wires_arr();
        for p in params.iter_mut() {
            *p = dfg
                .add_dataflow_op(LeafOp::Xor, [*p, *p])
                .unwrap()
                .out_wire(0);
        }
        let h = dfg.finish_hugr_with_outputs(params)?;

        let ser = serde_json::to_string(&h)?;
        let h_deser: Hugr = serde_json::from_str(&ser)?;

        // Check the canonicalization works
        let mut h_canon = h;
        h_canon.canonicalize_nodes(|_, _| {});

        for node in h_deser.nodes() {
            assert_eq!(h_deser.get_optype(node), h_canon.get_optype(node));
            assert_eq!(h_deser.get_parent(node), h_canon.get_parent(node));
        }

        Ok(())
    }

    #[test]
    fn hierarchy_order() {
        let qb = SimpleType::Qubit;
        let dfg = DFGBuilder::new([qb.clone()].to_vec(), [qb.clone()].to_vec()).unwrap();
        let [old_in, out] = dfg.io();
        let w = dfg.input_wires();
        let mut hugr = dfg.finish_hugr_with_outputs(w).unwrap();

        // Now add a new input
        let new_in = hugr.add_op(Input::new([qb].to_vec()));
        hugr.disconnect(old_in, Port::new_outgoing(0)).unwrap();
        hugr.connect(new_in, 0, out, 0).unwrap();
        hugr.move_before_sibling(new_in, old_in).unwrap();
        hugr.remove_node(old_in).unwrap();
        hugr.validate().unwrap();

        let ser = serde_json::to_vec(&hugr).unwrap();
        let new_hugr: Hugr = serde_json::from_slice(&ser).unwrap();
        new_hugr.validate().unwrap();

        // Check the canonicalization works
        let mut h_canon = hugr.clone();
        h_canon.canonicalize_nodes(|_, _| {});

        for node in new_hugr.nodes() {
            assert_eq!(new_hugr.get_optype(node), h_canon.get_optype(node));
            assert_eq!(new_hugr.get_parent(node), h_canon.get_parent(node));
        }
    }
}

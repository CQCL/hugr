//! Serialization definition for [`Hugr`]
//! [`Hugr`]: crate::hugr::Hugr

use std::collections::HashMap;
use thiserror::Error;

use crate::ops::OpTrait;
use crate::{hugr::Hugr, ops::OpType};
use portgraph::hierarchy::AttachError;
use portgraph::multiportgraph::MultiPortGraph;
use portgraph::{
    Direction, Hierarchy, LinkError, LinkView, NodeIndex, PortView, UnmanagedDenseMap,
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
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SerHugrV0 {
    /// For each node: (parent, num_inputs, num_outputs, node_operation)
    nodes: Vec<(NodeIndex, usize, usize, OpType)>,
    /// for each edge: (src, src_offset, tgt, tgt_offset)
    edges: Vec<[(NodeIndex, Option<u16>); 2]>,
    root: NodeIndex,
}

/// Errors that can occur while serializing a HUGR.
#[derive(Debug, Clone, PartialEq, Error)]
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
        node: NodeIndex,
        /// The operation type of the node.
        op_type: OpType,
    },
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
        // We compact the operation nodes during the serialization process,
        // and ignore the copy nodes.
        let mut node_rekey = HashMap::new();
        let mut nodes: Vec<_> = graph
            .nodes_iter()
            .enumerate()
            .map(|(i, n)| {
                node_rekey.insert(n, NodeIndex::new(i));
                // Note that we don't rekey the parent here, as we need to fully
                // populate `node_rekey` first.
                let parent = hierarchy.parent(n).unwrap_or_else(|| {
                    assert_eq!(*root, n);
                    n
                });
                let opt = &op_types[n];
                Ok((
                    parent,
                    graph.num_inputs(n),
                    graph.num_outputs(n),
                    opt.clone(),
                ))
            })
            .collect::<Result<_, Self::Error>>()?;
        for (parent, _, _, _) in &mut nodes {
            *parent = node_rekey[parent];
        }

        let find_offset = |node: NodeIndex, offset: usize, dir: Direction| {
            let sig = &op_types[node].signature();
            let offset = match offset < sig.port_count(dir) {
                true => Some(offset as u16),
                false => None,
            };
            (node, offset)
        };

        let edges: Vec<_> = graph
            .nodes_iter()
            .flat_map(|node| {
                graph
                    .outputs(node)
                    .enumerate()
                    .flat_map(move |(src_offset, port)| {
                        let src = find_offset(node, src_offset, Direction::Outgoing);
                        graph.port_links(port).map(move |(_, tgt)| {
                            let tgt_node = graph.port_node(tgt).unwrap();
                            let tgt_offset = graph.port_offset(tgt).unwrap().index();
                            let tgt = find_offset(tgt_node, tgt_offset, Direction::Incoming);
                            [src, tgt]
                        })
                    })
            })
            .collect();

        Ok(Self {
            nodes,
            edges,
            root: *root,
        })
    }
}

impl TryFrom<SerHugrV0> for Hugr {
    type Error = HUGRSerializationError;
    fn try_from(SerHugrV0 { nodes, edges, root }: SerHugrV0) -> Result<Self, Self::Error> {
        let mut hierarchy = Hierarchy::new();

        // if there are any unconnected ports or copy nodes the capacity will be
        // an underestimate
        let mut graph = MultiPortGraph::with_capacity(nodes.len(), edges.len() * 2);
        let mut op_types_sec = UnmanagedDenseMap::with_capacity(nodes.len());
        for (parent, incoming, outgoing, typ) in nodes {
            let ni = graph.add_node(incoming, outgoing);
            if parent != ni {
                hierarchy.push_child(ni, parent)?;
            }
            op_types_sec[ni] = typ;
        }

        let unwrap_offset = |node, offset, dir| -> Result<usize, Self::Error> {
            let offset = match offset {
                Some(offset) => offset as usize,
                None => op_types_sec[node]
                    .other_port_index(dir)
                    .ok_or(HUGRSerializationError::MissingPortOffset {
                        node,
                        op_type: op_types_sec[node].clone(),
                    })?
                    .index(),
            };
            Ok(offset)
        };
        for [(srcn, from_offset), (tgtn, to_offset)] in edges {
            let from_offset = unwrap_offset(srcn, from_offset, Direction::Outgoing)?;
            let to_offset = unwrap_offset(tgtn, to_offset, Direction::Incoming)?;
            assert!(from_offset < graph.num_outputs(srcn));
            assert!(to_offset < graph.num_inputs(tgtn));
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
    use crate::{
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        ops::{dataflow::IOTrait, Input, LeafOp, Module, Output, DFG},
        types::{ClassicType, LinearType, Signature, SimpleType},
    };
    use itertools::Itertools;
    use portgraph::proptest::gen_portgraph;
    use proptest::prelude::*;
    proptest! {
        #[test]
        // miri fails due to proptest filesystem access
        #[cfg_attr(miri, ignore)]
        fn prop_serialization(graph in gen_portgraph(100, 50, 1000)) {
            let mut graph : MultiPortGraph = graph.into();
            let root = graph.add_node(0, 0);
            let mut hierarchy = Hierarchy::new();
            let mut op_types = UnmanagedDenseMap::new();
            for n in graph.nodes_iter() {
                if n != root {
                    hierarchy.push_child(n, root).unwrap();
                }
                op_types[n] = gen_optype(&graph, n);
            }

            let hugr = Hugr { graph, hierarchy, root, op_types };

            prop_assert_eq!(ser_roundtrip(&hugr), hugr);
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

        let a = g.add_node(1, 1);
        let b = g.add_node(3, 2);
        let c = g.add_node(1, 1);
        let root = g.add_node(0, 0);

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
        };

        let v = rmp_serde::to_vec_named(&hg).unwrap();

        let newhg = rmp_serde::from_slice(&v[..]).unwrap();
        assert_eq!(hg, newhg);
    }

    #[test]
    fn weighted_hugr_ser() {
        const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
        const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

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
                            LeafOp::Noop(f_build.get_wire_type(in_wire).unwrap()),
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
}

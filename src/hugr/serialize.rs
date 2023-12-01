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

use super::{HugrError, HugrMut, HugrView};

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
    input_extensions: Option<ExtensionSet>,
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
    //
    // TODO: Update to Vec<Option<Map<String,Value>>> to more closely
    // match the internal representation.
    #[serde(default)]
    metadata: Vec<serde_json::Value>,
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
        for (order, node) in hugr.canonical_order(hugr.root()).enumerate() {
            node_rekey.insert(node, portgraph::NodeIndex::new(order).into());
        }

        let mut nodes = vec![None; hugr.node_count()];
        let mut metadata = vec![serde_json::Value::Null; hugr.node_count()];
        for n in hugr.nodes() {
            let parent = node_rekey[&hugr.get_parent(n).unwrap_or(n)];
            let opt = hugr.get_nodetype(n);
            let new_node = node_rekey[&n].index();
            nodes[new_node] = Some(NodeSer {
                parent,
                input_extensions: opt.input_extensions.clone(),
                op: opt.op.clone(),
            });
            let node_metadata = hugr.metadata.get(n.pg_index()).clone();
            metadata[new_node] = match node_metadata {
                Some(m) => serde_json::Value::Object(m.clone()),
                None => serde_json::Value::Null,
            };
        }
        let nodes = nodes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .expect("Could not reach one of the nodes");

        let find_offset = |node: Node, offset: usize, dir: Direction, hugr: &Hugr| {
            let sig = hugr.signature(node).unwrap_or_default();
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
                            let tgt = find_offset(tgt_node, tgt.index(), Direction::Incoming, hugr);
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
            )?;
        }

        for (node, metadata) in metadata.into_iter().enumerate() {
            let node = portgraph::NodeIndex::new(node);
            hugr.metadata[node] = metadata.as_object().cloned();
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
    use crate::builder::{
        test::closed_dfg_root_hugr, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::BOOL_T;
    use crate::extension::{EMPTY_REG, PRELUDE_REGISTRY};
    use crate::hugr::hugrmut::sealed::HugrMutInternals;
    use crate::hugr::NodeType;
    use crate::ops::{dataflow::IOTrait, Input, LeafOp, Module, Output, DFG};
    use crate::types::{FunctionType, Type};
    use crate::OutgoingPort;
    use itertools::Itertools;
    use portgraph::LinkView;
    use portgraph::{
        multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, PortView, UnmanagedDenseMap,
    };

    const NAT: Type = crate::extension::prelude::USIZE_T;
    const QB: Type = crate::extension::prelude::QB_T;

    #[test]
    fn empty_hugr_serialize() {
        let hg = Hugr::default();
        assert_eq!(ser_roundtrip(&hg), hg);
    }

    /// Serialize and deserialize a value.
    pub fn ser_roundtrip<T: Serialize + serde::de::DeserializeOwned>(g: &T) -> T {
        let v = rmp_serde::to_vec_named(g).unwrap();
        rmp_serde::from_slice(&v[..]).unwrap()
    }

    /// Serialize and deserialize a HUGR, and check that the result is the same as the original.
    ///
    /// Returns the deserialized HUGR.
    pub fn check_hugr_roundtrip(hugr: &Hugr) -> Hugr {
        let new_hugr: Hugr = ser_roundtrip(hugr);

        // Original HUGR, with canonicalized node indices
        //
        // The internal port indices may still be different.
        let mut h_canon = hugr.clone();
        h_canon.canonicalize_nodes(|_, _| {});

        assert_eq!(new_hugr.root, h_canon.root);
        assert_eq!(new_hugr.hierarchy, h_canon.hierarchy);
        assert_eq!(new_hugr.op_types, h_canon.op_types);
        assert_eq!(new_hugr.metadata, h_canon.metadata);

        // Check that the graphs are equivalent up to port renumbering.
        let new_graph = &new_hugr.graph;
        let old_graph = &h_canon.graph;
        assert_eq!(new_graph.node_count(), old_graph.node_count());
        assert_eq!(new_graph.port_count(), old_graph.port_count());
        assert_eq!(new_graph.link_count(), old_graph.link_count());
        for n in old_graph.nodes_iter() {
            assert_eq!(new_graph.num_inputs(n), old_graph.num_inputs(n));
            assert_eq!(new_graph.num_outputs(n), old_graph.num_outputs(n));
            assert_eq!(
                new_graph.output_neighbours(n).collect_vec(),
                old_graph.output_neighbours(n).collect_vec()
            );
        }

        new_hugr
    }

    /// Generate an optype for a node with a matching amount of inputs and outputs.
    fn gen_optype(g: &MultiPortGraph, node: portgraph::NodeIndex) -> OpType {
        let inputs = g.num_inputs(node);
        let outputs = g.num_outputs(node);
        match (inputs == 0, outputs == 0) {
            (false, false) => DFG {
                signature: FunctionType::new(vec![NAT; inputs - 1], vec![NAT; outputs - 1]),
            }
            .into(),
            (true, false) => Input::new(vec![NAT; outputs - 1]).into(),
            (false, true) => Output::new(vec![NAT; inputs - 1]).into(),
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

        op_types[root] = NodeType::new_open(gen_optype(&g, root));

        for n in [a, b, c] {
            h.push_child(n, root).unwrap();
            op_types[n] = NodeType::new_pure(gen_optype(&g, n));
        }

        let hugr = Hugr {
            graph: g,
            hierarchy: h,
            root,
            op_types,
            metadata: Default::default(),
        };

        check_hugr_roundtrip(&hugr);
    }

    #[test]
    fn weighted_hugr_ser() {
        let hugr = {
            let mut module_builder = ModuleBuilder::new();
            module_builder.set_metadata("name", "test");

            let t_row = vec![Type::new_sum(vec![NAT, QB])];
            let mut f_build = module_builder
                .define_function("main", FunctionType::new(t_row.clone(), t_row).into())
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
            f_build.set_metadata("val", 42);
            f_build.finish_with_outputs(outputs).unwrap();

            module_builder.finish_prelude_hugr().unwrap()
        };

        check_hugr_roundtrip(&hugr);
    }

    #[test]
    fn dfg_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let tp: Vec<Type> = vec![BOOL_T; 2];
        let mut dfg = DFGBuilder::new(FunctionType::new(tp.clone(), tp))?;
        let mut params: [_; 2] = dfg.input_wires_arr();
        for p in params.iter_mut() {
            *p = dfg
                .add_dataflow_op(LeafOp::Noop { ty: BOOL_T }, [*p])
                .unwrap()
                .out_wire(0);
        }
        let hugr = dfg.finish_hugr_with_outputs(params, &EMPTY_REG)?;

        check_hugr_roundtrip(&hugr);
        Ok(())
    }

    #[test]
    fn canonicalisation() {
        let mut hugr = closed_dfg_root_hugr(FunctionType::new(vec![QB], vec![QB]));
        let [old_in, out] = hugr.get_io(hugr.root()).unwrap();
        hugr.connect(old_in, 0, out, 0).unwrap();

        // Now add a new input
        let new_in = hugr.add_node(Input::new([QB].to_vec()).into());
        hugr.disconnect(old_in, OutgoingPort::from(0)).unwrap();
        hugr.connect(new_in, 0, out, 0).unwrap();
        hugr.move_before_sibling(new_in, old_in).unwrap();
        hugr.remove_node(old_in).unwrap();
        hugr.update_validate(&PRELUDE_REGISTRY).unwrap();

        let new_hugr: Hugr = check_hugr_roundtrip(&hugr);
        new_hugr.validate(&EMPTY_REG).unwrap_err();
        new_hugr.validate(&PRELUDE_REGISTRY).unwrap();
    }
}

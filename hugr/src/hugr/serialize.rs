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

use super::{HugrMut, HugrView};

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
    V0,
    /// Version 1 of the HUGR serialization format.
    V1(SerHugrV1),

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

/// Version 1 of the HUGR serialization format.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct SerHugrV1 {
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
        let versioned = Versioned::V1(shg);
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

        Ok(Self {
            nodes,
            edges,
            metadata,
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

#[cfg(test)]
pub mod test {

    use super::*;
    use crate::builder::{
        test::closed_dfg_root_hugr, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::BOOL_T;
    use crate::extension::simple_op::MakeRegisteredOp;
    use crate::extension::{EMPTY_REG, PRELUDE_REGISTRY};
    use crate::hugr::hugrmut::sealed::HugrMutInternals;
    use crate::hugr::NodeType;
    use crate::ops::custom::{ExtensionOp, OpaqueOp};
    use crate::ops::{dataflow::IOTrait, Input, Module, Noop, Output, DFG};
    use crate::std_extensions::arithmetic::float_ops::FLOAT_OPS_REGISTRY;
    use crate::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
    use crate::std_extensions::logic::NotOp;
    use crate::types::{FunctionType, Type};
    use crate::{type_row, OutgoingPort};
    use itertools::Itertools;
    use jsonschema::{Draft, JSONSchema};
    use lazy_static::lazy_static;
    use portgraph::LinkView;
    use portgraph::{
        multiportgraph::MultiPortGraph, Hierarchy, LinkMut, PortMut, PortView, UnmanagedDenseMap,
    };

    const NAT: Type = crate::extension::prelude::USIZE_T;
    const QB: Type = crate::extension::prelude::QB_T;

    lazy_static! {
        static ref SCHEMA: JSONSchema = {
            let schema_val: serde_json::Value = serde_json::from_str(include_str!(
                "../../../specification/schema/hugr_schema_v1.json"
            ))
            .unwrap();
            JSONSchema::options()
                .with_draft(Draft::Draft7)
                .compile(&schema_val)
                .expect("Schema is invalid.")
        };
    }

    #[test]
    fn empty_hugr_serialize() {
        let hg = Hugr::default();
        assert_eq!(ser_roundtrip(&hg), hg);
    }

    /// Serialize and deserialize a value.
    pub fn ser_roundtrip<T: Serialize + serde::de::DeserializeOwned>(g: &T) -> T {
        ser_roundtrip_validate(g, None)
    }

    /// Serialize and deserialize a value, optionally validating against a schema.
    pub fn ser_roundtrip_validate<T: Serialize + serde::de::DeserializeOwned>(
        g: &T,
        schema: Option<&JSONSchema>,
    ) -> T {
        let s = serde_json::to_string(g).unwrap();
        let val: serde_json::Value = serde_json::from_str(&s).unwrap();

        if let Some(schema) = schema {
            let validate = schema.validate(&val);

            if let Err(errors) = validate {
                // errors don't necessarily implement Debug
                for error in errors {
                    println!("Validation error: {}", error);
                    println!("Instance path: {}", error.instance_path);
                }
                panic!("Serialization test failed.");
            }
        }
        serde_json::from_str(&s).unwrap()
    }

    /// Serialize and deserialize a HUGR, and check that the result is the same as the original.
    /// Checks the serialized json against the in-tree schema.
    ///
    /// Returns the deserialized HUGR.
    pub fn check_hugr_schema_roundtrip(hugr: &Hugr) -> Hugr {
        check_hugr_roundtrip(hugr, true)
    }

    /// Serialize and deserialize a HUGR, and check that the result is the same as the original.
    ///
    /// If `check_schema` is true, checks the serialized json against the in-tree schema.
    ///
    /// Returns the deserialized HUGR.
    pub fn check_hugr_roundtrip(hugr: &Hugr, check_schema: bool) -> Hugr {
        let new_hugr: Hugr = ser_roundtrip_validate(hugr, check_schema.then_some(&SCHEMA));

        // Original HUGR, with canonicalized node indices
        //
        // The internal port indices may still be different.
        let mut h_canon = hugr.clone();
        h_canon.canonicalize_nodes(|_, _| {});

        assert_eq!(new_hugr.root, h_canon.root);
        assert_eq!(new_hugr.hierarchy, h_canon.hierarchy);
        assert_eq!(new_hugr.metadata, h_canon.metadata);

        // Extension operations may have been downgraded to opaque operations.
        for node in new_hugr.nodes() {
            let new_op = new_hugr.get_optype(node);
            let old_op = h_canon.get_optype(node);
            assert_eq!(new_op, old_op);
        }

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

        check_hugr_schema_roundtrip(&hugr);
    }

    #[test]
    fn weighted_hugr_ser() {
        let hugr = {
            let mut module_builder = ModuleBuilder::new();
            module_builder.set_metadata("name", "test");

            let t_row = vec![Type::new_sum([type_row![NAT], type_row![QB]])];
            let mut f_build = module_builder
                .define_function("main", FunctionType::new(t_row.clone(), t_row).into())
                .unwrap();

            let outputs = f_build
                .input_wires()
                .map(|in_wire| {
                    f_build
                        .add_dataflow_op(
                            Noop {
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

        check_hugr_schema_roundtrip(&hugr);
    }

    #[test]
    fn dfg_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let tp: Vec<Type> = vec![BOOL_T; 2];
        let mut dfg = DFGBuilder::new(FunctionType::new(tp.clone(), tp))?;
        let mut params: [_; 2] = dfg.input_wires_arr();
        for p in params.iter_mut() {
            *p = dfg
                .add_dataflow_op(Noop { ty: BOOL_T }, [*p])
                .unwrap()
                .out_wire(0);
        }
        let hugr = dfg.finish_hugr_with_outputs(params, &EMPTY_REG)?;

        check_hugr_schema_roundtrip(&hugr);
        Ok(())
    }

    #[test]
    fn opaque_ops() -> Result<(), Box<dyn std::error::Error>> {
        let tp: Vec<Type> = vec![BOOL_T; 1];
        let mut dfg = DFGBuilder::new(FunctionType::new_endo(tp))?;
        let [wire] = dfg.input_wires_arr();

        // Add an extension operation
        let extension_op: ExtensionOp = NotOp.to_extension_op().unwrap();
        let wire = dfg
            .add_dataflow_op(extension_op.clone(), [wire])
            .unwrap()
            .out_wire(0);

        // Add an unresolved opaque operation
        let opaque_op: OpaqueOp = extension_op.into();
        let wire = dfg.add_dataflow_op(opaque_op, [wire]).unwrap().out_wire(0);

        let hugr = dfg.finish_hugr_with_outputs([wire], &PRELUDE_REGISTRY)?;

        check_hugr_schema_roundtrip(&hugr);
        Ok(())
    }

    #[test]
    fn function_type() -> Result<(), Box<dyn std::error::Error>> {
        let fn_ty = Type::new_function(FunctionType::new_endo(type_row![BOOL_T]));
        let mut bldr = DFGBuilder::new(FunctionType::new_endo(vec![fn_ty.clone()]))?;
        let op = bldr.add_dataflow_op(Noop { ty: fn_ty }, bldr.input_wires())?;
        let h = bldr.finish_prelude_hugr_with_outputs(op.outputs())?;

        check_hugr_schema_roundtrip(&h);
        Ok(())
    }

    #[test]
    fn hierarchy_order() -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr = closed_dfg_root_hugr(FunctionType::new(vec![QB], vec![QB]));
        let [old_in, out] = hugr.get_io(hugr.root()).unwrap();
        hugr.connect(old_in, 0, out, 0);

        // Now add a new input
        let new_in = hugr.add_node(Input::new([QB].to_vec()).into());
        hugr.disconnect(old_in, OutgoingPort::from(0));
        hugr.connect(new_in, 0, out, 0);
        hugr.move_before_sibling(new_in, old_in);
        hugr.remove_node(old_in);
        hugr.update_validate(&PRELUDE_REGISTRY)?;

        let new_hugr: Hugr = check_hugr_schema_roundtrip(&hugr);
        new_hugr.validate(&EMPTY_REG).unwrap_err();
        new_hugr.validate(&PRELUDE_REGISTRY)?;
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore = "Extension ops cannot be used with miri.")]
    // Miri doesn't run the extension registration required by `typetag` for registering `CustomConst`s.
    // https://github.com/rust-lang/miri/issues/450
    fn constants_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = DFGBuilder::new(FunctionType::new(vec![], vec![FLOAT64_TYPE])).unwrap();
        let w = builder.add_load_const(ConstF64::new(0.5));
        let hugr = builder.finish_hugr_with_outputs([w], &FLOAT_OPS_REGISTRY)?;

        let ser = serde_json::to_string(&hugr)?;
        let deser = serde_json::from_str(&ser)?;

        assert_eq!(hugr, deser);

        Ok(())
    }
}

use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::views::HugrView;
use crate::hugr::{NodeMetadata, ValidationError};
use crate::ops::{self, MakeTuple, OpTag, OpTrait, OpType, Tag};
use crate::utils::collect_array;
use crate::{IncomingPort, Node, OutgoingPort};

use std::iter;

use super::{
    handle::{BuildHandle, Outputs},
    CircuitBuilder,
};
use super::{BuilderWiringError, FunctionBuilder};

use crate::{
    hugr::NodeType,
    ops::handle::{ConstID, DataflowOpID, FuncID, NodeHandle},
    types::EdgeKind,
};

use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE_REGISTRY};
use crate::types::{PolyFuncType, FunctionType, Type, TypeArg, TypeRow};

use itertools::Itertools;

use super::{
    cfg::CFGBuilder, conditional::ConditionalBuilder, dataflow::DFGBuilder,
    tail_loop::TailLoopBuilder, BuildError, Wire,
};

use crate::Hugr;

use crate::hugr::HugrMut;

/// Trait for HUGR container builders.
/// Containers are nodes that are parents of sibling graphs.
/// Implementations of this trait allow the child sibling graph to be added to
/// the HUGR.
pub trait Container {
    /// The container node.
    fn container_node(&self) -> Node;
    /// The underlying [`Hugr`] being built
    fn hugr_mut(&mut self) -> &mut Hugr;
    /// Immutable reference to HUGR being built
    fn hugr(&self) -> &Hugr;
    /// Add an [`OpType`] as the final child of the container.
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Node {
        let parent = self.container_node();
        self.hugr_mut().add_node_with_parent(parent, op)
    }
    /// Add a [`NodeType`] as the final child of the container.
    fn add_child_node(&mut self, node: NodeType) -> Node {
        let parent = self.container_node();
        self.hugr_mut().add_node_with_parent(parent, node)
    }

    /// Adds a non-dataflow edge between two nodes. The kind is given by the operation's [`other_inputs`] or  [`other_outputs`]
    ///
    /// [`other_inputs`]: crate::ops::OpTrait::other_input
    /// [`other_outputs`]: crate::ops::OpTrait::other_output
    fn add_other_wire(&mut self, src: Node, dst: Node) -> Wire {
        let (src_port, _) = self.hugr_mut().add_other_edge(src, dst);
        Wire::new(src, src_port)
    }

    /// Add a constant value to the container and return a handle to it.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::Const`] node.
    fn add_constant(&mut self, constant: impl Into<ops::Const>) -> ConstID {
        self.add_child_node(NodeType::new_pure(constant.into()))
            .into()
    }

    /// Add a [`ops::FuncDefn`] node and returns a builder to define the function
    /// body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    fn define_function(
        &mut self,
        name: impl Into<String>,
        signature: PolyFuncType<false>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let body = signature.body().clone();
        let f_node = self.add_child_node(NodeType::new_pure(ops::FuncDefn {
            name: name.into(),
            signature,
        }));

        let db =
            DFGBuilder::create_with_io(self.hugr_mut(), f_node, body, Some(ExtensionSet::new()))?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Insert a HUGR as a child of the container.
    fn add_hugr(&mut self, child: Hugr) -> InsertionResult {
        let parent = self.container_node();
        self.hugr_mut().insert_hugr(parent, child)
    }

    /// Insert a copy of a HUGR as a child of the container.
    fn add_hugr_view(&mut self, child: &impl HugrView) -> InsertionResult {
        let parent = self.container_node();
        self.hugr_mut().insert_from_view(parent, child)
    }

    /// Add metadata to the container node.
    fn set_metadata(&mut self, key: impl AsRef<str>, meta: impl Into<NodeMetadata>) {
        let parent = self.container_node();
        // Implementor's container_node() should be a valid node
        self.hugr_mut().set_metadata(parent, key, meta);
    }

    /// Add metadata to a child node.
    ///
    /// Returns an error if the specified `child` is not a child of this container
    fn set_child_metadata(
        &mut self,
        child: Node,
        key: impl AsRef<str>,
        meta: impl Into<NodeMetadata>,
    ) {
        self.hugr_mut().set_metadata(child, key, meta);
    }
}

/// Types implementing this trait can be used to build complete HUGRs
/// (with varying root node types)
pub trait HugrBuilder: Container {
    /// Finish building the HUGR, perform any validation checks and return it.
    fn finish_hugr(self, extension_registry: &ExtensionRegistry) -> Result<Hugr, ValidationError>;

    /// Finish building the HUGR (as [HugrBuilder::finish_hugr]),
    /// validating against the [prelude] extension only
    ///
    /// [prelude]: crate::extension::prelude
    fn finish_prelude_hugr(self) -> Result<Hugr, ValidationError>
    where
        Self: Sized,
    {
        self.finish_hugr(&PRELUDE_REGISTRY)
    }
}

/// Types implementing this trait build a container graph region by borrowing a HUGR
pub trait SubContainer: Container {
    /// A handle to the finished container node, typically returned when the
    /// child graph has been finished.
    type ContainerHandle;
    /// Consume the container builder and return the handle, may perform some
    /// checks before finishing.
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError>;
}
/// Trait for building dataflow regions of a HUGR.
pub trait Dataflow: Container {
    /// Return the number of inputs to the dataflow sibling graph.
    fn num_inputs(&self) -> usize;
    /// Return indices of input and output nodes.
    fn io(&self) -> [Node; 2] {
        self.hugr()
            .children(self.container_node())
            .take(2)
            .collect_vec()
            .try_into()
            .expect("First two children should be IO")
    }
    /// Handle to input node.
    fn input(&self) -> BuildHandle<DataflowOpID> {
        (self.io()[0], self.num_inputs()).into()
    }
    /// Handle to output node.
    fn output(&self) -> DataflowOpID {
        self.io()[1].into()
    }
    /// Return iterator over all input Value wires.
    fn input_wires(&self) -> Outputs {
        self.input().outputs()
    }
    /// Add a dataflow op to the sibling graph, wiring up the `input_wires` to the
    /// incoming ports of the resulting node.
    ///
    /// # Errors
    ///
    /// Returns a [`BuildError::OperationWiring`] error if the `input_wires` cannot be connected.
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        self.add_dataflow_node(NodeType::new_auto(op), input_wires)
    }

    /// Add a dataflow [`NodeType`] to the sibling graph, wiring up the `input_wires` to the
    /// incoming ports of the resulting node.
    ///
    /// # Errors
    ///
    /// Returns a [`BuildError::OperationWiring`] error if the `input_wires` cannot be connected.
    fn add_dataflow_node(
        &mut self,
        nodetype: NodeType,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let outs = add_node_with_wires(self, nodetype, input_wires)?;

        Ok(outs.into())
    }

    /// Insert a hugr-defined op to the sibling graph, wiring up the
    /// `input_wires` to the incoming ports of the resulting root node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the
    /// node.
    fn add_hugr_with_wires(
        &mut self,
        hugr: Hugr,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let optype = hugr.get_optype(hugr.root()).clone();
        let num_outputs = optype.value_output_count();
        let node = self.add_hugr(hugr).new_root;

        wire_up_inputs(input_wires, node, self)
            .map_err(|error| BuildError::OperationWiring { op: optype, error })?;

        Ok((node, num_outputs).into())
    }

    /// Copy a hugr-defined op into the sibling graph, wiring up the
    /// `input_wires` to the incoming ports of the resulting root node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the
    /// node.
    fn add_hugr_view_with_wires(
        &mut self,
        hugr: &impl HugrView,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let node = self.add_hugr_view(hugr).new_root;
        let optype = hugr.get_optype(hugr.root()).clone();
        let num_outputs = optype.value_output_count();

        wire_up_inputs(input_wires, node, self)
            .map_err(|error| BuildError::OperationWiring { op: optype, error })?;

        Ok((node, num_outputs).into())
    }

    /// Wire up the `output_wires` to the input ports of the Output node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when wiring up.
    fn set_outputs(
        &mut self,
        output_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        let [_, out] = self.io();
        wire_up_inputs(output_wires.into_iter().collect_vec(), out, self).map_err(|error| {
            BuildError::OutputWiring {
                container_op: self.hugr().get_optype(self.container_node()).clone(),
                container_node: self.container_node(),
                error,
            }
        })
    }

    /// Return an array of the input wires.
    ///
    /// # Panics
    ///
    /// Panics if the number of input Wires does not match the size of the array.
    fn input_wires_arr<const N: usize>(&self) -> [Wire; N] {
        collect_array(self.input_wires())
    }

    /// Return a builder for a [`crate::ops::DFG`] node, i.e. a nested dataflow subgraph.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the DFG node.
    // TODO: Should this be one function, or should there be a temporary "op" one like with the others?
    fn dfg_builder(
        &mut self,
        signature: FunctionType,
        input_extensions: Option<ExtensionSet>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<DFGBuilder<&mut Hugr>, BuildError> {
        let op = ops::DFG {
            signature: signature.clone(),
        };
        let nodetype = NodeType::new(op, input_extensions.clone());
        let (dfg_n, _) = add_node_with_wires(self, nodetype, input_wires)?;

        DFGBuilder::create_with_io(self.hugr_mut(), dfg_n, signature, input_extensions)
    }

    /// Return a builder for a [`crate::ops::CFG`] node,
    /// i.e. a nested controlflow subgraph.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the CFG node.
    fn cfg_builder(
        &mut self,
        inputs: impl IntoIterator<Item = (Type, Wire)>,
        input_extensions: impl Into<Option<ExtensionSet>>,
        output_types: TypeRow,
        extension_delta: ExtensionSet,
    ) -> Result<CFGBuilder<&mut Hugr>, BuildError> {
        let (input_types, input_wires): (Vec<Type>, Vec<Wire>) = inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (cfg_node, _) = add_node_with_wires(
            self,
            NodeType::new(
                ops::CFG {
                    signature: FunctionType::new(inputs.clone(), output_types.clone())
                        .with_extension_delta(extension_delta),
                },
                input_extensions.into(),
            ),
            input_wires,
        )?;
        CFGBuilder::create(self.hugr_mut(), cfg_node, inputs, output_types)
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`OpType::LoadConstant`] node.
    fn load_const(&mut self, cid: &ConstID) -> Wire {
        let const_node = cid.node();
        let nodetype = self.hugr().get_nodetype(const_node);
        let op: ops::Const = nodetype
            .op()
            .clone()
            .try_into()
            .expect("ConstID does not refer to Const op.");

        let load_n = self
            .add_dataflow_op(
                ops::LoadConstant {
                    datatype: op.get_type().clone(),
                },
                // Constant wire from the constant value node
                vec![Wire::new(const_node, OutgoingPort::from(0))],
            )
            .expect("The constant type should match the LoadConstant type.");

        load_n.out_wire(0)
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`ops::Const`] and a [`ops::LoadConstant`] node.
    fn add_load_const(&mut self, constant: impl Into<ops::Const>) -> Wire {
        let cid = self.add_constant(constant);
        self.load_const(&cid)
    }

    /// Load a [`ops::Value`] and return the local dataflow wire for that constant.
    /// Adds a [`ops::Const`] and a [`ops::LoadConstant`] node.
    fn add_load_value(&mut self, constant: impl Into<ops::Value>) -> Wire {
        self.add_load_const(constant.into())
    }

    /// Load a static function and return the local dataflow wire for that function.
    /// Adds a [`OpType::LoadFunction`] node.
    ///
    /// The `DEF` const generic is used to indicate whether the function is defined
    /// or just declared.
    fn load_func<const DEFINED: bool>(
        &mut self,
        fid: &FuncID<DEFINED>,
        type_args: &[TypeArg],
        // Sadly required as we substituting in type_args may result in recomputing bounds of types:
        exts: &ExtensionRegistry,
    ) -> Result<Wire, BuildError> {
        let func_node = fid.node();
        let func_op = self.hugr().get_nodetype(func_node).op();
        let func_sig = match func_op {
            OpType::FuncDefn(ops::FuncDefn { signature, .. })
            | OpType::FuncDecl(ops::FuncDecl { signature, .. }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: func_node,
                    op_desc: "FuncDecl/FuncDefn",
                })
            }
        };

        let load_n = self.add_dataflow_op(
            ops::LoadFunction::try_new(func_sig, type_args, exts)?,
            // Static wire from the function node
            vec![Wire::new(func_node, func_op.static_output_port().unwrap())],
        )?;

        Ok(load_n.out_wire(0))
    }

    /// Return a builder for a [`crate::ops::TailLoop`] node.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the [`ops::TailLoop`] node.
    fn tail_loop_builder(
        &mut self,
        just_inputs: impl IntoIterator<Item = (Type, Wire)>,
        inputs_outputs: impl IntoIterator<Item = (Type, Wire)>,
        just_out_types: TypeRow,
    ) -> Result<TailLoopBuilder<&mut Hugr>, BuildError> {
        let (input_types, mut input_wires): (Vec<Type>, Vec<Wire>) =
            just_inputs.into_iter().unzip();
        let (rest_types, rest_input_wires): (Vec<Type>, Vec<Wire>) =
            inputs_outputs.into_iter().unzip();
        input_wires.extend(rest_input_wires);

        let tail_loop = ops::TailLoop {
            just_inputs: input_types.into(),
            just_outputs: just_out_types,
            rest: rest_types.into(),
        };
        // TODO: Make input extensions a parameter
        let (loop_node, _) = add_node_with_wires(self, tail_loop.clone(), input_wires)?;

        TailLoopBuilder::create_with_io(self.hugr_mut(), loop_node, &tail_loop)
    }

    /// Return a builder for a [`crate::ops::Conditional`] node.
    /// `sum_rows` and `sum_wire` define the type of the Sum
    /// variants and the wire carrying the Sum respectively.
    ///
    /// The `other_inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `outputs` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the Conditional node.
    fn conditional_builder(
        &mut self,
        (sum_rows, sum_wire): (impl IntoIterator<Item = TypeRow>, Wire),
        other_inputs: impl IntoIterator<Item = (Type, Wire)>,
        output_types: TypeRow,
        extension_delta: ExtensionSet,
    ) -> Result<ConditionalBuilder<&mut Hugr>, BuildError> {
        let mut input_wires = vec![sum_wire];
        let (input_types, rest_input_wires): (Vec<Type>, Vec<Wire>) =
            other_inputs.into_iter().unzip();

        input_wires.extend(rest_input_wires);
        let inputs: TypeRow = input_types.into();
        let sum_rows: Vec<_> = sum_rows.into_iter().collect();
        let n_cases = sum_rows.len();
        let n_out_wires = output_types.len();

        let conditional_id = self.add_dataflow_op(
            ops::Conditional {
                sum_rows,
                other_inputs: inputs,
                outputs: output_types,
                extension_delta,
            },
            input_wires,
        )?;

        Ok(ConditionalBuilder {
            base: self.hugr_mut(),
            conditional_node: conditional_id.node(),
            n_out_wires,
            case_nodes: vec![None; n_cases],
        })
    }

    /// Add an order edge from `before` to `after`. Assumes any additional edges
    /// to both nodes will be Order kind.
    fn set_order(&mut self, before: &impl NodeHandle, after: &impl NodeHandle) {
        self.add_other_wire(before.node(), after.node());
    }

    /// Get the type of a Value [`Wire`]. If not valid port or of Value kind, returns None.
    fn get_wire_type(&self, wire: Wire) -> Result<Type, BuildError> {
        let kind = self.hugr().get_optype(wire.node()).port_kind(wire.source());

        if let Some(EdgeKind::Value(typ)) = kind {
            Ok(typ)
        } else {
            Err(BuildError::WireNotFound(wire))
        }
    }

    /// Add a [`MakeTuple`] node and wire in the `values` Wires,
    /// returning the Wire corresponding to the tuple.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the
    /// [`MakeTuple`] node.
    fn make_tuple(&mut self, values: impl IntoIterator<Item = Wire>) -> Result<Wire, BuildError> {
        let values = values.into_iter().collect_vec();
        let types: Result<Vec<Type>, _> = values
            .iter()
            .map(|&wire| self.get_wire_type(wire))
            .collect();
        let types = types?.into();
        let make_op = self.add_dataflow_op(MakeTuple { tys: types }, values)?;
        Ok(make_op.out_wire(0))
    }

    /// Add a [`Tag`] node and wire in the `value` Wire,
    /// to make a value with Sum type, with `tag` and possible types described
    /// by `variants`.
    /// Returns the Wire corresponding to the Sum value.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the
    /// Tag node.
    fn make_sum(
        &mut self,
        tag: usize,
        variants: impl IntoIterator<Item = TypeRow>,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let make_op = self.add_dataflow_op(
            Tag {
                tag,
                variants: variants.into_iter().map(Into::into).collect_vec(),
            },
            values.into_iter().collect_vec(),
        )?;
        Ok(make_op.out_wire(0))
    }

    /// Use the wires in `values` to return a wire corresponding to the
    /// "Continue" variant of a [`ops::TailLoop`] with `loop_signature`.
    ///
    /// Packs the values in to a tuple and tags appropriately to generate a
    /// value of Sum type.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the nodes.
    fn make_continue(
        &mut self,
        tail_loop: ops::TailLoop,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_sum(0, [tail_loop.just_inputs, tail_loop.just_outputs], values)
    }

    /// Use the wires in `values` to return a wire corresponding to the
    /// "Break" variant of a [`ops::TailLoop`] with `loop_signature`.
    ///
    /// Packs the values in to a tuple and tags appropriately to generate a
    /// value of Sum type.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the nodes.
    fn make_break(
        &mut self,
        loop_op: ops::TailLoop,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_sum(1, [loop_op.just_inputs, loop_op.just_outputs], values)
    }

    /// Add a [`ops::Call`] node, calling `function`, with inputs
    /// specified by `input_wires`. Returns a handle to the corresponding Call node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the Call
    /// node, or if `function` does not refer to a [`ops::FuncDecl`] or
    /// [`ops::FuncDefn`] node.
    fn call<const DEFINED: bool>(
        &mut self,
        function: &FuncID<DEFINED>,
        type_args: &[TypeArg],
        input_wires: impl IntoIterator<Item = Wire>,
        // Sadly required as we substituting in type_args may result in recomputing bounds of types:
        exts: &ExtensionRegistry,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let hugr = self.hugr();
        let def_op = hugr.get_optype(function.node());
        let type_scheme = match def_op {
            OpType::FuncDefn(ops::FuncDefn { signature, .. })
            | OpType::FuncDecl(ops::FuncDecl { signature, .. }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: function.node(),
                    op_desc: "FuncDecl/FuncDefn",
                })
            }
        };
        let op: OpType = ops::Call::try_new(type_scheme, type_args, exts)?.into();
        let const_in_port = op.static_input_port().unwrap();
        let op_id = self.add_dataflow_op(op, input_wires)?;
        let src_port = self.hugr_mut().num_outputs(function.node()) - 1;

        self.hugr_mut()
            .connect(function.node(), src_port, op_id.node(), const_in_port);
        Ok(op_id)
    }

    /// For the vector of `wires`, produce a `CircuitBuilder` where ops can be
    /// added using indices in to the vector.
    fn as_circuit(&mut self, wires: impl IntoIterator<Item = Wire>) -> CircuitBuilder<Self> {
        CircuitBuilder::new(wires, self)
    }
}

/// Add a node to the graph, wiring up the `inputs` to the input ports of the resulting node.
///
/// # Errors
///
/// Returns a [`BuildError::OperationWiring`] if any of the connections produces an
/// invalid edge.
fn add_node_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    nodetype: impl Into<NodeType>,
    inputs: impl IntoIterator<Item = Wire>,
) -> Result<(Node, usize), BuildError> {
    let nodetype: NodeType = nodetype.into();
    let num_outputs = nodetype.op().value_output_count();
    let op_node = data_builder.add_child_node(nodetype.clone());

    wire_up_inputs(inputs, op_node, data_builder).map_err(|error| BuildError::OperationWiring {
        op: nodetype.into_op(),
        error,
    })?;

    Ok((op_node, num_outputs))
}

/// Connect each of the `inputs` wires sequentially to the input ports of
/// `op_node`.
///
/// # Errors
///
/// Returns a [`BuilderWiringError`] if any of the connections produces an
/// invalid edge.
fn wire_up_inputs<T: Dataflow + ?Sized>(
    inputs: impl IntoIterator<Item = Wire>,
    op_node: Node,
    data_builder: &mut T,
) -> Result<(), BuilderWiringError> {
    for (dst_port, wire) in inputs.into_iter().enumerate() {
        wire_up(data_builder, wire.node(), wire.source(), op_node, dst_port)?;
    }
    Ok(())
}

/// Add edge from src to dst.
///
/// # Errors
///
/// Returns a [`BuilderWiringError`] if the edge is invalid.
fn wire_up<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    src: Node,
    src_port: impl Into<OutgoingPort>,
    dst: Node,
    dst_port: impl Into<IncomingPort>,
) -> Result<bool, BuilderWiringError> {
    let src_port = src_port.into();
    let dst_port = dst_port.into();
    let base = data_builder.hugr_mut();

    let src_parent = base.get_parent(src);
    let src_parent_parent = src_parent.and_then(|src| base.get_parent(src));
    let dst_parent = base.get_parent(dst);
    let local_source = src_parent == dst_parent;
    if let EdgeKind::Value(typ) = base.get_optype(src).port_kind(src_port).unwrap() {
        if !local_source {
            // Non-local value sources require a state edge to an ancestor of dst
            if !typ.copyable() {
                return Err(BuilderWiringError::NonCopyableIntergraph {
                    src,
                    src_offset: src_port.into(),
                    dst,
                    dst_offset: dst_port.into(),
                    typ,
                });
            }

            let src_parent = src_parent.expect("Node has no parent");
            let Some(src_sibling) = iter::successors(dst_parent, |&p| base.get_parent(p))
                .tuple_windows()
                .find_map(|(ancestor, ancestor_parent)| {
                    (ancestor_parent == src_parent ||
                        // Dom edge - in CFGs
                        Some(ancestor_parent) == src_parent_parent)
                        .then_some(ancestor)
                })
            else {
                return Err(BuilderWiringError::NoRelationIntergraph {
                    src,
                    src_offset: src_port.into(),
                    dst,
                    dst_offset: dst_port.into(),
                });
            };

            if !OpTag::BasicBlock.is_superset(base.get_optype(src).tag())
                && !OpTag::BasicBlock.is_superset(base.get_optype(src_sibling).tag())
            {
                // Add a state order constraint unless one of the nodes is a CFG BasicBlock
                base.add_other_edge(src, src_sibling);
            }
        } else if !typ.copyable() & base.linked_ports(src, src_port).next().is_some() {
            // Don't copy linear edges.
            return Err(BuilderWiringError::NoCopyLinear {
                typ,
                src,
                src_offset: src_port.into(),
            });
        }
    }

    data_builder
        .hugr_mut()
        .connect(src, src_port, dst, dst_port);
    Ok(local_source
        && matches!(
            data_builder
                .hugr_mut()
                .get_optype(dst)
                .port_kind(dst_port)
                .unwrap(),
            EdgeKind::Value(_)
        ))
}

/// Trait implemented by builders of Dataflow Hugrs
pub trait DataflowHugr: HugrBuilder + Dataflow {
    /// Set outputs of dataflow HUGR and return validated HUGR
    /// # Errors
    ///
    /// * if there is an error when setting outputs
    /// * if the Hugr does not validate
    fn finish_hugr_with_outputs(
        mut self,
        outputs: impl IntoIterator<Item = Wire>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(outputs)?;
        Ok(self.finish_hugr(extension_registry)?)
    }

    /// Sets the outputs of a dataflow Hugr, validates against
    /// the [prelude] extension only, and return the Hugr
    ///
    /// [prelude]: crate::extension::prelude
    fn finish_prelude_hugr_with_outputs(
        self,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<Hugr, BuildError>
    where
        Self: Sized,
    {
        self.finish_hugr_with_outputs(outputs, &PRELUDE_REGISTRY)
    }
}

/// Trait implemented by builders of Dataflow container regions of a HUGR
pub trait DataflowSubContainer: SubContainer + Dataflow {
    /// Set the outputs of the graph and consume the builder, while returning a
    /// handle to the parent.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when setting outputs.
    fn finish_with_outputs(
        mut self,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<Self::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(outputs)?;
        self.finish_sub_container()
    }
}

impl<T: HugrBuilder + Dataflow> DataflowHugr for T {}
impl<T: SubContainer + Dataflow> DataflowSubContainer for T {}

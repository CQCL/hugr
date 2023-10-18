use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::validate::InterGraphEdgeError;
use crate::hugr::views::HugrView;
use crate::hugr::{IncomingPort, Node, NodeMetadata, OutgoingPort, ValidationError};
use crate::ops::{self, LeafOp, OpTrait, OpType};

use std::iter;

use super::FunctionBuilder;
use super::{
    handle::{BuildHandle, Outputs},
    CircuitBuilder,
};

use crate::{
    hugr::NodeType,
    ops::handle::{ConstID, DataflowOpID, FuncID, NodeHandle},
    types::EdgeKind,
};

use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE_REGISTRY};
use crate::types::{FunctionType, Signature, Type, TypeRow};

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
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<Node, BuildError> {
        let parent = self.container_node();
        Ok(self.hugr_mut().add_op_with_parent(parent, op)?)
    }
    /// Add a [`NodeType`] as the final child of the container.
    fn add_child_node(&mut self, node: NodeType) -> Result<Node, BuildError> {
        let parent = self.container_node();
        Ok(self.hugr_mut().add_node_with_parent(parent, node)?)
    }

    /// Adds a non-dataflow edge between two nodes. The kind is given by the operation's [`other_inputs`] or  [`other_outputs`]
    ///
    /// [`other_inputs`]: crate::ops::OpTrait::other_input
    /// [`other_outputs`]: crate::ops::OpTrait::other_output
    fn add_other_wire(&mut self, src: Node, dst: Node) -> Result<Wire, BuildError> {
        let (src_port, _) = self.hugr_mut().add_other_edge(src, dst)?;
        Ok(Wire::new(src, src_port.as_outgoing().unwrap()))
    }

    /// Add a constant value to the container and return a handle to it.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::Const`] node.
    fn add_constant(
        &mut self,
        constant: ops::Const,
        extensions: ExtensionSet,
    ) -> Result<ConstID, BuildError> {
        let const_n = self.add_child_node(NodeType::new(constant, extensions))?;

        Ok(const_n.into())
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
        signature: Signature,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let f_node = self.add_child_node(NodeType::new(
            ops::FuncDefn {
                name: name.into(),
                signature: signature.signature.clone(),
            },
            signature.input_extensions.clone(),
        ))?;

        let db = DFGBuilder::create_with_io(
            self.hugr_mut(),
            f_node,
            signature.signature,
            Some(signature.input_extensions),
        )?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Insert a HUGR as a child of the container.
    fn add_hugr(&mut self, child: Hugr) -> Result<InsertionResult, BuildError> {
        let parent = self.container_node();
        Ok(self.hugr_mut().insert_hugr(parent, child)?)
    }

    /// Insert a copy of a HUGR as a child of the container.
    fn add_hugr_view(&mut self, child: &impl HugrView) -> Result<InsertionResult, BuildError> {
        let parent = self.container_node();
        Ok(self.hugr_mut().insert_from_view(parent, child)?)
    }

    /// Add metadata to the container node.
    fn set_metadata(&mut self, meta: NodeMetadata) {
        let parent = self.container_node();
        // Implementor's container_node() should be a valid node
        self.hugr_mut().set_metadata(parent, meta).unwrap();
    }

    /// Add metadata to a child node.
    ///
    /// Returns an error if the specified `child` is not a child of this container
    fn set_child_metadata(&mut self, child: Node, meta: NodeMetadata) -> Result<(), BuildError> {
        self.hugr_mut().set_metadata(child, meta)?;
        Ok(())
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
    /// This function will return an error if there is an error when adding the node.
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        self.add_dataflow_node(NodeType::open_extensions(op), input_wires)
    }

    /// Add a dataflow [`NodeType`] to the sibling graph, wiring up the `input_wires` to the
    /// incoming ports of the resulting node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the node.
    fn add_dataflow_node(
        &mut self,
        nodetype: NodeType,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let outs = add_node_with_wires(self, nodetype, input_wires.into_iter().collect())?;

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
        let num_outputs = hugr.get_optype(hugr.root()).signature().output_count();
        let node = self.add_hugr(hugr)?.new_root;

        let inputs = input_wires.into_iter().collect();
        wire_up_inputs(inputs, node, self)?;

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
        let num_outputs = hugr.get_optype(hugr.root()).signature().output_count();
        let node = self.add_hugr_view(hugr)?.new_root;

        let inputs = input_wires.into_iter().collect();
        wire_up_inputs(inputs, node, self)?;

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
        wire_up_inputs(output_wires.into_iter().collect_vec(), out, self)
    }

    /// Return an array of the input wires.
    ///
    /// # Panics
    ///
    /// Panics if the number of input Wires does not match the size of the array.
    fn input_wires_arr<const N: usize>(&self) -> [Wire; N] {
        self.input_wires()
            .collect_vec()
            .try_into()
            .expect(&format!("Incorrect number of wires: {N}")[..])
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
        let (dfg_n, _) = add_node_with_wires(self, nodetype, input_wires.into_iter().collect())?;

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
                        .with_extension_delta(&extension_delta),
                },
                input_extensions,
            ),
            input_wires,
        )?;
        CFGBuilder::create(self.hugr_mut(), cfg_node, inputs, output_types)
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`OpType::LoadConstant`] node.
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the node.
    fn load_const(&mut self, cid: &ConstID) -> Result<Wire, BuildError> {
        let const_node = cid.node();
        let nodetype = self.hugr().get_nodetype(const_node);
        let input_extensions = nodetype.input_extensions().cloned();
        let op: &OpType = nodetype.into();
        let op: ops::Const = op
            .clone()
            .try_into()
            .expect("ConstID does not refer to Const op.");

        let load_n = self.add_dataflow_node(
            NodeType::new(
                ops::LoadConstant {
                    datatype: op.const_type().clone(),
                },
                input_extensions,
            ),
            // Constant wire from the constant value node
            vec![Wire::new(const_node, OutgoingPort::from(0))],
        )?;

        Ok(load_n.out_wire(0))
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`ops::LoadConstant`] node.
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the node.
    fn add_load_const(
        &mut self,
        constant: ops::Const,
        extensions: ExtensionSet,
    ) -> Result<Wire, BuildError> {
        let cid = self.add_constant(constant, extensions)?;
        self.load_const(&cid)
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
        let (loop_node, _) = add_op_with_wires(self, tail_loop.clone(), input_wires)?;

        TailLoopBuilder::create_with_io(self.hugr_mut(), loop_node, &tail_loop)
    }

    /// Return a builder for a [`crate::ops::Conditional`] node.
    /// `predicate_inputs` and `predicate_wire` define the type of the predicate
    /// variants and the wire carrying the predicate respectively.
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
        (predicate_inputs, predicate_wire): (impl IntoIterator<Item = TypeRow>, Wire),
        other_inputs: impl IntoIterator<Item = (Type, Wire)>,
        output_types: TypeRow,
        extension_delta: ExtensionSet,
    ) -> Result<ConditionalBuilder<&mut Hugr>, BuildError> {
        let mut input_wires = vec![predicate_wire];
        let (input_types, rest_input_wires): (Vec<Type>, Vec<Wire>) =
            other_inputs.into_iter().unzip();

        input_wires.extend(rest_input_wires);
        let inputs: TypeRow = input_types.into();
        let predicate_inputs: Vec<_> = predicate_inputs.into_iter().collect();
        let n_cases = predicate_inputs.len();
        let n_out_wires = output_types.len();

        let conditional_id = self.add_dataflow_op(
            ops::Conditional {
                predicate_inputs,
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
    fn set_order(
        &mut self,
        before: &impl NodeHandle,
        after: &impl NodeHandle,
    ) -> Result<(), BuildError> {
        self.add_other_wire(before.node(), after.node())?;

        Ok(())
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

    /// Add a [`LeafOp::MakeTuple`] node and wire in the `values` Wires,
    /// returning the Wire corresponding to the tuple.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the
    /// [`LeafOp::MakeTuple`] node.
    fn make_tuple(&mut self, values: impl IntoIterator<Item = Wire>) -> Result<Wire, BuildError> {
        let values = values.into_iter().collect_vec();
        let types: Result<Vec<Type>, _> = values
            .iter()
            .map(|&wire| self.get_wire_type(wire))
            .collect();
        let types = types?.into();
        let make_op = self.add_dataflow_op(LeafOp::MakeTuple { tys: types }, values)?;
        Ok(make_op.out_wire(0))
    }

    /// Add a [`LeafOp::Tag`] node and wire in the `value` Wire,
    /// to make a value with Sum type, with `tag` and possible types described
    /// by `variants`.
    /// Returns the Wire corresponding to the Sum value.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the
    /// Tag node.
    fn make_tag(
        &mut self,
        tag: usize,
        variants: impl Into<TypeRow>,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        let make_op = self.add_dataflow_op(
            LeafOp::Tag {
                tag,
                variants: variants.into(),
            },
            vec![value],
        )?;
        Ok(make_op.out_wire(0))
    }

    /// Add [`LeafOp::MakeTuple`] and [`LeafOp::Tag`] nodes to construct the
    /// `tag` variant of a predicate (sum-of-tuples) type.
    fn make_predicate(
        &mut self,
        tag: usize,
        predicate_variants: impl IntoIterator<Item = TypeRow>,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let tuple = self.make_tuple(values)?;
        let variants = crate::types::predicate_variants_row(predicate_variants);
        let make_op = self.add_dataflow_op(LeafOp::Tag { tag, variants }, vec![tuple])?;
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
        self.make_predicate(0, [tail_loop.just_inputs, tail_loop.just_outputs], values)
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
        self.make_predicate(1, [loop_op.just_inputs, loop_op.just_outputs], values)
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
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let hugr = self.hugr();
        let def_op = hugr.get_optype(function.node());
        let signature = match def_op {
            OpType::FuncDefn(ops::FuncDefn { signature, .. })
            | OpType::FuncDecl(ops::FuncDecl { signature, .. }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: function.node(),
                    op_desc: "FuncDecl/FuncDefn",
                })
            }
        };
        let const_in_port = signature.output.len();
        let op_id = self.add_dataflow_op(ops::Call { signature }, input_wires)?;
        let src_port = self.hugr_mut().num_outputs(function.node()) - 1;

        self.hugr_mut()
            .connect(function.node(), src_port, op_id.node(), const_in_port)?;
        Ok(op_id)
    }

    /// For the vector of `wires`, produce a `CircuitBuilder` where ops can be
    /// added using indices in to the vector.
    fn as_circuit(&mut self, wires: Vec<Wire>) -> CircuitBuilder<Self> {
        CircuitBuilder::new(wires, self)
    }
}

fn add_op_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    optype: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(Node, usize), BuildError> {
    add_node_with_wires(data_builder, NodeType::open_extensions(optype), inputs)
}

fn add_node_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    nodetype: NodeType,
    inputs: Vec<Wire>,
) -> Result<(Node, usize), BuildError> {
    let op_node = data_builder.add_child_node(nodetype.clone())?;
    let sig = nodetype.op_signature();

    wire_up_inputs(inputs, op_node, data_builder)?;

    Ok((op_node, sig.output().len()))
}

fn wire_up_inputs<T: Dataflow + ?Sized>(
    inputs: Vec<Wire>,
    op_node: Node,
    data_builder: &mut T,
) -> Result<(), BuildError> {
    for (dst_port, wire) in inputs.into_iter().enumerate() {
        wire_up(data_builder, wire.node(), wire.source(), op_node, dst_port)?;
    }
    Ok(())
}

/// Add edge from src to dst and report back if they do share a parent
fn wire_up<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    src: Node,
    src_port: impl Into<OutgoingPort>,
    dst: Node,
    dst_port: impl Into<IncomingPort>,
) -> Result<bool, BuildError> {
    let src_port: OutgoingPort = src_port.into();
    let dst_port: IncomingPort = dst_port.into();
    let base = data_builder.hugr_mut();

    let src_parent = base.get_parent(src);
    let dst_parent = base.get_parent(dst);
    let local_source = src_parent == dst_parent;
    if let EdgeKind::Value(typ) = base.get_optype(src).port_kind(src_port).unwrap() {
        if !local_source {
            // Non-local value sources require a state edge to an ancestor of dst
            if !typ.copyable() {
                let val_err: ValidationError = InterGraphEdgeError::NonCopyableData {
                    from: src,
                    from_offset: src_port.into(),
                    to: dst,
                    to_offset: dst_port.into(),
                    ty: EdgeKind::Value(typ),
                }
                .into();
                return Err(val_err.into());
            }

            let src_parent = src_parent.expect("Node has no parent");
            let Some(src_sibling) = iter::successors(dst_parent, |&p| base.get_parent(p))
                .tuple_windows()
                .find_map(|(ancestor, ancestor_parent)| {
                    (ancestor_parent == src_parent).then_some(ancestor)
                })
            else {
                let val_err: ValidationError = InterGraphEdgeError::NoRelation {
                    from: src,
                    from_offset: src_port.into(),
                    to: dst,
                    to_offset: dst_port.into(),
                }
                .into();
                return Err(val_err.into());
            };

            // TODO: Avoid adding duplicate edges
            // This should be easy with https://github.com/CQCL-DEV/hugr/issues/130
            base.add_other_edge(src, src_sibling)?;
        } else if !typ.copyable() & base.linked_ports(src, src_port).next().is_some() {
            // Don't copy linear edges.
            return Err(BuildError::NoCopyLinear(typ));
        }
    }

    data_builder
        .hugr_mut()
        .connect(src, src_port, dst, dst_port)?;
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

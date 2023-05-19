use crate::{
    hugr::{validate::InterGraphEdgeError, ValidationError},
    ops::controlflow::{ConditionalSignature, TailLoopSignature},
};

use std::iter;

use super::{
    handle::{BuildHandle, Outputs},
    CircuitBuilder,
};

use crate::{
    ops::handle::{ConstID, DataflowOpID, FuncID, NodeHandle},
    ops::{controlflow::ControlFlowOp, BasicBlockOp, DataflowOp, LeafOp, ModuleOp, OpType},
    types::{ClassicType, EdgeKind},
};

use crate::types::{Signature, SimpleType, TypeRow};

use itertools::Itertools;
use portgraph::{Direction, NodeIndex, PortOffset};

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
    /// A handle to the finished container node, typically returned when the
    /// child graph has been finished.
    type ContainerHandle;
    /// The container node.
    fn container_node(&self) -> NodeIndex;
    /// The underlying [`HugrMut`] being used to build the HUGR.
    fn base(&mut self) -> &mut HugrMut;
    /// Immutable reference to HUGR being built.
    fn hugr(&self) -> &Hugr;
    /// Add an [`OpType`] as the final child of the container.
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<NodeIndex, BuildError> {
        let parent = self.container_node();
        Ok(self.base().add_op_with_parent(parent, op)?)
    }

    /// Adds a non-dataflow edge between two nodes. The kind is given by the operation's [`OpType::other_inputs`] or  [`OpType::other_outputs`]
    ///
    /// [`OpType::other_inputs`]: crate::ops::OpType::other_inputs
    /// [`OpType::other_outputs`]: crate::ops::OpType::other_outputs
    fn add_other_wire(&mut self, src: NodeIndex, dst: NodeIndex) -> Result<Wire, BuildError> {
        let (src_port, _) = self.base().add_other_edge(src, dst)?;
        Ok(Wire::new(src, src_port))
    }

    /// Consume the container builder and return the handle, may perform some
    /// checks before finishing.
    fn finish(self) -> Self::ContainerHandle;
}

/// Trait for building dataflow regions of a HUGR.
pub trait Dataflow: Container {
    /// Return indices of input and output nodes.
    fn io(&self) -> [NodeIndex; 2];
    /// Return the number of inputs to the dataflow sibling graph.
    fn num_inputs(&self) -> usize;
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
    /// Add a dataflow op to the sibling graph, wiring up the input_wires to the
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
        let outs = add_op_with_wires(self, op, input_wires.into_iter().collect())?;

        Ok(outs.into())
    }

    /// Wire up the output_wires to the input ports of the Output node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when wiring up.
    fn set_outputs(
        &mut self,
        output_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        let [inp, out] = self.io();
        wire_up_inputs(output_wires.into_iter().collect_vec(), out, self, inp)
    }

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
        Ok(self.finish())
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
            .expect(&format!("Incorrect number of wires: {}", N)[..])
    }

    /// Return a builder for a [`crate::ops::dataflow::DataflowOp::DFG`] node, i.e. a nested dataflow subgraph.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the DFG node.
    fn dfg_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: impl IntoIterator<Item = (SimpleType, Wire)>,
        output_types: TypeRow,
    ) -> Result<DFGBuilder<'b>, BuildError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();
        let (dfg_n, _) = add_op_with_wires(
            self,
            OpType::Dataflow(DataflowOp::DFG {
                signature: Signature::new_df(input_types.clone(), output_types.clone()),
            }),
            input_wires,
        )?;

        DFGBuilder::create_with_io(self.base(), dfg_n, input_types.into(), output_types)
    }

    /// Return a builder for a [`crate::ops::controlflow::ControlFlowOp::CFG`] node,
    /// i.e. a nested controlflow subgraph.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the CFG node.
    fn cfg_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: impl IntoIterator<Item = (SimpleType, Wire)>,
        output_types: TypeRow,
    ) -> Result<CFGBuilder<'b>, BuildError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (cfg_node, _) = add_op_with_wires(
            self,
            OpType::Dataflow(DataflowOp::ControlFlow {
                op: ControlFlowOp::CFG {
                    inputs: inputs.clone(),
                    outputs: output_types.clone(),
                },
            }),
            input_wires,
        )?;

        let exit_block_type = OpType::BasicBlock(BasicBlockOp::Exit {
            cfg_outputs: output_types.clone(),
        });
        let exit_node = self.base().add_op_with_parent(cfg_node, exit_block_type)?;
        let n_out_wires = output_types.len();
        let cfg_builder = CFGBuilder {
            base: self.base(),
            cfg_node,
            n_out_wires,
            exit_node,
            inputs: Some(inputs),
        };

        Ok(cfg_builder)
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`DataflowOp::LoadConstant`] node.
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the node.
    fn load_const(&mut self, cid: &ConstID) -> Result<Wire, BuildError> {
        let cn = cid.node();
        let c_out = self.hugr().num_outputs(cn);

        self.base().add_ports(cn, Direction::Outgoing, 1);

        let load_n = self.add_dataflow_op(
            DataflowOp::LoadConstant {
                datatype: cid.const_type(),
            },
            // Constant wire from the constant value node
            vec![Wire::new(cn, c_out)],
        )?;

        Ok(load_n.out_wire(0))
    }

    /// Return a builder for a [`crate::ops::controlflow::ControlFlowOp::TailLoop`] node.
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the TailLoop node.
    fn tail_loop_builder<'a: 'b, 'b>(
        &'a mut self,
        just_inputs: impl IntoIterator<Item = (SimpleType, Wire)>,
        inputs_outputs: impl IntoIterator<Item = (SimpleType, Wire)>,
        just_out_types: TypeRow,
    ) -> Result<TailLoopBuilder<'b>, BuildError> {
        let (input_types, mut input_wires): (Vec<SimpleType>, Vec<Wire>) =
            just_inputs.into_iter().unzip();
        let (rest_types, rest_input_wires): (Vec<SimpleType>, Vec<Wire>) =
            inputs_outputs.into_iter().unzip();
        input_wires.extend(rest_input_wires.into_iter());

        let tail_loop_signature = TailLoopSignature {
            just_inputs: input_types.into(),
            just_outputs: just_out_types,
            rest: rest_types.into(),
        };
        let (loop_node, _) = add_op_with_wires(
            self,
            ControlFlowOp::TailLoop(tail_loop_signature.clone()),
            input_wires,
        )?;

        TailLoopBuilder::create_with_io(self.base(), loop_node, tail_loop_signature)
    }

    /// Return a builder for a [`crate::ops::controlflow::ControlFlowOp::Conditional`] node.
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
    fn conditional_builder<'a: 'b, 'b>(
        &'a mut self,
        (predicate_inputs, predicate_wire): (impl IntoIterator<Item = TypeRow>, Wire),
        other_inputs: impl IntoIterator<Item = (SimpleType, Wire)>,
        output_types: TypeRow,
    ) -> Result<ConditionalBuilder<'b>, BuildError> {
        let mut input_wires = vec![predicate_wire];
        let (input_types, rest_input_wires): (Vec<SimpleType>, Vec<Wire>) =
            other_inputs.into_iter().unzip();

        input_wires.extend(rest_input_wires);
        let inputs: TypeRow = input_types.into();
        let predicate_inputs: Vec<_> = predicate_inputs.into_iter().collect();

        let n_cases = predicate_inputs.len();
        let n_out_wires = output_types.len();
        let conditional_id = self.add_dataflow_op(
            ControlFlowOp::Conditional(ConditionalSignature {
                predicate_inputs,
                other_inputs: inputs,
                outputs: output_types,
            }),
            input_wires,
        )?;
        Ok(ConditionalBuilder {
            base: self.base(),
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
    fn get_wire_type(&self, wire: Wire) -> Result<SimpleType, BuildError> {
        let kind = self
            .hugr()
            .get_optype(wire.node())
            .port_kind(PortOffset::new_outgoing(wire.offset()));

        if let Some(EdgeKind::Value(typ)) = kind {
            Ok(typ)
        } else {
            Err(BuildError::WireNotFound(wire))
        }
    }

    /// Add a discard (0-arity copy) for a `wire` with a known type `typ`.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the
    /// copy node.
    fn discard_type(
        &mut self,
        wire: Wire,
        typ: ClassicType,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        self.add_dataflow_op(LeafOp::Copy { n_copies: 0, typ }, [wire])
    }

    /// Discard a value on a `wire` using [`Dataflow::discard_type`], retrieving
    /// the type of the Wire from it's source.
    ///
    /// # Errors
    ///
    /// This function will return an error if ther is an error when adding the
    /// copy node.
    fn discard(&mut self, wire: Wire) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let typ = self.get_wire_type(wire)?;
        let typ = match typ {
            SimpleType::Classic(typ) => typ,
            SimpleType::Linear(typ) => return Err(BuildError::NoCopyLinear(typ)),
        };
        self.discard_type(wire, typ)
    }

    /// Add a [`LeafOp::MakeTuple`] node and wire in the `values` Wires,
    /// returning the Wire corresponding to the tuple.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the
    /// MakeTuple node.
    fn make_tuple(&mut self, values: impl IntoIterator<Item = Wire>) -> Result<Wire, BuildError> {
        let values = values.into_iter().collect_vec();
        let types: Result<Vec<SimpleType>, _> = values
            .iter()
            .map(|&wire| self.get_wire_type(wire))
            .collect();
        let types = types?.into();
        let make_op = self.add_dataflow_op(LeafOp::MakeTuple(types), values)?;
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
    fn make_tag(&mut self, tag: usize, variants: TypeRow, value: Wire) -> Result<Wire, BuildError> {
        let make_op = self.add_dataflow_op(LeafOp::Tag { tag, variants }, vec![value])?;
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
        let variants = TypeRow::predicate_variants_row(predicate_variants);
        let make_op = self.add_dataflow_op(LeafOp::Tag { tag, variants }, vec![tuple])?;
        Ok(make_op.out_wire(0))
    }

    /// Use the wires in `values` to return a wire corresponding to the
    /// "Continue" variant of a TailLoop with `loop_signature`.
    ///
    /// Packs the values in to a tuple and tags appropriately to generate a
    /// value of Sum type.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the nodes.
    fn make_continue(
        &mut self,
        loop_signature: TailLoopSignature,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_predicate(
            0,
            [loop_signature.just_inputs, loop_signature.just_outputs],
            values,
        )
    }

    /// Use the wires in `values` to return a wire corresponding to the
    /// "Break" variant of a TailLoop with `loop_signature`.
    ///
    /// Packs the values in to a tuple and tags appropriately to generate a
    /// value of Sum type.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the nodes.
    fn make_break(
        &mut self,
        loop_signature: TailLoopSignature,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_predicate(
            1,
            [loop_signature.just_inputs, loop_signature.just_outputs],
            values,
        )
    }

    /// Add a [`DataflowOp::Call`] node, calling `function`, with inputs
    /// specified by `input_wires`. Returns a handle to the corresponding Call node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the Call
    /// node, or if `function` does not refer to a [`ModuleOp::Declare`] or
    /// [`ModuleOp::Def`] node.
    fn call<const DEFINED: bool>(
        &mut self,
        function: &FuncID<DEFINED>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let def_op: Result<&ModuleOp, ()> = self.hugr().get_optype(function.node()).try_into();
        let signature = match def_op {
            Ok(ModuleOp::Def { signature } | ModuleOp::Declare { signature }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: function.node(),
                    op_desc: "Declare/Def",
                })
            }
        };
        let const_in_port = signature.output.len();
        let op_id = self.add_dataflow_op(DataflowOp::Call { signature }, input_wires)?;
        let src_port: usize = self
            .base()
            .add_ports(function.node(), Direction::Outgoing, 1)
            .collect_vec()[0];

        self.base()
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
    op: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(NodeIndex, usize), BuildError> {
    let [inp, out] = data_builder.io();

    let op: OpType = op.into();
    let sig = op.signature();
    let op_node = data_builder.base().add_op_before(out, op)?;

    wire_up_inputs(inputs, op_node, data_builder, inp)?;

    Ok((op_node, sig.output.len()))
}

fn wire_up_inputs<T: Dataflow + ?Sized>(
    inputs: Vec<Wire>,
    op_node: NodeIndex,
    data_builder: &mut T,
    inp: NodeIndex,
) -> Result<(), BuildError> {
    let mut any_local_inputs = false;
    for (dst_port, wire) in inputs.into_iter().enumerate() {
        any_local_inputs |= wire_up(data_builder, wire.node(), wire.offset(), op_node, dst_port)?;
    }

    if !any_local_inputs {
        // If op has no inputs add a StateOrder edge from input to place in
        // causal cone of Input node
        data_builder.add_other_wire(inp, op_node)?;
    };
    Ok(())
}

/// Add edge from src to dst and report back if they do share a parent
fn wire_up<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    mut src: NodeIndex,
    mut src_port: usize,
    dst: NodeIndex,
    dst_port: usize,
) -> Result<bool, BuildError> {
    let base = data_builder.base();
    let src_offset = PortOffset::new_outgoing(src_port);

    let src_parent = base.hugr().get_parent(src);
    let dst_parent = base.hugr().get_parent(dst);
    let local_source = src_parent == dst_parent;
    if !local_source {
        if let Some(copy_port) = if_copy_add_port(base, src) {
            src_port = copy_port;
        } else if let Some(typ) = check_classical_value(base, src, src_offset)? {
            let src_parent = base.hugr().get_parent(src).expect("Node has no parent");

            let final_child = base
                .hugr()
                .children(src_parent)
                .next_back()
                .expect("Parent must have at least one child.");
            let copy_node = base.add_op_before(final_child, LeafOp::Copy { n_copies: 1, typ })?;

            base.connect(src, src_port, copy_node, 0)?;

            // Copy node has to have state edge to an ancestor of dst
            let Some(src_sibling) = iter::successors(dst_parent, |&p| base.hugr().get_parent(p))
                .tuple_windows()
                .find_map(|(ancestor, ancestor_parent)| {
                    (ancestor_parent == src_parent).then_some(ancestor)
                }) else {
                    let val_err: ValidationError = InterGraphEdgeError::NoRelation {
                        from: src,
                        from_offset: PortOffset::new_outgoing(src_port),
                        to: dst,
                        to_offset: PortOffset::new_incoming(dst_port),
                    }.into();
                    return Err(val_err.into());
                };

            base.add_other_edge(copy_node, src_sibling)?;

            src = copy_node;
            src_port = 0;
        }
    }

    if let Some((connected, connected_offset)) = base.hugr().linked_port(src, src_offset) {
        if let Some(copy_port) = if_copy_add_port(base, src) {
            src_port = copy_port;
            src = connected;
        }
        // Need to insert a copy - first check can be copied
        else if let Some(typ) = check_classical_value(base, src, src_offset)? {
            // TODO API consistency in using PortOffset vs. usize
            base.disconnect(src, src_port, Direction::Outgoing)?;

            let copy = data_builder.add_dataflow_op(
                LeafOp::Copy { n_copies: 2, typ },
                [Wire::new(src, src_port)],
            )?;

            let base = data_builder.base();
            base.connect(copy.node(), 0, connected, connected_offset.index())?;
            src = copy.node();
            src_port = 1;
        }
    }
    data_builder.base().connect(src, src_port, dst, dst_port)?;
    Ok(local_source)
}

/// Check the kind of a port is a classical Value and return it
/// Return None if Const kind
/// Panics if port not valid for Op or port is not Const/Value
fn check_classical_value(
    base: &HugrMut,
    src: NodeIndex,
    src_offset: PortOffset,
) -> Result<Option<ClassicType>, BuildError> {
    let wire_kind = base.hugr().get_optype(src).port_kind(src_offset).unwrap();
    let typ = match wire_kind {
        EdgeKind::Const(_) => None,
        EdgeKind::Value(simple_type) => match simple_type {
            SimpleType::Classic(typ) => Some(typ),
            SimpleType::Linear(typ) => return Err(BuildError::NoCopyLinear(typ)),
        },
        _ => {
            panic!("Wires can only be Value kind")
        }
    };

    Ok(typ)
}

// Return newly added port to copy node if src node is a copy
fn if_copy_add_port(base: &mut HugrMut, src: NodeIndex) -> Option<usize> {
    let src_op: Result<&LeafOp, ()> = base.hugr().get_optype(src).try_into();
    if let Ok(LeafOp::Copy { n_copies, typ }) = src_op {
        let copy_node = src;
        // If already connected to a copy node, add wire to the copy
        let n_copies = *n_copies;
        base.replace_op(
            copy_node,
            LeafOp::Copy {
                n_copies: n_copies + 1,
                typ: typ.clone(),
            },
        );
        base.add_ports(copy_node, Direction::Outgoing, 1).next()
    } else {
        None
    }
}

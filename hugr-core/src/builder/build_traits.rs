use crate::extension::prelude::MakeTuple;
use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::linking::{HugrLinking, NodeLinkingDirective};
use crate::hugr::views::HugrView;
use crate::hugr::{NodeMetadata, ValidationError};
use crate::ops::{self, OpTag, OpTrait, OpType, Tag, TailLoop};
use crate::utils::collect_array;
use crate::{Extension, IncomingPort, Node, OutgoingPort};

use std::collections::HashMap;
use std::iter;
use std::sync::Arc;

use super::{BuilderWiringError, ModuleBuilder};
use super::{
    CircuitBuilder,
    handle::{BuildHandle, Outputs},
};

use crate::{
    ops::handle::{ConstID, DataflowOpID, FuncID, NodeHandle},
    types::EdgeKind,
};

use crate::extension::ExtensionRegistry;
use crate::types::{Signature, Type, TypeArg, TypeRow};

use itertools::Itertools;

use super::{
    BuildError, Wire, cfg::CFGBuilder, conditional::ConditionalBuilder, dataflow::DFGBuilder,
    tail_loop::TailLoopBuilder,
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
    ///
    /// Adds the extensions required by the op to the HUGR, if they are not already present.
    fn add_child_node(&mut self, node: impl Into<OpType>) -> Node {
        let node: OpType = node.into();

        // Add the extension the operation is defined in to the HUGR.
        let used_extensions = node
            .used_extensions()
            .unwrap_or_else(|e| panic!("Build-time signatures should have valid extensions. {e}"));
        self.use_extensions(used_extensions);

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
    /// Adds the extensions required by the op to the HUGR, if they are not already present.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::Const`] node.
    fn add_constant(&mut self, constant: impl Into<ops::Const>) -> ConstID {
        self.add_child_node(constant.into()).into()
    }

    /// Insert a HUGR's entrypoint region as a child of the container.
    ///
    /// To insert an arbitrary region of a HUGR, use [`Container::add_hugr_region`].
    fn add_hugr(&mut self, child: Hugr) -> InsertionResult {
        let region = child.entrypoint();
        self.add_hugr_region(child, region)
    }

    /// Insert a HUGR region as a child of the container.
    ///
    /// To insert the entrypoint region of a HUGR, use [`Container::add_hugr`].
    fn add_hugr_region(&mut self, child: Hugr, region: Node) -> InsertionResult {
        let parent = self.container_node();
        self.hugr_mut().insert_region(parent, child, region)
    }

    /// Insert a copy of a HUGR as a child of the container.
    /// (Only the portion below the entrypoint will be inserted, with any incoming
    /// edges broken; see [Dataflow::add_link_view_by_node_with_wires])
    fn add_hugr_view<H: HugrView>(&mut self, child: &H) -> InsertionResult<H::Node, Node> {
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

    /// Add an extension to the set of extensions used by the hugr.
    fn use_extension(&mut self, ext: impl Into<Arc<Extension>>) {
        self.hugr_mut().use_extension(ext);
    }

    /// Extend the set of extensions used by the hugr with the extensions in the registry.
    fn use_extensions<Reg>(&mut self, registry: impl IntoIterator<Item = Reg>)
    where
        ExtensionRegistry: Extend<Reg>,
    {
        self.hugr_mut().use_extensions(registry);
    }
}

/// Types implementing this trait can be used to build complete HUGRs
/// (with varying entrypoint node types)
pub trait HugrBuilder: Container {
    /// Allows adding definitions to the module root of which
    /// this builder is building a part
    fn module_root_builder(&mut self) -> ModuleBuilder<&mut Hugr> {
        debug_assert!(
            self.hugr()
                .get_optype(self.hugr().module_root())
                .is_module()
        );
        ModuleBuilder(self.hugr_mut())
    }

    /// Finish building the HUGR, perform any validation checks and return it.
    fn finish_hugr(self) -> Result<Hugr, ValidationError<Node>>;
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
    /// Add a dataflow [`OpType`] to the sibling graph, wiring up the `input_wires` to the
    /// incoming ports of the resulting node.
    ///
    /// Adds the extensions required by the op to the HUGR, if they are not already present.
    ///
    /// # Errors
    ///
    /// Returns a [`BuildError::OperationWiring`] error if the `input_wires` cannot be connected.
    fn add_dataflow_op(
        &mut self,
        nodetype: impl Into<OpType>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let outs = add_node_with_wires(self, nodetype, input_wires)?;

        Ok(outs.into())
    }

    /// Insert a hugr-defined op to the sibling graph, wiring up the
    /// `input_wires` to the incoming ports of the resulting root node.
    ///
    /// Inserts everything from the entrypoint region of the HUGR.
    /// See [`Dataflow::add_hugr_region_with_wires`] for a generic version that allows
    /// inserting a region other than the entrypoint.
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
        let region = hugr.entrypoint();
        self.add_hugr_region_with_wires(hugr, region, input_wires)
    }

    /// Insert a hugr-defined op to the sibling graph, wiring up the
    /// `input_wires` to the incoming ports of the resulting root node.
    ///
    /// `region` must be a node in the `hugr`. See [`Dataflow::add_hugr_with_wires`]
    /// for a helper that inserts the entrypoint region to the HUGR.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when adding the
    /// node.
    fn add_hugr_region_with_wires(
        &mut self,
        hugr: Hugr,
        region: Node,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let node = self.add_hugr_region(hugr, region).inserted_entrypoint;

        wire_ins_return_outs(input_wires, node, self)
    }

    /// Insert a hugr, adding its entrypoint to the sibling graph and wiring up the
    /// `input_wires` to the incoming ports of the resulting root node. `defns` may
    /// contain other children of the module root of `hugr`, which will be added to
    /// the module root being built.
    fn add_link_hugr_by_node_with_wires(
        &mut self,
        hugr: Hugr,
        input_wires: impl IntoIterator<Item = Wire>,
        defns: HashMap<Node, NodeLinkingDirective>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let parent = Some(self.container_node());
        let ep = hugr.entrypoint();
        let node = self
            .hugr_mut()
            .insert_link_hugr_by_node(parent, hugr, defns)?
            .node_map[&ep];
        wire_ins_return_outs(input_wires, node, self)
    }

    /// Copy a hugr's entrypoint-subtree (only) into the sibling graph, wiring up the
    /// `input_wires` to the incoming ports of the node that was the entrypoint.
    /// (Note that any wires from outside the entrypoint-subtree are disconnected in the copy;
    /// see [Self::add_link_view_by_node_with_wires] for an alternative.)
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
        let node = self.add_hugr_view(hugr).inserted_entrypoint;
        wire_ins_return_outs(input_wires, node, self)
    }

    /// Copy a Hugr, adding its entrypoint into the sibling graph and wiring up the
    /// `input_wires` to the incoming ports. `defns` may contain other children of
    /// the module root of `hugr`, which will be added to the module root being built.
    fn add_link_view_by_node_with_wires<H: HugrView>(
        &mut self,
        hugr: &H,
        input_wires: impl IntoIterator<Item = Wire>,
        defns: HashMap<H::Node, NodeLinkingDirective>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let parent = Some(self.container_node());
        let node = self
            .hugr_mut()
            .insert_link_view_by_node(parent, hugr, defns)
            .map_err(|ins_err| BuildError::HugrViewInsertionError(ins_err.to_string()))?
            .node_map[&hugr.entrypoint()];
        wire_ins_return_outs(input_wires, node, self)
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
                container_op: Box::new(self.hugr().get_optype(self.container_node()).clone()),
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
    #[track_caller]
    fn input_wires_arr<const N: usize>(&self) -> [Wire; N] {
        collect_array(self.input_wires())
    }

    /// Return a builder for a [`crate::ops::DFG`] node, i.e. a nested dataflow subgraph,
    /// given a signature describing its input and output types and extension delta,
    /// and the input wires (which must match the input types)
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error when building
    /// the DFG node.
    // TODO: Should this be one function, or should there be a temporary "op" one like with the others?
    fn dfg_builder(
        &mut self,
        signature: Signature,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<DFGBuilder<&mut Hugr>, BuildError> {
        let op = ops::DFG {
            signature: signature.clone(),
        };
        let (dfg_n, _) = add_node_with_wires(self, op, input_wires)?;

        DFGBuilder::create_with_io(self.hugr_mut(), dfg_n, signature)
    }

    /// Return a builder for a [`crate::ops::DFG`] node, i.e. a nested dataflow subgraph,
    /// that is endomorphic (the output types are the same as the input types).
    /// The `inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    fn dfg_builder_endo(
        &mut self,
        inputs: impl IntoIterator<Item = (Type, Wire)>,
    ) -> Result<DFGBuilder<&mut Hugr>, BuildError> {
        let (types, input_wires): (Vec<Type>, Vec<Wire>) = inputs.into_iter().unzip();
        self.dfg_builder(Signature::new_endo(types), input_wires)
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
        output_types: TypeRow,
    ) -> Result<CFGBuilder<&mut Hugr>, BuildError> {
        let (input_types, input_wires): (Vec<Type>, Vec<Wire>) = inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (cfg_node, _) = add_node_with_wires(
            self,
            ops::CFG {
                signature: Signature::new(inputs.clone(), output_types.clone()),
            },
            input_wires,
        )?;
        CFGBuilder::create(self.hugr_mut(), cfg_node, inputs, output_types)
    }

    /// Load a static constant and return the local dataflow wire for that constant.
    /// Adds a [`OpType::LoadConstant`] node.
    fn load_const(&mut self, cid: &ConstID) -> Wire {
        let const_node = cid.node();
        let nodetype = self.hugr().get_optype(const_node);
        let op: ops::Const = nodetype
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
    ) -> Result<Wire, BuildError> {
        let func_node = fid.node();
        let func_op = self.hugr().get_optype(func_node);
        let func_sig = match func_op {
            OpType::FuncDefn(fd) => fd.signature().clone(),
            OpType::FuncDecl(fd) => fd.signature().clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: func_node,
                    op_desc: "FuncDecl/FuncDefn",
                });
            }
        };

        let load_n = self.add_dataflow_op(
            ops::LoadFunction::try_new(func_sig, type_args)?,
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
    ///
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
    /// `sum_input` is a tuple of the type of the Sum
    /// variants and the corresponding wire.
    ///
    /// The `other_inputs` must be an iterable over pairs of the type of the input and
    /// the corresponding wire.
    /// The `output_types` are the types of the outputs.
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
        let make_op = self.add_dataflow_op(MakeTuple(types), values)?;
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
                variants: variants.into_iter().collect_vec(),
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
        self.make_sum(
            TailLoop::CONTINUE_TAG,
            [tail_loop.just_inputs, tail_loop.just_outputs],
            values,
        )
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
        self.make_sum(
            TailLoop::BREAK_TAG,
            [loop_op.just_inputs, loop_op.just_outputs],
            values,
        )
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
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let hugr = self.hugr();
        let def_op = hugr.get_optype(function.node());
        let type_scheme = match def_op {
            OpType::FuncDefn(fd) => fd.signature().clone(),
            OpType::FuncDecl(fd) => fd.signature().clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: function.node(),
                    op_desc: "FuncDecl/FuncDefn",
                });
            }
        };
        let op: OpType = ops::Call::try_new(type_scheme, type_args)?.into();
        let const_in_port = op.static_input_port().unwrap();
        let op_id = self.add_dataflow_op(op, input_wires)?;
        let src_port = self.hugr_mut().num_outputs(function.node()) - 1;

        self.hugr_mut()
            .connect(function.node(), src_port, op_id.node(), const_in_port);
        Ok(op_id)
    }

    /// For the vector of `wires`, produce a `CircuitBuilder` where ops can be
    /// added using indices in to the vector.
    fn as_circuit(&mut self, wires: impl IntoIterator<Item = Wire>) -> CircuitBuilder<'_, Self> {
        CircuitBuilder::new(wires, self)
    }

    /// Add a [Barrier] to a set of wires and return them in the same order.
    ///
    /// [Barrier]: crate::extension::prelude::Barrier
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the Barrier node
    /// or retrieving the type of the incoming wires.
    fn add_barrier(
        &mut self,
        wires: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let wires = wires.into_iter().collect_vec();
        let types: Result<Vec<Type>, _> =
            wires.iter().map(|&wire| self.get_wire_type(wire)).collect();
        let types = types?;
        let barrier_op =
            self.add_dataflow_op(crate::extension::prelude::Barrier::new(types), wires)?;
        Ok(barrier_op)
    }
}

/// Add a node to the graph, wiring up the `inputs` to the input ports of the resulting node.
///
/// Adds the extensions required by the op to the HUGR, if they are not already present.
///
/// # Errors
///
/// Returns a [`BuildError::OperationWiring`] if any of the connections produces an
/// invalid edge.
fn add_node_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    nodetype: impl Into<OpType>,
    inputs: impl IntoIterator<Item = Wire>,
) -> Result<(Node, usize), BuildError> {
    let op: OpType = nodetype.into();
    let num_outputs = op.value_output_count();
    let op_node = data_builder.add_child_node(op.clone());

    wire_up_inputs(inputs, op_node, data_builder).map_err(|error| BuildError::OperationWiring {
        op: Box::new(op),
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

fn wire_ins_return_outs<T: Dataflow + ?Sized>(
    inputs: impl IntoIterator<Item = Wire>,
    node: Node,
    data_builder: &mut T,
) -> Result<BuildHandle<DataflowOpID>, BuildError> {
    let op = data_builder.hugr().get_optype(node).clone();
    let num_outputs = op.value_output_count();
    wire_up_inputs(inputs, node, data_builder).map_err(|error| BuildError::OperationWiring {
        op: Box::new(op),
        error,
    })?;
    Ok((node, num_outputs).into())
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
                    typ: Box::new(typ),
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

            if !OpTag::ControlFlowChild.is_superset(base.get_optype(src).tag())
                && !OpTag::ControlFlowChild.is_superset(base.get_optype(src_sibling).tag())
            {
                // Add a state order constraint unless one of the nodes is a CFG BasicBlock
                base.add_other_edge(src, src_sibling);
            }
        } else if !typ.copyable() & base.linked_ports(src, src_port).next().is_some() {
            // Don't copy linear edges.
            return Err(BuilderWiringError::NoCopyLinear {
                typ: Box::new(typ),
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
    ) -> Result<Hugr, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(outputs)?;
        Ok(self.finish_hugr()?)
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

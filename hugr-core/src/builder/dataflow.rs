//! Builders for dataflow regions.

use itertools::Itertools;

use super::build_traits::{HugrBuilder, SubContainer};
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{HugrView, ValidationError};
use crate::ops::handle::NodeHandle;
use crate::ops::{self, DFG, FuncDefn, NamedOp, OpParent, OpType};
use crate::types::{PolyFuncType, Signature, Type};
use crate::{Direction, Hugr, IncomingPort, Node, OutgoingPort, Visibility, Wire, hugr::HugrMut};

/// Builder for a [`ops::DFG`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct DFGBuilder<T> {
    pub(crate) base: T,
    pub(crate) dfg_node: Node,
    pub(crate) num_in_wires: usize,
    pub(crate) num_out_wires: usize,
}

/// Error returned by [`DFGBuilder::add_input`] and [`DFGBuilder::add_output`].
#[derive(Debug, Clone, PartialEq, derive_more::Display, derive_more::Error)]
#[non_exhaustive]
pub enum DFGAddPortError {
    /// The parent optype is not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot update its signature.
    #[display("Adding port to an optype {op} is not supported.")]
    ParentOpNotSupported {
        /// The name of the unsupported operation.
        op: String,
    },
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> DFGBuilder<T> {
    /// Returns a new `DFGBuilder` with the given base and parent node.
    ///
    /// Sets up the input and output nodes of the region. If `parent` already has
    /// input and output nodes, use [`DFGBuilder::create`] instead.
    pub fn create_with_io(
        mut base: T,
        parent: Node,
        signature: Signature,
    ) -> Result<Self, BuildError> {
        debug_assert_eq!(base.as_ref().children(parent).count(), 0);

        let num_in_wires = signature.input_count();
        let num_out_wires = signature.output_count();
        let input = ops::Input {
            types: signature.input().clone(),
        };
        let output = ops::Output {
            types: signature.output().clone(),
        };
        base.as_mut().add_node_with_parent(parent, input);
        base.as_mut().add_node_with_parent(parent, output);

        Ok(Self {
            base,
            dfg_node: parent,
            num_in_wires,
            num_out_wires,
        })
    }

    /// Returns a new `DFGBuilder` with the given base and parent node.
    ///
    /// The parent node may be any `DataflowParent` node.
    ///
    /// If `parent` doesn't have input and output nodes, use
    /// [`DFGBuilder::create_with_io`] instead.
    pub fn create(base: T, parent: Node) -> Result<Self, BuildError> {
        let sig = base
            .as_ref()
            .get_optype(parent)
            .inner_function_type()
            .expect("DFG parent must have an inner function signature.");
        let num_in_wires = sig.input_count();
        let num_out_wires = sig.output_count();

        Ok(Self {
            base,
            dfg_node: parent,
            num_in_wires,
            num_out_wires,
        })
    }

    /// Add a new input to the dataflow being constructed.
    ///
    /// Updates the parent's optype to include the new input type.
    ///
    /// # Returns
    ///
    /// - The new wire from the input node.
    ///
    /// # Errors
    ///
    /// - [`DFGAddPortError::ParentOpNotSupported`] if the container optype is not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot add an input.
    ///   In this case, the Hugr will not be updated.
    pub fn add_input(&mut self, input_type: Type) -> Result<Wire, DFGAddPortError> {
        let [inp_node, _] = self.io();

        // Update the parent's root type
        if !self.update_parent_signature(|mut s| {
            s.input.to_mut().push(input_type.clone());
            s
        }) {
            return Err(DFGAddPortError::ParentOpNotSupported {
                op: self
                    .hugr()
                    .get_optype(self.container_node())
                    .name()
                    .to_string(),
            });
        }

        // Update the inner input node
        let OpType::Input(inp) = self.hugr_mut().optype_mut(inp_node) else {
            panic!("Input node {inp_node} is not an Input");
        };
        inp.types.to_mut().push(input_type);

        let mut new_port = self.hugr_mut().add_ports(inp_node, Direction::Outgoing, 1);
        let new_port = new_port.next().unwrap();

        // The last port in an input/output node is an order edge port, so we must shift any connections to it.
        let new_value_port: OutgoingPort = (new_port - 1).into();
        let new_order_port: OutgoingPort = new_port.into();
        let order_edge_targets = self
            .hugr()
            .linked_inputs(inp_node, new_value_port)
            .collect_vec();
        self.hugr_mut().disconnect(inp_node, new_value_port);
        for (tgt_node, tgt_port) in order_edge_targets {
            self.hugr_mut()
                .connect(inp_node, new_order_port, tgt_node, tgt_port);
        }

        // Update the builder metadata
        self.num_in_wires += 1;

        Ok(self.input_wires().next_back().unwrap())
    }

    /// Add a new output to the function being constructed.
    ///
    /// Updates the parent's optype to include the new input type, if possible.
    /// If the container optype is not a [`FuncDefn`], [`DFG`], or dataflow
    /// block, the optype will not be updated.
    ///
    /// # Errors
    ///
    /// - [`DFGAddPortError::ParentOpNotSupported`] if the container optype is not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot add an input.
    ///   In this case, the Hugr will not be updated.
    pub fn add_output(&mut self, output_type: Type) -> Result<(), DFGAddPortError> {
        let [_, out_node] = self.io();

        // Update the parent's root type
        if !self.update_parent_signature(|mut s| {
            s.output.to_mut().push(output_type.clone());
            s
        }) {
            return Err(DFGAddPortError::ParentOpNotSupported {
                op: self
                    .hugr()
                    .get_optype(self.container_node())
                    .name()
                    .to_string(),
            });
        }

        // Update the inner input node
        let OpType::Output(out) = self.hugr_mut().optype_mut(out_node) else {
            panic!("Output node {out_node} is not an Output");
        };
        out.types.to_mut().push(output_type);

        let mut new_port = self.hugr_mut().add_ports(out_node, Direction::Incoming, 1);
        let new_port = new_port.next().unwrap();

        // The last port in an input/output node is an order edge port, so we must shift any connections to it.
        let new_value_port: IncomingPort = (new_port - 1).into();
        let new_order_port: IncomingPort = new_port.into();
        let order_edge_sources = self
            .hugr()
            .linked_outputs(out_node, new_value_port)
            .collect_vec();
        self.hugr_mut().disconnect(out_node, new_value_port);
        for (src_node, src_port) in order_edge_sources {
            self.hugr_mut()
                .connect(src_node, src_port, out_node, new_order_port);
        }

        // Update the builder metadata
        self.num_out_wires += 1;

        Ok(())
    }

    /// Update the container's parent signature, if it is a function definition, DFG region, or dataflow block.
    ///
    /// Internal function used in [`add_input`] and [`add_output`].
    ///
    /// Does not update the input and output nodes.
    ///
    /// # Returns
    ///
    /// - `true` if the parent signature was updated.
    /// - `false` if the parent optype is not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot update the signature.
    fn update_parent_signature(&mut self, f: impl FnOnce(Signature) -> Signature) -> bool {
        let parent = self.container_node();

        match self.hugr_mut().optype_mut(parent) {
            ops::OpType::FuncDefn(fd) => {
                let mut sig = std::mem::take(fd.signature_mut());
                let body = std::mem::take(sig.body_mut());
                *sig.body_mut() = f(body);
                *fd.signature_mut() = sig;
            }
            ops::OpType::DFG(dfg) => {
                let sig = std::mem::take(&mut dfg.signature);
                dfg.signature = f(sig);
            }
            ops::OpType::DataflowBlock(dfb) => {
                let inp = std::mem::take(&mut dfb.inputs);
                let other_outputs = std::mem::take(&mut dfb.other_outputs);
                let sig = f(Signature::new(inp, other_outputs));
                dfb.inputs = sig.input;
                dfb.other_outputs = sig.output;
            }
            _ => return false,
        }
        true
    }
}

impl DFGBuilder<Hugr> {
    /// Begin building a new DFG-rooted HUGR given its inputs, outputs,
    /// and extension delta.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(signature: Signature) -> Result<DFGBuilder<Hugr>, BuildError> {
        let dfg_op: DFG = ops::DFG {
            signature: signature.clone(),
        };
        let base = Hugr::new_with_entrypoint(dfg_op).expect("DFG entrypoint should be valid");
        let root = base.entrypoint();
        DFGBuilder::create_with_io(base, root, signature)
    }
}

impl HugrBuilder for DFGBuilder<Hugr> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError<Node>> {
        self.base.validate()?;
        Ok(self.base)
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> Container for DFGBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.dfg_node
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.base.as_mut()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.as_ref()
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> SubContainer for DFGBuilder<T> {
    type ContainerHandle = BuildHandle<DfgID>;
    #[inline]
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
        Ok((self.dfg_node, self.num_out_wires).into())
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> Dataflow for DFGBuilder<T> {
    #[inline]
    fn num_inputs(&self) -> usize {
        self.num_in_wires
    }
}

/// Wrapper around [`DFGBuilder`] used to build other dataflow regions.
// Stores option of DFGBuilder so it can be taken out without moving.
#[derive(Debug, Clone, PartialEq)]
pub struct DFGWrapper<B, T>(DFGBuilder<B>, PhantomData<T>);

impl<B, T: NodeHandle> DFGWrapper<B, T> {
    /// Wrap a [`DFGBuilder`] into a [`DFGWrapper`], with the given handle type.
    ///
    /// The caller must ensure that the DFGBuilder's parent node has the correct
    /// handle type given by `T::TAG`.
    pub fn from_dfg_builder(db: DFGBuilder<B>) -> Self {
        Self(db, PhantomData)
    }

    /// Unwrap the [`DFGWrapper`] into a [`DFGBuilder`].
    pub fn into_dfg_builder(self) -> DFGBuilder<B> {
        self.0
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T> DFGWrapper<B, T> {
    /// Add a new input to the dataflow being constructed.
    ///
    /// Updates the parent's optype to include the new input type, if possible.
    /// If the container optype is not a [`FuncDefn`] or a [`DFG`], the optype
    /// will not be updated.
    ///
    /// # Returns
    ///
    /// - The new wire from the input node.
    ///
    /// # Errors
    ///
    /// - [`DFGAddPortError::ParentOpNotSupported`] if the container optype is
    ///   not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot add an
    ///   input. In this case, the Hugr will not be updated.
    pub fn add_input(&mut self, input_type: Type) -> Result<Wire, DFGAddPortError> {
        self.0.add_input(input_type)
    }

    /// Add a new output to the dataflow being constructed.
    ///
    /// Updates the parent's optype to include the new output type, if possible.
    /// If the container optype is not a [`FuncDefn`] or a [`DFG`], the optype
    /// will not be updated.
    ///
    /// # Errors
    ///
    /// - [`DFGAddPortError::ParentOpNotSupported`] if the container optype is
    ///   not a [`FuncDefn`], [`DFG`], or dataflow block so we cannot add an
    ///   input. In this case, the Hugr will not be updated.
    pub fn add_output(&mut self, output_type: Type) -> Result<(), DFGAddPortError> {
        self.0.add_output(output_type)
    }
}

/// Builder for a [`ops::FuncDefn`] node
pub type FunctionBuilder<B> = DFGWrapper<B, BuildHandle<FuncID<true>>>;

impl FunctionBuilder<Hugr> {
    /// Initialize a builder for a [`FuncDefn`]-rooted HUGR; the function will
    /// be private. (See also [Self::new_vis].)
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
    ) -> Result<Self, BuildError> {
        Self::new_with_op(FuncDefn::new(name, signature))
    }

    /// Initialize a builder for a FuncDefn-rooted HUGR, with the specified
    /// [Visibility].
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new_vis(
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
        visibility: Visibility,
    ) -> Result<Self, BuildError> {
        Self::new_with_op(FuncDefn::new_vis(name, signature, visibility))
    }

    fn new_with_op(op: FuncDefn) -> Result<Self, BuildError> {
        let body = op.signature().body().clone();

        let base = Hugr::new_with_entrypoint(op).expect("FuncDefn entrypoint should be valid");
        let root = base.entrypoint();

        let db = DFGBuilder::create_with_io(base, root, body)?;
        Ok(Self::from_dfg_builder(db))
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>> FunctionBuilder<B> {
    /// Initialize a new function definition on the root module of an existing HUGR.
    ///
    /// The HUGR's entrypoint will **not** be modified.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn with_hugr(
        mut hugr: B,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
    ) -> Result<Self, BuildError> {
        let signature: PolyFuncType = signature.into();
        let body = signature.body().clone();
        let op = ops::FuncDefn::new(name, signature);

        let module = hugr.as_ref().module_root();
        let func = hugr.as_mut().add_node_with_parent(module, op);

        let db = DFGBuilder::create_with_io(hugr, func, body)?;
        Ok(Self::from_dfg_builder(db))
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T> Container for DFGWrapper<B, T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.0.container_node()
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.0.hugr_mut()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.0.hugr()
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T> Dataflow for DFGWrapper<B, T> {
    #[inline]
    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T: From<BuildHandle<DfgID>>> SubContainer for DFGWrapper<B, T> {
    type ContainerHandle = T;

    #[inline]
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
        self.0.finish_sub_container().map(Into::into)
    }
}

impl<T> HugrBuilder for DFGWrapper<Hugr, T> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError<Node>> {
        self.0.finish_hugr()
    }
}

#[cfg(test)]
pub(crate) mod test {
    use cool_asserts::assert_matches;
    use rstest::rstest;
    use serde_json::json;
    use std::collections::HashMap;

    use crate::builder::build_traits::DataflowHugr;
    use crate::builder::test::dfg_calling_defn_decl;
    use crate::builder::{
        BuilderWiringError, CFGBuilder, DataflowSubContainer, ModuleBuilder, TailLoopBuilder,
        endo_sig, inout_sig,
    };
    use crate::extension::SignatureError;
    use crate::extension::prelude::Noop;
    use crate::extension::prelude::{bool_t, qb_t, usize_t};
    use crate::hugr::linking::NodeLinkingDirective;
    use crate::hugr::validate::InterGraphEdgeError;
    use crate::ops::{FuncDecl, FuncDefn, OpParent, OpTag, OpTrait, Value, handle::NodeHandle};

    use crate::std_extensions::logic::test::and_op;
    use crate::types::type_param::TypeParam;
    use crate::types::{EdgeKind, FuncValueType, RowVariable, Signature, Type, TypeBound, TypeRV};
    use crate::utils::test_quantum_extension::h_gate;
    use crate::{Wire, builder::test::n_identity, type_row};

    use super::super::test::simple_dfg_hugr;
    use super::*;
    #[test]
    fn nested_identity() -> Result<(), BuildError> {
        let build_result = {
            let mut outer_builder = DFGBuilder::new(endo_sig(vec![usize_t(), qb_t()]))?;

            let [int, qb] = outer_builder.input_wires_arr();

            let q_out = outer_builder.add_dataflow_op(h_gate(), vec![qb])?;

            let inner_builder = outer_builder.dfg_builder_endo([(usize_t(), int)])?;
            let inner_id = n_identity(inner_builder)?;

            outer_builder.finish_hugr_with_outputs(inner_id.outputs().chain(q_out.outputs()))
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }

    // Scaffolding for copy insertion tests
    fn copy_scaffold<F>(f: F, msg: &'static str) -> Result<(), BuildError>
    where
        F: FnOnce(&mut DFGBuilder<Hugr>) -> Result<(), BuildError>,
    {
        let build_result = {
            let mut builder = DFGBuilder::new(inout_sig(bool_t(), vec![bool_t(), bool_t()]))?;

            f(&mut builder)?;

            builder.finish_hugr()
        };
        assert_matches!(build_result, Ok(_), "Failed on example: {}", msg);

        Ok(())
    }
    #[test]
    fn copy_insertion() -> Result<(), BuildError> {
        copy_scaffold(
            |f_build| {
                let [b1] = f_build.input_wires_arr();
                f_build.set_outputs([b1, b1])
            },
            "Copy input and output",
        )?;

        copy_scaffold(
            |f_build| {
                let [b1] = f_build.input_wires_arr();
                let xor = f_build.add_dataflow_op(and_op(), [b1, b1])?;
                f_build.set_outputs([xor.out_wire(0), b1])
            },
            "Copy input and use with binary function",
        )?;

        copy_scaffold(
            |f_build| {
                let [b1] = f_build.input_wires_arr();
                let xor1 = f_build.add_dataflow_op(and_op(), [b1, b1])?;
                let xor2 = f_build.add_dataflow_op(and_op(), [b1, xor1.out_wire(0)])?;
                f_build.set_outputs([xor2.out_wire(0), b1])
            },
            "Copy multiple times",
        )?;

        Ok(())
    }

    #[test]
    fn copy_insertion_qubit() {
        let builder = || {
            let mut module_builder = ModuleBuilder::new();

            let f_build = module_builder
                .define_function("main", Signature::new(vec![qb_t()], vec![qb_t(), qb_t()]))?;

            let [q1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([q1, q1])?;

            Ok(module_builder.finish_hugr()?)
        };

        assert_matches!(
            builder(),
            Err(BuildError::OutputWiring {
                error: BuilderWiringError::NoCopyLinear { typ, .. },
                ..
            })
            if *typ == qb_t()
        );
    }

    #[test]
    fn simple_inter_graph_edge() {
        let builder = || -> Result<Hugr, BuildError> {
            let mut f_build =
                FunctionBuilder::new("main", Signature::new(vec![bool_t()], vec![bool_t()]))?;

            let [i1] = f_build.input_wires_arr();
            let noop = f_build.add_dataflow_op(Noop(bool_t()), [i1])?;
            let i1 = noop.out_wire(0);

            let mut nested =
                f_build.dfg_builder(Signature::new(type_row![], vec![bool_t()]), [])?;

            let id = nested.add_dataflow_op(Noop(bool_t()), [i1])?;

            let nested = nested.finish_with_outputs([id.out_wire(0)])?;

            f_build.finish_hugr_with_outputs([nested.out_wire(0)])
        };

        assert_matches!(builder(), Ok(_));
    }

    #[rstest]
    #[case::function(|sig| FunctionBuilder::new("main", sig).unwrap().into_dfg_builder())]
    #[case::dfg(|sig| DFGBuilder::new(sig).unwrap())]
    #[case::df_block(|sig: Signature| {
        let outs = sig.output.clone();
        let mut h = CFGBuilder::new(sig).unwrap();
        let entry_node = h.entry_builder([], outs).unwrap().finish_sub_container().unwrap().node();
        let hugr = std::mem::take(h.hugr_mut());
        DFGBuilder::create(hugr, entry_node).unwrap()
    })]
    fn add_inputs_outputs(#[case] builder: impl FnOnce(Signature) -> DFGBuilder<Hugr>) {
        let builder = || -> Result<(Hugr, Node, Signature), BuildError> {
            let mut dfg = builder(Signature::new(vec![bool_t()], vec![bool_t()]));
            let dfg_node = dfg.container_node();

            // Initial inner signature, before any inputs or outputs are added
            let initial_sig = dfg
                .hugr()
                .get_optype(dfg_node)
                .inner_function_type()
                .unwrap()
                .into_owned();

            let [i0] = dfg.input_wires_arr();
            let noop0 = dfg.add_dataflow_op(Noop(bool_t()), [i0])?;

            // Some some order edges
            dfg.set_order(&dfg.io()[0], &noop0.node());
            dfg.set_order(&noop0.node(), &dfg.io()[1]);

            // Add a new input and output, and connect them with a noop in between
            dfg.add_output(qb_t()).unwrap();
            let i1 = dfg.add_input(qb_t()).unwrap();
            let noop1 = dfg.add_dataflow_op(Noop(qb_t()), [i1])?;

            // Do not validate the final hugr, as it may have disconnected inputs.
            dfg.set_outputs([noop0.out_wire(0), noop1.out_wire(0)])?;
            let hugr = std::mem::take(dfg.hugr_mut());
            Ok((hugr, dfg_node, initial_sig))
        };

        let (hugr, dfg_node, initial_sig) = builder().unwrap_or_else(|e| panic!("{e}"));

        let container_sig = hugr.get_optype(dfg_node).inner_function_type().unwrap();
        let mut expected_sig = initial_sig;
        expected_sig.input.to_mut().push(qb_t());
        expected_sig.output.to_mut().push(qb_t());
        assert_eq!(
            container_sig.io(),
            (&expected_sig.input, &expected_sig.output),
            "Got signature: {container_sig}, expected: {expected_sig}",
        );
    }

    #[rstest]
    #[case::tail_loop(|sig: Signature| TailLoopBuilder::new(sig.input, vec![], sig.output).unwrap().into_dfg_builder())]
    fn add_inputs_outputs_unsupported(#[case] builder: impl FnOnce(Signature) -> DFGBuilder<Hugr>) {
        let mut dfg = builder(Signature::new(vec![bool_t()], vec![bool_t()]));

        // Add a new input and output, and connect them with a noop in between
        assert!(dfg.add_output(qb_t()).is_err());
        assert!(dfg.add_input(qb_t()).is_err());
    }

    #[test]
    fn error_on_linear_inter_graph_edge() -> Result<(), BuildError> {
        let mut f_build = FunctionBuilder::new("main", Signature::new(vec![qb_t()], vec![qb_t()]))?;

        let [i1] = f_build.input_wires_arr();
        let noop = f_build.add_dataflow_op(Noop(qb_t()), [i1])?;
        let i1 = noop.out_wire(0);

        let mut nested = f_build.dfg_builder(Signature::new(type_row![], vec![qb_t()]), [])?;

        let id_res = nested.add_dataflow_op(Noop(qb_t()), [i1]);

        // The error would anyway be caught in validation when we finish the Hugr,
        // but the builder catches it earlier
        assert_matches!(
            id_res.map(|bh| bh.handle().node()), // Transform into something that impl's Debug
            Err(BuildError::OperationWiring {
                error: BuilderWiringError::NonCopyableIntergraph { .. },
                ..
            })
        );

        Ok(())
    }

    #[rstest]
    fn dfg_hugr(simple_dfg_hugr: Hugr) {
        assert_eq!(simple_dfg_hugr.num_nodes(), 7);
        assert_eq!(simple_dfg_hugr.entry_descendants().count(), 3);
        assert_matches!(simple_dfg_hugr.entrypoint_optype().tag(), OpTag::Dfg);
    }

    #[test]
    fn add_hugr() -> Result<(), BuildError> {
        // Create a simple DFG
        let mut dfg_builder = DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t()]))?;
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.set_metadata("x", 42);
        let dfg_hugr = dfg_builder.finish_hugr_with_outputs([i1])?;

        // Create a module, and insert the DFG into it
        let mut module_builder = ModuleBuilder::new();

        let (dfg_node, f_node) = {
            let mut f_build =
                module_builder.define_function("main", Signature::new_endo(bool_t()))?;

            let [i1] = f_build.input_wires_arr();
            let dfg = f_build.add_hugr_with_wires(dfg_hugr, [i1])?;
            let f = f_build.finish_with_outputs([dfg.out_wire(0)])?;
            module_builder.set_child_metadata(f.node(), "x", "hi");
            (dfg.node(), f.node())
        };

        let hugr = module_builder.finish_hugr()?;
        assert_eq!(hugr.entry_descendants().count(), 7);

        assert_eq!(hugr.get_metadata(hugr.entrypoint(), "x"), None);
        assert_eq!(hugr.get_metadata(dfg_node, "x").cloned(), Some(json!(42)));
        assert_eq!(hugr.get_metadata(f_node, "x").cloned(), Some(json!("hi")));

        Ok(())
    }

    #[rstest]
    fn add_hugr_link_nodes(
        #[values(false, true)] replace: bool,
        #[values(true, false)] view: bool,
    ) {
        let mut fb = FunctionBuilder::new("main", Signature::new_endo(bool_t())).unwrap();
        let my_decl = fb
            .module_root_builder()
            .declare("func1", Signature::new_endo(bool_t()).into())
            .unwrap();
        let (insert, ins_defn, ins_decl) = dfg_calling_defn_decl();
        let ins_defn_name = insert
            .get_optype(ins_defn.node())
            .as_func_defn()
            .unwrap()
            .func_name()
            .clone();
        let ins_decl_name = insert
            .get_optype(ins_decl.node())
            .as_func_decl()
            .unwrap()
            .func_name()
            .clone();
        let decl_mode = if replace {
            NodeLinkingDirective::UseExisting(my_decl.node())
        } else {
            NodeLinkingDirective::add()
        };
        let link_spec = HashMap::from([
            (ins_defn.node(), NodeLinkingDirective::add()),
            (ins_decl.node(), decl_mode),
        ]);
        let inserted = if view {
            fb.add_link_view_by_node_with_wires(&insert, [], link_spec)
                .unwrap()
        } else {
            fb.add_link_hugr_by_node_with_wires(insert, [], link_spec)
                .unwrap()
        };
        let h = fb.finish_hugr_with_outputs(inserted.outputs()).unwrap();
        let defn_names = h
            .nodes()
            .filter_map(|n| h.get_optype(n).as_func_defn().map(FuncDefn::func_name))
            .collect_vec();
        assert_eq!(defn_names, [&"main".to_string(), &ins_defn_name]);
        let decl_names = h
            .nodes()
            .filter_map(|n| h.get_optype(n).as_func_decl().map(FuncDecl::func_name))
            .cloned()
            .collect_vec();
        let mut expected_decl_names = vec!["func1".to_string()];
        if !replace {
            expected_decl_names.push(ins_decl_name)
        }
        assert_eq!(decl_names, expected_decl_names);
    }

    #[test]
    fn barrier_node() -> Result<(), BuildError> {
        let mut parent = DFGBuilder::new(endo_sig(bool_t()))?;

        let [w] = parent.input_wires_arr();

        let mut dfg_b = parent.dfg_builder(endo_sig(bool_t()), [w])?;
        let [w] = dfg_b.input_wires_arr();

        let barr0 = dfg_b.add_barrier([w])?;
        let [w] = barr0.outputs_arr();

        let barr1 = dfg_b.add_barrier([w])?;
        let [w] = barr1.outputs_arr();

        let dfg = dfg_b.finish_with_outputs([w])?;
        let [w] = dfg.outputs_arr();

        let mut dfg2_b = parent.dfg_builder(endo_sig(vec![bool_t(), bool_t()]), [w, w])?;
        let [w1, w2] = dfg2_b.input_wires_arr();
        let barr2 = dfg2_b.add_barrier([w1, w2])?;
        let wires: Vec<Wire> = barr2.outputs().collect();

        let dfg2 = dfg2_b.finish_with_outputs(wires)?;
        let [w, _] = dfg2.outputs_arr();
        parent.finish_hugr_with_outputs([w])?;

        Ok(())
    }

    #[test]
    fn non_cfg_ancestor() -> Result<(), BuildError> {
        let unit_sig = Signature::new(type_row![Type::UNIT], type_row![Type::UNIT]);
        let mut b = DFGBuilder::new(unit_sig.clone())?;
        let b_child = b.dfg_builder(unit_sig.clone(), [b.input().out_wire(0)])?;
        let b_child_in_wire = b_child.input().out_wire(0);
        b_child.finish_with_outputs([])?;
        let b_child_2 = b.dfg_builder(unit_sig.clone(), [])?;

        // DFG block has edge coming a sibling block, which is only valid for
        // CFGs
        let b_child_2_handle = b_child_2.finish_with_outputs([b_child_in_wire])?;

        let res = b.finish_hugr_with_outputs([b_child_2_handle.out_wire(0)]);

        assert_matches!(
            res,
            Err(BuildError::InvalidHUGR(
                ValidationError::InterGraphEdgeError(InterGraphEdgeError::NonCFGAncestor { .. })
            ))
        );
        Ok(())
    }

    #[test]
    fn no_relation_edge() -> Result<(), BuildError> {
        let unit_sig = Signature::new(type_row![Type::UNIT], type_row![Type::UNIT]);
        let mut b = DFGBuilder::new(unit_sig.clone())?;
        let mut b_child = b.dfg_builder(unit_sig.clone(), [b.input().out_wire(0)])?;
        let b_child_child = b_child.dfg_builder(unit_sig.clone(), [b_child.input().out_wire(0)])?;
        let b_child_child_in_wire = b_child_child.input().out_wire(0);

        b_child_child.finish_with_outputs([])?;
        b_child.finish_with_outputs([])?;

        let mut b_child_2 = b.dfg_builder(unit_sig.clone(), [])?;
        let b_child_2_child =
            b_child_2.dfg_builder(unit_sig.clone(), [b_child_2.input().out_wire(0)])?;

        let res = b_child_2_child.finish_with_outputs([b_child_child_in_wire]);

        assert_matches!(
            res.map(|h| h.handle().node()), // map to something that implements Debug
            Err(BuildError::OutputWiring {
                error: BuilderWiringError::NoRelationIntergraph { .. },
                ..
            })
        );
        Ok(())
    }

    #[test]
    fn no_outer_row_variables() -> Result<(), BuildError> {
        let e = crate::hugr::validate::test::extension_with_eval_parallel();
        let tv = TypeRV::new_row_var_use(0, TypeBound::Copyable);
        // Can *declare* a function that takes a function-value of unknown #args
        FunctionBuilder::new(
            "bad_eval",
            PolyFuncType::new(
                [TypeParam::new_list_type(TypeBound::Copyable)],
                Signature::new(
                    Type::new_function(FuncValueType::new(usize_t(), tv.clone())),
                    vec![],
                ),
            ),
        )?;

        // But cannot eval it...
        let ev = e.instantiate_extension_op(
            "eval",
            [vec![usize_t().into()].into(), vec![tv.into()].into()],
        );
        assert_eq!(
            ev,
            Err(SignatureError::RowVarWhereTypeExpected {
                var: RowVariable(0, TypeBound::Copyable)
            })
        );
        Ok(())
    }

    #[test]
    fn order_edges() {
        let (mut hugr, load_constant, call) = {
            let mut builder = ModuleBuilder::new();
            let func = builder
                .declare("func", Signature::new_endo(bool_t()).into())
                .unwrap();
            let (load_constant, call) = {
                let mut builder = builder
                    .define_function("main", Signature::new(Type::EMPTY_TYPEROW, bool_t()))
                    .unwrap();
                let load_constant = builder.add_load_value(Value::true_val());
                let [r] = builder
                    .call(&func, &[], [load_constant])
                    .unwrap()
                    .outputs_arr();
                builder.finish_with_outputs([r]).unwrap();
                (load_constant.node(), r.node())
            };
            (builder.finish_hugr().unwrap(), load_constant, call)
        };

        let lc_optype = hugr.get_optype(load_constant);
        let call_optype = hugr.get_optype(call);
        assert_eq!(EdgeKind::StateOrder, lc_optype.other_input().unwrap());
        assert_eq!(EdgeKind::StateOrder, lc_optype.other_output().unwrap());
        assert_eq!(EdgeKind::StateOrder, call_optype.other_input().unwrap());
        assert_eq!(EdgeKind::StateOrder, call_optype.other_output().unwrap());

        hugr.connect(
            load_constant,
            lc_optype.other_output_port().unwrap(),
            call,
            call_optype.other_input_port().unwrap(),
        );

        hugr.validate().unwrap();
    }
}

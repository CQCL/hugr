use itertools::Itertools;

use super::build_traits::{HugrBuilder, SubContainer};
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::hugr::internal::HugrMutInternals;
use crate::hugr::{HugrView, ValidationError};
use crate::ops::{self, DataflowParent, FuncDefn, Input, OpParent, Output};
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

impl<T: AsMut<Hugr> + AsRef<Hugr>> DFGBuilder<T> {
    /// Initialize a new DFG container in an existing Hugr.
    ///
    /// The HUGR's entrypoint will **not** be modified.
    ///
    /// # Args
    ///
    /// - `parent` must be the parent of an existing dataflow region in the HUGR,
    ///   which will contain the new DFG.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn with_hugr(mut hugr: T, parent: Node, signature: Signature) -> Result<Self, BuildError> {
        let op = ops::DFG {
            signature: signature.clone(),
        };
        let dfg = hugr.as_mut().add_node_with_parent(parent, op);

        DFGBuilder::create_with_io(hugr, dfg, signature)
    }

    /// Returns a new `DFGBuilder` with the given base and parent node.
    ///
    /// Sets up the input and output nodes of the region. If `parent` already has
    /// input and output nodes, use [`DFGBuilder::create`] instead.
    pub(super) fn create_with_io(
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
    pub(super) fn create(base: T, parent: Node) -> Result<Self, BuildError> {
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
}

impl DFGBuilder<Hugr> {
    /// Begin building a new DFG-rooted HUGR given its inputs, outputs,
    /// and extension delta.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(signature: Signature) -> Result<DFGBuilder<Hugr>, BuildError> {
        let dfg_op = ops::DFG {
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

impl<B, T> DFGWrapper<B, T> {
    pub(super) fn from_dfg_builder(db: DFGBuilder<B>) -> Self {
        Self(db, PhantomData)
    }
}

/// Builder for a [`ops::FuncDefn`] node
pub type FunctionBuilder<B> = DFGWrapper<B, BuildHandle<FuncID<true>>>;

impl FunctionBuilder<Hugr> {
    /// Initialize a builder for a [`FuncDefn`](ops::FuncDefn)-rooted HUGR;
    /// the function will be private. (See also [Self::new_vis].)
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

    /// Add a new input to the function being constructed.
    ///
    /// Returns the new wire from the input node.
    pub fn add_input(&mut self, input_type: Type) -> Wire {
        let [inp_node, _] = self.io();

        // Update the parent's root type
        let new_optype = self.update_fn_signature(|mut s| {
            s.input.to_mut().push(input_type);
            s
        });

        // Update the inner input node
        let types = new_optype.signature().body().input.clone();
        self.hugr_mut().replace_op(inp_node, Input { types });
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
        self.0.num_in_wires += 1;

        self.input_wires().next_back().unwrap()
    }

    /// Add a new output to the function being constructed.
    pub fn add_output(&mut self, output_type: Type) {
        let [_, out_node] = self.io();

        // Update the parent's root type
        let new_optype = self.update_fn_signature(|mut s| {
            s.output.to_mut().push(output_type);
            s
        });

        // Update the inner input node
        let types = new_optype.signature().body().output.clone();
        self.hugr_mut().replace_op(out_node, Output { types });
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
        self.0.num_out_wires += 1;
    }

    /// Update the function builder's parent signature.
    ///
    /// Internal function used in [`add_input`] and [`add_output`].
    ///
    /// Does not update the input and output nodes.
    ///
    /// Returns a reference to the new optype.
    fn update_fn_signature(&mut self, f: impl FnOnce(Signature) -> Signature) -> &ops::FuncDefn {
        let parent = self.container_node();

        let ops::OpType::FuncDefn(fd) = self.hugr_mut().optype_mut(parent) else {
            panic!("FunctionBuilder node must be a FuncDefn")
        };
        *fd.signature_mut() = f(fd.inner_signature().into_owned()).into();
        &*fd
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
    use ops::OpParent;
    use rstest::rstest;
    use serde_json::json;

    use crate::builder::build_traits::DataflowHugr;
    use crate::builder::{
        BuilderWiringError, DataflowSubContainer, ModuleBuilder, endo_sig, inout_sig,
    };
    use crate::extension::SignatureError;
    use crate::extension::prelude::Noop;
    use crate::extension::prelude::{bool_t, qb_t, usize_t};
    use crate::hugr::validate::InterGraphEdgeError;
    use crate::ops::{OpTag, handle::NodeHandle};
    use crate::ops::{OpTrait, Value};

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

    #[test]
    fn add_inputs_outputs() {
        let builder = || -> Result<(Hugr, Node), BuildError> {
            let mut f_build =
                FunctionBuilder::new("main", Signature::new(vec![bool_t()], vec![bool_t()]))?;
            let f_node = f_build.container_node();

            let [i0] = f_build.input_wires_arr();
            let noop0 = f_build.add_dataflow_op(Noop(bool_t()), [i0])?;

            // Some some order edges
            f_build.set_order(&f_build.io()[0], &noop0.node());
            f_build.set_order(&noop0.node(), &f_build.io()[1]);

            // Add a new input and output, and connect them with a noop in between
            f_build.add_output(qb_t());
            let i1 = f_build.add_input(qb_t());
            let noop1 = f_build.add_dataflow_op(Noop(qb_t()), [i1])?;

            let hugr = f_build.finish_hugr_with_outputs([noop0.out_wire(0), noop1.out_wire(0)])?;
            Ok((hugr, f_node))
        };

        let (hugr, f_node) = builder().unwrap_or_else(|e| panic!("{e}"));

        let func_sig = hugr.get_optype(f_node).inner_function_type().unwrap();
        assert_eq!(
            func_sig.io(),
            (
                &vec![bool_t(), qb_t()].into(),
                &vec![bool_t(), qb_t()].into()
            )
        );
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
    fn insert_hugr() -> Result<(), BuildError> {
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

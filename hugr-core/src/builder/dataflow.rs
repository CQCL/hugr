use super::build_traits::{HugrBuilder, SubContainer};
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::hugr::{HugrView, ValidationError};
use crate::ops;

use crate::types::{FunctionType, TypeScheme};

use crate::extension::ExtensionRegistry;
use crate::Node;
use crate::{hugr::HugrMut, Hugr};

/// Builder for a [`ops::DFG`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct DFGBuilder<T> {
    pub(crate) base: T,
    pub(crate) dfg_node: Node,
    pub(crate) num_in_wires: usize,
    pub(crate) num_out_wires: usize,
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> DFGBuilder<T> {
    pub(super) fn create_with_io(
        mut base: T,
        parent: Node,
        signature: FunctionType,
    ) -> Result<Self, BuildError> {
        let num_in_wires = signature.input().len();
        let num_out_wires = signature.output().len();
        /* For a given dataflow graph with extension requirements IR -> IR + dR,
         - The output node's extension requirements are IR + dR -> IR + dR
           (but we expect no output wires)

         - The input node's extension requirements are IR -> IR, though we
           expect no input wires. We must avoid the case where the difference
           in extensions is an open variable, as it would be if the requirements
           were 0 -> IR.
           N.B. This means that for input nodes, we can't infer the extensions
           from the input wires as we normally expect, but have to infer the
           output wires and make use of the equality between the two.
        */
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
}

impl DFGBuilder<Hugr> {
    /// Begin building a new DFG-rooted HUGR given its inputs, outputs,
    /// and extension delta.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(signature: FunctionType) -> Result<DFGBuilder<Hugr>, BuildError> {
        let dfg_op = ops::DFG {
            signature: signature.clone(),
        };
        let base = Hugr::new(dfg_op);
        let root = base.root();
        DFGBuilder::create_with_io(base, root, signature)
    }
}

impl HugrBuilder for DFGBuilder<Hugr> {
    fn finish_hugr(
        mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, ValidationError> {
        self.base.update_validate(extension_registry)?;
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
    /// Initialize a builder for a FuncDefn rooted HUGR
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(
        name: impl Into<String>,
        signature: impl Into<TypeScheme>,
    ) -> Result<Self, BuildError> {
        let signature = signature.into();
        let body = signature.body().clone();
        let op = ops::FuncDefn {
            signature,
            name: name.into(),
        };

        let base = Hugr::new(op);
        let root = base.root();

        let db = DFGBuilder::create_with_io(base, root, body)?;
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
    fn finish_hugr(self, extension_registry: &ExtensionRegistry) -> Result<Hugr, ValidationError> {
        self.0.finish_hugr(extension_registry)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use cool_asserts::assert_matches;
    use rstest::rstest;
    use serde_json::json;

    use crate::builder::build_traits::DataflowHugr;
    use crate::builder::{ft1, BuilderWiringError, DataflowSubContainer, ModuleBuilder};
    use crate::extension::prelude::{BOOL_T, USIZE_T};
    use crate::extension::{ExtensionId, EMPTY_REG, PRELUDE_REGISTRY};
    use crate::hugr::validate::InterGraphEdgeError;
    use crate::ops::OpTrait;
    use crate::ops::{handle::NodeHandle, Lift, Noop, OpTag};

    use crate::std_extensions::logic::test::and_op;
    use crate::types::type_param::TypeParam;
    use crate::types::{FunctionType, FunctionTypeRV, Type, TypeBound, TypeRV};
    use crate::utils::test_quantum_extension::h_gate;
    use crate::{
        builder::test::{n_identity, BIT, NAT, QB},
        type_row, Wire,
    };

    use super::super::test::simple_dfg_hugr;
    use super::*;
    #[test]
    fn nested_identity() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let _f_id = {
                let mut func_builder = module_builder.define_function(
                    "main",
                    FunctionType::new(type_row![NAT, QB], type_row![NAT, QB]),
                )?;

                let [int, qb] = func_builder.input_wires_arr();

                let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb])?;

                let inner_builder = func_builder
                    .dfg_builder(FunctionType::new(type_row![NAT], type_row![NAT]), [int])?;
                let inner_id = n_identity(inner_builder)?;

                func_builder.finish_with_outputs(inner_id.outputs().chain(q_out.outputs()))?
            };
            module_builder.finish_prelude_hugr()
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }

    // Scaffolding for copy insertion tests
    fn copy_scaffold<F>(f: F, msg: &'static str) -> Result<(), BuildError>
    where
        F: FnOnce(FunctionBuilder<&mut Hugr>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
    {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_build = module_builder.define_function(
                "main",
                FunctionType::new(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]),
            )?;

            f(f_build)?;

            module_builder.finish_hugr(&EMPTY_REG)
        };
        assert_matches!(build_result, Ok(_), "Failed on example: {}", msg);

        Ok(())
    }
    #[test]
    fn copy_insertion() -> Result<(), BuildError> {
        copy_scaffold(
            |f_build| {
                let [b1] = f_build.input_wires_arr();
                f_build.finish_with_outputs([b1, b1])
            },
            "Copy input and output",
        )?;

        copy_scaffold(
            |mut f_build| {
                let [b1] = f_build.input_wires_arr();
                let xor = f_build.add_dataflow_op(and_op(), [b1, b1])?;
                f_build.finish_with_outputs([xor.out_wire(0), b1])
            },
            "Copy input and use with binary function",
        )?;

        copy_scaffold(
            |mut f_build| {
                let [b1] = f_build.input_wires_arr();
                let xor1 = f_build.add_dataflow_op(and_op(), [b1, b1])?;
                let xor2 = f_build.add_dataflow_op(and_op(), [b1, xor1.out_wire(0)])?;
                f_build.finish_with_outputs([xor2.out_wire(0), b1])
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
                .define_function("main", FunctionType::new(type_row![QB], type_row![QB, QB]))?;

            let [q1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([q1, q1])?;

            Ok(module_builder.finish_prelude_hugr()?)
        };

        assert_matches!(
            builder(),
            Err(BuildError::OutputWiring {
                error: BuilderWiringError::NoCopyLinear { typ, .. },
                ..
            })
            if typ == QB
        );
    }

    #[test]
    fn simple_inter_graph_edge() {
        let builder = || -> Result<Hugr, BuildError> {
            let mut f_build =
                FunctionBuilder::new("main", FunctionType::new(type_row![BIT], type_row![BIT]))?;

            let [i1] = f_build.input_wires_arr();
            let noop = f_build.add_dataflow_op(Noop { ty: BIT }, [i1])?;
            let i1 = noop.out_wire(0);

            let mut nested =
                f_build.dfg_builder(FunctionType::new(type_row![], type_row![BIT]), [])?;

            let id = nested.add_dataflow_op(Noop { ty: BIT }, [i1])?;

            let nested = nested.finish_with_outputs([id.out_wire(0)])?;

            f_build.finish_hugr_with_outputs([nested.out_wire(0)], &EMPTY_REG)
        };

        assert_matches!(builder(), Ok(_));
    }

    #[test]
    fn error_on_linear_inter_graph_edge() -> Result<(), BuildError> {
        let mut f_build =
            FunctionBuilder::new("main", FunctionType::new(type_row![QB], type_row![QB]))?;

        let [i1] = f_build.input_wires_arr();
        let noop = f_build.add_dataflow_op(Noop { ty: QB }, [i1])?;
        let i1 = noop.out_wire(0);

        let mut nested = f_build.dfg_builder(FunctionType::new(type_row![], type_row![QB]), [])?;

        let id_res = nested.add_dataflow_op(Noop { ty: QB }, [i1]);

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
        assert_eq!(simple_dfg_hugr.node_count(), 3);
        assert_matches!(simple_dfg_hugr.root_type().tag(), OpTag::Dfg);
    }

    #[test]
    fn insert_hugr() -> Result<(), BuildError> {
        // Create a simple DFG
        let mut dfg_builder = DFGBuilder::new(FunctionType::new(type_row![BIT], type_row![BIT]))?;
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.set_metadata("x", 42);
        let dfg_hugr = dfg_builder.finish_hugr_with_outputs([i1], &EMPTY_REG)?;

        // Create a module, and insert the DFG into it
        let mut module_builder = ModuleBuilder::new();

        let (dfg_node, f_node) = {
            let mut f_build = module_builder
                .define_function("main", FunctionType::new(type_row![BIT], type_row![BIT]))?;

            let [i1] = f_build.input_wires_arr();
            let dfg = f_build.add_hugr_with_wires(dfg_hugr, [i1])?;
            let f = f_build.finish_with_outputs([dfg.out_wire(0)])?;
            module_builder.set_child_metadata(f.node(), "x", "hi");
            (dfg.node(), f.node())
        };

        let hugr = module_builder.finish_hugr(&EMPTY_REG)?;
        assert_eq!(hugr.node_count(), 7);

        assert_eq!(hugr.get_metadata(hugr.root(), "x"), None);
        assert_eq!(hugr.get_metadata(dfg_node, "x").cloned(), Some(json!(42)));
        assert_eq!(hugr.get_metadata(f_node, "x").cloned(), Some(json!("hi")));

        Ok(())
    }

    #[test]
    fn lift_node() -> Result<(), BuildError> {
        let xa: ExtensionId = "A".try_into().unwrap();
        let xb: ExtensionId = "B".try_into().unwrap();
        let xc: ExtensionId = "C".try_into().unwrap();

        let mut parent = DFGBuilder::new(ft1(BIT))?;

        let [w] = parent.input_wires_arr();

        // A box which adds extensions A and B, via child Lift nodes
        let mut add_ab = parent.dfg_builder(ft1(BIT), [w])?;
        let [w] = add_ab.input_wires_arr();

        let lift_a = add_ab.add_dataflow_op(
            Lift {
                type_row: type_row![BIT],
                new_extension: xa.clone(),
            },
            [w],
        )?;
        let [w] = lift_a.outputs_arr();

        let lift_b = add_ab.add_dataflow_op(
            Lift {
                type_row: type_row![BIT],
                new_extension: xb,
            },
            [w],
        )?;
        let [w] = lift_b.outputs_arr();

        let add_ab = add_ab.finish_with_outputs([w])?;
        let [w] = add_ab.outputs_arr();

        // Add another node (a sibling to add_ab) which adds extension C
        // via a child lift node
        let mut add_c = parent.dfg_builder(ft1(BIT), [w])?;
        let [w] = add_c.input_wires_arr();
        let lift_c = add_c.add_dataflow_op(
            Lift {
                type_row: type_row![BIT],
                new_extension: xc,
            },
            [w],
        )?;
        let wires: Vec<Wire> = lift_c.outputs().collect();

        let add_c = add_c.finish_with_outputs(wires)?;
        let [w] = add_c.outputs_arr();
        parent.finish_hugr_with_outputs([w], &EMPTY_REG)?;

        Ok(())
    }

    #[test]
    fn non_cfg_ancestor() -> Result<(), BuildError> {
        let unit_sig = FunctionType::new(type_row![Type::UNIT], type_row![Type::UNIT]);
        let mut b = DFGBuilder::new(unit_sig.clone())?;
        let b_child = b.dfg_builder(unit_sig.clone(), [b.input().out_wire(0)])?;
        let b_child_in_wire = b_child.input().out_wire(0);
        b_child.finish_with_outputs([])?;
        let b_child_2 = b.dfg_builder(unit_sig.clone(), [])?;

        // DFG block has edge coming a sibling block, which is only valid for
        // CFGs
        let b_child_2_handle = b_child_2.finish_with_outputs([b_child_in_wire])?;

        let res = b.finish_prelude_hugr_with_outputs([b_child_2_handle.out_wire(0)]);

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
        let unit_sig = FunctionType::new(type_row![Type::UNIT], type_row![Type::UNIT]);
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
            TypeScheme::new(
                [TypeParam::new_list(TypeBound::Copyable)],
                FunctionType::new(
                    Type::new_function(FunctionTypeRV::new(USIZE_T, tv.clone())),
                    vec![],
                ),
            ),
        )?;

        // But cannot eval it...
        let ev = e.instantiate_extension_op(
            "eval",
            [vec![USIZE_T.into()].into(), vec![tv.into()].into()],
            &PRELUDE_REGISTRY,
        );
        assert!(ev.is_err());
        Ok(())
    }
}

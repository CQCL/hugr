use hugr::builder::{
    BuildError, BuildHandle, CFGBuilder, CircuitBuilder, Container, DFGBuilder, Dataflow,
    DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder, SubContainer,
};
use hugr::extension::{prelude::BOOL_T, ExtensionSet, EMPTY_REG};
use hugr::ops::handle::FuncID;
use hugr::ops::LeafOp;
use hugr::types::{FunctionType, Signature, Type, TypeArg, TypeBound};
use hugr::{hugr::CircuitUnit, ops, type_row, Hugr, Wire};

/// Wire up inputs of a Dataflow container to the outputs.
pub fn n_identity<T: DataflowSubContainer>(
    dataflow_builder: T,
) -> Result<T::ContainerHandle, BuildError> {
    let w = dataflow_builder.input_wires();
    dataflow_builder.finish_with_outputs(w)
}

pub fn build_main(
    signature: Signature,
    f: impl FnOnce(FunctionBuilder<&mut Hugr>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
) -> Result<Hugr, BuildError> {
    let mut module_builder = ModuleBuilder::new();
    let f_builder = module_builder.define_function("main", signature)?;

    f(f_builder)?;
    Ok(module_builder.finish_prelude_hugr()?)
}

fn build_basic_cfg<T: AsMut<Hugr> + AsRef<Hugr>>(
    cfg_builder: &mut CFGBuilder<T>,
) -> Result<(), BuildError> {
    use examples::NAT;
    let sum2_variants = vec![type_row![NAT], type_row![NAT]];
    let mut entry_b =
        cfg_builder.entry_builder(sum2_variants.clone(), type_row![], ExtensionSet::new())?;
    let entry = {
        let [inw] = entry_b.input_wires_arr();

        let sum = entry_b.make_tuple_sum(1, sum2_variants, [inw])?;
        entry_b.finish_with_outputs(sum, [])?
    };
    let mut middle_b =
        cfg_builder.simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
    let middle = {
        let c = middle_b.add_load_const(ops::Const::unary_unit_sum(), ExtensionSet::new())?;
        let [inw] = middle_b.input_wires_arr();
        middle_b.finish_with_outputs(c, [inw])?
    };
    let exit = cfg_builder.exit_block();
    cfg_builder.branch(&entry, 0, &middle)?;
    cfg_builder.branch(&middle, 0, &exit)?;
    cfg_builder.branch(&entry, 1, &exit)?;
    Ok(())
}

fn get_gate(gate_name: &str) -> ops::LeafOp {
    hugr::std_extensions::quantum::EXTENSION
        .instantiate_extension_op(gate_name, [], &EMPTY_REG)
        .unwrap()
        .into()
}

// Scaffolding for copy insertion tests
fn copy_scaffold<F>(f: F) -> Result<Hugr, BuildError>
where
    F: FnOnce(FunctionBuilder<&mut Hugr>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
{
    let mut module_builder = ModuleBuilder::new();

    let f_build = module_builder.define_function(
        "main",
        FunctionType::new(type_row![BOOL_T], type_row![BOOL_T, BOOL_T]).pure(),
    )?;

    f(f_build)?;

    module_builder.finish_hugr(&EMPTY_REG).map_err(Into::into)
}

pub mod examples {
    pub mod logic {
        use super::*;
        use hugr::std_extensions::logic;
        /// Generate a logic extension and "and" operation over [`crate::prelude::BOOL_T`]
        pub fn and_op() -> LeafOp {
            logic::EXTENSION
                .instantiate_extension_op(
                    logic::AND_NAME,
                    [TypeArg::BoundedNat { n: 2 }],
                    &EMPTY_REG,
                )
                .unwrap()
                .into()
        }

        /// Generate a logic extension and "not" operation over [`crate::prelude::BOOL_T`]
        pub fn not_op() -> LeafOp {
            logic::EXTENSION
                .instantiate_extension_op(logic::NOT_NAME, [], &EMPTY_REG)
                .unwrap()
                .into()
        }
    }
    pub mod quantum {
        use super::*;
        pub fn h_gate() -> ops::LeafOp {
            get_gate("H")
        }

        pub fn cx_gate() -> ops::LeafOp {
            get_gate("CX")
        }

        pub fn measure() -> ops::LeafOp {
            get_gate("Measure")
        }
    }

    use super::*;

    pub const NAT: Type = hugr::extension::prelude::USIZE_T;
    pub const BIT: Type = hugr::extension::prelude::BOOL_T;
    pub const QB: Type = hugr::extension::prelude::QB_T;

    pub fn local_def() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let mut f_build = module_builder.define_function(
            "main",
            FunctionType::new(type_row![NAT], type_row![NAT]).pure(),
        )?;
        let local_build = f_build.define_function(
            "local",
            FunctionType::new(type_row![NAT], type_row![NAT]).pure(),
        )?;
        let [wire] = local_build.input_wires_arr();
        let f_id = local_build.finish_with_outputs([wire])?;

        let call = f_build.call(f_id.handle(), f_build.input_wires())?;

        f_build.finish_with_outputs(call.outputs())?;
        module_builder.finish_prelude_hugr().map_err(Into::into)
    }

    pub fn simple_alias() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let qubit_state_type = module_builder.add_alias_declare("qubit_state", TypeBound::Any)?;

        let f_build = module_builder.define_function(
            "main",
            FunctionType::new(
                vec![qubit_state_type.get_alias_type()],
                vec![qubit_state_type.get_alias_type()],
            )
            .pure(),
        )?;
        n_identity(f_build)?;
        module_builder.finish_hugr(&EMPTY_REG).map_err(Into::into)
    }

    pub fn basic_recurse() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let f_id = module_builder.declare(
            "main",
            FunctionType::new(type_row![NAT], type_row![NAT]).pure(),
        )?;

        let mut f_build = module_builder.define_declaration(&f_id)?;
        let call = f_build.call(&f_id, f_build.input_wires())?;

        f_build.finish_with_outputs(call.outputs())?;
        module_builder.finish_prelude_hugr().map_err(Into::into)
    }

    pub fn basic_cfg() -> Result<Hugr, BuildError> {
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
        build_basic_cfg(&mut cfg_builder)?;
        cfg_builder.finish_prelude_hugr().map_err(Into::into)
    }

    pub fn basic_module_cfg() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let mut func_builder = module_builder
            .define_function("main", FunctionType::new(vec![NAT], type_row![NAT]).pure())?;
        let _f_id = {
            let [int] = func_builder.input_wires_arr();

            let cfg_id = {
                let mut cfg_builder = func_builder.cfg_builder(
                    vec![(NAT, int)],
                    None,
                    type_row![NAT],
                    ExtensionSet::new(),
                )?;
                build_basic_cfg(&mut cfg_builder)?;

                cfg_builder.finish_sub_container()?
            };

            func_builder.finish_with_outputs(cfg_id.outputs())?
        };
        module_builder.finish_prelude_hugr().map_err(Into::into)
    }

    pub fn simple_linear() -> Result<Hugr, BuildError> {
        use quantum::{cx_gate, h_gate};
        build_main(
            FunctionType::new(type_row![QB, QB], type_row![QB, QB]).pure(),
            |mut f_build| {
                let wires = f_build.input_wires().collect();

                let mut linear = CircuitBuilder::new(wires, &mut f_build);

                assert_eq!(linear.n_wires(), 2);

                linear
                    .append(h_gate(), [0])?
                    .append(cx_gate(), [0, 1])?
                    .append(cx_gate(), [1, 0])?;

                let outs = linear.finish();
                f_build.finish_with_outputs(outs)
            },
        )
    }

    pub fn with_nonlinear_and_outputs() -> Result<Hugr, BuildError> {
        use quantum::{cx_gate, measure};
        let my_custom_op = ops::LeafOp::CustomOp(
            crate::ops::custom::ExternalOp::Opaque(ops::custom::OpaqueOp::new(
                "MissingRsrc".try_into().unwrap(),
                "MyOp",
                "unknown op".to_string(),
                vec![],
                Some(FunctionType::new(vec![QB, NAT], vec![QB])),
            ))
            .into(),
        );
        build_main(
            FunctionType::new(type_row![QB, QB, NAT], type_row![QB, QB, BOOL_T]).pure(),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

                let mut linear = f_build.as_circuit(vec![q0, q1]);

                let measure_out = linear
                    .append(cx_gate(), [0, 1])?
                    .append_and_consume(
                        my_custom_op,
                        [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
                    )?
                    .append_with_outputs(measure(), [0])?;

                let out_qbs = linear.finish();
                f_build.finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            },
        )
    }

    pub fn nested_identity() -> Result<Hugr, BuildError> {
        use quantum::h_gate;
        let mut module_builder = ModuleBuilder::new();

        let _f_id = {
            let mut func_builder = module_builder.define_function(
                "main",
                FunctionType::new(type_row![NAT, QB], type_row![NAT, QB]).pure(),
            )?;

            let [int, qb] = func_builder.input_wires_arr();
            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb])?;

            let inner_builder = func_builder.dfg_builder(
                FunctionType::new(type_row![NAT], type_row![NAT]),
                None,
                [int],
            )?;
            let inner_id = n_identity(inner_builder)?;

            func_builder.finish_with_outputs(inner_id.outputs().chain(q_out.outputs()))?
        };
        module_builder.finish_prelude_hugr().map_err(Into::into)
    }

    pub fn copy_input_and_output() -> Result<Hugr, BuildError> {
        copy_scaffold(|f_build| {
            let [b1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([b1, b1])
        })
    }

    pub fn copy_input_and_use_with_binary_function() -> Result<Hugr, BuildError> {
        use logic::and_op;
        copy_scaffold(|mut f_build| {
            let [b1] = f_build.input_wires_arr();
            let xor = f_build.add_dataflow_op(and_op(), [b1, b1])?;
            f_build.finish_with_outputs([xor.out_wire(0), b1])
        })
    }

    pub fn copy_multiple_times() -> Result<Hugr, BuildError> {
        use logic::and_op;
        copy_scaffold(|mut f_build| {
            let [b1] = f_build.input_wires_arr();
            let xor1 = f_build.add_dataflow_op(and_op(), [b1, b1])?;
            let xor2 = f_build.add_dataflow_op(and_op(), [b1, xor1.out_wire(0)])?;
            f_build.finish_with_outputs([xor2.out_wire(0), b1])
        })
    }

    pub fn simple_inter_graph_edge() -> Result<Hugr, BuildError> {
        let mut f_build = FunctionBuilder::new(
            "main",
            FunctionType::new(type_row![BIT], type_row![BIT]).pure(),
        )?;

        let [i1] = f_build.input_wires_arr();
        let noop = f_build.add_dataflow_op(LeafOp::Noop { ty: BIT }, [i1])?;
        let i1 = noop.out_wire(0);

        let mut nested =
            f_build.dfg_builder(FunctionType::new(type_row![], type_row![BIT]), None, [])?;

        let id = nested.add_dataflow_op(LeafOp::Noop { ty: BIT }, [i1])?;

        let nested = nested.finish_with_outputs([id.out_wire(0)])?;

        f_build.finish_hugr_with_outputs([nested.out_wire(0)], &EMPTY_REG)
    }
}

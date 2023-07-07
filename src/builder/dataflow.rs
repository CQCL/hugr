use super::build_traits::{HugrBuilder, SubContainer};
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::hugr::{HugrView, ValidationError};
use crate::ops;

use crate::types::{Signature, TypeRow};

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
        signature: Signature,
    ) -> Result<Self, BuildError> {
        let num_in_wires = signature.input.len();
        let num_out_wires = signature.output.len();
        base.as_mut().add_op_with_parent(
            parent,
            ops::Input {
                types: signature.input.clone(),
                resources: signature.input_resources,
            },
        )?;
        base.as_mut().add_op_with_parent(
            parent,
            ops::Output {
                types: signature.output.clone(),
                resources: signature.output_resources,
            },
        )?;

        Ok(Self {
            base,
            dfg_node: parent,
            num_in_wires,
            num_out_wires,
        })
    }
}

impl DFGBuilder<Hugr> {
    /// Begin building a new DFG rooted HUGR.
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
    ) -> Result<DFGBuilder<Hugr>, BuildError> {
        let input = input.into();
        let output = output.into();
        let signature = Signature::new_df(input, output);
        let dfg_op = ops::DFG {
            signature: signature.clone(),
        };
        let base = Hugr::new(dfg_op);
        let root = base.root();
        DFGBuilder::create_with_io(base, root, signature)
    }
}

impl HugrBuilder for DFGBuilder<Hugr> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError> {
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
    /// Initialize a builder for a FuncDefn rooted HUGR
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(name: impl Into<String>, signature: Signature) -> Result<Self, BuildError> {
        let op = ops::FuncDefn {
            signature: signature.clone(),
            name: name.into(),
        };

        let base = Hugr::new(op);
        let root = base.root();

        let db = DFGBuilder::create_with_io(base, root, signature)?;
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
    fn finish_hugr(self) -> Result<Hugr, ValidationError> {
        self.0.finish_hugr()
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;
    use serde_json::json;

    use crate::builder::build_traits::DataflowHugr;
    use crate::builder::{DataflowSubContainer, ModuleBuilder};
    use crate::ops::tag::OpTag;
    use crate::ops::OpTrait;
    use crate::types::SimpleType;
    use crate::{
        builder::{
            test::{n_identity, BIT, NAT, QB},
            BuildError,
        },
        ops::LeafOp,
        resource::ResourceSet,
        type_row,
        types::Signature,
        Wire,
    };

    use super::*;
    #[test]
    fn nested_identity() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let _f_id = {
                let mut func_builder = module_builder.define_function(
                    "main",
                    Signature::new_df(type_row![NAT, QB], type_row![NAT, QB]),
                )?;

                let [int, qb] = func_builder.input_wires_arr();

                let q_out = func_builder.add_dataflow_op(LeafOp::H, vec![qb])?;

                let inner_builder = func_builder
                    .dfg_builder(Signature::new_df(type_row![NAT], type_row![NAT]), [int])?;
                let inner_id = n_identity(inner_builder)?;

                func_builder.finish_with_outputs(inner_id.outputs().chain(q_out.outputs()))?
            };
            module_builder.finish_hugr()
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
                Signature::new_df(type_row![BIT], type_row![BIT, BIT]),
            )?;

            f(f_build)?;

            module_builder.finish_hugr()
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
                let xor = f_build.add_dataflow_op(LeafOp::Xor, [b1, b1])?;
                f_build.finish_with_outputs([xor.out_wire(0), b1])
            },
            "Copy input and use with binary function",
        )?;

        copy_scaffold(
            |mut f_build| {
                let [b1] = f_build.input_wires_arr();
                let xor1 = f_build.add_dataflow_op(LeafOp::Xor, [b1, b1])?;
                let xor2 = f_build.add_dataflow_op(LeafOp::Xor, [b1, xor1.out_wire(0)])?;
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
                .define_function("main", Signature::new_df(type_row![QB], type_row![QB, QB]))?;

            let [q1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([q1, q1])?;

            Ok(module_builder.finish_hugr()?)
        };

        assert_eq!(builder(), Err(BuildError::NoCopyLinear(SimpleType::Qubit)));
    }

    #[test]
    fn simple_inter_graph_edge() {
        let builder = || -> Result<Hugr, BuildError> {
            let mut f_build =
                FunctionBuilder::new("main", Signature::new_df(type_row![BIT], type_row![BIT]))?;

            let [i1] = f_build.input_wires_arr();
            let noop = f_build.add_dataflow_op(LeafOp::Noop { ty: BIT }, [i1])?;
            let i1 = noop.out_wire(0);

            let mut nested =
                f_build.dfg_builder(Signature::new_df(type_row![], type_row![BIT]), [])?;

            let id = nested.add_dataflow_op(LeafOp::Noop { ty: BIT }, [i1])?;

            let nested = nested.finish_with_outputs([id.out_wire(0)])?;

            f_build.finish_hugr_with_outputs([nested.out_wire(0)])
        };

        assert_matches!(builder(), Ok(_));
    }

    #[test]
    fn dfg_hugr() -> Result<(), BuildError> {
        let dfg_builder = DFGBuilder::new(type_row![BIT], type_row![BIT])?;

        let [i1] = dfg_builder.input_wires_arr();
        let hugr = dfg_builder.finish_hugr_with_outputs([i1])?;

        assert_eq!(hugr.node_count(), 3);
        assert_matches!(hugr.root_type().tag(), OpTag::Dfg);

        Ok(())
    }

    #[test]
    fn insert_hugr() -> Result<(), BuildError> {
        // Create a simple DFG
        let mut dfg_builder = DFGBuilder::new(type_row![BIT], type_row![BIT])?;
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.set_metadata(json!(42));
        let dfg_hugr = dfg_builder.finish_hugr_with_outputs([i1])?;

        // Create a module, and insert the DFG into it
        let mut module_builder = ModuleBuilder::new();

        {
            let mut f_build = module_builder
                .define_function("main", Signature::new_df(type_row![BIT], type_row![BIT]))?;

            let [i1] = f_build.input_wires_arr();
            let id = f_build.add_hugr_with_wires(dfg_hugr, [i1])?;
            f_build.finish_with_outputs([id.out_wire(0)])?;
        }

        assert_eq!(module_builder.finish_hugr()?.node_count(), 7);

        Ok(())
    }

    #[test]
    fn lift_node() -> Result<(), BuildError> {
        let mut module_builder = ModuleBuilder::new();

        let ab_resources = ResourceSet::from_iter(["A".into(), "B".into()]);
        let c_resources = ResourceSet::singleton(&"C".into());
        let abc_resources = ab_resources.clone().union(&c_resources);

        let mut parent_sig = Signature::new_df(type_row![BIT], type_row![BIT]);
        parent_sig.output_resources = abc_resources.clone();
        let mut parent = module_builder.define_function("parent", parent_sig)?;

        let mut add_c_sig = Signature::new_df(type_row![BIT], type_row![BIT]);
        add_c_sig.input_resources = ab_resources.clone();
        add_c_sig.output_resources = abc_resources;

        let [w] = parent.input_wires_arr();

        let mut add_ab_sig = Signature::new_df(type_row![BIT], type_row![BIT]);
        add_ab_sig.output_resources = ab_resources.clone();

        // A box which adds resources A and B, via child Lift nodes
        let mut add_ab = parent.dfg_builder(add_ab_sig, [w])?;
        let [w] = add_ab.input_wires_arr();

        let lift_a = add_ab.add_dataflow_op(
            LeafOp::Lift {
                type_row: type_row![BIT],
                input_resources: ResourceSet::new(),
                new_resource: "A".into(),
            },
            [w],
        )?;
        let [w] = lift_a.outputs_arr();

        let lift_b = add_ab.add_dataflow_op(
            LeafOp::Lift {
                type_row: type_row![BIT],
                input_resources: ResourceSet::from_iter(["A".into()]),
                new_resource: "B".into(),
            },
            [w],
        )?;
        let [w] = lift_b.outputs_arr();

        let add_ab = add_ab.finish_with_outputs([w])?;
        let [w] = add_ab.outputs_arr();

        // Add another node (a sibling to add_ab) which adds resource C
        // via a child lift node
        let mut add_c = parent.dfg_builder(add_c_sig, [w])?;
        let [w] = add_c.input_wires_arr();
        let lift_c = add_c.add_dataflow_op(
            LeafOp::Lift {
                type_row: type_row![BIT],
                input_resources: ab_resources,
                new_resource: "C".into(),
            },
            [w],
        )?;
        let wires: Vec<Wire> = lift_c.outputs().collect();

        let add_c = add_c.finish_with_outputs(wires)?;
        let [w] = add_c.outputs_arr();
        parent.finish_with_outputs([w])?;
        module_builder.finish_hugr()?;

        Ok(())
    }
}

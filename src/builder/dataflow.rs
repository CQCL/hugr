use super::build_traits::HugrBuilder;
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID, HugrMutRef};

use std::marker::PhantomData;

use crate::hugr::{HugrView, ValidationError};
use crate::ops::{DataflowOp, OpType};

use crate::types::{Signature, TypeRow};

use crate::Node;
use crate::{hugr::HugrMut, Hugr};

/// Builder for a [`crate::ops::dataflow::DataflowOp::DFG`] node.
pub struct DFGBuilder<T> {
    pub(crate) base: T,
    pub(crate) dfg_node: Node,
    pub(crate) num_in_wires: usize,
    pub(crate) num_out_wires: usize,
    pub(crate) io: [Node; 2],
}

impl<T: HugrMutRef> DFGBuilder<T> {
    pub(super) fn create_with_io(
        mut base: T,
        parent: Node,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, BuildError> {
        let num_in_wires = inputs.len();
        let num_out_wires = outputs.len();
        let i = base.as_mut().add_op_with_parent(
            parent,
            OpType::Dataflow(DataflowOp::Input { types: inputs }),
        )?;
        let o = base.as_mut().add_op_with_parent(
            parent,
            OpType::Dataflow(DataflowOp::Output { types: outputs }),
        )?;

        Ok(Self {
            base,
            dfg_node: parent,
            io: [i, o],
            num_in_wires,
            num_out_wires,
        })
    }
}
impl DFGBuilder<HugrMut> {
    /// Begin building a new DFG rooted HUGR.
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
    ) -> Result<DFGBuilder<HugrMut>, BuildError> {
        let input = input.into();
        let output = output.into();
        let dfg_op = DataflowOp::DFG {
            signature: Signature::new_df(input.clone(), output.clone()),
        };
        let base = HugrMut::new(dfg_op);
        let root = base.hugr().root();
        DFGBuilder::create_with_io(base, root, input, output)
    }
}

impl HugrBuilder for DFGBuilder<HugrMut> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError> {
        self.base.finish()
    }
}

impl<T: HugrMutRef> Container for DFGBuilder<T> {
    type ContainerHandle = BuildHandle<DfgID>;
    #[inline]
    fn container_node(&self) -> Node {
        self.dfg_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base.as_mut()
    }
    #[inline]
    fn finish_container(self) -> Result<Self::ContainerHandle, BuildError> {
        Ok((self.dfg_node, self.num_out_wires).into())
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.as_ref().hugr()
    }
}

impl<T: HugrMutRef> Dataflow for DFGBuilder<T> {
    #[inline]
    fn io(&self) -> [Node; 2] {
        self.io
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        self.num_in_wires
    }
}

/// Wrapper around [`DFGBuilder`] used to build other dataflow regions.
// Stores option of DFGBuilder so it can be taken out without moving.
pub struct DFGWrapper<B, T>(Option<DFGBuilder<B>>, PhantomData<T>);

/// Builder for a [`crate::ops::module::ModuleOp::Def`] node
///
/// The `DEF` const generic is used to indicate whether the function is
/// defined or just declared.
pub type FunctionBuilder<B, const DEF: bool> = DFGWrapper<B, BuildHandle<FuncID<DEF>>>;

impl<B, T> DFGWrapper<B, T> {
    pub(super) fn new(db: DFGBuilder<B>) -> Self {
        Self(Some(db), PhantomData)
    }
}

impl<B: HugrMutRef, T: From<BuildHandle<DfgID>>> Container for DFGWrapper<B, T> {
    type ContainerHandle = T;

    #[inline]
    fn container_node(&self) -> Node {
        self.0.as_ref().unwrap().container_node()
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.0.as_mut().unwrap().base()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.0.as_ref().unwrap().hugr()
    }
    #[inline]
    fn finish_container(mut self) -> Result<Self::ContainerHandle, BuildError> {
        let dfg = self.0.take().expect("Already finished.");
        dfg.finish_container().map(Into::into)
    }
}

impl<B: HugrMutRef, T: From<BuildHandle<DfgID>>> Dataflow for DFGWrapper<B, T> {
    #[inline]
    fn io(&self) -> [Node; 2] {
        self.0.as_ref().unwrap().io
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        self.0.as_ref().unwrap().num_inputs()
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::builder::build_traits::DataflowHugrBuilder;
    use crate::builder::ModuleBuilder;
    use crate::hugr::HugrView;
    use crate::{
        builder::{
            test::{n_identity, BIT, NAT, QB},
            BuildError,
        },
        ops::LeafOp,
        type_row,
        types::{LinearType, Signature},
    };

    use super::*;
    #[test]
    fn nested_identity() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let _f_id = {
                let mut func_builder = module_builder.declare_and_def(
                    "main",
                    Signature::new_df(type_row![NAT, QB], type_row![NAT, QB]),
                )?;

                let [int, qb] = func_builder.input_wires_arr();

                let q_out = func_builder.add_dataflow_op(
                    OpType::Dataflow(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![qb],
                )?;

                let inner_builder = func_builder.dfg_builder(vec![(NAT, int)], type_row![NAT])?;
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
        F: FnOnce(
            FunctionBuilder<&mut HugrMut, true>,
        ) -> Result<BuildHandle<FuncID<true>>, BuildError>,
    {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_build = module_builder.declare_and_def(
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
                .declare_and_def("main", Signature::new_df(type_row![QB], type_row![QB, QB]))?;

            let [q1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([q1, q1])?;

            Ok(module_builder.finish_hugr()?)
        };

        assert_eq!(builder(), Err(BuildError::NoCopyLinear(LinearType::Qubit)));
    }

    #[test]
    fn simple_inter_graph_edge() {
        let builder = || -> Result<Hugr, BuildError> {
            let mut module_builder = ModuleBuilder::new();

            let mut f_build = module_builder
                .declare_and_def("main", Signature::new_df(type_row![BIT], type_row![BIT]))?;

            let [i1] = f_build.input_wires_arr();
            let noop = f_build.add_dataflow_op(LeafOp::Noop(BIT), [i1])?;
            let i1 = noop.out_wire(0);

            let mut nested = f_build.dfg_builder(vec![], type_row![BIT])?;

            let id = nested.add_dataflow_op(LeafOp::Noop(BIT), [i1])?;

            let nested = nested.finish_with_outputs([id.out_wire(0)])?;

            f_build.finish_with_outputs([nested.out_wire(0)])?;

            Ok(module_builder.finish_hugr()?)
        };

        assert_matches!(builder(), Ok(_));
    }

    #[test]
    fn dfg_hugr() -> Result<(), BuildError> {
        let dfg_builder = DFGBuilder::new(type_row![BIT], type_row![BIT])?;

        let [i1] = dfg_builder.input_wires_arr();
        let hugr = dfg_builder.finish_hugr_with_outputs([i1])?;

        assert_eq!(hugr.node_count(), 3);
        assert_matches!(hugr.root_type(), OpType::Dataflow(DataflowOp::DFG { .. }));

        Ok(())
    }
}

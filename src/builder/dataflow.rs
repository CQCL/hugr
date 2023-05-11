use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::ops::{DataflowOp, OpType};

use crate::types::TypeRow;

use portgraph::NodeIndex;

use crate::{hugr::HugrMut, Hugr};

/// Builder for a [`crate::ops::dataflow::DataflowOp::DFG`] node.
pub struct DFGBuilder<'f> {
    pub(crate) base: &'f mut HugrMut,
    pub(crate) dfg_node: NodeIndex,
    pub(crate) num_in_wires: usize,
    pub(crate) num_out_wires: usize,
    pub(crate) io: [NodeIndex; 2],
}

impl<'f> DFGBuilder<'f> {
    pub(super) fn create_with_io(
        base: &'f mut HugrMut,
        parent: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, BuildError> {
        let num_in_wires = inputs.len();
        let num_out_wires = outputs.len();
        let i = base.add_op_with_parent(
            parent,
            OpType::Dataflow(DataflowOp::Input { types: inputs }),
        )?;
        let o = base.add_op_with_parent(
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

impl<'f> Container for DFGBuilder<'f> {
    type ContainerHandle = DfgID;
    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.dfg_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }
    #[inline]
    fn finish(self) -> DfgID {
        (self.dfg_node, self.num_out_wires).into()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.hugr()
    }
}

impl<'f> Dataflow for DFGBuilder<'f> {
    #[inline]
    fn io(&self) -> [NodeIndex; 2] {
        self.io
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        self.num_in_wires
    }
}

pub struct DFGWrapper<'b, T>(DFGBuilder<'b>, PhantomData<T>);

/// Builder for a [`crate::ops::module::ModuleOp::Def`] node
pub type FunctionBuilder<'b> = DFGWrapper<'b, FuncID>;

impl<'b, T> DFGWrapper<'b, T> {
    pub(super) fn new(db: DFGBuilder<'b>) -> Self {
        Self(db, PhantomData)
    }
}

impl<'b, T: From<DfgID>> Container for DFGWrapper<'b, T> {
    type ContainerHandle = T;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.0.container_node()
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.0.base()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.0.hugr()
    }
    #[inline]
    fn finish(self) -> Self::ContainerHandle {
        self.0.finish().into()
    }
}

impl<'b, T: From<DfgID>> Dataflow for DFGWrapper<'b, T> {
    #[inline]
    fn io(&self) -> [NodeIndex; 2] {
        self.0.io
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            module_builder::ModuleBuilder,
            test::{n_identity, BIT, NAT, QB},
            BuildError, BuildHandle,
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
            module_builder.finish()
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }

    // Scaffolding for copy insertion tests
    fn copy_scaffold<F>(f: F, msg: &'static str) -> Result<(), BuildError>
    where
        F: FnOnce(FunctionBuilder) -> Result<FuncID, BuildError>,
    {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_build = module_builder.declare_and_def(
                "main",
                Signature::new_df(type_row![BIT], type_row![BIT, BIT]),
            )?;

            f(f_build)?;

            module_builder.finish()
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

            module_builder.finish()
        };

        assert_eq!(builder(), Err(BuildError::NoCopyLinear(LinearType::Qubit)));
    }

    #[test]
    fn simple_inter_graph_edge() {
        let builder = || {
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

            module_builder.finish()
        };

        assert_matches!(builder(), Ok(_));
    }
}

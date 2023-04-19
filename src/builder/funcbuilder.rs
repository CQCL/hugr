use portgraph::NodeIndex;

use crate::hugr::BaseBuilder;
use crate::ops::DataflowOp;
use crate::types::{Signature, SimpleType, TypeRow};
use crate::{hugr::HugrError, ops::OpType};

#[derive(Clone, Copy)]
pub struct Wire(NodeIndex, usize);
pub struct DeltaID {
    node: NodeIndex,
    out_wires: Vec<Wire>,
}

pub struct DeltaBuilder<'f> {
    base: &'f mut BaseBuilder,
    delt_node: NodeIndex,
    internal_in_wires: Vec<Wire>,
    external_out_wires: Vec<Wire>,
    io: [NodeIndex; 2],
}

pub trait DataflowIO<'f> {
    fn add_io_children(
        base: &'f mut BaseBuilder,
        cont_node: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<[NodeIndex; 2], HugrError> {
        let i = base.add_op_with_parent(
            cont_node,
            OpType::Function(DataflowOp::Input { types: inputs }),
        )?;
        let o = base.add_op_with_parent(
            cont_node,
            OpType::Function(DataflowOp::Output { types: outputs }),
        )?;

        Ok([i, o])
    }
}

pub trait ContainerBuilder {
    type ContainerHandle;
    fn parent(&self) -> NodeIndex;
    fn base(&mut self) -> &mut BaseBuilder;
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<NodeIndex, HugrError> {
        let parent = self.parent();
        self.base().add_op_with_parent(parent, op)
    }
    fn finish(self) -> Self::ContainerHandle;
}

pub trait DataflowBuilder: ContainerBuilder {
    fn io(&self) -> [NodeIndex; 2];
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        inputs: Vec<Wire>,
    ) -> Result<Vec<Wire>, HugrError> {
        let (_, wires) = add_op_with_wires(self, op, inputs)?;
        Ok(wires)
    }

    fn set_outputs(&mut self, outputs: Vec<Wire>) -> Result<(), HugrError> {
        let [_, out] = self.io();
        let base = self.base();
        for (dst_port, Wire(src, src_port)) in outputs.into_iter().enumerate() {
            base.connect(src, src_port, out, dst_port)?;
        }
        Ok(())
    }

    fn input_wires(&self) -> &[Wire];

    fn input_wires_arr<const N: usize>(&self) -> [Wire; N] {
        self.input_wires()
            .try_into()
            .expect(&format!("Incorrect number of wires: {}", N)[..])
    }

    fn delta_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: Vec<(SimpleType, Wire)>,
        outputs: TypeRow,
    ) -> Result<DeltaBuilder<'b>, HugrError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();
        let (deltn, _) = add_op_with_wires(
            self,
            OpType::Function(DataflowOp::Nested {
                signature: Signature::new(input_types.clone(), outputs.clone(), None),
            }),
            input_wires,
        )?;

        DeltaBuilder::create_with_io(self.base(), deltn, input_types.into(), outputs)
    }
}

fn add_op_with_wires<T: DataflowBuilder + ?Sized>(
    dbuild: &mut T,
    op: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(NodeIndex, Vec<Wire>), HugrError> {
    let [_, out] = dbuild.io();
    let base = dbuild.base();
    let op: OpType = op.into();
    let sig = op.signature();
    let opn = base.add_op_before(out, op)?;
    for (dst_port, Wire(src, src_port)) in inputs.into_iter().enumerate() {
        base.connect(src, src_port, opn, dst_port)?;
    }
    let wires = (0..sig.output.len()).map(|i| Wire(opn, i)).collect();

    Ok((opn, wires))
}

impl<'f> DeltaBuilder<'f> {
    pub fn create_with_io(
        base: &'f mut BaseBuilder,
        parent: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, HugrError> {
        let ilen = inputs.len();
        let olen = outputs.len();
        let i = base.add_op_with_parent(
            parent,
            OpType::Function(DataflowOp::Input { types: inputs }),
        )?;
        let o = base.add_op_with_parent(
            parent,
            OpType::Function(DataflowOp::Output { types: outputs }),
        )?;

        let internal_in_wires = (0..ilen).map(|port| Wire(i, port)).collect();
        let external_out_wires = (0..olen).map(|port| Wire(parent, port)).collect();
        Ok(Self {
            base,
            delt_node: parent,
            io: [i, o],
            internal_in_wires,
            external_out_wires,
        })
    }
}

impl<'f> ContainerBuilder for DeltaBuilder<'f> {
    type ContainerHandle = DeltaID;
    fn parent(&self) -> NodeIndex {
        self.delt_node
    }

    fn base(&mut self) -> &mut BaseBuilder {
        self.base
    }
    fn finish(self) -> DeltaID {
        DeltaID {
            node: self.delt_node,
            out_wires: self.external_out_wires,
        }
    }
}

impl<'f> DataflowBuilder for DeltaBuilder<'f> {
    fn io(&self) -> [NodeIndex; 2] {
        self.io
    }

    fn input_wires(&self) -> &[Wire] {
        &self.internal_in_wires[..]
    }
}

#[cfg(test)]
mod test {

    use crate::{
        ops::LeafOp,
        type_row,
        types::{ClassicType, QuantumType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::Nat);
    const QB: SimpleType = SimpleType::Quantum(QuantumType::Qubit);

    #[test]
    fn nested_identity() -> Result<(), HugrError> {
        let mut base = BaseBuilder::new();
        let root = base.root();

        let mut outbuild =
            DeltaBuilder::create_with_io(&mut base, root, type_row![NAT, QB], type_row![NAT, QB])?;

        let [int, qb] = outbuild.input_wires_arr();

        let qout = outbuild.add_dataflow_op(
            OpType::Function(DataflowOp::Leaf { op: LeafOp::H }),
            vec![qb],
        )?;

        let indelt: DeltaID;
        {
            let mut inbuilder = outbuild.delta_builder(vec![(NAT, int)], type_row![NAT])?;
            inbuilder.set_outputs(inbuilder.input_wires().iter().cloned().collect())?;
            indelt = inbuilder.finish();
        }

        outbuild.set_outputs([indelt.out_wires, qout].concat())?;

        let _ = outbuild.finish();

        let buildres = base.finish();
        // println!("{}", buildres.unwrap().dot_string());
        assert!(buildres.is_ok());

        Ok(())
    }
}

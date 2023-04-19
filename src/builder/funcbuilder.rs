use portgraph::NodeIndex;

use crate::hugr::{BuildError, HugrMut};
use crate::ops::{DataflowOp, ModuleOp};
use crate::types::{Signature, SimpleType, TypeRow};
use crate::Hugr;
use crate::{hugr::HugrError, ops::OpType};

#[derive(Clone, Copy)]
pub struct Wire(NodeIndex, usize);
pub struct DeltaID {
    node: NodeIndex,
    out_wires: Vec<Wire>,
}

pub struct FuncID(NodeIndex);

#[derive(Default)]
pub struct ModuleBuilder(HugrMut);

impl ModuleBuilder {
    pub fn new() -> Self {
        Self(HugrMut::new())
    }
}

pub struct DeltaBuilder<'f> {
    base: &'f mut HugrMut,
    delt_node: NodeIndex,
    internal_in_wires: Vec<Wire>,
    external_out_wires: Vec<Wire>,
    io: [NodeIndex; 2],
}

pub trait Container {
    type ContainerHandle;
    fn container_node(&self) -> NodeIndex;
    fn base(&mut self) -> &mut HugrMut;
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<NodeIndex, HugrError> {
        let parent = self.container_node();
        self.base().add_op_with_parent(parent, op)
    }

    fn finish(self) -> Self::ContainerHandle;
}

pub trait Dataflow: Container {
    fn io(&self) -> [NodeIndex; 2];
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        inputs: Vec<Wire>,
    ) -> Result<Vec<Wire>, HugrError> {
        let (_, wires) = add_op_with_wires(self, op, inputs)?;
        Ok(wires)
    }

    fn set_outputs(&mut self, outputs: impl IntoIterator<Item = Wire>) -> Result<(), HugrError> {
        let [_, out] = self.io();
        let base = self.base();
        for (dst_port, Wire(src, src_port)) in outputs.into_iter().enumerate() {
            base.connect(src, src_port, out, dst_port)?;
        }
        Ok(())
    }

    fn finish_with_outputs(
        mut self,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<Self::ContainerHandle, HugrError>
    where
        Self: Sized,
    {
        self.set_outputs(outputs)?;
        Ok(self.finish())
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

fn add_op_with_wires<T: Dataflow + ?Sized>(
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
        base: &'f mut HugrMut,
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

impl<'f> Container for DeltaBuilder<'f> {
    type ContainerHandle = DeltaID;
    fn container_node(&self) -> NodeIndex {
        self.delt_node
    }

    fn base(&mut self) -> &mut HugrMut {
        self.base
    }
    fn finish(self) -> DeltaID {
        DeltaID {
            node: self.delt_node,
            out_wires: self.external_out_wires,
        }
    }
}

impl<'f> Dataflow for DeltaBuilder<'f> {
    fn io(&self) -> [NodeIndex; 2] {
        self.io
    }

    fn input_wires(&self) -> &[Wire] {
        &self.internal_in_wires[..]
    }
}

impl Container for ModuleBuilder {
    type ContainerHandle = Result<Hugr, BuildError>;

    fn container_node(&self) -> NodeIndex {
        self.0.root()
    }

    fn base(&mut self) -> &mut HugrMut {
        &mut self.0
    }

    fn finish(self) -> Self::ContainerHandle {
        self.0.finish()
    }
}

pub struct FunctionBuilder<'b>(DeltaBuilder<'b>);

impl<'b> Container for FunctionBuilder<'b> {
    type ContainerHandle = FuncID;

    fn container_node(&self) -> NodeIndex {
        self.0.container_node()
    }

    fn base(&mut self) -> &mut HugrMut {
        self.0.base()
    }

    fn finish(self) -> Self::ContainerHandle {
        let n = self.0.finish();
        FuncID(n.node)
    }
}

impl<'b> Dataflow for FunctionBuilder<'b> {
    fn io(&self) -> [NodeIndex; 2] {
        self.0.io
    }

    fn input_wires(&self) -> &[Wire] {
        self.0.input_wires()
    }
}

impl ModuleBuilder {
    pub fn function_builder<'a: 'b, 'b>(
        &'a mut self,
        _name: impl Into<String>,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<FunctionBuilder<'b>, HugrError> {
        let defn = self.add_child_op(OpType::Module(ModuleOp::Def {
            signature: Signature::new(inputs.clone(), outputs.clone(), None),
        }))?;

        let db = DeltaBuilder::create_with_io(self.base(), defn, inputs, outputs)?;
        Ok(FunctionBuilder(db))
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
        let buildres = {
            let mut modbuilder = ModuleBuilder::new();

            let _fdef = {
                let mut fbuild =
                    modbuilder.function_builder("main", type_row![NAT, QB], type_row![NAT, QB])?;

                let [int, qb] = fbuild.input_wires_arr();

                let qout = fbuild.add_dataflow_op(
                    OpType::Function(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![qb],
                )?;

                let indelt: DeltaID = {
                    let inbuilder = fbuild.delta_builder(vec![(NAT, int)], type_row![NAT])?;
                    let [iw] = inbuilder.input_wires_arr();
                    inbuilder.finish_with_outputs([iw])?
                };

                fbuild.finish_with_outputs([indelt.out_wires, qout].concat())?
            };
            modbuilder.finish()
        };

        println!("{}", buildres.clone().unwrap().dot_string());
        assert!(buildres.is_ok());

        Ok(())
    }
}

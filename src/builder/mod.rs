use std::marker::PhantomData;

use portgraph::NodeIndex;

use crate::hugr::{BuildError, HugrMut};
use crate::ops::controlflow::ControlFlowOp;
use crate::ops::{BasicBlockOp, DataflowOp, ModuleOp};
use crate::types::{Signature, SimpleType, TypeRow};
use crate::Hugr;
use crate::{hugr::HugrError, ops::OpType};
use nodehandle::{BetaID, DeltaID, FuncID, KappaID};

use self::nodehandle::BuildHandle;

pub mod nodehandle;
#[derive(Clone, Copy)]
pub struct Wire(NodeIndex, usize);

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

    fn kappa_builder<'a: 'b, 'b>(
        &'a mut self,
        entry_inputs: Vec<(SimpleType, Wire)>,
        entry_outputs: TypeRow,
        exit_inputs: TypeRow,
        exit_outputs: TypeRow,
    ) -> Result<KappaBuilder<'b>, HugrError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) =
            entry_inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (kapn, wirs) = add_op_with_wires(
            self,
            OpType::Function(DataflowOp::ControlFlow {
                op: ControlFlowOp::CFG {
                    inputs: inputs.clone(),
                    outputs: exit_outputs.clone(),
                },
            }),
            input_wires,
        )?;
        let entryop = OpType::BasicBlock(BasicBlockOp {
            inputs: inputs.clone(),
            outputs: entry_outputs.clone(),
        });
        let exitop = OpType::BasicBlock(BasicBlockOp {
            inputs: exit_inputs.clone(),
            outputs: exit_outputs.clone(),
        });

        let entry_n = self.base().add_op_with_parent(kapn, entryop)?;
        let exit_n = self.base().add_op_with_parent(kapn, exitop)?;

        let kb = KappaBuilder {
            base: self.base(),
            kapp_node: kapn,
            external_out_wires: wirs,
            entry_exit: [
                (entry_n, inputs, entry_outputs),
                (exit_n, exit_inputs, exit_outputs),
            ],
        };

        Ok(kb)
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
    fn create_with_io(
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
    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.delt_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }
    #[inline]
    fn finish(self) -> DeltaID {
        (self.delt_node, self.external_out_wires).into()
    }
}

impl<'f> Dataflow for DeltaBuilder<'f> {
    #[inline]
    fn io(&self) -> [NodeIndex; 2] {
        self.io
    }

    #[inline]
    fn input_wires(&self) -> &[Wire] {
        &self.internal_in_wires[..]
    }
}

impl Container for ModuleBuilder {
    type ContainerHandle = Result<Hugr, BuildError>;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.0.root()
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        &mut self.0
    }

    #[inline]
    fn finish(self) -> Self::ContainerHandle {
        self.0.finish()
    }
}

pub struct DeltaWrapper<'b, T>(DeltaBuilder<'b>, PhantomData<T>);

pub type FunctionBuilder<'b> = DeltaWrapper<'b, FuncID>;
pub type BetaBuilder<'b> = DeltaWrapper<'b, BetaID>;

impl<'b, T> DeltaWrapper<'b, T> {
    fn new(db: DeltaBuilder<'b>) -> Self {
        Self(db, PhantomData)
    }
}

impl<'b, T: From<DeltaID>> Container for DeltaWrapper<'b, T> {
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
    fn finish(self) -> Self::ContainerHandle {
        self.0.finish().into()
    }
}

impl<'b, T: From<DeltaID>> Dataflow for DeltaWrapper<'b, T> {
    #[inline]
    fn io(&self) -> [NodeIndex; 2] {
        self.0.io
    }

    #[inline]
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
        // TODO add name and param names to metadata
        let defn = self.add_child_op(OpType::Module(ModuleOp::Def {
            signature: Signature::new(inputs.clone(), outputs.clone(), None),
        }))?;

        let db = DeltaBuilder::create_with_io(self.base(), defn, inputs, outputs)?;
        Ok(FunctionBuilder::new(db))
    }
}

pub struct KappaBuilder<'f> {
    base: &'f mut HugrMut,
    kapp_node: NodeIndex,
    external_out_wires: Vec<Wire>,
    entry_exit: [(NodeIndex, TypeRow, TypeRow); 2],
}

impl<'f> Container for KappaBuilder<'f> {
    type ContainerHandle = KappaID;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.kapp_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }

    #[inline]
    fn finish(self) -> Self::ContainerHandle {
        (self.kapp_node, self.external_out_wires).into()
    }
}

impl<'f> KappaBuilder<'f> {
    pub fn beta_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<BetaBuilder<'b>, HugrError> {
        let op = OpType::BasicBlock(BasicBlockOp {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        });
        let [_, (exit, ..)] = self.entry_exit;
        let base = self.base();
        let opn = base.add_op_before(exit, op)?;

        let db = DeltaBuilder::create_with_io(self.base(), opn, inputs, outputs)?;
        Ok(BetaBuilder::new(db))
    }

    fn entry_exit_builder<'a: 'b, 'b, const N: usize>(
        &'a mut self,
    ) -> Result<BetaBuilder<'b>, HugrError> {
        let (betn, inputs, outputs) = self.entry_exit[N].clone();
        DeltaBuilder::create_with_io(self.base(), betn, inputs, outputs).map(BetaBuilder::new)
    }
    #[inline]
    pub fn entry_builder<'a: 'b, 'b>(&'a mut self) -> Result<BetaBuilder<'b>, HugrError> {
        self.entry_exit_builder::<0>()
    }

    #[inline]
    pub fn exit_builder<'a: 'b, 'b>(&'a mut self) -> Result<BetaBuilder<'b>, HugrError> {
        self.entry_exit_builder::<1>()
    }

    pub fn branch(&mut self, from: &BetaID, to: &BetaID) -> Result<(), HugrError> {
        let from = from.node();
        let to = to.node();
        let base = &mut self.base;
        let hugr = base.hugr();
        let fin = hugr.num_inputs(from);
        let fout = hugr.num_outputs(from);
        let tin = hugr.num_inputs(to);
        let tout = hugr.num_outputs(to);

        base.set_num_ports(from, fin, fout + 1);
        base.set_num_ports(to, tin + 1, tout);
        base.connect(from, fout, to, tin)
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

                let inbuilder = fbuild.delta_builder(vec![(NAT, int)], type_row![NAT])?;
                let indelt = nat_identity(inbuilder)?;

                fbuild.finish_with_outputs([indelt.sig_out_wires(), &qout].concat())?
            };
            modbuilder.finish()
        };

        // crate::utils::test::viz_dotstr(&buildres.clone().unwrap().dot_string());
        assert!(buildres.is_ok());

        Ok(())
    }

    fn nat_identity<T: Dataflow>(inbuilder: T) -> Result<T::ContainerHandle, HugrError> {
        let [iw] = inbuilder.input_wires_arr();
        inbuilder.finish_with_outputs([iw])
    }

    #[test]
    fn basic_cfg() -> Result<(), HugrError> {
        let buildres = {
            let mut modbuilder = ModuleBuilder::new();

            let _fdef = {
                let mut fbuild =
                    modbuilder.function_builder("main", type_row![NAT], type_row![NAT])?;

                let [int] = fbuild.input_wires_arr();

                let inkapp: KappaID = {
                    let mut cfgbuilder = fbuild.kappa_builder(
                        vec![(NAT, int)],
                        type_row![NAT],
                        type_row![NAT],
                        type_row![NAT],
                    )?;
                    let entrybuild = cfgbuilder.entry_builder()?;

                    let entry = nat_identity(entrybuild)?;

                    let middlebuild = cfgbuilder.beta_builder(type_row![NAT], type_row![NAT])?;

                    let middle = nat_identity(middlebuild)?;

                    let exitbuild = cfgbuilder.exit_builder()?;

                    let exit = nat_identity(exitbuild)?;

                    cfgbuilder.branch(&entry, &middle)?;
                    cfgbuilder.branch(&middle, &exit)?;

                    cfgbuilder.finish()
                };

                fbuild.finish_with_outputs(Vec::from(inkapp.sig_out_wires()))?
            };
            modbuilder.finish()
        };

        // crate::utils::test::viz_dotstr(&buildres.clone().unwrap().dot_string());
        assert!(buildres.is_ok());

        Ok(())
    }
}

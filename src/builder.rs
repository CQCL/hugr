use std::marker::PhantomData;

use portgraph::NodeIndex;
use thiserror::Error;

use crate::hugr::{HugrMut, ValidationError};
use crate::ops::controlflow::ControlFlowOp;
use crate::ops::{BasicBlockOp, ConstValue, DataflowOp, ModuleOp};
use crate::types::{Signature, SimpleType, TypeRow};
use crate::Hugr;
use crate::{hugr::HugrError, ops::OpType};
use nodehandle::{BetaID, DeltaID, FuncID, KappaID};

use self::nodehandle::{BuildHandle, ConstID};

pub mod nodehandle;

#[derive(Clone, Copy, Debug)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(NodeIndex, usize);

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError),
    /// HUGR construction error.
    #[error("Error when mutating HUGR: {0}.")]
    ConstructError(#[from] HugrError),
    /// CFG can only have one entry.
    #[error("CFG entry node already built for CFG node: {0:?}.")]
    EntryBuiltError(NodeIndex),
    /// FuncID does not refer to declare.
    #[error("FuncID containing node {0:?} does not refer to a Declare as expected")]
    NotDeclareError(NodeIndex),
}

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
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<NodeIndex, BuildError> {
        let parent = self.container_node();
        Ok(self.base().add_op_with_parent(parent, op)?)
    }
    fn finish(self) -> Self::ContainerHandle;
}

pub trait Dataflow: Container {
    fn io(&self) -> [NodeIndex; 2];
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        inputs: Vec<Wire>,
    ) -> Result<Vec<Wire>, BuildError> {
        let (_, wires) = add_op_with_wires(self, op, inputs)?;
        Ok(wires)
    }

    fn set_outputs(&mut self, outputs: impl IntoIterator<Item = Wire>) -> Result<(), BuildError> {
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
    ) -> Result<Self::ContainerHandle, BuildError>
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
    ) -> Result<DeltaBuilder<'b>, BuildError> {
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
        inputs: Vec<(SimpleType, Wire)>,
        outputs: TypeRow,
    ) -> Result<KappaBuilder<'b>, BuildError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (kapn, _) = add_op_with_wires(
            self,
            OpType::Function(DataflowOp::ControlFlow {
                op: ControlFlowOp::CFG {
                    inputs: inputs.clone(),
                    outputs: outputs.clone(),
                },
            }),
            input_wires,
        )?;

        let exitbeta = OpType::BasicBlock(BasicBlockOp::Exit {
            cfg_outputs: outputs.clone(),
        });
        let exit_node = self.base().add_op_with_parent(kapn, exitbeta)?;
        let n_out_wires = outputs.len();
        let kb = KappaBuilder {
            base: self.base(),
            kapp_node: kapn,
            n_out_wires,
            exit_node,
            inputs: Some(inputs),
        };

        Ok(kb)
    }

    fn load_const(&mut self, cid: &ConstID) -> Result<Wire, BuildError> {
        let cn = cid.node();
        let cout = self.base().hugr().num_outputs(cn);

        self.base().set_num_ports(cn, 0, cout + 1);

        let mut loadn = self.add_dataflow_op(
            DataflowOp::LoadConstant {
                datatype: cid.const_type(),
            },
            vec![Wire(cn, cout)],
        )?;

        Ok(loadn.remove(0))
    }
}

fn add_op_with_wires<T: Dataflow + ?Sized>(
    dbuild: &mut T,
    op: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(NodeIndex, Vec<Wire>), BuildError> {
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
    ) -> Result<Self, BuildError> {
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
        Ok(self.0.finish()?)
    }
}

pub struct DeltaWrapper<'b, T>(DeltaBuilder<'b>, PhantomData<T>);

pub type FunctionBuilder<'b> = DeltaWrapper<'b, FuncID>;

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
    pub fn define_function<'a: 'b, 'b>(
        &'a mut self,
        fid: &FuncID,
    ) -> Result<FunctionBuilder<'b>, BuildError> {
        let fnode = fid.node();
        let (inputs, outputs) = if let OpType::Module(ModuleOp::Declare { signature }) =
            self.base().hugr().get_optype(fnode)
        {
            (signature.input.clone(), signature.output.clone())
        } else {
            return Err(BuildError::NotDeclareError(fid.node()));
        };
        self.base().replace_op(
            fnode,
            OpType::Module(ModuleOp::Def {
                signature: Signature::new(inputs.clone(), outputs.clone(), None),
            }),
        );

        let db = DeltaBuilder::create_with_io(self.base(), fnode, inputs, outputs)?;
        Ok(FunctionBuilder::new(db))
    }

    pub fn declare_and_def<'a: 'b, 'b>(
        &'a mut self,
        _name: impl Into<String>,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<FunctionBuilder<'b>, BuildError> {
        let fid = self.declare(_name, inputs, outputs)?;
        self.define_function(&fid)
    }

    pub fn declare(
        &mut self,
        _name: impl Into<String>,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<FuncID, BuildError> {
        // TODO add name and param names to metadata
        let decln = self.add_child_op(ModuleOp::Declare {
            signature: Signature::new(inputs, outputs, None),
        })?;

        Ok(decln.into())
    }

    pub fn constant(&mut self, val: ConstValue) -> Result<ConstID, BuildError> {
        let typ = val.const_type();
        let cn = self.add_child_op(ModuleOp::Const(val))?;

        Ok((cn, typ).into())
    }
}

pub struct KappaBuilder<'f> {
    base: &'f mut HugrMut,
    kapp_node: NodeIndex,
    inputs: Option<TypeRow>,
    exit_node: NodeIndex,
    n_out_wires: usize,
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
        let wirs = (0..self.n_out_wires)
            .map(|i| Wire(self.kapp_node, i))
            .collect();
        (self.kapp_node, wirs).into()
    }
}

impl<'f> KappaBuilder<'f> {
    pub fn beta_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: TypeRow,
        outputs: TypeRow,
        n_branches: usize,
    ) -> Result<BetaBuilder<'b>, BuildError> {
        let predtype = SimpleType::new_predicate(n_branches);
        let outputs: TypeRow = [&[predtype], outputs.as_ref()].concat().into();
        let op = OpType::BasicBlock(BasicBlockOp::Beta {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        });
        let exit = self.exit_node;
        let beta_n = self.base().add_op_before(exit, op)?;

        self.base().set_num_ports(beta_n, 0, n_branches);

        let db = DeltaBuilder::create_with_io(self.base(), beta_n, inputs, outputs)?;
        Ok(BetaBuilder::new(db))
    }

    pub fn entry_builder<'a: 'b, 'b>(
        &'a mut self,
        outputs: TypeRow,
        n_branches: usize,
    ) -> Result<BetaBuilder<'b>, BuildError> {
        let inputs = self
            .inputs
            .take()
            .ok_or(BuildError::EntryBuiltError(self.kapp_node))?;
        self.beta_builder(inputs, outputs, n_branches)
    }

    pub fn exit_block(&self) -> BetaID {
        self.exit_node.into()
    }

    pub fn branch(
        &mut self,
        pred: &BetaID,
        branch: usize,
        succ: &BetaID,
    ) -> Result<(), BuildError> {
        let from = pred.node();
        let to = succ.node();
        let base = &mut self.base;
        let hugr = base.hugr();
        let tin = hugr.num_inputs(to);
        let tout = hugr.num_outputs(to);

        base.set_num_ports(to, tin + 1, tout);
        Ok(base.connect(from, branch, to, tin)?)
    }
}

pub type BetaBuilder<'b> = DeltaWrapper<'b, BetaID>;

impl<'b> BetaBuilder<'b> {
    pub fn set_outputs(
        &mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [branch_wire].into_iter().chain(outputs.into_iter()))
    }
    pub fn finish_with_outputs(
        mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<<DeltaWrapper<'b, BetaID> as Container>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(branch_wire, outputs)?;
        Ok(self.finish())
    }
}

#[cfg(test)]
mod test {

    use crate::{
        ops::LeafOp,
        type_row,
        types::{ClassicType, LinearType},
    };

    use super::*;

    const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

    #[test]
    fn nested_identity() -> Result<(), BuildError> {
        let buildres = {
            let mut modbuilder = ModuleBuilder::new();

            let _fdef = {
                let mut fbuild =
                    modbuilder.declare_and_def("main", type_row![NAT, QB], type_row![NAT, QB])?;

                let [int, qb] = fbuild.input_wires_arr();

                let qout = fbuild.add_dataflow_op(
                    OpType::Function(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![qb],
                )?;

                let inbuilder = fbuild.delta_builder(vec![(NAT, int)], type_row![NAT])?;
                let indelt = n_identity(inbuilder)?;

                fbuild.finish_with_outputs([indelt.sig_out_wires(), &qout].concat())?
            };
            modbuilder.finish()
        };

        assert_eq!(buildres.err(), None);

        Ok(())
    }

    fn n_identity<T: Dataflow>(inbuilder: T) -> Result<T::ContainerHandle, BuildError> {
        let w = Vec::from(inbuilder.input_wires());
        inbuilder.finish_with_outputs(w)
    }

    #[test]
    fn basic_cfg() -> Result<(), BuildError> {
        let sum2_type = SimpleType::new_predicate(2);

        let buildres = {
            let mut modbuilder = ModuleBuilder::new();
            let main =
                modbuilder.declare("main", vec![sum2_type.clone(), NAT].into(), type_row![NAT])?;
            let s1 = modbuilder.constant(ConstValue::predicate(0, 1))?;
            let _fdef = {
                let mut fbuild = modbuilder.define_function(&main)?;
                let [flag, int] = fbuild.input_wires_arr();

                let inkapp: KappaID = {
                    let mut cfgbuilder = fbuild
                        .kappa_builder(vec![(sum2_type, flag), (NAT, int)], type_row![NAT])?;
                    let entrybuild = cfgbuilder.entry_builder(type_row![NAT], 2)?;

                    let entry = n_identity(entrybuild)?;

                    let mut middlebuild =
                        cfgbuilder.beta_builder(type_row![NAT], type_row![NAT], 1)?;

                    let middle = {
                        let c = middlebuild.load_const(&s1)?;
                        let [inw] = middlebuild.input_wires_arr();
                        middlebuild.finish_with_outputs(c, [inw])?
                    };

                    let exit = cfgbuilder.exit_block();

                    cfgbuilder.branch(&entry, 0, &middle)?;
                    cfgbuilder.branch(&middle, 0, &exit)?;
                    cfgbuilder.branch(&entry, 1, &exit)?;

                    cfgbuilder.finish()
                };

                fbuild.finish_with_outputs(Vec::from(inkapp.sig_out_wires()))?
            };
            modbuilder.finish()
        };

        assert_eq!(buildres.err(), None);

        Ok(())
    }
}

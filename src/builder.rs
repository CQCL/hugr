use std::collections::HashSet;
use std::marker::PhantomData;

use itertools::Itertools;
use portgraph::{Direction, NodeIndex, PortOffset};
use thiserror::Error;

use crate::hugr::{HugrMut, ValidationError};
use crate::ops::controlflow::ControlFlowOp;
use crate::ops::{BasicBlockOp, BranchOp, ConstValue, DataflowOp, LeafOp, ModuleOp};
use crate::types::{ClassicType, EdgeKind, LinearType, Signature, SimpleType, TypeRow};
use crate::Hugr;
use crate::{hugr::HugrError, ops::OpType};
use nodehandle::{BetaID, DeltaID, FuncID, KappaID, OpID};

use self::nodehandle::{BuildHandle, ConstID, GammaID, ThetaID};

pub mod nodehandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Node was expected to have a certain type but was found to not.
    #[error("Node with index {node:?} does not have type {op_desc:?} as expected.")]
    UnexpectedType {
        node: NodeIndex,
        op_desc: &'static str,
    },
    /// Error building gamma node
    #[error("Error building gamma node: {0}.")]
    GammaError(#[from] GammaBuildError),

    /// Wire not found in Hugr
    #[error("Wire not found in Hugr: {0:?}.")]
    WireNotFound(Wire),

    /// Can't copy a linear type
    #[error("Can't copy linear type: {0:?}.")]
    NoCopyLinear(LinearType),
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
    delta_node: NodeIndex,
    num_in_wires: usize,
    num_out_wires: usize,
    io: [NodeIndex; 2],
}

pub trait Container {
    type ContainerHandle;
    fn container_node(&self) -> NodeIndex;
    fn base(&mut self) -> &mut HugrMut;
    fn hugr(&self) -> &Hugr;
    fn add_child_op(&mut self, op: impl Into<OpType>) -> Result<NodeIndex, BuildError> {
        let parent = self.container_node();
        Ok(self.base().add_op_with_parent(parent, op)?)
    }

    /// Adds a non-dataflow edge between two nodes. The kind is given by the operation's [`OpType::other_inputs`] or  [`OpType::other_outputs`]
    ///
    /// [`OpType::other_inputs`]: crate::ops::OpType::other_inputs
    /// [`OpType::other_outputs`]: crate::ops::OpType::other_outputs
    fn add_other_wire(&mut self, src: NodeIndex, dst: NodeIndex) -> Result<Wire, BuildError> {
        let (src_port, _) = self.base().add_other_edge(src, dst)?;
        Ok(Wire(src, src_port))
    }

    fn finish(self) -> Self::ContainerHandle;
}

pub trait Dataflow: Container {
    fn io(&self) -> [NodeIndex; 2];
    fn num_inputs(&self) -> usize;

    fn input(&self) -> OpID {
        (self.io()[0], self.num_inputs()).into()
    }

    fn input_wires(&self) -> Vec<Wire> {
        self.input().outputs()
    }
    fn add_dataflow_op(
        &mut self,
        op: impl Into<OpType>,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<OpID, BuildError> {
        let outs = add_op_with_wires(self, op, input_wires.into_iter().collect())?;

        Ok(outs.into())
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
        let (delta_n, _) = add_op_with_wires(
            self,
            OpType::Function(DataflowOp::Nested {
                signature: Signature::new(input_types.clone(), outputs.clone(), None),
            }),
            input_wires,
        )?;

        DeltaBuilder::create_with_io(self.base(), delta_n, input_types.into(), outputs)
    }

    fn kappa_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: Vec<(SimpleType, Wire)>,
        outputs: TypeRow,
    ) -> Result<KappaBuilder<'b>, BuildError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();

        let inputs: TypeRow = input_types.into();

        let (kappa_node, _) = add_op_with_wires(
            self,
            OpType::Function(DataflowOp::ControlFlow {
                op: ControlFlowOp::CFG {
                    inputs: inputs.clone(),
                    outputs: outputs.clone(),
                },
            }),
            input_wires,
        )?;

        let exit_beta = OpType::BasicBlock(BasicBlockOp::Exit {
            cfg_outputs: outputs.clone(),
        });
        let exit_node = self.base().add_op_with_parent(kappa_node, exit_beta)?;
        let n_out_wires = outputs.len();
        let kb = KappaBuilder {
            base: self.base(),
            kappa_node,
            n_out_wires,
            exit_node,
            inputs: Some(inputs),
        };

        Ok(kb)
    }

    fn load_const(&mut self, cid: &ConstID) -> Result<Wire, BuildError> {
        let cn = cid.node();
        let c_out = self.hugr().num_outputs(cn);

        self.base().add_ports(cn, Direction::Outgoing, 1);

        let load_n = self.add_dataflow_op(
            DataflowOp::LoadConstant {
                datatype: cid.const_type(),
            },
            // Constant wire from the constant value node
            vec![Wire(cn, c_out)],
        )?;

        // Add the required incoming order wire
        self.set_order(&self.input(), &load_n)?;

        Ok(load_n.out_wire(0))
    }

    fn theta_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: Vec<(SimpleType, Wire)>,
        outputs: TypeRow,
    ) -> Result<ThetaBuilder<'b>, BuildError> {
        let (input_types, input_wires): (Vec<SimpleType>, Vec<Wire>) = inputs.into_iter().unzip();
        let (theta_node, _) = add_op_with_wires(
            self,
            OpType::Function(
                ControlFlowOp::Loop {
                    inputs: input_types.clone().into(),
                    outputs: outputs.clone(),
                }
                .into(),
            ),
            input_wires,
        )?;

        let input: TypeRow = input_types.into();

        let delta_build = DeltaBuilder::create_with_io(
            self.base(),
            theta_node,
            input.clone(),
            vec![SimpleType::new_sum(theta_sum_variants(input, outputs))].into(),
        )?;

        Ok(ThetaBuilder::new(delta_build))
    }

    fn gamma_builder<'a: 'b, 'b>(
        &'a mut self,
        (predicate_inputs, predicate_wire): (TypeRow, Wire),
        other_inputs: Vec<(SimpleType, Wire)>,
        outputs: TypeRow,
    ) -> Result<GammaBuilder<'b>, BuildError> {
        let (input_types, mut input_wires): (Vec<SimpleType>, Vec<Wire>) =
            other_inputs.into_iter().unzip();

        input_wires.insert(0, predicate_wire);

        let inputs: TypeRow = input_types.into();
        let n_branches = predicate_inputs.len();
        let n_out_wires = outputs.len();

        let gamma_node = self.add_dataflow_op(
            ControlFlowOp::Conditional {
                predicate_inputs,
                inputs,
                outputs,
            },
            input_wires,
        )?;
        Ok(GammaBuilder {
            base: self.base(),
            gamma_node: gamma_node.node(),
            n_out_wires,
            remaining_branches: (0..n_branches).collect(),
        })
    }

    /// Add an order edge from `before` to `after`. Assumes any additional edges
    /// to both nodes will be Order kind.
    fn set_order(
        &mut self,
        before: &impl BuildHandle,
        after: &impl BuildHandle,
    ) -> Result<(), BuildError> {
        self.add_other_wire(before.node(), after.node())?;

        Ok(())
    }

    fn get_wire_type(&self, wire: Wire) -> Option<SimpleType> {
        let kind = self
            .hugr()
            .get_optype(wire.0)
            .port_kind(PortOffset::new_outgoing(wire.1))?;

        if let EdgeKind::Value(typ) = kind {
            Some(typ)
        } else {
            None
        }
    }

    fn discard_type(&mut self, wire: Wire, typ: ClassicType) -> Result<OpID, BuildError> {
        self.add_dataflow_op(LeafOp::Copy { n_copies: 0, typ }, [wire])
    }

    fn discard(&mut self, wire: Wire) -> Result<OpID, BuildError> {
        let typ = self
            .get_wire_type(wire)
            .ok_or(BuildError::WireNotFound(wire))?;
        let typ = match typ {
            SimpleType::Classic(typ) => typ,
            SimpleType::Linear(typ) => return Err(BuildError::NoCopyLinear(typ)),
        };
        self.discard_type(wire, typ)
    }
}

fn add_op_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    op: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(NodeIndex, usize), BuildError> {
    let [inp, out] = data_builder.io();

    let base = data_builder.base();
    let op: OpType = op.into();
    let sig = op.signature();
    let opn = base.add_op_before(out, op)?;
    let no_inputs = inputs.is_empty();

    for (dst_port, Wire(src, src_port)) in inputs.into_iter().enumerate() {
        base.connect(src, src_port, opn, dst_port)?;
    }

    if no_inputs {
        data_builder.add_other_wire(inp, opn)?;
    }

    Ok((opn, sig.output.len()))
}

impl<'f> DeltaBuilder<'f> {
    fn create_with_io(
        base: &'f mut HugrMut,
        parent: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, BuildError> {
        let num_in_wires = inputs.len();
        let num_out_wires = outputs.len();
        let i = base.add_op_with_parent(
            parent,
            OpType::Function(DataflowOp::Input { types: inputs }),
        )?;
        let o = base.add_op_with_parent(
            parent,
            OpType::Function(DataflowOp::Output { types: outputs }),
        )?;

        Ok(Self {
            base,
            delta_node: parent,
            io: [i, o],
            num_in_wires,
            num_out_wires,
        })
    }
}

impl<'f> Container for DeltaBuilder<'f> {
    type ContainerHandle = DeltaID;
    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.delta_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }
    #[inline]
    fn finish(self) -> DeltaID {
        (self.delta_node, self.num_out_wires).into()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.hugr()
    }
}

impl<'f> Dataflow for DeltaBuilder<'f> {
    #[inline]
    fn io(&self) -> [NodeIndex; 2] {
        self.io
    }

    #[inline]
    fn num_inputs(&self) -> usize {
        self.num_in_wires
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

    fn hugr(&self) -> &Hugr {
        self.0.hugr()
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
    fn hugr(&self) -> &Hugr {
        self.0.hugr()
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
    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }
}

impl ModuleBuilder {
    pub fn define_function<'a: 'b, 'b>(
        &'a mut self,
        f_id: &FuncID,
    ) -> Result<FunctionBuilder<'b>, BuildError> {
        let f_node = f_id.node();
        let (inputs, outputs) = if let OpType::Module(ModuleOp::Declare { signature }) =
            self.hugr().get_optype(f_node)
        {
            (signature.input.clone(), signature.output.clone())
        } else {
            return Err(BuildError::UnexpectedType {
                node: f_node,
                op_desc: "ModuleOp::Declare",
            });
        };
        self.base().replace_op(
            f_node,
            OpType::Module(ModuleOp::Def {
                signature: Signature::new(inputs.clone(), outputs.clone(), None),
            }),
        );

        let db = DeltaBuilder::create_with_io(self.base(), f_node, inputs, outputs)?;
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
        let declare_n = self.add_child_op(ModuleOp::Declare {
            signature: Signature::new(inputs, outputs, None),
        })?;

        Ok(declare_n.into())
    }

    pub fn constant(&mut self, val: ConstValue) -> Result<ConstID, BuildError> {
        let typ = val.const_type();
        let const_n = self.add_child_op(ModuleOp::Const(val))?;

        Ok((const_n, typ).into())
    }
}

pub struct KappaBuilder<'f> {
    base: &'f mut HugrMut,
    kappa_node: NodeIndex,
    inputs: Option<TypeRow>,
    exit_node: NodeIndex,
    n_out_wires: usize,
}

impl<'f> Container for KappaBuilder<'f> {
    type ContainerHandle = KappaID;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.kappa_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.hugr()
    }

    #[inline]
    fn finish(self) -> Self::ContainerHandle {
        (self.kappa_node, self.n_out_wires).into()
    }
}

impl<'f> KappaBuilder<'f> {
    pub fn beta_builder<'a: 'b, 'b>(
        &'a mut self,
        inputs: TypeRow,
        outputs: TypeRow,
        n_branches: usize,
    ) -> Result<BetaBuilder<'b>, BuildError> {
        let op = OpType::BasicBlock(BasicBlockOp::Beta {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            n_branches,
        });
        let exit = self.exit_node;
        let beta_n = self.base().add_op_before(exit, op)?;

        self.base().set_num_ports(beta_n, 0, n_branches);

        // The node outputs a predicate before the data outputs of the beta node
        let predicate_type = SimpleType::new_predicate(n_branches);
        let node_outputs: TypeRow = [&[predicate_type], outputs.as_ref()].concat().into();
        let db = DeltaBuilder::create_with_io(self.base(), beta_n, inputs, node_outputs)?;
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
            .ok_or(BuildError::EntryBuiltError(self.kappa_node))?;
        self.beta_builder(inputs, outputs, n_branches)
    }

    pub fn exit_block(&self) -> BetaID {
        self.exit_node.into()
    }

    pub fn branch(
        &mut self,
        predicate: &BetaID,
        branch: usize,
        successor: &BetaID,
    ) -> Result<(), BuildError> {
        let from = predicate.node();
        let to = successor.node();
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
    ) -> Result<<BetaBuilder<'b> as Container>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(branch_wire, outputs)?;
        Ok(self.finish())
    }
}

pub type ThetaBuilder<'b> = DeltaWrapper<'b, ThetaID>;

impl<'b> ThetaBuilder<'b> {
    pub fn set_outputs(&mut self, out_variant: Wire) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [out_variant])
    }

    pub fn finish_with_outputs(
        mut self,
        out_variant: Wire,
    ) -> Result<<ThetaBuilder<'b> as Container>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(out_variant)?;
        Ok(self.finish())
    }

    fn theta_signature(&self) -> Result<Signature, BuildError> {
        let hugr = self.hugr();

        if let OpType::Function(DataflowOp::ControlFlow {
            op: ControlFlowOp::Loop { inputs, outputs },
        }) = hugr.get_optype(self.container_node())
        {
            Ok(Signature::new_df(inputs.clone(), outputs.clone()))
        } else {
            Err(BuildError::UnexpectedType {
                node: self.container_node(),
                op_desc: "ControlFlowOp::Loop",
            })
        }
    }

    fn make_out_variant<const N: usize>(
        &mut self,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let Signature { input, output, .. } = self.theta_signature()?;
        let sig = (if N == 1 { &output } else { &input }).clone();
        let make_op = self.add_dataflow_op(LeafOp::MakeTuple(sig), values)?;
        let tuple = make_op.out_wire(0);
        let variants = theta_sum_variants(input, output);

        let tag_op = self.add_dataflow_op(LeafOp::Tag { tag: N, variants }, vec![tuple])?;

        Ok(tag_op.out_wire(0))
    }

    pub fn make_continue(
        &mut self,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_out_variant::<0>(values)
    }

    pub fn make_break(
        &mut self,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_out_variant::<1>(values)
    }
}

fn theta_sum_variants(input: TypeRow, output: TypeRow) -> TypeRow {
    vec![SimpleType::new_tuple(input), SimpleType::new_tuple(output)].into()
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GammaBuildError {
    /// Branch already built.
    #[error("Branch {branch} of gamma node {gamma:?} has already been built.")]
    BranchBuiltError { gamma: NodeIndex, branch: usize },
    /// Branch already built.
    #[error("Gamma node {gamma:?} has no branch with index {branch}.")]
    NotBranchError { gamma: NodeIndex, branch: usize },
    /// Not all branches of gamma built.
    #[error("Branches {branches:?} of gamma node {gamma:?} have not been built.")]
    NotAllBranchesBuiltError {
        gamma: NodeIndex,
        branches: HashSet<usize>,
    },
}
pub struct GammaBuilder<'f> {
    base: &'f mut HugrMut,
    gamma_node: NodeIndex,
    n_out_wires: usize,
    remaining_branches: HashSet<usize>,
}

impl<'f> Container for GammaBuilder<'f> {
    type ContainerHandle = GammaID;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.gamma_node
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.base
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.hugr()
    }

    fn finish(self) -> Self::ContainerHandle {
        // TODO check all branches built
        (self.gamma_node, self.n_out_wires).into()
    }
}

impl<'f> GammaBuilder<'f> {
    pub fn branch_builder<'a: 'b, 'b>(
        &'a mut self,
        branch: usize,
    ) -> Result<DeltaBuilder<'b>, BuildError> {
        let gamma = self.gamma_node;
        let control_op: Result<ControlFlowOp, ()> =
            self.hugr().get_optype(self.gamma_node).clone().try_into();

        let Ok(ControlFlowOp::Conditional {
            predicate_inputs,
            inputs,
            outputs,
        }) = control_op else {panic!("Gamma node does not have gamma optype.")};
        let sum_input = predicate_inputs
            .get(branch)
            .ok_or(GammaBuildError::NotBranchError { gamma, branch })?
            .clone();

        if !self.remaining_branches.remove(&branch) {
            return Err(GammaBuildError::BranchBuiltError { gamma, branch }.into());
        }

        let inputs: TypeRow = [vec![sum_input], inputs.iter().cloned().collect_vec()]
            .concat()
            .into();

        let branch_node = self.add_child_op(OpType::Branch(BranchOp {
            signature: Signature::new_df(inputs.clone(), outputs.clone()),
        }))?;

        DeltaBuilder::create_with_io(self.base(), branch_node, inputs, outputs)
    }
}

#[cfg(test)]
mod test {

    use cool_asserts::assert_matches;

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
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let _f_id = {
                let mut func_builder = module_builder.declare_and_def(
                    "main",
                    type_row![NAT, QB],
                    type_row![NAT, QB],
                )?;

                let [int, qb] = func_builder.input_wires_arr();

                let q_out = func_builder.add_dataflow_op(
                    OpType::Function(DataflowOp::Leaf { op: LeafOp::H }),
                    vec![qb],
                )?;

                let inner_builder = func_builder.delta_builder(vec![(NAT, int)], type_row![NAT])?;
                let inner_id = n_identity(inner_builder)?;

                func_builder.finish_with_outputs([inner_id.outputs(), q_out.outputs()].concat())?
            };
            module_builder.finish()
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }

    fn n_identity<T: Dataflow>(dataflow_builder: T) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        dataflow_builder.finish_with_outputs(w)
    }

    #[test]
    fn basic_cfg() -> Result<(), BuildError> {
        let sum2_type = SimpleType::new_predicate(2);

        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder.declare(
                "main",
                vec![sum2_type.clone(), NAT].into(),
                type_row![NAT],
            )?;
            let s1 = module_builder.constant(ConstValue::predicate(0, 1))?;
            let _f_id = {
                let mut func_builder = module_builder.define_function(&main)?;
                let [flag, int] = func_builder.input_wires_arr();

                let kappa_id: KappaID = {
                    let mut cfg_builder = func_builder
                        .kappa_builder(vec![(sum2_type, flag), (NAT, int)], type_row![NAT])?;
                    let entry_b = cfg_builder.entry_builder(type_row![NAT], 2)?;

                    let entry = n_identity(entry_b)?;

                    let mut middle_b =
                        cfg_builder.beta_builder(type_row![NAT], type_row![NAT], 1)?;

                    let middle = {
                        let c = middle_b.load_const(&s1)?;
                        let [inw] = middle_b.input_wires_arr();
                        middle_b.finish_with_outputs(c, [inw])?
                    };

                    let exit = cfg_builder.exit_block();

                    cfg_builder.branch(&entry, 0, &middle)?;
                    cfg_builder.branch(&middle, 0, &exit)?;
                    cfg_builder.branch(&entry, 1, &exit)?;

                    cfg_builder.finish()
                };

                func_builder.finish_with_outputs(kappa_id.outputs())?
            };
            module_builder.finish()
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }

    #[test]
    fn basic_theta() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder.declare("main", type_row![], type_row![NAT])?;
            let s1 = module_builder.constant(ConstValue::Int(1))?;
            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;

                let theta: ThetaID = {
                    let mut theta_b = fbuild.theta_builder(vec![], type_row![NAT])?;

                    let const_wire = theta_b.load_const(&s1)?;

                    let break_wire = theta_b.make_break([const_wire])?;

                    theta_b.finish_with_outputs(break_wire)?
                };

                fbuild.finish_with_outputs(theta.outputs().iter().cloned())?
            };
            // crate::utils::test::viz_dotstr(&module_builder.hugr().dot_string());
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    fn basic_gamma() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder.declare("main", type_row![NAT], type_row![NAT])?;
            let tru_const = module_builder.constant(ConstValue::predicate(1, 2))?;
            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;

                let const_wire = fbuild.load_const(&tru_const)?;
                let [int] = fbuild.input_wires_arr();
                let gamma: GammaID = {
                    let predicate_inputs = vec![SimpleType::new_unit(); 2].into();
                    let other_inputs = vec![(NAT, int)];
                    let outputs = vec![SimpleType::new_unit(), NAT].into();
                    let mut gamma_b = fbuild.gamma_builder(
                        (predicate_inputs, const_wire),
                        other_inputs,
                        outputs,
                    )?;

                    // let branch_0 = gamma_b.branch_builder(0)?;
                    n_identity(gamma_b.branch_builder(0)?)?;
                    n_identity(gamma_b.branch_builder(1)?)?;

                    gamma_b.finish()
                };
                let [unit, int] = gamma.outputs_arr();
                fbuild.discard(unit)?;
                fbuild.finish_with_outputs([int])?
            };
            // crate::utils::test::viz_dotstr(&module_builder.hugr().dot_string());
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    fn theta_with_gamma() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder.declare("main", type_row![], type_row![NAT])?;
            let s1 = module_builder.constant(ConstValue::Int(1))?;
            let s2 = module_builder.constant(ConstValue::Int(2))?;
            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;

                let theta: ThetaID = {
                    let mut theta_b = fbuild.theta_builder(vec![], type_row![NAT])?;

                    let const_wire = theta_b.load_const(&s1)?;

                    let break_wire = theta_b.make_break([const_wire])?;

                    theta_b.finish_with_outputs(break_wire)?
                };

                fbuild.finish_with_outputs(theta.outputs().iter().cloned())?
            };
            // crate::utils::test::viz_dotstr(&module_builder.hugr().dot_string());
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }
}

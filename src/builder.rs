use std::collections::HashSet;
use std::iter;
use std::marker::PhantomData;

use itertools::Itertools;
use portgraph::{Direction, NodeIndex, PortOffset};
use smol_str::SmolStr;
use thiserror::Error;

use crate::hugr::validate::InterGraphEdgeError;
use crate::hugr::{HugrMut, ValidationError};
use crate::ops::controlflow::ControlFlowOp;
use crate::ops::{BasicBlockOp, BranchOp, ConstValue, DataflowOp, LeafOp, ModuleOp};
use crate::types::{ClassicType, EdgeKind, LinearType, Signature, SimpleType, TypeRow};
use crate::Hugr;
use crate::{hugr::HugrError, ops::OpType};
use nodehandle::{BetaID, DeltaID, FuncID, KappaID, OpID};

use self::nodehandle::{BuildHandle, ConstID, GammaID, LambdaID, NewTypeID, Outputs, ThetaID};

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

    fn output(&self) -> OpID {
        (self.io()[1], 0).into()
    }

    fn input_wires(&self) -> Outputs {
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
        let [inp, out] = self.io();
        wire_up_inputs(outputs.into_iter().collect_vec(), out, self, inp)
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
            .collect_vec()
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
        ThetaBuilder::create_with_io(self.base(), theta_node, input, outputs)
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
            branch_nodes: vec![None; n_branches],
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

    fn make_tuple(
        &mut self,
        types: TypeRow,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let make_op: OpID = self.add_dataflow_op(LeafOp::MakeTuple(types), values)?;
        Ok(make_op.out_wire(0))
    }

    fn make_tag(&mut self, tag: usize, variants: TypeRow, value: Wire) -> Result<Wire, BuildError> {
        let make_op: OpID = self.add_dataflow_op(LeafOp::Tag { tag, variants }, vec![value])?;
        Ok(make_op.out_wire(0))
    }

    fn make_new_type(&mut self, new_type: &NewTypeID, value: Wire) -> Result<Wire, BuildError> {
        let name = new_type.get_name().clone();
        let typ = new_type.get_core_type().clone();
        let make_op: OpID = self.add_dataflow_op(LeafOp::MakeNewType { name, typ }, [value])?;
        Ok(make_op.out_wire(0))
    }

    fn make_out_variant<const N: usize>(
        &mut self,
        signature: Signature,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let Signature { input, output, .. } = signature;
        let sig = (if N == 1 { &output } else { &input }).clone();
        let tuple = self.make_tuple(sig, values)?;
        let variants = theta_sum_variants(input, output);

        self.make_tag(N, variants, tuple)
    }

    fn make_continue(
        &mut self,
        signature: Signature,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_out_variant::<0>(signature, values)
    }

    fn make_break(
        &mut self,
        signature: Signature,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.make_out_variant::<1>(signature, values)
    }

    fn call(
        &mut self,
        function: &FuncID,
        input_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<OpID, BuildError> {
        let def_op: Result<&ModuleOp, ()> = self.hugr().get_optype(function.node()).try_into();
        let signature = match def_op {
            Ok(ModuleOp::Def { signature } | ModuleOp::Declare { signature }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: function.node(),
                    op_desc: "Declare/Def",
                })
            }
        };
        let const_in_port = signature.output.len();
        let op_id = self.add_dataflow_op(DataflowOp::Call { signature }, input_wires)?;
        let src_port: usize = self
            .base()
            .add_ports(function.node(), Direction::Outgoing, 1)
            .collect_vec()[0];

        self.base()
            .connect(function.node(), src_port, op_id.node(), const_in_port)?;
        Ok(op_id)
    }
}

fn add_op_with_wires<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    op: impl Into<OpType>,
    inputs: Vec<Wire>,
) -> Result<(NodeIndex, usize), BuildError> {
    let [inp, out] = data_builder.io();

    let op: OpType = op.into();
    let sig = op.signature();
    let op_node = data_builder.base().add_op_before(out, op)?;

    wire_up_inputs(inputs, op_node, data_builder, inp)?;

    Ok((op_node, sig.output.len()))
}

fn wire_up_inputs<T: Dataflow + ?Sized>(
    inputs: Vec<Wire>,
    op_node: NodeIndex,
    data_builder: &mut T,
    inp: NodeIndex,
) -> Result<(), BuildError> {
    let mut any_local_inputs = false;
    for (dst_port, Wire(src, src_port)) in inputs.into_iter().enumerate() {
        any_local_inputs |= wire_up(data_builder, src, src_port, op_node, dst_port)?;
    }

    if !any_local_inputs {
        // If op has no inputs add a StateOrder edge from input to place in
        // causal cone of Input node
        data_builder.add_other_wire(inp, op_node)?;
    };
    Ok(())
}

/// Add edge from src to dst and report back if they do share a parent
fn wire_up<T: Dataflow + ?Sized>(
    data_builder: &mut T,
    mut src: NodeIndex,
    mut src_port: usize,
    dst: NodeIndex,
    dst_port: usize,
) -> Result<bool, BuildError> {
    let base = data_builder.base();
    let src_offset = PortOffset::new_outgoing(src_port);

    let src_parent = base.hugr().get_parent(src);
    let dst_parent = base.hugr().get_parent(dst);
    let local_source = src_parent == dst_parent;
    if !local_source {
        if let Some(copy_port) = if_copy_add_port(base, src) {
            src_port = copy_port;
        } else if let Some(typ) = check_classical_value(base, src, src_offset)? {
            let src_parent = base.hugr().get_parent(src).expect("Node has no parent");

            let final_child = base
                .hugr()
                .children(src_parent)
                .next_back()
                .expect("Parent must have at least one child.");
            let copy_node = base.add_op_before(final_child, LeafOp::Copy { n_copies: 1, typ })?;

            base.connect(src, src_port, copy_node, 0)?;

            // Copy node has to have state edge to an ancestor of dst
            let Some(src_sibling) = iter::successors(dst_parent, |&p| base.hugr().get_parent(p))
                .tuple_windows()
                .find_map(|(ancestor, ancestor_parent)| {
                    (ancestor_parent == src_parent).then_some(ancestor)
                }) else {
                    let val_err: ValidationError = InterGraphEdgeError::NoRelation {
                        from: src,
                        from_offset: PortOffset::new_outgoing(src_port),
                        to: dst,
                        to_offset: PortOffset::new_incoming(dst_port),
                    }.into();
                    return Err(val_err.into());
                };

            base.add_other_edge(copy_node, src_sibling)?;

            src = copy_node;
            src_port = 0;
        }
    }

    if let Some((connected, connected_offset)) = base.hugr().linked_port(src, src_offset) {
        if let Some(copy_port) = if_copy_add_port(base, src) {
            src_port = copy_port;
            src = connected;
        }
        // Need to insert a copy - first check can be copied
        else if let Some(typ) = check_classical_value(base, src, src_offset)? {
            // TODO API consistency in using PortOffset vs. usize
            base.disconnect(src, src_port, Direction::Outgoing)?;

            let copy = data_builder
                .add_dataflow_op(LeafOp::Copy { n_copies: 2, typ }, [Wire(src, src_port)])?;

            let base = data_builder.base();
            base.connect(copy.node(), 0, connected, connected_offset.index())?;
            src = copy.node();
            src_port = 1;
        }
    }
    data_builder.base().connect(src, src_port, dst, dst_port)?;
    Ok(local_source)
}

/// Check the kind of a port is a classical Value and return it
/// Return None if Const kind
/// Panics if port not valid for Op or port is not Const/Value
fn check_classical_value(
    base: &HugrMut,
    src: NodeIndex,
    src_offset: PortOffset,
) -> Result<Option<ClassicType>, BuildError> {
    let wire_kind = base.hugr().get_optype(src).port_kind(src_offset).unwrap();
    let typ = match wire_kind {
        EdgeKind::Const(_) => None,
        EdgeKind::Value(simple_type) => match simple_type {
            SimpleType::Classic(typ) => Some(typ),
            SimpleType::Linear(typ) => return Err(BuildError::NoCopyLinear(typ)),
        },
        _ => {
            panic!("Wires can only be Value kind")
        }
    };

    Ok(typ)
}

// Return newly added port to copy node if src node is a copy
fn if_copy_add_port(base: &mut HugrMut, src: NodeIndex) -> Option<usize> {
    let src_op: Result<&LeafOp, ()> = base.hugr().get_optype(src).try_into();
    if let Ok(LeafOp::Copy { n_copies, typ }) = src_op {
        let copy_node = src;
        // If already connected to a copy node, add wire to the copy
        let n_copies = *n_copies;
        base.replace_op(
            copy_node,
            LeafOp::Copy {
                n_copies: n_copies + 1,
                typ: typ.clone(),
            },
        );
        base.add_ports(copy_node, Direction::Outgoing, 1).next()
    } else {
        None
    }
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

    /// Add a NewType node and return a handle to the NewType
    pub fn add_new_type(
        &mut self,
        name: impl Into<SmolStr>,
        typ: SimpleType,
    ) -> Result<NewTypeID, BuildError> {
        let name: SmolStr = name.into();

        let node = self.add_child_op(ModuleOp::NewType {
            name: name.clone(),
            definition: typ.clone(),
        })?;

        Ok((node, name, typ).into())
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
    fn create_with_io(
        base: &'b mut HugrMut,
        theta_node: NodeIndex,
        inputs: TypeRow,
        outputs: TypeRow,
    ) -> Result<Self, BuildError> {
        let delta_build = DeltaBuilder::create_with_io(
            base,
            theta_node,
            inputs.clone(),
            theta_output_row(inputs, outputs),
        )?;

        Ok(ThetaBuilder::new(delta_build))
    }
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

    pub fn theta_signature(&self) -> Result<Signature, BuildError> {
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

    pub fn internal_output_row(&self) -> Result<TypeRow, BuildError> {
        let Signature { input, output, .. } = self.theta_signature()?;

        Ok(theta_output_row(input, output))
    }
}

fn theta_output_row(input: TypeRow, output: TypeRow) -> TypeRow {
    vec![SimpleType::new_sum(theta_sum_variants(input, output))].into()
}

fn theta_sum_variants(input: TypeRow, output: TypeRow) -> TypeRow {
    vec![SimpleType::new_tuple(input), SimpleType::new_tuple(output)].into()
}

pub type LambdaBuilder<'b> = DeltaWrapper<'b, LambdaID>;

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
    branch_nodes: Vec<Option<NodeIndex>>,
}

impl<'f> Container for GammaBuilder<'f> {
    type ContainerHandle = Result<GammaID, GammaBuildError>;

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
        let branches: HashSet<usize> = self
            .branch_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| if node.is_none() { Some(i) } else { None })
            .collect();
        if !branches.is_empty() {
            return Err(GammaBuildError::NotAllBranchesBuiltError {
                gamma: self.gamma_node,
                branches,
            });
        }
        Ok((self.gamma_node, self.n_out_wires).into())
    }
}

impl<'f> GammaBuilder<'f> {
    pub fn branch_builder<'a: 'b, 'b>(
        &'a mut self,
        branch: usize,
    ) -> Result<LambdaBuilder<'b>, BuildError> {
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

        if self.branch_nodes.get(branch).unwrap().is_some() {
            return Err(GammaBuildError::BranchBuiltError { gamma, branch }.into());
        }

        let inputs: TypeRow = [vec![sum_input], inputs.iter().cloned().collect_vec()]
            .concat()
            .into();

        let bb_op = OpType::Branch(BranchOp {
            signature: Signature::new_df(inputs.clone(), outputs.clone()),
        });
        let branch_node =
            // add branch before any existing subsequent branches
            if let Some(&sibling_node) = self.branch_nodes[branch + 1..].iter().flatten().next() {
                self.base().add_op_before(sibling_node, bb_op)?
            } else {
                self.add_child_op(bb_op)?
            };

        self.branch_nodes[branch] = Some(branch_node);

        let delta_builder =
            DeltaBuilder::create_with_io(self.base(), branch_node, inputs, outputs)?;

        Ok(LambdaBuilder::new(delta_builder))
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
    const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());
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

                func_builder.finish_with_outputs(inner_id.outputs().chain(q_out.outputs()))?
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

                    let break_wire =
                        theta_b.make_break(theta_b.theta_signature()?, [const_wire])?;

                    theta_b.finish_with_outputs(break_wire)?
                };

                fbuild.finish_with_outputs(theta.outputs())?
            };
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

                    gamma_b.finish()?
                };
                let [unit, int] = gamma.outputs_arr();
                fbuild.discard(unit)?;
                fbuild.finish_with_outputs([int])?
            };
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    fn theta_with_gamma() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let main = module_builder.declare("main", type_row![BIT], type_row![NAT])?;

            let s2 = module_builder.constant(ConstValue::Int(2))?;
            let tru_const = module_builder.constant(ConstValue::predicate(1, 2))?;

            let _fdef = {
                let mut fbuild = module_builder.define_function(&main)?;
                let [b1] = fbuild.input_wires_arr();
                let theta: ThetaID = {
                    let mut theta_b = fbuild.theta_builder(vec![(BIT, b1)], type_row![NAT])?;
                    let signature = theta_b.theta_signature()?;
                    let const_wire = theta_b.load_const(&tru_const)?;
                    let [b1] = theta_b.input_wires_arr();
                    let gamma: GammaID = {
                        let predicate_inputs = vec![SimpleType::new_unit(); 2].into();
                        let output_row = theta_b.internal_output_row()?;
                        let mut gamma_b = theta_b.gamma_builder(
                            (predicate_inputs, const_wire),
                            vec![(BIT, b1)],
                            output_row,
                        )?;

                        let mut branch_0 = gamma_b.branch_builder(0)?;
                        let [pred, b1] = branch_0.input_wires_arr();
                        branch_0.discard(pred)?;

                        let continue_wire = branch_0.make_continue(signature.clone(), [b1])?;
                        branch_0.finish_with_outputs([continue_wire])?;

                        let mut branch_1 = gamma_b.branch_builder(1)?;
                        let [pred, b1] = branch_1.input_wires_arr();

                        branch_1.discard(pred)?;
                        branch_1.discard(b1)?;

                        let wire = branch_1.load_const(&s2)?;
                        let break_wire = branch_1.make_break(signature, [wire])?;
                        branch_1.finish_with_outputs([break_wire])?;

                        gamma_b.finish()?
                    };

                    theta_b.finish_with_outputs(gamma.out_wire(0))?
                };

                fbuild.finish_with_outputs(theta.outputs())?
            };
            module_builder.finish()
        };

        assert_matches!(build_result, Ok(_));

        Ok(())
    }

    #[test]
    fn basic_recurse() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_id = module_builder.declare("main", type_row![NAT], type_row![NAT])?;

            let mut f_build = module_builder.define_function(&f_id)?;
            let call = f_build.call(&f_id, f_build.input_wires())?;

            f_build.finish_with_outputs(call.outputs())?;
            module_builder.finish()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn simple_newtype() -> Result<(), BuildError> {
        let inputs = type_row![QB, BIT];
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let qubit_state_type = module_builder
                .add_new_type("qubit_state", SimpleType::new_tuple(inputs.clone()))?;

            let mut f_build = module_builder.declare_and_def(
                "main",
                inputs.clone(),
                vec![qubit_state_type.get_new_type()].into(),
            )?;
            {
                let tuple = f_build.make_tuple(inputs, f_build.input_wires())?;
                let q_s_val = f_build.make_new_type(&qubit_state_type, tuple)?;
                f_build.finish_with_outputs([q_s_val])?;
            }

            module_builder.finish()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    // Scaffolding for copy insertion tests
    fn copy_scaffold<F>(f: F, msg: &'static str) -> Result<(), BuildError>
    where
        F: FnOnce(FunctionBuilder) -> Result<FuncID, BuildError>,
    {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_build =
                module_builder.declare_and_def("main", type_row![BIT], type_row![BIT, BIT])?;

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

            let f_build =
                module_builder.declare_and_def("main", type_row![QB], type_row![QB, QB])?;

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

            let mut f_build =
                module_builder.declare_and_def("main", type_row![BIT], type_row![BIT])?;

            let [i1] = f_build.input_wires_arr();
            let noop = f_build.add_dataflow_op(LeafOp::Noop(BIT), [i1])?;
            let i1 = noop.out_wire(0);

            let mut nested = f_build.delta_builder(vec![], type_row![BIT])?;

            let id = nested.add_dataflow_op(LeafOp::Noop(BIT), [i1])?;

            let nested = nested.finish_with_outputs([id.out_wire(0)])?;

            f_build.finish_with_outputs([nested.out_wire(0)])?;

            module_builder.finish()
        };

        assert_matches!(builder(), Ok(_));
    }
}

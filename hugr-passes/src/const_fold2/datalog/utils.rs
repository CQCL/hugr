use ascent::lattice::{ord_lattice::OrdLattice, BoundedLattice, Dual, Lattice};
use either::Either;
use hugr_core::{
    ops::OpTrait as _,
    partial_value::{PartialValue, ValueHandle},
    types::{EdgeKind, TypeRow},
    HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _,
};
use itertools::zip_eq;

use super::context::DFContext;

#[derive(PartialEq, Eq, Hash, PartialOrd, Clone, Debug)]
pub struct PV(PartialValue);

impl From<PartialValue> for PV {
    fn from(inner: PartialValue) -> Self {
        Self(inner)
    }
}

impl PV {
    pub fn tuple_field_value(&self, idx: usize) -> Self {
        self.variant_field_value(0, idx)
    }

    /// TODO the arguments here are not  pretty, two usizes, better not mix them
    /// up!!!
    pub fn variant_field_value(&self, variant: usize, idx: usize) -> Self {
        self.0.variant_field_value(variant, idx).into()
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.supports_tag(tag)
    }
}

impl From<PV> for PartialValue {
    fn from(value: PV) -> Self {
        value.0
    }
}

impl From<ValueHandle> for PV {
    fn from(inner: ValueHandle) -> Self {
        Self(inner.into())
    }
}

impl Lattice for PV {
    fn meet(self, other: Self) -> Self {
        self.0.meet(other.0).into()
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        self.0.meet_mut(other.0)
    }

    fn join(self, other: Self) -> Self {
        self.0.join(other.0).into()
    }

    fn join_mut(&mut self, other: Self) -> bool {
        self.0.join_mut(other.0)
    }
}

impl BoundedLattice for PV {
    fn bottom() -> Self {
        PartialValue::bottom().into()
    }

    fn top() -> Self {
        PartialValue::top().into()
    }
}

#[derive(PartialEq, Clone, Eq, Hash, PartialOrd)]
pub struct ValueRow(Vec<PV>);

impl ValueRow {
    fn new(len: usize) -> Self {
        Self(vec![PV::bottom(); len])
    }

    fn singleton(len: usize, idx: usize, v: PV) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
    }

    fn singleton_from_row(r: &TypeRow, idx: usize, v: PV) -> Self {
        Self::singleton(r.len(), idx, v)
    }

    fn bottom_from_row(r: &TypeRow) -> Self {
        Self::new(r.len())
    }

    fn iter<'b>(
        &'b self,
        context: &'b impl DFContext,
        n: Node,
    ) -> impl Iterator<Item = (IncomingPort, &PV)> + 'b {
        zip_eq(value_inputs(context, n), self.0.iter())
    }

    // fn initialised(&self) -> bool {
    //     self.0.iter().all(|x| x != &PV::top())
    // }
}

impl Lattice for ValueRow {
    fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
    }

    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn join_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.join_mut(v2);
        }
        changed
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());
        let mut changed = false;
        for (v1, v2) in zip_eq(self.0.iter_mut(), other.0.into_iter()) {
            changed |= v1.meet_mut(v2);
        }
        changed
    }
}

impl IntoIterator for ValueRow {
    type Item = PV;

    type IntoIter = <Vec<PV> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub(super) fn bottom_row(context: &impl DFContext, n: Node) -> ValueRow {
    if let Some(sig) = context.hugr().signature(n) {
        ValueRow::new(sig.input_count())
    } else {
        ValueRow::new(0)
    }
}

pub(super) fn singleton_in_row(
    context: &impl DFContext,
    n: &Node,
    ip: &IncomingPort,
    v: PV,
) -> ValueRow {
    let Some(sig) = context.hugr().signature(*n) else {
        panic!("dougrulz");
    };
    if sig.input_count() <= ip.index() {
        panic!(
            "bad port index: {} >= {}: {}",
            ip.index(),
            sig.input_count(),
            context.hugr().get_optype(*n).description()
        );
    }
    ValueRow::singleton_from_row(&context.hugr().signature(*n).unwrap().input, ip.index(), v)
}

pub(super) fn partial_value_from_load_constant(context: &impl DFContext, node: Node) -> PV {
    let load_op = context.hugr().get_optype(node).as_load_constant().unwrap();
    let const_node = context
        .hugr()
        .single_linked_output(node, load_op.constant_port())
        .unwrap()
        .0;
    let const_op = context.hugr().get_optype(const_node).as_const().unwrap();
    context
        .node_value_handle(const_node, const_op.value())
        .into()
}

pub(super) fn partial_value_tuple_from_value_row(r: ValueRow) -> PV {
    PartialValue::variant(0, r.into_iter().map(|x| x.0)).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IO {
    Input,
    Output,
}

pub(super) fn value_inputs(
    context: &impl DFContext,
    n: Node,
) -> impl Iterator<Item = IncomingPort> + '_ {
    context.hugr().in_value_types(n).map(|x| x.0)
}

pub(super) fn value_outputs(
    context: &impl DFContext,
    n: Node,
) -> impl Iterator<Item = OutgoingPort> + '_ {
    context.hugr().out_value_types(n).map(|x| x.0)
}

// todo this should work for dataflowblocks too
pub(super) fn tail_loop_worker<'a>(
    context: &'a impl DFContext,
    n: Node,
    output_p: IncomingPort,
    control_variant: usize,
    v: &'a PV,
) -> impl Iterator<Item = (OutgoingPort, PV)> + 'a {
    let tail_loop_op = context.hugr().get_optype(n).as_tail_loop().unwrap();
    let num_variant_vals = if control_variant == 0 {
        tail_loop_op.just_inputs.len()
    } else {
        tail_loop_op.just_outputs.len()
    };
    let hugr = context.hugr();
    if output_p.index() == 0 {
        Either::Left(
            (0..num_variant_vals)
                .map(move |i| (i.into(), v.variant_field_value(control_variant, i))),
        )
    } else {
        let v = if v.supports_tag(control_variant) {
            v.clone()
        } else {
            PV::bottom()
        };
        Either::Right(std::iter::once((
            (num_variant_vals + output_p.index() - 1).into(),
            v,
        )))
    }
    .inspect(move |x| {
        assert!(matches!(
            hugr.get_optype(n).port_kind(x.0),
            Some(EdgeKind::Value(_))
        ))
    })
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub enum TailLoopTermination {
    NeverTerminates,
    SingleIteration,
    Terminates,
}

impl TailLoopTermination {
    pub fn from_control_value(v: &PV) -> Self {
        if v.supports_tag(1) && !v.supports_tag(0) {
            Self::SingleIteration
        } else if v.supports_tag(1) {
            Self::Terminates
        } else {
            Self::NeverTerminates
        }
    }
}

impl From<TailLoopTermination> for OrdLattice<TailLoopTermination> {
    fn from(value: TailLoopTermination) -> Self {
        Self(value)
    }
}

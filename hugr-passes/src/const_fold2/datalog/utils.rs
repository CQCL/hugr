// proptest-derive generates many of these warnings.
// https://github.com/rust-lang/rust/issues/120363
// https://github.com/proptest-rs/proptest/issues/447
#![cfg_attr(test, allow(non_local_definitions))]

use std::{cmp::Ordering, ops::Index, sync::Arc};

use ascent::lattice::{BoundedLattice, Lattice};
use itertools::{zip_eq, Either};

use crate::const_fold2::partial_value::{PartialValue, ValueHandle};
use hugr_core::{
    ops::OpTrait as _, types::TypeRow, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _,
};

#[cfg(test)]
use proptest_derive::Arbitrary;

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

    pub fn variant_values(&self, variant: usize, len: usize) -> Option<Vec<PV>> {
        Some(
            self.0
                .variant_values(variant, len)?
                .into_iter()
                .map(PV::from)
                .collect(),
        )
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

    pub fn iter(&self) -> impl Iterator<Item = &PV> {
        self.0.iter()
    }

    pub fn iter_with_ports<'b>(
        &'b self,
        h: &'b impl HugrView,
        n: Node,
    ) -> impl Iterator<Item = (IncomingPort, &PV)> + 'b {
        zip_eq(value_inputs(h, n), self.0.iter())
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

impl<Idx> Index<Idx> for ValueRow
where
    Vec<PV>: Index<Idx>,
{
    type Output = <Vec<PV> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
    }
}

pub(super) fn bottom_row(h: &impl HugrView, n: Node) -> ValueRow {
    if let Some(sig) = h.signature(n) {
        ValueRow::new(sig.input_count())
    } else {
        ValueRow::new(0)
    }
}

pub(super) fn singleton_in_row(h: &impl HugrView, n: &Node, ip: &IncomingPort, v: PV) -> ValueRow {
    let Some(sig) = h.signature(*n) else {
        panic!("dougrulz");
    };
    if sig.input_count() <= ip.index() {
        panic!(
            "bad port index: {} >= {}: {}",
            ip.index(),
            sig.input_count(),
            h.get_optype(*n).description()
        );
    }
    ValueRow::singleton_from_row(&h.signature(*n).unwrap().input, ip.index(), v)
}

pub(super) fn partial_value_from_load_constant(h: &impl HugrView, node: Node) -> PV {
    let load_op = h.get_optype(node).as_load_constant().unwrap();
    let const_node = h
        .single_linked_output(node, load_op.constant_port())
        .unwrap()
        .0;
    let const_op = h.get_optype(const_node).as_const().unwrap();
    ValueHandle::new(const_node.into(), Arc::new(const_op.value().clone())).into()
}

pub(super) fn partial_value_tuple_from_value_row(r: ValueRow) -> PV {
    PartialValue::variant(0, r.into_iter().map(|x| x.0)).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IO {
    Input,
    Output,
}

pub(super) fn value_inputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = IncomingPort> + '_ {
    h.in_value_types(n).map(|x| x.0)
}

pub(super) fn value_outputs(h: &impl HugrView, n: Node) -> impl Iterator<Item = OutgoingPort> + '_ {
    h.out_value_types(n).map(|x| x.0)
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum TailLoopTermination {
    Bottom,
    ExactlyZeroContinues,
    Top,
}

impl TailLoopTermination {
    pub fn from_control_value(v: &PV) -> Self {
        let (may_continue, may_break) = (v.supports_tag(0), v.supports_tag(1));
        if may_break && !may_continue {
            Self::ExactlyZeroContinues
        } else if may_break && may_continue {
            Self::top()
        } else {
            Self::bottom()
        }
    }
}

impl PartialOrd for TailLoopTermination {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            return Some(std::cmp::Ordering::Equal);
        };
        match (self, other) {
            (Self::Bottom, _) => Some(Ordering::Less),
            (_, Self::Bottom) => Some(Ordering::Greater),
            (Self::Top, _) => Some(Ordering::Greater),
            (_, Self::Top) => Some(Ordering::Less),
            _ => None,
        }
    }
}

impl Lattice for TailLoopTermination {
    fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
    }

    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        // let new_self = &mut self;
        match (*self).partial_cmp(&other) {
            Some(Ordering::Greater) => {
                *self = other;
                true
            }
            Some(_) => false,
            _ => {
                *self = Self::Bottom;
                true
            }
        }
    }

    fn join_mut(&mut self, other: Self) -> bool {
        match (*self).partial_cmp(&other) {
            Some(Ordering::Less) => {
                *self = other;
                true
            }
            Some(_) => false,
            _ => {
                *self = Self::Top;
                true
            }
        }
    }
}

impl BoundedLattice for TailLoopTermination {
    fn bottom() -> Self {
        Self::Bottom
    }

    fn top() -> Self {
        Self::Top
    }
}

#[cfg(test)]
#[cfg_attr(test, allow(non_local_definitions))]
mod test {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn bounded_lattice(v: TailLoopTermination) {
            prop_assert!(v <= TailLoopTermination::top());
            prop_assert!(v >= TailLoopTermination::bottom());
        }

        #[test]
        fn meet_join_self_noop(v1: TailLoopTermination) {
            let mut subject = v1.clone();

            assert_eq!(v1.clone(), v1.clone().join(v1.clone()));
            assert!(!subject.join_mut(v1.clone()));
            assert_eq!(subject, v1);

            assert_eq!(v1.clone(), v1.clone().meet(v1.clone()));
            assert!(!subject.meet_mut(v1.clone()));
            assert_eq!(subject, v1);
        }

        #[test]
        fn lattice(v1: TailLoopTermination, v2: TailLoopTermination) {
            let meet = v1.clone().meet(v2.clone());
            prop_assert!(meet <= v1, "meet not less <=: {:#?}", &meet);
            prop_assert!(meet <= v2, "meet not less <=: {:#?}", &meet);

            let join = v1.clone().join(v2.clone());
            prop_assert!(join >= v1, "join not >=: {:#?}", &join);
            prop_assert!(join >= v2, "join not >=: {:#?}", &join);
        }
    }
}

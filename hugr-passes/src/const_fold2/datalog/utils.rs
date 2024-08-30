// proptest-derive generates many of these warnings.
// https://github.com/rust-lang/rust/issues/120363
// https://github.com/proptest-rs/proptest/issues/447
#![cfg_attr(test, allow(non_local_definitions))]

use std::{
    cmp::Ordering,
    ops::{Index, IndexMut},
};

use ascent::lattice::{BoundedLattice, Lattice};
use itertools::zip_eq;

use super::{partial_value::PartialValue, AbstractValue};
use hugr_core::{
    ops::OpTrait as _,
    types::{Signature, TypeRow},
    HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _,
};

#[cfg(test)]
use proptest_derive::Arbitrary;

impl<V: AbstractValue> Lattice for PartialValue<V> {
    fn meet(self, other: Self) -> Self {
        self.meet(other)
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        self.meet_mut(other)
    }

    fn join(self, other: Self) -> Self {
        self.join(other)
    }

    fn join_mut(&mut self, other: Self) -> bool {
        self.join_mut(other)
    }
}

impl<V: AbstractValue> BoundedLattice for PartialValue<V> {
    fn bottom() -> Self {
        Self::bottom()
    }

    fn top() -> Self {
        Self::top()
    }
}

#[derive(PartialEq, Clone, Eq, Hash)]
pub struct ValueRow<V>(Vec<PartialValue<V>>);

impl<V: AbstractValue> ValueRow<V> {
    pub fn new(len: usize) -> Self {
        Self(vec![PartialValue::bottom(); len])
    }

    fn single_among_bottoms(len: usize, idx: usize, v: PartialValue<V>) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
    }

    fn bottom_from_row(r: &TypeRow) -> Self {
        Self::new(r.len())
    }

    pub fn iter(&self) -> impl Iterator<Item = &PartialValue<V>> {
        self.0.iter()
    }

    pub fn unpack_first(
        &self,
        variant: usize,
        len: usize,
    ) -> Option<impl Iterator<Item = PartialValue<V>> + '_> {
        self[0]
            .variant_values(variant, len)
            .map(|vals| vals.into_iter().chain(self.iter().skip(1).cloned()))
    }

    // fn initialised(&self) -> bool {
    //     self.0.iter().all(|x| x != &PV::top())
    // }
}

impl<V> FromIterator<PartialValue<V>> for ValueRow<V> {
    fn from_iter<T: IntoIterator<Item = PartialValue<V>>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<V: PartialEq> PartialOrd for ValueRow<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<V: AbstractValue> Lattice for ValueRow<V> {
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

impl<V> IntoIterator for ValueRow<V> {
    type Item = PartialValue<V>;

    type IntoIter = <Vec<PartialValue<V>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<V, Idx> Index<Idx> for ValueRow<V>
where
    Vec<PartialValue<V>>: Index<Idx>,
{
    type Output = <Vec<PartialValue<V>> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
    }
}

impl<V, Idx> IndexMut<Idx> for ValueRow<V>
where
    Vec<PartialValue<V>>: IndexMut<Idx>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

pub(super) fn bottom_row<V: AbstractValue>(h: &impl HugrView, n: Node) -> ValueRow<V> {
    ValueRow::new(
        h.signature(n)
            .as_ref()
            .map(Signature::input_count)
            .unwrap_or(0),
    )
}

pub(super) fn singleton_in_row<V: AbstractValue>(
    h: &impl HugrView,
    n: &Node,
    ip: &IncomingPort,
    v: PartialValue<V>,
) -> ValueRow<V> {
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
    ValueRow::single_among_bottoms(h.signature(*n).unwrap().input.len(), ip.index(), v)
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
    pub fn from_control_value<V: AbstractValue>(v: &PartialValue<V>) -> Self {
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

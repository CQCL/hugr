// proptest-derive generates many of these warnings.
// https://github.com/rust-lang/rust/issues/120363
// https://github.com/proptest-rs/proptest/issues/447
#![cfg_attr(test, allow(non_local_definitions))]

use std::cmp::Ordering;

use ascent::lattice::{BoundedLattice, Lattice};

use super::super::partial_value::{AbstractValue, PartialValue};
use hugr_core::{types::Signature, HugrView, IncomingPort, Node, OutgoingPort};

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

pub(super) fn input_count(h: &impl HugrView, n: Node) -> usize {
    h.signature(n)
        .as_ref()
        .map(Signature::input_count)
        .unwrap_or(0)
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

// Really this is part of partial_value.rs

use std::{
    cmp::Ordering,
    ops::{Index, IndexMut},
};

use ascent::lattice::{BoundedLattice, Lattice};
use itertools::zip_eq;

use super::partial_value::{AbstractValue, PartialValue};

#[derive(PartialEq, Clone, Eq, Hash)]
pub struct ValueRow<V>(Vec<PartialValue<V>>);

impl<V: AbstractValue> ValueRow<V> {
    pub fn new(len: usize) -> Self {
        Self(vec![PartialValue::bottom(); len])
    }

    pub fn single_known(len: usize, idx: usize, v: PartialValue<V>) -> Self {
        assert!(idx < len);
        let mut r = Self::new(len);
        r.0[idx] = v;
        r
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

// Wrap a (known-length) row of values into a lattice.

use std::{
    cmp::Ordering,
    ops::{Index, IndexMut},
};

use ascent::Lattice;
use itertools::zip_eq;

use super::{AbstractValue, PartialValue};

#[derive(PartialEq, Clone, Debug, Eq, Hash)]
pub(super) struct ValueRow<V, N>(Vec<PartialValue<V, N>>);

impl<V: AbstractValue, N: Clone> ValueRow<V, N> {
    pub fn new(len: usize) -> Self {
        Self(vec![PartialValue::Bottom; len])
    }

    pub fn set(mut self, idx: usize, v: PartialValue<V, N>) -> Self {
        *self.0.get_mut(idx).unwrap() = v;
        self
    }

    pub fn singleton(v: PartialValue<V, N>) -> Self {
        Self(vec![v])
    }

    /// The first value in this `ValueRow` must be a sum;
    /// returns a new `ValueRow` given by unpacking the elements of the specified variant of said first value,
    /// then appending the rest of the values in this row.
    pub fn unpack_first(
        &self,
        variant: usize,
        len: usize,
    ) -> Option<impl Iterator<Item = PartialValue<V, N>> + use<V, N>> {
        let vals = self[0].variant_values(variant, len)?;
        Some(vals.into_iter().chain(self.0[1..].to_owned()))
    }
}

impl<V, N> FromIterator<PartialValue<V, N>> for ValueRow<V, N> {
    fn from_iter<T: IntoIterator<Item = PartialValue<V, N>>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<V: PartialEq, N: PartialEq + PartialOrd> PartialOrd for ValueRow<V, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<V: AbstractValue, N: PartialEq + PartialOrd> Lattice for ValueRow<V, N> {
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

impl<V, N> IntoIterator for ValueRow<V, N> {
    type Item = PartialValue<V, N>;

    type IntoIter = <Vec<PartialValue<V, N>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<V, N, Idx> Index<Idx> for ValueRow<V, N>
where
    Vec<PartialValue<V, N>>: Index<Idx>,
{
    type Output = <Vec<PartialValue<V, N>> as Index<Idx>>::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        self.0.index(index)
    }
}

impl<V, N, Idx> IndexMut<Idx> for ValueRow<V, N>
where
    Vec<PartialValue<V, N>>: IndexMut<Idx>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

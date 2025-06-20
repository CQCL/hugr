use itertools::Itertools;
use std::{
    fmt::{self, Write as _},
    ops::Index,
    sync::Arc,
};
use thiserror::Error;

use crate::utils::display_list;

use super::{Substitution, Term, Type, type_param::SeqPart};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Hash)]
pub struct ClosedList<T>(Arc<[T]>);

impl<T> ClosedList<T> {
    pub fn new(items: impl IntoIterator<Item = T>) -> Self {
        let items: Vec<_> = items.into_iter().collect();
        Self(items.into())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.0.iter()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index)
    }
}

impl<T> Default for ClosedList<T> {
    fn default() -> Self {
        Self(Arc::default())
    }
}

impl<T> fmt::Display for ClosedList<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.iter(), f)?;
        f.write_char(']')
    }
}

impl<T> From<ClosedList<T>> for Term
where
    T: Into<Term> + Clone,
{
    fn from(value: ClosedList<T>) -> Self {
        Term::new_list(value.0.iter().cloned().map_into())
    }
}

impl<T> From<Vec<T>> for ClosedList<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value.into())
    }
}

impl<T, const N: usize> From<[T; N]> for ClosedList<T> {
    fn from(value: [T; N]) -> Self {
        Self(value.into())
    }
}

impl<T> Index<usize> for ClosedList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl ClosedList<Type> {
    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self::new(self.iter().map(|t| t.substitute1(tr)))
    }
}

impl ClosedList<Term> {
    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self::new(self.iter().map(|t| t.substitute(tr)))
    }
}

#[derive(Debug, Clone, Error)]
pub enum ClosedListError<E> {
    #[error("expected closed list")]
    NotClosed,
    #[error(transparent)]
    Item(E),
}

impl<T> TryFrom<Term> for ClosedList<T>
where
    T: TryFrom<Term>,
{
    type Error = ClosedListError<T::Error>;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        let items: Vec<_> = value
            .into_list_parts()
            .map(|part| match part {
                SeqPart::Item(item) => Ok(item.try_into().map_err(ClosedListError::Item)?),
                SeqPart::Splice(_) => Err(ClosedListError::NotClosed),
            })
            .try_collect()?;
        Ok(Self(items.into()))
    }
}

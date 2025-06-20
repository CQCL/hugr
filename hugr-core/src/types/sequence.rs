use itertools::Itertools;
use std::{ops::Index, sync::Arc};
use thiserror::Error;

use super::{Term, type_param::SeqPart};

#[derive(Debug, Clone, Default)]
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
}

impl<T> From<ClosedList<T>> for Term
where
    T: Into<Term> + Clone,
{
    fn from(value: ClosedList<T>) -> Self {
        Term::new_list(value.0.iter().cloned().map_into())
    }
}

impl<T> Index<usize> for ClosedList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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

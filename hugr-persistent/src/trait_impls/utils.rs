use std::iter::FusedIterator;

/// An adapter that allows an `Iterator` to implement `DoubleEndedIterator`
/// by collecting its items when `next_back` is called.
///
/// If `next_back` is never called, the original iterator is used directly,
/// avoiding the overhead of collecting items into a vector.
#[derive(Clone, Debug)]
pub enum DoubleEndedIteratorAdapter<I: Iterator> {
    Iter(I),
    CollectedIter(std::vec::IntoIter<I::Item>),
}

impl<I: Iterator> DoubleEndedIteratorAdapter<I> {
    /// Creates a new `DoubleEndedIteratorAdapter` from the given iterator.
    pub fn new(iter: I) -> Self {
        Self::Iter(iter)
    }

    fn collect_self(&mut self) {
        match self {
            DoubleEndedIteratorAdapter::Iter(iter) => {
                let collected: Vec<I::Item> = iter.collect();
                *self = DoubleEndedIteratorAdapter::CollectedIter(collected.into_iter());
            }
            DoubleEndedIteratorAdapter::CollectedIter(_) => {}
        }
    }
}

impl<I: Iterator> From<I> for DoubleEndedIteratorAdapter<I> {
    fn from(iter: I) -> Self {
        Self::new(iter)
    }
}

impl<I: Iterator> Iterator for DoubleEndedIteratorAdapter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DoubleEndedIteratorAdapter::Iter(iter) => iter.next(),
            DoubleEndedIteratorAdapter::CollectedIter(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            DoubleEndedIteratorAdapter::Iter(iter) => iter.size_hint(),
            DoubleEndedIteratorAdapter::CollectedIter(iter) => iter.size_hint(),
        }
    }
}

impl<I: FusedIterator> FusedIterator for DoubleEndedIteratorAdapter<I> {}

impl<I: Iterator> DoubleEndedIterator for DoubleEndedIteratorAdapter<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.collect_self();
        match self {
            DoubleEndedIteratorAdapter::CollectedIter(iter) => iter.next_back(),
            _ => unreachable!("just collected self"),
        }
    }
}

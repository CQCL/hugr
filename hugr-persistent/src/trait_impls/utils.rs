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
#[cfg(test)]
mod tests {
    use super::*;
    use hugr_core::Direction;
    use rstest::rstest;

    #[rstest]
    #[case(Direction::Outgoing, Box::new(0..3) as Box<dyn Iterator<Item = i32>>, vec![0, 1, 2])]
    #[case(Direction::Incoming, Box::new(0..3) as Box<dyn Iterator<Item = i32>>, vec![2, 1, 0])]
    #[case(Direction::Outgoing, Box::new(std::iter::empty()) as Box<dyn Iterator<Item = i32>>, vec![])]
    #[case(Direction::Incoming, Box::new(std::iter::empty()) as Box<dyn Iterator<Item = i32>>, vec![])]
    #[case(Direction::Outgoing, Box::new(std::iter::once(42)) as Box<dyn Iterator<Item = i32>>, vec![42])]
    #[case(Direction::Incoming, Box::new(std::iter::once(42)) as Box<dyn Iterator<Item = i32>>, vec![42])]
    fn test_double_ended_iterator_adapter(
        #[case] direction: Direction,
        #[case] iter: Box<dyn Iterator<Item = i32>>,
        #[case] expected: Vec<i32>,
    ) {
        let adapter = DoubleEndedIteratorAdapter::from(iter);
        let collected: Vec<_> = match direction {
            Direction::Outgoing => adapter.collect(),
            Direction::Incoming => adapter.rev().collect(),
        };
        assert_eq!(collected, expected);
    }
}

use fxhash::FxHasher;
use hugr_model::v0::{Literal, SymbolName, VarName};
use servo_arc::ThinArc;
use std::hash::{Hash, Hasher};

use crate::Node;

/// ...
///
/// To pattern match the values of a term use [`Term::view`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term(ThinArc<TermHeader, Term>);

/// The constant-sized part of the [`Term`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TermData {
    Apply(SymbolName),
    Literal(Literal),
    List,
    ListConcat,
    Var(Var),
}

impl TermData {
    pub fn split<'a>(view: TermView<'a>) -> (Self, &'a [Term]) {
        match view {
            TermView::Apply(symbol, terms) => (TermData::Apply(symbol.clone()), terms),
            TermView::Literal(literal) => (TermData::Literal(literal.clone()), &[] as &[_]),
            TermView::List(terms) => (TermData::List, terms),
            TermView::ListConcat(terms) => (TermData::ListConcat, terms),
            TermView::Var(var) => (TermData::Var(var.clone()), &[] as &[_]),
        }
    }
}

/// The constant-sized part of the [`Term`] together with derived data.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TermHeader {
    /// Cached hash code (derived).
    ///
    /// With a cached hash code we can calculate a [`Term`]s hash quickly
    /// without having to traverse the entire term. The cached hash code can
    /// also speed up the comparison of terms in the case where the compared
    /// terms are not equal.
    hash: u64,

    /// The constant-sized part without derived data.
    data: TermData,
}

impl TermHeader {
    pub fn new(data: TermData, terms: &[Term]) -> Self {
        let hash = {
            let mut hasher = FxHasher::default();
            data.hash(&mut hasher);
            terms.hash(&mut hasher);
            hasher.finish()
        };

        Self { data, hash }
    }
}

impl Term {
    /// Create a new [`Term`].
    pub fn new(view: TermView) -> Self {
        let (data, terms) = TermData::split(view);
        Self(ThinArc::from_header_and_iter(
            TermHeader::new(data, terms),
            terms.iter().cloned(),
        ))
    }

    /// Create a new static [`Term`] without reference counting.
    ///
    /// Terms created with this method will never be deallocated. This is useful
    /// for [`Term`]s that are known at Rust compile time.
    pub fn new_static(view: TermView) -> Self {
        let (data, terms) = TermData::split(view);
        Self(ThinArc::from_header_and_iter_alloc(
            |layout| unsafe { std::alloc::alloc(layout) },
            TermHeader::new(data, terms),
            terms.iter().cloned(),
            terms.len(),
            true,
        ))
    }

    /// Borrow this term's data as a [`TermView`].
    #[inline]
    pub fn view(&self) -> TermView {
        match &self.0.header.data {
            TermData::Apply(symbol) => TermView::Apply(symbol, self.0.slice()),
            TermData::Literal(literal) => TermView::Literal(literal),
            TermData::List => TermView::List(self.0.slice()),
            TermData::ListConcat => TermView::ListConcat(self.0.slice()),
            TermData::Var(var) => TermView::Var(var),
        }
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.header.hash);
    }
}

/// A borrowed view into a [`Term`].
#[derive(Debug, Clone)]
pub enum TermView<'a> {
    Apply(&'a SymbolName, &'a [Term]),
    Literal(&'a Literal),
    List(&'a [Term]),
    ListConcat(&'a [Term]),
    Var(&'a Var),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    name: VarName,
    node: Node,
    index: u16,
}

use fxhash::FxHasher;
use hugr_model::v0::ast;
use hugr_model::v0::{Literal, SymbolName, VarName};
use once_cell::sync::Lazy;
use servo_arc::ThinArc;
use std::error::Error;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use thiserror::Error;
use tinyvec::TinyVec;

use crate::Node;

pub mod core;
pub mod util;

/// A term in the static language that is used to parameterise `Hugr`s.
///
/// To pattern match the values of a term use [`Term::get`].
///
/// Terms are immutable and have no identity beyond structural equality. Two
/// structurally equal terms should be considered logically indistinguishable.
/// Equal terms can share their representation and this sharing can be observed
/// to enable optimisations.
///
/// Terms can be cloned cheaply without requiring a copy of the term's data.
/// Equality comparisons and hash codes are optimised so that [`Term`]s can be
/// used as keys in hash tables.
///
/// A term has a unique type within the context of a hugr. In particular, local
/// variables refer to their binding site. The same term may have different
/// types when interpreted in the context of different hugrs. This enables
/// caching types and constraints in a side table instead of requiring them to
/// be stored inline with the term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term(ThinArc<TermHeader, Term>);

/// Assert that [`Term`] is the same size as a pointer.
const _: () = assert!(std::mem::size_of::<Term>() == std::mem::size_of::<usize>());

impl From<Term> for ast::Term {
    fn from(value: Term) -> Self {
        Self::from(value.get())
    }
}

impl From<&Term> for ast::Term {
    fn from(value: &Term) -> Self {
        Self::from(value.get())
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.get(), f)
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.header.hash);
    }
}

/// The constant-sized part of the [`Term`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
enum TermData {
    #[default]
    Wildcard,
    Apply(SymbolName),
    Literal(Literal),
    ListEmpty(Term),
    ListCons(Term, Term),
    ListConcat(Term, Term),
    Tuple,
    TupleConcat,
    Var(Var),
}

impl TermData {
    pub fn split<'a>(view: TermKind<'a>) -> (Self, &'a [Term]) {
        match view {
            TermKind::Wildcard => (TermData::Wildcard, &[]),
            TermKind::Apply(symbol, terms) => (TermData::Apply(symbol.clone()), terms),
            TermKind::Literal(literal) => (TermData::Literal(literal.clone()), &[] as &[_]),
            TermKind::List(_) => todo!(),
            TermKind::Tuple(terms) => (TermData::Tuple, terms),
            TermKind::TupleConcat(terms) => (TermData::TupleConcat, terms),
            TermKind::Var(var) => (TermData::Var(var.clone()), &[] as &[_]),
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
    pub fn new_apply(symbol: &SymbolName, args: &[Term]) -> Self {
        Self::new(TermData::Apply(symbol.clone()), args)
    }

    pub fn new_apply_static(symbol: &SymbolName, args: &[Term]) -> Self {
        Self::new_static(TermData::Apply(symbol.clone()), args)
    }

    /// Create a new [`Term`].
    fn new(data: TermData, terms: &[Term]) -> Self {
        // TODO: Normalise list and tuple terms
        Self(ThinArc::from_header_and_iter(
            TermHeader::new(data, terms),
            terms.iter().cloned(),
        ))
    }

    /// Create a new static [`Term`] without reference counting.
    ///
    /// Terms created with this method will never be deallocated. This is useful
    /// for [`Term`]s that are known at Rust compile time.
    fn new_static(data: TermData, terms: &[Term]) -> Self {
        Self(ThinArc::from_header_and_iter_alloc(
            |layout| unsafe { std::alloc::alloc(layout) },
            TermHeader::new(data, terms),
            terms.iter().cloned(),
            terms.len(),
            true,
        ))
    }

    /// Borrow this term's data as a [`TermKind`].
    #[inline]
    pub fn get(&self) -> TermKind {
        match &self.0.header.data {
            TermData::Wildcard => TermKind::Wildcard,
            TermData::Apply(symbol) => TermKind::Apply(symbol, self.0.slice()),
            TermData::Literal(literal) => TermKind::Literal(literal),
            TermData::ListEmpty(_) => TermKind::List(List(self.clone())),
            TermData::ListCons(_, _) => TermKind::List(List(self.clone())),
            TermData::ListConcat(_, _) => TermKind::List(List(self.clone())),
            TermData::Tuple => TermKind::Tuple(self.0.slice()),
            TermData::TupleConcat => TermKind::TupleConcat(self.0.slice()),
            TermData::Var(var) => TermKind::Var(var),
        }
    }

    #[inline]
    pub fn view<V: View>(&self) -> Result<V, ViewError> {
        V::view(self)
    }

    pub fn view_apply<const N: usize>(&self, symbol: &SymbolName) -> Result<[Term; N], ViewError> {
        let TermKind::Apply(term_symbol, term_args) = self.get() else {
            return Err(ViewError::Mismatch);
        };

        if symbol != term_symbol {
            return Err(ViewError::Mismatch);
        }

        if term_args.len() > N {
            return Err(ViewError::Mismatch);
        }

        let args = std::array::from_fn(|i| {
            (i + term_args.len())
                .checked_sub(N)
                .map(|i| term_args[i].clone())
                .unwrap_or_default()
        });

        Ok(args)
    }

    pub fn view_list_prefix<const N: usize>(&self) -> Result<([Term; N], Term), ViewError> {
        todo!()
    }

    pub fn view_list_exact<const N: usize>(&self) -> Result<[Term; N], ViewError> {
        todo!()
    }
}

impl From<Literal> for Term {
    fn from(value: Literal) -> Self {
        Term::new(TermData::Literal(value), &[])
    }
}

/// The default [`Term`] is a wildcard term.
///
/// ```
/// # use hugr_core::terms::{Term, TermKind};
/// assert_eq!(Term::default().get(), TermKind::Wildcard);
/// ```
impl Default for Term {
    fn default() -> Self {
        static WILDCARD: Lazy<Term> = Lazy::new(|| Term::new_static(TermData::Wildcard, &[]));
        WILDCARD.clone()
    }
}

/// A borrowed view into a [`Term`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TermKind<'a> {
    #[default]
    Wildcard,
    Apply(&'a SymbolName, &'a [Term]),
    Literal(&'a Literal),
    List(List),
    Tuple(&'a [Term]),
    TupleConcat(&'a [Term]),
    Var(&'a Var),
}

impl From<TermKind<'_>> for ast::Term {
    fn from(value: TermKind<'_>) -> Self {
        match value {
            TermKind::Wildcard => ast::Term::Wildcard,
            TermKind::Apply(symbol, args) => {
                let symbol = symbol.clone();
                let args = args.iter().map(ast::Term::from).collect();
                ast::Term::Apply(symbol, args)
            }
            TermKind::Literal(literal) => ast::Term::Literal(literal.clone()),
            TermKind::List(list) => list.into(),
            TermKind::Var(var) => ast::Term::Var(var.name().clone()),
            TermKind::Tuple(items) => ast::Term::Tuple(
                items
                    .iter()
                    .map(|item| ast::SeqPart::Item(item.into()))
                    .collect(),
            ),
            TermKind::TupleConcat(lists) => ast::Term::Tuple(
                lists
                    .iter()
                    .map(|list| ast::SeqPart::Splice(list.into()))
                    .collect(),
            ),
        }
    }
}

impl Display for TermKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let term: ast::Term = self.clone().into();
        Display::fmt(&term, f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct List(Term);

impl From<List> for ast::Term {
    fn from(value: List) -> Self {
        ast::Term::List(value.iter().map(ast::SeqPart::from).collect())
    }
}

impl From<List> for Term {
    fn from(value: List) -> Self {
        value.0
    }
}

impl List {
    pub fn new<I>(parts: I, item_type: Term) -> Self
    where
        I: IntoIterator<Item = SeqPart>,
    {
        Self::new_prepend(parts, Self::new_empty(item_type).into())
    }

    fn new_prepend<I>(parts: I, tail: Term) -> Self
    where
        I: IntoIterator<Item = SeqPart>,
    {
        let mut parts: TinyVec<[SeqPart; 8]> = parts.into_iter().collect();
        let mut list = tail;

        while let Some(part) = parts.pop() {
            match part {
                SeqPart::Item(term) => list = Term::new(TermData::ListCons(term, list), &[]),
                SeqPart::Splice(term) => match term.get() {
                    TermKind::List(list) => parts.extend(list.iter()),
                    _ => list = Term::new(TermData::ListConcat(term, list), &[]),
                },
            }
        }

        Self(list)
    }

    pub fn new_empty(item_type: Term) -> Self {
        Self(Term::new(TermData::ListEmpty(item_type), &[]))
    }

    pub fn new_cons(head: Term, tail: Term) -> Self {
        Self(Term::new(TermData::ListCons(head, tail), &[]))
    }

    pub fn new_concat(first: Term, second: Term) -> Self {
        Self::new_prepend([SeqPart::Splice(first)], second)
    }

    pub fn iter(&self) -> ListIter {
        ListIter {
            term: self.0.clone(),
        }
    }
}

impl IntoIterator for List {
    type Item = SeqPart;
    type IntoIter = ListIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct ListIter {
    term: Term,
}

impl Iterator for ListIter {
    type Item = SeqPart;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.term.0.header.data {
            TermData::ListEmpty(_) => None,
            TermData::ListCons(head, tail) => {
                let item = head.clone();
                self.term = tail.clone();
                Some(SeqPart::Item(item))
            }
            TermData::ListConcat(splice, rest) => {
                let splice = splice.clone();
                self.term = rest.clone();
                Some(SeqPart::Splice(splice))
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SeqPart {
    Item(Term),
    Splice(Term),
}

impl From<SeqPart> for ast::SeqPart {
    fn from(value: SeqPart) -> Self {
        match value {
            SeqPart::Item(term) => ast::SeqPart::Item(term.into()),
            SeqPart::Splice(term) => ast::SeqPart::Item(term.into()),
        }
    }
}

impl Default for SeqPart {
    fn default() -> Self {
        Self::Splice(Term::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    name: VarName,
    node: Node,
    index: u16,
}

impl Var {
    #[inline]
    pub fn name(&self) -> &VarName {
        &self.name
    }

    #[inline]
    pub fn node(&self) -> Node {
        self.node
    }

    #[inline]
    pub fn index(&self) -> u16 {
        self.index
    }
}

pub trait View: Sized {
    fn view(term: &Term) -> Result<Self, ViewError>;
}

impl View for Term {
    fn view(term: &Term) -> Result<Self, ViewError> {
        Ok(term.clone())
    }
}

impl View for u64 {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.get() {
            TermKind::Literal(literal) => match literal {
                Literal::Nat(value) => Ok(*value),
                _ => Err(ViewError::Mismatch),
            },
            _ => Err(ViewError::Mismatch),
        }
    }
}

impl From<u64> for Term {
    fn from(value: u64) -> Self {
        Literal::Nat(value).into()
    }
}

#[derive(Debug, Error)]
pub enum ViewError {
    #[error("the term does not match the pattern of the view")]
    Mismatch,
    // TODO: Error type to account for situations in which the term is too incomplete,
    // e.g. by containing variables or uninferred wildcards.
}

type AnyError = Box<dyn Error + Send + Sync>;

/// Constructor is applied to wrong number of arguments.
#[derive(Debug, Error)]
#[error(
    "`{constructor}` expects `{expected}` arguments but got `{actual}` in term:\n```\n{term}\n```"
)]
pub struct ArityError {
    /// The number of arguments that the constructor expects.
    pub expected: usize,
    /// The number of arguments that were passed to the constructor.
    pub actual: usize,
    /// The term that caused the error.
    pub term: Term,
    /// The constructor that is applied to the wrong number of arguments.
    pub constructor: SymbolName,
}

/// There is an error within a field of a constructor.
#[derive(Debug, Error)]
#[error(
    "invalid field `{name}` (index {index}) of constructor `{constructor}` in term:\n```\n{term}\n```"
)]
pub struct FieldError {
    /// The original error in the field.
    #[source]
    pub error: AnyError,
    /// The index of the field within the constructor's parameter list.
    pub index: usize,
    /// The name of the field.
    pub name: VarName,
    /// The name of the constructor.
    pub constructor: SymbolName,
    /// The constructor application that caused the error.
    pub term: Term,
}

/// A particular term constructor was expected.
#[derive(Debug, Error)]
#[error("expected constructor `{constructor}` but got term:\n```\n{term}\n```")]
pub struct ConstructorError {
    /// The constructor that was expected.
    pub constructor: SymbolName,
    /// The term that caused the error.
    pub term: Term,
}

#[macro_export]
macro_rules! term_view_ctr {
    (
        $name:expr;
        $(#[$attr:meta])*
        $vis:vis struct $ident:ident {
            $($(#[$field_meta:meta])* pub $field_name:ident: $field_type:ty,)*
        }
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $ident {
            $($(#[$field_meta])* pub $field_name: $field_type,)*
        }

        impl $ident {
            /// The name of the term constructor for this type.
            pub const CTR_NAME: ::hugr_model::v0::SymbolName = ::hugr_model::v0::SymbolName::new_static($name);
        }

        impl $crate::terms::View for $ident {
            fn view(term: &$crate::terms::Term) -> Result<Self, $crate::terms::ViewError> {
                let [$($field_name),*] = term.view_apply(&Self::CTR_NAME)?;
                $(let $field_name = $field_name.view()?;)*
                Ok(Self { $($field_name),* })
            }
        }

        impl From<$ident> for $crate::terms::Term {
            fn from(value: $ident) -> Self {
                $crate::terms::Term::new_apply(
                    &$ident::CTR_NAME,
                    &[$(value.$field_name.into()),*],
                )
            }
        }
    };

    (
        $name:expr;
        $(#[$attr:meta])*
        $vis:vis struct $ident:ident;
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy)]
        pub struct $ident;

        impl $ident {
            /// The name of the term constructor for this type.
            pub const CTR_NAME: ::hugr_model::v0::SymbolName = ::hugr_model::v0::SymbolName::new_static($name);
        }

        impl $crate::terms::View for $ident {
            fn view(term: &$crate::terms::Term) -> Result<Self, $crate::terms::ViewError> {
                let [] = term.view_apply(&Self::CTR_NAME)?;
                Ok(Self)
            }
        }

        impl From<$ident> for $crate::terms::Term {
            fn from(_: $ident) -> Self {
                static TERM: ::once_cell::sync::Lazy<$crate::terms::Term> =
                    ::once_cell::sync::Lazy::new(|| {
                        $crate::terms::Term::new_apply_static(&$ident::CTR_NAME, &[])
                    });
                TERM.clone()
            }
        }
    };
}
